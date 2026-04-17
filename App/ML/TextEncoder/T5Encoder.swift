// T5Encoder.swift — Phase 1
//
// T5-v1.1-XXL encoder for LTX-Video text conditioning.
// Property names match the quantized safetensors key structure exactly:
//   encoder.block.{N}.attention.{q,k,v,o}  (QuantizedLinear)
//   encoder.block.{N}.dense.{wi_0,wi_1,wo} (QuantizedLinear)
//   encoder.block.{N}.ln1, ln2             (weight only)
//   encoder.embed_tokens                    (QuantizedEmbedding)
//   encoder.relative_attention_bias.embeddings (Embedding)
//   encoder.final_layer_norm               (weight only)
//
// Peak memory (4-bit): ~2.68 GB

import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct T5Config: Sendable {
    public let vocabSize: Int
    public let dModel: Int
    public let dKV: Int
    public let dFF: Int
    public let numHeads: Int
    public let numLayers: Int
    public let relativeAttentionNumBuckets: Int
    public let relativeAttentionMaxDistance: Int
    public let layerNormEpsilon: Float

    public static let xxl = T5Config(
        vocabSize: 32128, dModel: 4096, dKV: 64, dFF: 10240,
        numHeads: 64, numLayers: 24,
        relativeAttentionNumBuckets: 32, relativeAttentionMaxDistance: 128,
        layerNormEpsilon: 1e-6
    )
}

// MARK: - RMS Layer Norm

public class T5LayerNorm: Module {
    let weight: MLXArray
    let eps: Float

    public init(dModel: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dModel])
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let variance: MLXArray = (x * x).mean(axis: -1, keepDims: true)
        return weight * (x * rsqrt(variance + eps))
    }
}

// MARK: - Relative Position Bias

public class T5RelativePositionBiasEmbeddings: Module {
    let weight: MLXArray  // [numBuckets, numHeads] = [32, 64]
    let numBuckets: Int
    let maxDistance: Int
    let numHeads: Int

    public init(numBuckets: Int = 32, maxDistance: Int = 128, numHeads: Int = 64) {
        self.numBuckets = numBuckets
        self.maxDistance = maxDistance
        self.numHeads = numHeads
        self.weight = MLXArray.zeros([numBuckets, numHeads])
    }

    public func callAsFunction(queryLen: Int, keyLen: Int) -> MLXArray {
        let contextPos: MLXArray = MLXArray(Array(0..<queryLen).map { Int32($0) })
        let memoryPos: MLXArray = MLXArray(Array(0..<keyLen).map { Int32($0) })
        let relativePosition: MLXArray = expandedDimensions(memoryPos, axis: 0) - expandedDimensions(contextPos, axis: 1)

        let numBucketsHalf: Int = numBuckets / 2
        let maxExact: Int = numBucketsHalf / 2

        let isNegative: MLXArray = relativePosition .< MLXArray(Int32(0))
        let absPos: MLXArray = abs(relativePosition)
        let isSmall: MLXArray = absPos .< MLXArray(Int32(maxExact))

        let logPos: MLXArray = log(absPos.asType(.float32) / Float(maxExact))
        let logMax: Float = log(Float(maxDistance) / Float(maxExact))
        let largeBuckets: MLXArray = (logPos / logMax * Float(numBucketsHalf - maxExact)).asType(.int32)
        let clampedLarge: MLXArray = minimum(largeBuckets, MLXArray(Int32(numBucketsHalf - 1)))
        let largeFinal: MLXArray = clampedLarge + MLXArray(Int32(maxExact))

        let absBuckets: MLXArray = which(isSmall, absPos.asType(.int32), largeFinal)
        let buckets: MLXArray = which(isNegative, absBuckets + MLXArray(Int32(numBucketsHalf)), absBuckets)

        // Look up: [queryLen, keyLen] -> index into [numBuckets, numHeads]
        let flat: MLXArray = buckets.reshaped(-1)
        let values: MLXArray = weight[flat].reshaped(queryLen, keyLen, numHeads)
        return expandedDimensions(values.transposed(2, 0, 1), axis: 0)
    }
}

// Wrapper to match key path: encoder.relative_attention_bias.embeddings
public class T5RelativeAttentionBias: Module {
    let embeddings: T5RelativePositionBiasEmbeddings

    public init(config: T5Config) {
        self.embeddings = T5RelativePositionBiasEmbeddings(
            numBuckets: config.relativeAttentionNumBuckets,
            maxDistance: config.relativeAttentionMaxDistance,
            numHeads: config.numHeads
        )
    }
}

// MARK: - T5 Attention

/// Property names match: encoder.block.{N}.attention.{q,k,v,o}
public class T5Attention: Module {
    let q: QuantizedLinear
    let k: QuantizedLinear
    let v: QuantizedLinear
    let o: QuantizedLinear
    let numHeads: Int
    let dKV: Int

    public init(config: T5Config) {
        let innerDim: Int = config.numHeads * config.dKV
        self.q = QuantizedLinear(config.dModel, innerDim, bias: false, groupSize: 64, bits: 4)
        self.k = QuantizedLinear(config.dModel, innerDim, bias: false, groupSize: 64, bits: 4)
        self.v = QuantizedLinear(config.dModel, innerDim, bias: false, groupSize: 64, bits: 4)
        self.o = QuantizedLinear(innerDim, config.dModel, bias: false, groupSize: 64, bits: 4)
        self.numHeads = config.numHeads
        self.dKV = config.dKV
    }

    public func callAsFunction(_ x: MLXArray, positionBias: MLXArray) -> MLXArray {
        let batchSize: Int = x.dim(0)
        let seqLen: Int = x.dim(1)

        let qProj: MLXArray = q(x).reshaped(batchSize, seqLen, numHeads, dKV).transposed(0, 2, 1, 3)
        let kProj: MLXArray = k(x).reshaped(batchSize, seqLen, numHeads, dKV).transposed(0, 2, 1, 3)
        let vProj: MLXArray = v(x).reshaped(batchSize, seqLen, numHeads, dKV).transposed(0, 2, 1, 3)

        // T5 does NOT scale by 1/sqrt(d_k)
        var scores: MLXArray = matmul(qProj, kProj.transposed(0, 1, 3, 2))
        scores = scores + positionBias

        let attnWeights: MLXArray = softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        let attnOutput: MLXArray = matmul(attnWeights, vProj)
        let merged: MLXArray = attnOutput.transposed(0, 2, 1, 3).reshaped(batchSize, seqLen, numHeads * dKV)

        return o(merged)
    }
}

// MARK: - Gated Feed-Forward

/// Property names match: encoder.block.{N}.dense.{wi_0,wi_1,wo}
public class T5Dense: Module {
    let wi_0: QuantizedLinear
    let wi_1: QuantizedLinear
    let wo: QuantizedLinear

    public init(config: T5Config) {
        self.wi_0 = QuantizedLinear(config.dModel, config.dFF, bias: false, groupSize: 64, bits: 4)
        self.wi_1 = QuantizedLinear(config.dModel, config.dFF, bias: false, groupSize: 64, bits: 4)
        self.wo = QuantizedLinear(config.dFF, config.dModel, bias: false, groupSize: 64, bits: 4)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return wo(gelu(wi_0(x)) * wi_1(x))
    }
}

// MARK: - T5 Encoder Block

/// Property names match: encoder.block.{N}.{attention, dense, ln1, ln2}
public class T5Block: Module {
    let attention: T5Attention
    let dense: T5Dense
    let ln1: T5LayerNorm  // pre-attention norm
    let ln2: T5LayerNorm  // pre-ff norm

    public init(config: T5Config) {
        self.attention = T5Attention(config: config)
        self.dense = T5Dense(config: config)
        self.ln1 = T5LayerNorm(dModel: config.dModel, eps: config.layerNormEpsilon)
        self.ln2 = T5LayerNorm(dModel: config.dModel, eps: config.layerNormEpsilon)
    }

    public func callAsFunction(_ x: MLXArray, positionBias: MLXArray) -> MLXArray {
        // Pre-norm self-attention + residual
        let attnOut: MLXArray = attention(ln1(x), positionBias: positionBias)
        let afterAttn: MLXArray = x + attnOut

        // Pre-norm feed-forward + residual
        let ffOut: MLXArray = dense(ln2(afterAttn))
        return afterAttn + ffOut
    }
}

// MARK: - T5 Encoder

/// Complete encoder. Property structure matches safetensors keys:
///   encoder.embed_tokens.{weight,scales,biases}
///   encoder.block.{0-23}.{attention,dense,ln1,ln2}
///   encoder.relative_attention_bias.embeddings.weight
///   encoder.final_layer_norm.weight
public class T5Encoder: Module {
    let embed_tokens: QuantizedEmbedding
    let block: [T5Block]
    let relative_attention_bias: T5RelativeAttentionBias
    let final_layer_norm: T5LayerNorm
    let config: T5Config

    public init(config: T5Config = .xxl) {
        self.config = config
        self.embed_tokens = QuantizedEmbedding(
            embeddingCount: config.vocabSize, dimensions: config.dModel,
            groupSize: 64, bits: 4
        )

        var blocks: [T5Block] = []
        for _ in 0..<config.numLayers {
            blocks.append(T5Block(config: config))
        }
        self.block = blocks

        self.relative_attention_bias = T5RelativeAttentionBias(config: config)
        self.final_layer_norm = T5LayerNorm(dModel: config.dModel, eps: config.layerNormEpsilon)
    }

    /// Encode token IDs to hidden states.
    /// - Parameter inputIds: [batch, seqLen] Int32
    /// - Returns: [batch, seqLen, 4096]
    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let seqLen: Int = inputIds.dim(1)
        var hidden: MLXArray = embed_tokens(inputIds)

        // Compute position bias once from block 0's perspective
        let positionBias: MLXArray = relative_attention_bias.embeddings(
            queryLen: seqLen, keyLen: seqLen
        )

        for blk in block {
            hidden = blk(hidden, positionBias: positionBias)
        }

        return final_layer_norm(hidden)
    }
}

// MARK: - Top-level wrapper

/// Matches key path root: "encoder.{...}"
public class T5EncoderModel: Module {
    let encoder: T5Encoder

    public init(config: T5Config = .xxl) {
        self.encoder = T5Encoder(config: config)
    }

    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        return encoder(inputIds)
    }
}
