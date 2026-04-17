// T5Encoder.swift — Phase 1
//
// Port of T5-v1.1-XXL encoder for LTX-Video text conditioning.
// Reference: HuggingFace transformers T5EncoderModel
// Weight format: safetensors with keys like "encoder.block.{N}.layer.{L}.SelfAttention.q.weight"
//
// Architecture: Pre-norm encoder-only transformer
// - RMS LayerNorm (no bias, no mean subtraction)
// - Relative position bias (only in block 0, reused by all blocks)
// - Gated GeLU feed-forward (wi_0 * gelu, wi_1, wo)
// - No sqrt(d_k) scaling in attention (T5 convention)
//
// Memory estimate: ~2.38 GB at 4-bit quantization for all 4.76B parameters.

import Foundation
import MLX
import MLXNN

// MARK: - Configuration

/// T5-v1.1-XXL encoder configuration.
/// All values from google/t5-v1_1-xxl config.json.
public struct T5Config {
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
        vocabSize: 32128,
        dModel: 4096,
        dKV: 64,
        dFF: 10240,
        numHeads: 64,
        numLayers: 24,
        relativeAttentionNumBuckets: 32,
        relativeAttentionMaxDistance: 128,
        layerNormEpsilon: 1e-6
    )
}

// MARK: - RMS Layer Norm

/// T5-style RMS normalization: no bias, no mean subtraction.
/// weight * (x / rms(x)), where rms(x) = sqrt(mean(x^2) + eps)
public class T5LayerNorm: Module {
    let weight: MLXArray  // [dModel]
    let eps: Float

    public init(dModel: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dModel])
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Variance = mean(x^2) over last dimension
        let variance: MLXArray = (x * x).mean(axis: -1, keepDims: true)
        // RMS normalize then scale by weight
        let normalized: MLXArray = x * rsqrt(variance + eps)
        return weight * normalized
    }
}

// MARK: - Relative Position Bias

/// Computes relative position bias for T5 self-attention.
/// Only instantiated in block 0; other blocks reuse its output.
public class T5RelativePositionBias: Module {
    let embedding: Embedding  // [numBuckets, numHeads]
    let numBuckets: Int
    let maxDistance: Int
    let numHeads: Int
    let bidirectional: Bool

    public init(numBuckets: Int = 32, maxDistance: Int = 128,
                numHeads: Int = 64, bidirectional: Bool = true) {
        self.numBuckets = numBuckets
        self.maxDistance = maxDistance
        self.numHeads = numHeads
        self.bidirectional = bidirectional
        self.embedding = Embedding(embeddingCount: numBuckets, dimensions: numHeads)
    }

    /// Compute relative position bucket indices.
    /// Returns: [queryLen, keyLen] of Int32 bucket indices
    private func relativeBuckets(queryLen: Int, keyLen: Int) -> MLXArray {
        // Build relative position matrix: context[i] - memory[j]
        let contextPos: MLXArray = MLXArray(Array(0..<queryLen), dtype: .int32)
        let memoryPos: MLXArray = MLXArray(Array(0..<keyLen), dtype: .int32)

        // relativePosition[i,j] = memoryPos[j] - contextPos[i]
        let relativePosition: MLXArray = expandedDims(memoryPos, axis: 0) - expandedDims(contextPos, axis: 1)

        var buckets: MLXArray = MLXArray.zeros([queryLen, keyLen], dtype: .int32)

        if bidirectional {
            let numBucketsHalf: Int = numBuckets / 2
            let isNegative: MLXArray = relativePosition .< MLXArray(Int32(0))
            let absPos: MLXArray = abs(relativePosition)

            // Exact buckets for small distances
            let maxExact: Int = numBucketsHalf / 2
            let isSmall: MLXArray = absPos .< MLXArray(Int32(maxExact))

            // Log-spaced buckets for large distances
            let logPos: MLXArray = log(absPos.asType(.float32) / Float(maxExact))
            let logMax: Float = log(Float(maxDistance) / Float(maxExact))
            let largeBuckets: MLXArray = (logPos / logMax * Float(numBucketsHalf - maxExact)).asType(.int32)
            let clampedLarge: MLXArray = minimum(largeBuckets, MLXArray(Int32(numBucketsHalf - 1)))
            let largeFinal: MLXArray = clampedLarge + MLXArray(Int32(maxExact))

            let absBuckets: MLXArray = which(isSmall, absPos.asType(.int32), largeFinal)

            // Offset negative positions by numBucketsHalf
            buckets = which(isNegative, absBuckets + MLXArray(Int32(numBucketsHalf)), absBuckets)
        }

        return buckets
    }

    /// Returns position bias: [1, numHeads, queryLen, keyLen]
    public func callAsFunction(queryLen: Int, keyLen: Int) -> MLXArray {
        let buckets: MLXArray = relativeBuckets(queryLen: queryLen, keyLen: keyLen)
        // Look up bias values: [queryLen, keyLen, numHeads]
        let values: MLXArray = embedding(buckets)
        // Transpose to [numHeads, queryLen, keyLen] then add batch dim
        let transposed: MLXArray = values.transposed(2, 0, 1)
        return expandedDims(transposed, axis: 0)
    }
}

// MARK: - T5 Self-Attention

/// T5 self-attention with relative position bias.
/// No sqrt(d_k) scaling — T5 convention.
public class T5Attention: Module {
    let q: Linear  // [dModel, innerDim]
    let k: Linear
    let v: Linear
    let o: Linear  // [innerDim, dModel]

    let numHeads: Int
    let dKV: Int

    /// Only block 0 has its own position bias; others receive it as input.
    let positionBias: T5RelativePositionBias?

    public init(config: T5Config, hasPositionBias: Bool) {
        let innerDim: Int = config.numHeads * config.dKV
        self.q = Linear(config.dModel, innerDim, bias: false)
        self.k = Linear(config.dModel, innerDim, bias: false)
        self.v = Linear(config.dModel, innerDim, bias: false)
        self.o = Linear(innerDim, config.dModel, bias: false)
        self.numHeads = config.numHeads
        self.dKV = config.dKV

        if hasPositionBias {
            self.positionBias = T5RelativePositionBias(
                numBuckets: config.relativeAttentionNumBuckets,
                maxDistance: config.relativeAttentionMaxDistance,
                numHeads: config.numHeads
            )
        } else {
            self.positionBias = nil
        }
    }

    /// Forward pass.
    /// - Parameters:
    ///   - x: input [batch, seqLen, dModel]
    ///   - existingBias: position bias from block 0 if this isn't block 0
    /// - Returns: (output, positionBias) — bias is passed to subsequent blocks
    public func callAsFunction(_ x: MLXArray, existingBias: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let batchSize: Int = x.dim(0)
        let seqLen: Int = x.dim(1)

        // Project Q, K, V: [batch, seqLen, dModel] -> [batch, numHeads, seqLen, dKV]
        let qProj: MLXArray = q(x).reshaped(batchSize, seqLen, numHeads, dKV).transposed(0, 2, 1, 3)
        let kProj: MLXArray = k(x).reshaped(batchSize, seqLen, numHeads, dKV).transposed(0, 2, 1, 3)
        let vProj: MLXArray = v(x).reshaped(batchSize, seqLen, numHeads, dKV).transposed(0, 2, 1, 3)

        // Attention scores: [batch, numHeads, seqLen, seqLen]
        // NOTE: T5 does NOT scale by 1/sqrt(d_k)
        var scores: MLXArray = matmul(qProj, kProj.transposed(0, 1, 3, 2))

        // Add relative position bias
        let bias: MLXArray
        if let existingBias = existingBias {
            bias = existingBias
        } else if let posBias = positionBias {
            bias = posBias(queryLen: seqLen, keyLen: seqLen)
        } else {
            fatalError("T5Attention: no position bias available")
        }
        scores = scores + bias

        // Softmax in float32 for numerical stability
        let attnWeights: MLXArray = softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)

        // Weighted sum: [batch, numHeads, seqLen, dKV]
        let attnOutput: MLXArray = matmul(attnWeights, vProj)

        // Reshape back: [batch, seqLen, dModel]
        let merged: MLXArray = attnOutput.transposed(0, 2, 1, 3).reshaped(batchSize, seqLen, numHeads * dKV)

        return (o(merged), bias)
    }
}

// MARK: - Gated Feed-Forward

/// T5 gated GeLU feed-forward: gate = gelu(wi_0(x)), hidden = wi_1(x), out = wo(gate * hidden)
public class T5FeedForward: Module {
    let wi0: Linear  // gate projection [dModel, dFF]
    let wi1: Linear  // value projection [dModel, dFF]
    let wo: Linear   // output projection [dFF, dModel]

    public init(config: T5Config) {
        self.wi0 = Linear(config.dModel, config.dFF, bias: false)
        self.wi1 = Linear(config.dModel, config.dFF, bias: false)
        self.wo = Linear(config.dFF, config.dModel, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate: MLXArray = gelu(wi0(x))
        let hidden: MLXArray = wi1(x)
        return wo(gate * hidden)
    }
}

// MARK: - T5 Encoder Block

/// Single T5 encoder block: pre-norm self-attention + pre-norm feed-forward.
public class T5Block: Module {
    let attnNorm: T5LayerNorm
    let attention: T5Attention
    let ffNorm: T5LayerNorm
    let feedForward: T5FeedForward

    public init(config: T5Config, hasPositionBias: Bool) {
        self.attnNorm = T5LayerNorm(dModel: config.dModel, eps: config.layerNormEpsilon)
        self.attention = T5Attention(config: config, hasPositionBias: hasPositionBias)
        self.ffNorm = T5LayerNorm(dModel: config.dModel, eps: config.layerNormEpsilon)
        self.feedForward = T5FeedForward(config: config)
    }

    /// Forward pass.
    /// - Parameters:
    ///   - x: [batch, seqLen, dModel]
    ///   - positionBias: from block 0 (nil for block 0 itself)
    /// - Returns: (output, positionBias)
    public func callAsFunction(_ x: MLXArray, positionBias: MLXArray? = nil) -> (MLXArray, MLXArray) {
        // Pre-norm self-attention with residual
        let normed1: MLXArray = attnNorm(x)
        let (attnOut, bias) = attention(normed1, existingBias: positionBias)
        let afterAttn: MLXArray = x + attnOut

        // Pre-norm feed-forward with residual
        let normed2: MLXArray = ffNorm(afterAttn)
        let ffOut: MLXArray = feedForward(normed2)
        let output: MLXArray = afterAttn + ffOut

        return (output, bias)
    }
}

// MARK: - T5 Encoder Model

/// Complete T5-v1.1-XXL encoder: embedding + 24 blocks + final layer norm.
/// Outputs last_hidden_state of shape [batch, seqLen, 4096].
///
/// Peak memory estimate (4-bit quantized): ~2.38 GB during forward pass.
public class T5EncoderModel: Module {
    let embedding: Embedding  // [vocabSize, dModel]
    let blocks: [T5Block]
    let finalNorm: T5LayerNorm
    let config: T5Config

    public init(config: T5Config = .xxl) {
        self.config = config
        self.embedding = Embedding(embeddingCount: config.vocabSize, dimensions: config.dModel)

        var blocks: [T5Block] = []
        for i in 0..<config.numLayers {
            blocks.append(T5Block(config: config, hasPositionBias: i == 0))
        }
        self.blocks = blocks
        self.finalNorm = T5LayerNorm(dModel: config.dModel, eps: config.layerNormEpsilon)
    }

    /// Encode text token IDs to hidden states.
    /// - Parameter inputIds: [batch, seqLen] of Int32 token IDs
    /// - Returns: [batch, seqLen, dModel] hidden states
    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        var hidden: MLXArray = embedding(inputIds)
        var positionBias: MLXArray? = nil

        for block in blocks {
            let (output, bias) = block(hidden, positionBias: positionBias)
            hidden = output
            positionBias = bias
        }

        return finalNorm(hidden)
    }
}
