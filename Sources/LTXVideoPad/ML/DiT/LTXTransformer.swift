// LTXTransformer.swift — Phase 3
//
// LTX-Video DiT backbone: 28-block transformer with:
// - Self-attention with 3D RoPE
// - Cross-attention to T5 text embeddings
// - AdaLN-Single timestep conditioning
// - Flow matching velocity prediction
//
// Config: 1.92B params, inner_dim=2048, 32 heads, head_dim=64
// Memory at 4-bit: ~0.96 GB
//
// Property names match quantized safetensors keys exactly.
// Top-level prefix: model.diffusion_model.*
//
// Reference: diffusers LTXVideoTransformer3DModel

import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct DiTConfig: Sendable {
    public let numLayers: Int           // 28
    public let innerDim: Int            // 2048
    public let numHeads: Int            // 32
    public let headDim: Int             // 64
    public let captionChannels: Int     // 4096 (T5 output dim)
    public let crossAttentionDim: Int   // 2048 (projected text dim)
    public let inChannels: Int          // 128 (VAE latent channels)
    public let outChannels: Int         // 128
    public let mlpRatio: Int            // 4
    public let normEps: Float           // 1e-6

    public static let v096 = DiTConfig(
        numLayers: 28, innerDim: 2048, numHeads: 32, headDim: 64,
        captionChannels: 4096, crossAttentionDim: 2048,
        inChannels: 128, outChannels: 128, mlpRatio: 4, normEps: 1e-6
    )
}

// MARK: - RMS Norm (no learnable weight)

public class DiTRMSNorm: Module {
    let eps: Float

    public init(eps: Float = 1e-6) {
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let variance: MLXArray = (x * x).mean(axis: -1, keepDims: true)
        return x * rsqrt(variance + eps)
    }
}

// MARK: - Timestep Embedder
// Key path: adaln_single.emb.timestep_embedder.{linear_1, linear_2}

public class DiTTimestepEmbedder: Module {
    let linear_1: QuantizedLinear
    let linear_2: QuantizedLinear
    let dim: Int

    public init(innerDim: Int = 2048) {
        self.dim = 256
        self.linear_1 = QuantizedLinear(dim, innerDim, bias: true, groupSize: 64, bits: 4)
        self.linear_2 = QuantizedLinear(innerDim, innerDim, bias: true, groupSize: 64, bits: 4)
    }

    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        let halfDim: Int = dim / 2
        let logScale: Float = -log(10000.0) / Float(halfDim - 1)
        let freqs: MLXArray = exp(MLXArray(Array(0..<halfDim).map { Float($0) * logScale }))
        let tFloat: MLXArray = expandedDimensions(t.asType(.float32), axis: -1)
        let angles: MLXArray = tFloat * freqs
        let embedding: MLXArray = concatenated([sin(angles), cos(angles)], axis: -1)

        var h: MLXArray = silu(linear_1(embedding))
        h = silu(linear_2(h))
        return h
    }
}

// MARK: - AdaLN Emb wrapper
// Key path: adaln_single.emb.timestep_embedder

public class DiTAdaLNEmb: Module {
    let timestep_embedder: DiTTimestepEmbedder

    public init(innerDim: Int = 2048) {
        self.timestep_embedder = DiTTimestepEmbedder(innerDim: innerDim)
    }

    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        return timestep_embedder(t)
    }
}

// MARK: - AdaLN Single
// Key path: adaln_single.{emb, linear}

public class DiTAdaLNSingle: Module {
    let emb: DiTAdaLNEmb
    let linear: QuantizedLinear  // innerDim -> 6 * innerDim

    public init(innerDim: Int = 2048) {
        self.emb = DiTAdaLNEmb(innerDim: innerDim)
        self.linear = QuantizedLinear(innerDim, 6 * innerDim, bias: true, groupSize: 64, bits: 4)
    }

    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        let embedded: MLXArray = emb(t)
        return linear(embedded)
    }
}

// MARK: - Caption Projection
// Key path: caption_projection.{linear_1, linear_2}

public class DiTCaptionProjection: Module {
    let linear_1: QuantizedLinear
    let linear_2: QuantizedLinear

    public init(captionChannels: Int = 4096, innerDim: Int = 2048) {
        self.linear_1 = QuantizedLinear(captionChannels, innerDim, bias: true, groupSize: 64, bits: 4)
        self.linear_2 = QuantizedLinear(innerDim, innerDim, bias: true, groupSize: 64, bits: 4)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h: MLXArray = silu(linear_1(x))
        h = linear_2(h)
        return h
    }
}

// MARK: - Attention
// Key path: transformer_blocks.{N}.attn1 or attn2
// Sub-keys: to_q, to_k, to_v, to_out (array, index 0), norm_q, norm_k

public class DiTAttention: Module {
    let to_q: QuantizedLinear
    let to_k: QuantizedLinear
    let to_v: QuantizedLinear
    let to_out: [QuantizedLinear]  // [0] = output projection
    let norm_q: DiTRMSNorm
    let norm_k: DiTRMSNorm
    let numHeads: Int
    let headDim: Int

    public init(dim: Int, numHeads: Int = 32, headDim: Int = 64) {
        let innerDim: Int = numHeads * headDim
        self.to_q = QuantizedLinear(dim, innerDim, bias: true, groupSize: 64, bits: 4)
        self.to_k = QuantizedLinear(dim, innerDim, bias: true, groupSize: 64, bits: 4)
        self.to_v = QuantizedLinear(dim, innerDim, bias: true, groupSize: 64, bits: 4)
        self.to_out = [QuantizedLinear(innerDim, dim, bias: true, groupSize: 64, bits: 4)]
        self.norm_q = DiTRMSNorm()
        self.norm_k = DiTRMSNorm()
        self.numHeads = numHeads
        self.headDim = headDim
    }

    /// Self-attention or cross-attention.
    public func callAsFunction(_ x: MLXArray, context: MLXArray? = nil,
                                rope: MLXArray? = nil) -> MLXArray {
        let kv: MLXArray = context ?? x
        let b: Int = x.dim(0)
        let sq: Int = x.dim(1)
        let sk: Int = kv.dim(1)

        var q: MLXArray = to_q(x).reshaped(b, sq, numHeads, headDim).transposed(0, 2, 1, 3)
        var k: MLXArray = to_k(kv).reshaped(b, sk, numHeads, headDim).transposed(0, 2, 1, 3)
        let v: MLXArray = to_v(kv).reshaped(b, sk, numHeads, headDim).transposed(0, 2, 1, 3)

        // QK normalization
        q = norm_q(q)
        k = norm_k(k)

        // Apply RoPE to self-attention Q/K (not cross-attention)
        if let rope = rope, context == nil {
            q = applyRoPE(q, freqs: rope)
            k = applyRoPE(k, freqs: rope)
        }

        // Scaled dot-product attention
        let scale: Float = 1.0 / sqrt(Float(headDim))
        let scores: MLXArray = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        let attnWeights: MLXArray = softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        let attnOut: MLXArray = matmul(attnWeights, v)

        // Merge heads
        let merged: MLXArray = attnOut.transposed(0, 2, 1, 3).reshaped(b, sq, numHeads * headDim)
        return to_out[0](merged)
    }

    /// Apply rotary position embeddings.
    /// x: [B, heads, seqLen, headDim]
    /// freqs: [1, seqLen, halfDim, 2] where last dim is (cos, sin)
    private func applyRoPE(_ x: MLXArray, freqs: MLXArray) -> MLXArray {
        let halfDim: Int = headDim / 2
        let x1: MLXArray = x[0..., 0..., 0..., 0..<halfDim]   // [B, heads, seqLen, halfDim]
        let x2: MLXArray = x[0..., 0..., 0..., halfDim...]     // [B, heads, seqLen, halfDim]

        // freqs: [1, seqLen, halfDim, 2] → extract cos/sin as [1, 1, seqLen, halfDim]
        let cos_: MLXArray = freqs[0..., 0..., 0..., 0].reshaped(1, 1, -1, halfDim)
        let sin_: MLXArray = freqs[0..., 0..., 0..., 1].reshaped(1, 1, -1, halfDim)

        let out1: MLXArray = x1 * cos_ - x2 * sin_
        let out2: MLXArray = x2 * cos_ + x1 * sin_

        return concatenated([out1, out2], axis: -1)
    }
}

// MARK: - Feed-Forward
// Key path: transformer_blocks.{N}.ff.net.{0,2}
// net[0] has .proj (gate projection), net[1] is GELU (no params), net[2] is output

/// Gate projection wrapper: ff.net.0.proj
public class DiTFFNGate: Module {
    let proj: QuantizedLinear

    public init(dim: Int, mlpDim: Int) {
        self.proj = QuantizedLinear(dim, mlpDim, bias: true, groupSize: 64, bits: 4)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return proj(x)
    }
}

/// Placeholder for GELU activation at net[1] — no parameters.
public class DiTFFNActivation: Module {
    public override init() {
        super.init()
    }
}

/// Feed-forward block: transformer_blocks.{N}.ff
/// Key path: ff.net.[0].proj, ff.net.[1] (no params), ff.net.[2]
public class DiTFeedForward: Module {
    let net: [Module]  // [0]=gate(with .proj), [1]=activation(no params), [2]=output linear

    public init(dim: Int, mlpDim: Int) {
        self.net = [
            DiTFFNGate(dim: dim, mlpDim: mlpDim),
            DiTFFNActivation(),
            QuantizedLinear(mlpDim, dim, bias: true, groupSize: 64, bits: 4)
        ]
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate: DiTFFNGate = net[0] as! DiTFFNGate
        let outLinear: QuantizedLinear = net[2] as! QuantizedLinear
        return outLinear(gelu(gate(x)))
    }
}

// MARK: - Transformer Block
// Key path: transformer_blocks.{N}

public class LTXTransformerBlock: Module {
    let scale_shift_table: MLXArray  // [6, innerDim]
    let attn1: DiTAttention          // self-attention
    let attn2: DiTAttention          // cross-attention
    let ff: DiTFeedForward

    public init(config: DiTConfig) {
        self.scale_shift_table = MLXArray.zeros([6, config.innerDim])
        self.attn1 = DiTAttention(dim: config.innerDim, numHeads: config.numHeads, headDim: config.headDim)
        self.attn2 = DiTAttention(dim: config.innerDim, numHeads: config.numHeads, headDim: config.headDim)
        self.ff = DiTFeedForward(dim: config.innerDim, mlpDim: config.innerDim * config.mlpRatio)
    }

    /// Forward pass.
    /// - Parameters:
    ///   - x: [B, seqLen, innerDim]
    ///   - textEmbeds: [B, textLen, innerDim] projected text
    ///   - adaln: [B, 6*innerDim] timestep conditioning
    ///   - rope: RoPE frequencies
    public func callAsFunction(_ x: MLXArray, textEmbeds: MLXArray,
                                adaln: MLXArray, rope: MLXArray? = nil) -> MLXArray {
        // Split adaLN parameters: [B, 6, innerDim]
        let adalnReshaped: MLXArray = adaln.reshaped(x.dim(0), 6, -1)
        // Add per-block scale_shift_table
        let params: MLXArray = adalnReshaped + scale_shift_table

        let shiftMSA: MLXArray = params[0..., 0]
        let scaleMSA: MLXArray = params[0..., 1]
        let gateMSA: MLXArray = params[0..., 2]
        let shiftMLP: MLXArray = params[0..., 3]
        let scaleMLP: MLXArray = params[0..., 4]
        let gateMLP: MLXArray = params[0..., 5]

        // Self-attention with adaLN (pre-norm is implicit via scale/shift)
        var h: MLXArray = x * expandedDimensions(1 + scaleMSA, axis: 1) + expandedDimensions(shiftMSA, axis: 1)
        h = attn1(h, rope: rope)
        var out: MLXArray = x + expandedDimensions(gateMSA, axis: 1) * h

        // Cross-attention (no adaLN modulation)
        out = out + attn2(out, context: textEmbeds)

        // Feed-forward with adaLN
        h = out * expandedDimensions(1 + scaleMLP, axis: 1) + expandedDimensions(shiftMLP, axis: 1)
        h = ff(h)
        out = out + expandedDimensions(gateMLP, axis: 1) * h

        return out
    }
}

// MARK: - Diffusion Model (inner)
// Key path: model.diffusion_model.*
//
// Peak memory (4-bit): ~0.96 GB

public class LTXDiffusionModel: Module {
    let patchify_proj: QuantizedLinear       // 128 -> 2048
    let proj_out: QuantizedLinear            // 2048 -> 128
    let adaln_single: DiTAdaLNSingle
    let caption_projection: DiTCaptionProjection
    let transformer_blocks: [LTXTransformerBlock]
    let scale_shift_table: MLXArray          // global [2, 2048] for final output
    let config: DiTConfig

    public init(config: DiTConfig = .v096) {
        self.config = config
        self.patchify_proj = QuantizedLinear(config.inChannels, config.innerDim, bias: true, groupSize: 64, bits: 4)
        self.proj_out = QuantizedLinear(config.innerDim, config.outChannels, bias: true, groupSize: 64, bits: 4)
        self.adaln_single = DiTAdaLNSingle(innerDim: config.innerDim)
        self.caption_projection = DiTCaptionProjection(
            captionChannels: config.captionChannels,
            innerDim: config.innerDim
        )
        self.scale_shift_table = MLXArray.zeros([2, config.innerDim])

        var blocks: [LTXTransformerBlock] = []
        for _ in 0..<config.numLayers {
            blocks.append(LTXTransformerBlock(config: config))
        }
        self.transformer_blocks = blocks
    }

    /// Denoise one step.
    /// - Parameters:
    ///   - latents: [B, seqLen, inChannels] flattened noisy latents
    ///   - textEmbeds: [B, textLen, captionChannels] T5 embeddings
    ///   - timestep: [B] timestep values
    ///   - rope: precomputed 3D RoPE [1, seqLen, headDim/2, 2] (optional but recommended)
    /// - Returns: [B, seqLen, outChannels] predicted velocity
    public func callAsFunction(latents: MLXArray, textEmbeds: MLXArray,
                                timestep: MLXArray,
                                rope: MLXArray? = nil) -> MLXArray {
        // Project latents to inner dim
        var hidden: MLXArray = patchify_proj(latents)

        // Project text embeddings
        let projectedText: MLXArray = caption_projection(textEmbeds)

        // Timestep -> adaLN parameters
        let adaln: MLXArray = adaln_single(timestep)

        // Transformer blocks
        for block in transformer_blocks {
            hidden = block(hidden, textEmbeds: projectedText, adaln: adaln, rope: rope)
        }

        // Final adaLN + project out
        let outShift: MLXArray = scale_shift_table[0]
        let outScale: MLXArray = scale_shift_table[1]

        hidden = hidden * expandedDimensions(1 + outScale, axis: 0) + expandedDimensions(outShift, axis: 0)
        let output: MLXArray = proj_out(hidden)

        return output
    }
}

// MARK: - Top-level Model wrapper
// Key path: model.diffusion_model

public class LTXDiffusionModelWrapper: Module {
    let diffusion_model: LTXDiffusionModel

    public init(config: DiTConfig = .v096) {
        self.diffusion_model = LTXDiffusionModel(config: config)
    }

    public func callAsFunction(latents: MLXArray, textEmbeds: MLXArray,
                                timestep: MLXArray,
                                rope: MLXArray? = nil) -> MLXArray {
        return diffusion_model(latents: latents, textEmbeds: textEmbeds,
                               timestep: timestep, rope: rope)
    }
}

/// Matches safetensors root: model.diffusion_model.*
public class LTXVideoTransformer: Module {
    let model: LTXDiffusionModelWrapper

    public init(config: DiTConfig = .v096) {
        self.model = LTXDiffusionModelWrapper(config: config)
    }

    public func callAsFunction(latents: MLXArray, textEmbeds: MLXArray,
                                timestep: MLXArray,
                                rope: MLXArray? = nil) -> MLXArray {
        return model(latents: latents, textEmbeds: textEmbeds,
                     timestep: timestep, rope: rope)
    }
}
