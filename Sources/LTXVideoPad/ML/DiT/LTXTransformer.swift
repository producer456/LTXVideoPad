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

// MARK: - Timestep Embedding

/// Sinusoidal timestep embedding → MLP → AdaLN parameters.
public class TimestepEmbedding: Module {
    let linear1: Linear
    let linear2: Linear

    public init(dim: Int = 256, outerDim: Int = 2048) {
        self.linear1 = Linear(dim, outerDim, bias: true)
        self.linear2 = Linear(outerDim, outerDim, bias: true)
    }

    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        // Sinusoidal embedding
        let halfDim: Int = 128  // 256 / 2
        let freqs: MLXArray = MLXArray(Float(-log(10000.0) / Float(halfDim - 1)))
        let indices: MLXArray = MLXArray(Array(0..<halfDim).map { Float($0) })
        let emb: MLXArray = exp(indices * freqs)
        let tExpanded: MLXArray = expandedDimensions(t.asType(.float32), axis: -1)
        let sinEmb: MLXArray = sin(tExpanded * emb)
        let cosEmb: MLXArray = cos(tExpanded * emb)
        let embedding: MLXArray = concatenated([sinEmb, cosEmb], axis: -1)

        // MLP
        var h: MLXArray = silu(linear1(embedding))
        h = silu(linear2(h))
        return h
    }
}

/// Full time embedding block that produces 6*innerDim adaLN parameters.
public class TimeEmbed: Module {
    let emb: TimestepEmbedder
    let linear: Linear

    public init(innerDim: Int = 2048) {
        self.emb = TimestepEmbedder(innerDim: innerDim)
        self.linear = Linear(innerDim, 6 * innerDim, bias: true)
    }

    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        let embedded: MLXArray = emb(t)
        return linear(embedded)
    }
}

/// Timestep embedder sub-module.
public class TimestepEmbedder: Module {
    let linear_1: Linear
    let linear_2: Linear
    let dim: Int

    public init(innerDim: Int = 2048) {
        self.dim = 256
        self.linear_1 = Linear(dim, innerDim, bias: true)
        self.linear_2 = Linear(innerDim, innerDim, bias: true)
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

// MARK: - Caption Projection

/// Projects T5 embeddings (4096-dim) to inner_dim (2048-dim).
public class CaptionProjection: Module {
    let linear_1: Linear
    let linear_2: Linear

    public init(captionChannels: Int = 4096, innerDim: Int = 2048) {
        self.linear_1 = Linear(captionChannels, innerDim, bias: true)
        self.linear_2 = Linear(innerDim, innerDim, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h: MLXArray = silu(linear_1(x))
        h = linear_2(h)
        return h
    }
}

// MARK: - DiT RMS Norm

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

// MARK: - Attention

/// Multi-head attention with QK normalization.
public class DiTAttention: Module {
    let to_q: Linear
    let to_k: Linear
    let to_v: Linear
    let to_out_0: Linear  // to_out.0
    let norm_q: DiTRMSNorm
    let norm_k: DiTRMSNorm
    let numHeads: Int
    let headDim: Int

    public init(dim: Int, numHeads: Int = 32, headDim: Int = 64) {
        let innerDim: Int = numHeads * headDim
        self.to_q = Linear(dim, innerDim, bias: true)
        self.to_k = Linear(dim, innerDim, bias: true)
        self.to_v = Linear(dim, innerDim, bias: true)
        self.to_out_0 = Linear(innerDim, dim, bias: true)
        self.norm_q = DiTRMSNorm()
        self.norm_k = DiTRMSNorm()
        self.numHeads = numHeads
        self.headDim = headDim
    }

    /// Self-attention or cross-attention.
    /// - Parameters:
    ///   - x: query input [B, seqLen, dim]
    ///   - context: key/value source (nil = self-attention, otherwise cross-attention)
    ///   - rope: optional RoPE frequencies for self-attention
    public func callAsFunction(_ x: MLXArray, context: MLXArray? = nil,
                                rope: MLXArray? = nil) -> MLXArray {
        let kv: MLXArray = context ?? x
        let b: Int = x.dim(0)
        let sq: Int = x.dim(1)
        let sk: Int = kv.dim(1)

        // Project Q, K, V
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
        var scores: MLXArray = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        let attnWeights: MLXArray = softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        let attnOut: MLXArray = matmul(attnWeights, v)

        // Merge heads
        let merged: MLXArray = attnOut.transposed(0, 2, 1, 3).reshaped(b, sq, numHeads * headDim)
        return to_out_0(merged)
    }

    /// Apply rotary position embeddings.
    private func applyRoPE(_ x: MLXArray, freqs: MLXArray) -> MLXArray {
        // x: [B, heads, seqLen, headDim]
        // freqs: [seqLen, headDim/2, 2] or similar
        // Standard complex rotation: x_rot = x * cos(theta) + rotate_half(x) * sin(theta)
        let halfDim: Int = headDim / 2
        let x1: MLXArray = x[0..., 0..., 0..., 0..<halfDim]
        let x2: MLXArray = x[0..., 0..., 0..., halfDim...]

        let cos_: MLXArray = freqs[0..., 0..., 0]  // cos component
        let sin_: MLXArray = freqs[0..., 0..., 1]  // sin component

        let out1: MLXArray = x1 * cos_ - x2 * sin_
        let out2: MLXArray = x2 * cos_ + x1 * sin_

        return concatenated([out1, out2], axis: -1)
    }
}

// MARK: - Feed-Forward

public class DiTFeedForward: Module {
    let proj: Linear   // net.0.proj
    let out: Linear    // net.2

    public init(dim: Int, mlpDim: Int) {
        self.proj = Linear(dim, mlpDim, bias: true)
        self.out = Linear(mlpDim, dim, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return out(gelu(proj(x)))
    }
}

// MARK: - Transformer Block

/// Single DiT block with AdaLN-Single conditioning.
public class LTXTransformerBlock: Module {
    let scale_shift_table: MLXArray  // [6, innerDim]
    let attn1: DiTAttention          // self-attention
    let attn2: DiTAttention          // cross-attention
    let ff: DiTFeedForward
    let norm1: DiTRMSNorm            // pre-self-attn
    let norm2: DiTRMSNorm            // pre-ffn

    public init(config: DiTConfig) {
        self.scale_shift_table = MLXArray.zeros([6, config.innerDim])
        self.attn1 = DiTAttention(dim: config.innerDim, numHeads: config.numHeads, headDim: config.headDim)
        self.attn2 = DiTAttention(dim: config.innerDim, numHeads: config.numHeads, headDim: config.headDim)
        self.ff = DiTFeedForward(dim: config.innerDim, mlpDim: config.innerDim * config.mlpRatio)
        self.norm1 = DiTRMSNorm(eps: config.normEps)
        self.norm2 = DiTRMSNorm(eps: config.normEps)
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

        // Self-attention with adaLN
        var h: MLXArray = norm1(x)
        h = h * expandedDimensions(1 + scaleMSA, axis: 1) + expandedDimensions(shiftMSA, axis: 1)
        h = attn1(h, rope: rope)
        var out: MLXArray = x + expandedDimensions(gateMSA, axis: 1) * h

        // Cross-attention (no adaLN modulation)
        out = out + attn2(out, context: textEmbeds)

        // Feed-forward with adaLN
        h = norm2(out)
        h = h * expandedDimensions(1 + scaleMLP, axis: 1) + expandedDimensions(shiftMLP, axis: 1)
        h = ff(h)
        out = out + expandedDimensions(gateMLP, axis: 1) * h

        return out
    }
}

// MARK: - Full Transformer Model

/// LTX-Video DiT: proj_in → 28 transformer blocks → proj_out
///
/// Peak memory (4-bit): ~0.96 GB
public class LTXVideoTransformer: Module {
    let proj_in: Linear                  // 128 → 2048
    let proj_out: Linear                 // 2048 → 128
    let time_embed: TimeEmbed
    let caption_projection: CaptionProjection
    let transformer_blocks: [LTXTransformerBlock]
    let scale_shift_table: MLXArray      // global [2, 2048] for final output
    let config: DiTConfig

    public init(config: DiTConfig = .v096) {
        self.config = config
        self.proj_in = Linear(config.inChannels, config.innerDim, bias: true)
        self.proj_out = Linear(config.innerDim, config.outChannels, bias: true)
        self.time_embed = TimeEmbed(innerDim: config.innerDim)
        self.caption_projection = CaptionProjection(
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
    /// - Returns: [B, seqLen, outChannels] predicted velocity
    public func callAsFunction(latents: MLXArray, textEmbeds: MLXArray,
                                timestep: MLXArray) -> MLXArray {
        // Project latents to inner dim
        var hidden: MLXArray = proj_in(latents)

        // Project text embeddings
        let projectedText: MLXArray = caption_projection(textEmbeds)

        // Timestep → adaLN parameters
        let adaln: MLXArray = time_embed(timestep)

        // TODO: Compute 3D RoPE frequencies from latent spatial dims
        let rope: MLXArray? = nil

        // Transformer blocks
        for block in transformer_blocks {
            hidden = block(hidden, textEmbeds: projectedText, adaln: adaln, rope: rope)
        }

        // Final adaLN + project out
        let finalParams: MLXArray = adaln.reshaped(hidden.dim(0), 6, -1)
        // Use last 2 params from global scale_shift_table for output modulation
        let outShift: MLXArray = scale_shift_table[0]
        let outScale: MLXArray = scale_shift_table[1]

        hidden = hidden * expandedDimensions(1 + outScale, axis: 0) + expandedDimensions(outShift, axis: 0)
        let output: MLXArray = proj_out(hidden)

        return output
    }
}
