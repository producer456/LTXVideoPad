// RoPE3D.swift
//
// 3D Rotary Position Embeddings for LTX-Video DiT.
// Encodes position along three dimensions: frames (F'), height (H'), width (W').
//
// The head_dim is split into 3 equal parts, one for each spatial dimension.
// Each part gets its own frequency basis computed as:
//   freq_i = 1 / (theta^(2i / d_part)) for i in 0..<d_part/2
//
// For each token position (f, h, w), we compute (cos, sin) pairs that are
// applied to Q and K in self-attention via the standard rotary embedding formula.
//
// Reference: diffusers LTXVideoRotaryPosEmbed
//
// Memory: negligible — frequencies are small tensors

import Foundation
import MLX

public enum RoPE3D {

    /// Compute 3D rotary position embeddings for the given latent dimensions.
    ///
    /// - Parameters:
    ///   - framesF: number of latent frames (F')
    ///   - heightH: number of latent height patches (H')
    ///   - widthW: number of latent width patches (W')
    ///   - headDim: dimension per attention head (must be divisible by 2)
    ///   - theta: base frequency (default 10000.0)
    /// - Returns: MLXArray of shape [1, seqLen, headDim/2, 2] where last dim is (cos, sin).
    ///            seqLen = F' * H' * W'.
    public static func build(
        framesF: Int,
        heightH: Int,
        widthW: Int,
        headDim: Int,
        theta: Float = 10000.0
    ) -> MLXArray {
        // LTX-Video splits head_dim into 3 parts for the 3 spatial dims.
        // The rope_dim_list in the config is typically [headDim/6, headDim/6, headDim/6]
        // repeated to cover headDim/2. Each dimension gets headDim/6 frequency components.
        //
        // For headDim=64: each dim gets 64/6 ~ 10 freqs, total 30 (with remainder to last dim).
        // Actually diffusers uses: d_t = d_h = d_w = headDim / 6 (integer division).
        // But let's match exactly: rope_dim_list = [d//6, d//6, d//6] where the rest goes unused.
        // Actually in LTX-Video the split is equal thirds of headDim/2.

        let halfDim: Int = headDim / 2
        // Split into 3 parts. For headDim=64: halfDim=32, each part gets ~10-11 frequencies.
        let dF: Int = halfDim / 3
        let dH: Int = halfDim / 3
        let dW: Int = halfDim - dF - dH  // remainder goes to width

        // Compute frequency bases for each dimension
        let freqsF: MLXArray = computeFreqs(dimPart: dF, theta: theta)  // [dF]
        let freqsH: MLXArray = computeFreqs(dimPart: dH, theta: theta)  // [dH]
        let freqsW: MLXArray = computeFreqs(dimPart: dW, theta: theta)  // [dW]

        // Create position indices for each dimension
        let posF: MLXArray = MLXArray(Array(0..<framesF).map { Float($0) })  // [F']
        let posH: MLXArray = MLXArray(Array(0..<heightH).map { Float($0) })  // [H']
        let posW: MLXArray = MLXArray(Array(0..<widthW).map { Float($0) })   // [W']

        // Outer products: position x frequency
        // anglesF[f, i] = posF[f] * freqsF[i]
        let anglesF: MLXArray = outerProduct(posF, freqsF)  // [F', dF]
        let anglesH: MLXArray = outerProduct(posH, freqsH)  // [H', dH]
        let anglesW: MLXArray = outerProduct(posW, freqsW)  // [W', dW]

        // Build the full 3D grid of angles.
        // For each (f, h, w) position, concatenate the angles from all 3 dims.
        // Total seqLen = F' * H' * W'
        let seqLen: Int = framesF * heightH * widthW

        // Expand each angle tensor to cover the full 3D grid:
        // anglesF: [F', 1, 1, dF] -> broadcast over H', W'
        // anglesH: [1, H', 1, dH] -> broadcast over F', W'
        // anglesW: [1, 1, W', dW] -> broadcast over F', H'
        let aF: MLXArray = broadcast(
            anglesF.reshaped(framesF, 1, 1, dF),
            to: [framesF, heightH, widthW, dF]
        ).reshaped(seqLen, dF)

        let aH: MLXArray = broadcast(
            anglesH.reshaped(1, heightH, 1, dH),
            to: [framesF, heightH, widthW, dH]
        ).reshaped(seqLen, dH)

        let aW: MLXArray = broadcast(
            anglesW.reshaped(1, 1, widthW, dW),
            to: [framesF, heightH, widthW, dW]
        ).reshaped(seqLen, dW)

        // Concatenate along frequency dimension: [seqLen, halfDim]
        let allAngles: MLXArray = concatenated([aF, aH, aW], axis: 1)

        // Compute cos and sin, stack to get [seqLen, halfDim, 2]
        let cosVals: MLXArray = cos(allAngles)
        let sinVals: MLXArray = sin(allAngles)
        let cosSin: MLXArray = stacked([cosVals, sinVals], axis: -1)

        // Add batch dimension: [1, seqLen, halfDim, 2]
        return expandedDimensions(cosSin, axis: 0)
    }

    // MARK: - Private

    /// Compute frequency bases: freq_i = 1 / (theta^(2i / dimPart))
    /// for i in 0..<dimPart.
    private static func computeFreqs(dimPart: Int, theta: Float) -> MLXArray {
        let indices: [Float] = (0..<dimPart).map { Float($0) }
        let exponents: [Float] = indices.map { 2.0 * $0 / Float(dimPart) }
        let freqs: [Float] = exponents.map { 1.0 / pow(theta, $0) }
        return MLXArray(freqs)
    }

    /// Outer product of two 1D arrays.
    private static func outerProduct(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        let aExpanded: MLXArray = expandedDimensions(a, axis: -1)  // [N, 1]
        let bExpanded: MLXArray = expandedDimensions(b, axis: 0)   // [1, M]
        return aExpanded * bExpanded  // [N, M]
    }
}
