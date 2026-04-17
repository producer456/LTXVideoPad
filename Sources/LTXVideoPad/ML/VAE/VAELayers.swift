// VAELayers.swift
//
// Core building blocks for LTX-Video's 3D Causal Video Autoencoder.
// No attention layers — all convolution-based.
//
// Reference: diffusers/models/autoencoders/autoencoder_kl_ltx.py
//            mlx_video/models/ltx_2/video_vae/convolution.py

import Foundation
import MLX
import MLXNN

// MARK: - VAE RMS Norm (no affine parameters)

/// RMS normalization without learnable parameters (elementwise_affine=False).
/// Used inside ResnetBlock3D. Different from T5's RMSNorm which has weights.
public class VAERMSNorm: Module {
    let eps: Float

    public init(eps: Float = 1e-8) {
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let variance: MLXArray = (x * x).mean(axis: -1, keepDims: true)
        return x * rsqrt(variance + eps)
    }
}

// MARK: - Causal Conv3d

/// 3D convolution with causal temporal padding.
/// Encoder uses causal=true (pad only past), decoder uses causal=false (symmetric).
///
/// For kernel_size=3, stride=1:
///   Causal:     temporal pad = (2, 0), spatial pad = (1, 1)
///   Non-causal: temporal pad = (1, 1), spatial pad = (1, 1)
///
/// For stride=2 downsample (encoder):
///   Causal: temporal pad = (2, 0), spatial pad = (0, 1) on each side
public class CausalConv3d: Module {
    let conv: Conv3d
    let causal: Bool
    let kernelSize: Int
    let stride: (Int, Int, Int)
    let temporalPad: (Int, Int)
    let spatialPad: (Int, Int)

    public init(inChannels: Int, outChannels: Int,
                kernelSize: Int = 3, stride: (Int, Int, Int) = (1, 1, 1),
                causal: Bool = true, bias: Bool = true) {

        self.causal = causal
        self.kernelSize = kernelSize
        self.stride = stride

        // Calculate padding
        if causal {
            // Causal: all temporal padding on the left (past)
            self.temporalPad = (kernelSize - 1, 0)
        } else {
            // Non-causal: symmetric temporal padding
            let tPad: Int = (kernelSize - 1) / 2
            self.temporalPad = (tPad, tPad)
        }

        // Spatial padding
        if stride.1 > 1 || stride.2 > 1 {
            // Strided: asymmetric spatial padding
            self.spatialPad = (0, kernelSize - 1)
        } else {
            let sPad: Int = (kernelSize - 1) / 2
            self.spatialPad = (sPad, sPad)
        }

        // Create conv with no padding (we handle it manually)
        self.conv = Conv3d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: .init(kernelSize, kernelSize, kernelSize),
            stride: .init(stride.0, stride.1, stride.2),
            padding: .init(0),
            bias: bias
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x shape: [B, D, H, W, C] (MLX channels-last)
        // Apply padding manually
        var padded: MLXArray = x

        // Temporal padding (axis 1 = D)
        if temporalPad.0 > 0 || temporalPad.1 > 0 {
            padded = padded(temporalPad.0, temporalPad.1, axis: 1)
        }

        // Spatial padding (axis 2 = H, axis 3 = W)
        if spatialPad.0 > 0 || spatialPad.1 > 0 {
            padded = padded(spatialPad.0, spatialPad.1, axis: 2)
            padded = padded(spatialPad.0, spatialPad.1, axis: 3)
        }

        return conv(padded)
    }

    /// Pad a specific axis with replicate padding (repeat edge values)
    private func padded(_ x: MLXArray, _ left: Int, _ right: Int, axis: Int) -> MLXArray {
        // For causal padding, replicate the first frame
        var parts: [MLXArray] = []

        if left > 0 {
            // Replicate the first slice along this axis
            let firstSlice: MLXArray = x.take(MLXArray([Int32(0)]), axis: axis)
            for _ in 0..<left {
                parts.append(firstSlice)
            }
        }

        parts.append(x)

        if right > 0 {
            let lastIdx: Int = x.dim(axis) - 1
            let lastSlice: MLXArray = x.take(MLXArray([Int32(lastIdx)]), axis: axis)
            for _ in 0..<right {
                parts.append(lastSlice)
            }
        }

        return concatenated(parts, axis: axis)
    }
}

// MARK: - ResNet Block 3D

/// Residual block with two CausalConv3d layers and RMSNorm.
/// Optional 1x1x1 shortcut convolution when channel dims differ.
///
/// Structure: RMSNorm → SiLU → Conv3d → RMSNorm → SiLU → Conv3d + skip
public class ResnetBlock3D: Module {
    let norm1: VAERMSNorm
    let conv1: CausalConv3d
    let norm2: VAERMSNorm
    let conv2: CausalConv3d
    let shortcut: CausalConv3d?
    let shortcutNorm: LayerNorm?

    public init(inChannels: Int, outChannels: Int, causal: Bool = true) {
        self.norm1 = VAERMSNorm()
        self.conv1 = CausalConv3d(inChannels: inChannels, outChannels: outChannels,
                                   kernelSize: 3, causal: causal)
        self.norm2 = VAERMSNorm()
        self.conv2 = CausalConv3d(inChannels: outChannels, outChannels: outChannels,
                                   kernelSize: 3, causal: causal)

        if inChannels != outChannels {
            self.shortcut = CausalConv3d(inChannels: inChannels, outChannels: outChannels,
                                          kernelSize: 1, stride: (1, 1, 1), causal: causal)
            self.shortcutNorm = LayerNorm(dimensions: outChannels)
        } else {
            self.shortcut = nil
            self.shortcutNorm = nil
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h: MLXArray = norm1(x)
        h = silu(h)
        h = conv1(h)
        h = norm2(h)
        h = silu(h)
        h = conv2(h)

        // Skip connection
        var skip: MLXArray = x
        if let shortcutConv = shortcut, let shortcutLN = shortcutNorm {
            skip = shortcutConv(x)
            skip = shortcutLN(skip)
        }

        return h + skip
    }
}
