// VAELayers.swift — Core building blocks for LTX-Video 3D Causal VAE
// Property names match safetensors weight keys exactly for direct loading.
// All convolution-based, no attention layers.

import Foundation
import MLX
import MLXNN

// MARK: - VAE RMS Norm (no learnable parameters)

public class VAERMSNorm: Module, UnaryLayer {
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

/// 3D convolution with causal temporal padding (replicate first frame).
/// Input/output in MLX channels-last format: [B, D, H, W, C]
///
/// The inner Conv3d is exposed as `.conv` so the weight key path
/// `conv1.conv.weight` maps to `conv1.conv.weight` on the module tree.
public class CausalConv3d: Module, UnaryLayer {
    let conv: Conv3d
    let causal: Bool
    let kernelSize: Int
    let strideVal: (Int, Int, Int)
    let temporalPadLeft: Int
    let temporalPadRight: Int
    let spatialPadLeft: Int
    let spatialPadRight: Int

    public init(inChannels: Int, outChannels: Int,
                kernelSize: Int = 3, stride: (Int, Int, Int) = (1, 1, 1),
                causal: Bool = true, bias: Bool = true) {
        self.causal = causal
        self.kernelSize = kernelSize
        self.strideVal = stride

        if causal {
            self.temporalPadLeft = kernelSize - 1
            self.temporalPadRight = 0
        } else {
            let tPad: Int = (kernelSize - 1) / 2
            self.temporalPadLeft = tPad
            self.temporalPadRight = tPad
        }

        if stride.1 > 1 || stride.2 > 1 {
            self.spatialPadLeft = 0
            self.spatialPadRight = kernelSize - 1
        } else {
            let sPad: Int = (kernelSize - 1) / 2
            self.spatialPadLeft = sPad
            self.spatialPadRight = sPad
        }

        // Conv with zero padding — we handle padding manually
        self.conv = Conv3d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrTriple((kernelSize, kernelSize, kernelSize)),
            stride: IntOrTriple(stride),
            padding: 0,
            bias: bias
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h: MLXArray = x

        // Temporal padding via replication (axis 1 = depth)
        if temporalPadLeft > 0 {
            let first: MLXArray = h[0..., 0..<1]  // [B, 1, H, W, C]
            let repeated: [MLXArray] = Array(repeating: first, count: temporalPadLeft)
            h = concatenated(repeated + [h], axis: 1)
        }
        if temporalPadRight > 0 {
            let last: MLXArray = h[0..., (h.dim(1) - 1)..<h.dim(1)]
            let repeated: [MLXArray] = Array(repeating: last, count: temporalPadRight)
            h = concatenated([h] + repeated, axis: 1)
        }

        // Spatial padding via replication (axis 2 = H, axis 3 = W)
        if spatialPadLeft > 0 || spatialPadRight > 0 {
            h = MLX.padded(h, widths: [
                .init(0),                                              // batch
                .init(0),                                              // depth
                .init((spatialPadLeft, spatialPadRight)),              // height
                .init((spatialPadLeft, spatialPadRight)),              // width
                .init(0)                                               // channels
            ], mode: .edge)
        }

        return conv(h)
    }
}

// MARK: - ResNet Block 3D

/// Two CausalConv3d layers with RMSNorm + SiLU, residual skip connection.
/// Weight key paths:  conv1.conv.{weight,bias}, conv2.conv.{weight,bias}
/// No learnable norm parameters (VAERMSNorm is parameter-free).
public class ResnetBlock3D: Module {
    let norm1: VAERMSNorm
    let conv1: CausalConv3d
    let norm2: VAERMSNorm
    let conv2: CausalConv3d

    public init(inChannels: Int, outChannels: Int, causal: Bool = true) {
        self.norm1 = VAERMSNorm()
        self.conv1 = CausalConv3d(inChannels: inChannels, outChannels: outChannels,
                                   kernelSize: 3, causal: causal)
        self.norm2 = VAERMSNorm()
        self.conv2 = CausalConv3d(inChannels: outChannels, outChannels: outChannels,
                                   kernelSize: 3, causal: causal)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h: MLXArray = norm1(x)
        h = silu(h)
        h = conv1(h)
        h = norm2(h)
        h = silu(h)
        h = conv2(h)
        return h + x
    }
}

// MARK: - Channel Projection Block

/// ResNet-style block for channel changes between stages.
/// Weight key paths: conv1.conv.{w,b}, conv2.conv.{w,b},
///   conv_shortcut.conv.{w,b}, norm3.{weight,bias}
/// Property names match safetensors keys: conv1, conv2, conv_shortcut, norm3.
public class ChannelProjection: Module {
    let norm1: VAERMSNorm
    let conv1: CausalConv3d
    let norm2: VAERMSNorm
    let conv2: CausalConv3d
    // swiftlint:disable:next identifier_name
    let conv_shortcut: CausalConv3d
    let norm3: LayerNorm

    public init(inChannels: Int, outChannels: Int, causal: Bool = true) {
        self.norm1 = VAERMSNorm()
        self.conv1 = CausalConv3d(inChannels: inChannels, outChannels: outChannels,
                                   kernelSize: 3, causal: causal)
        self.norm2 = VAERMSNorm()
        self.conv2 = CausalConv3d(inChannels: outChannels, outChannels: outChannels,
                                   kernelSize: 3, causal: causal)
        self.conv_shortcut = CausalConv3d(inChannels: inChannels, outChannels: outChannels,
                                           kernelSize: 1, stride: (1, 1, 1), causal: causal)
        self.norm3 = LayerNorm(dimensions: inChannels)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed: MLXArray = norm3(x)

        var h: MLXArray = norm1(normed)
        h = silu(h)
        h = conv1(h)
        h = norm2(h)
        h = silu(h)
        h = conv2(h)

        let skip: MLXArray = conv_shortcut(normed)
        return h + skip
    }
}

// MARK: - Upsampler wrapper

/// Wraps a CausalConv3d as `.conv` to match the safetensors key path
/// `upsamplers.0.conv.conv.weight`.
public class Upsampler: Module {
    let conv: CausalConv3d

    public init(inChannels: Int, outChannels: Int, causal: Bool = false) {
        self.conv = CausalConv3d(inChannels: inChannels, outChannels: outChannels,
                                  kernelSize: 3, causal: causal)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return conv(x)
    }
}
