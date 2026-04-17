// VideoEncoder.swift
//
// LTX-Video VAE Encoder: image/video → latent space.
// Patchify → Conv3d → 4 DownBlocks → MidBlock → Conv_out → mean/logvar
//
// Property names match safetensors weight keys exactly.
// Key prefix: "encoder."
//
// Input:  [B, 3, F, H, W] video (normalized to [-1, 1])
// Output: [B, 128, (F-1)/8+1, H/32, W/32] latent
//
// Memory estimate at FP16: ~361 MB for encoder weights.
// Peak during forward: ~600 MB (weights + activations for 512x320).

import Foundation
import MLX
import MLXNN

// MARK: - Encoder Down Block

/// Encoder down block with property names matching safetensors keys:
///   resnets[i].conv{1,2}.conv.{weight,bias}
///   downsamplers[0].conv.{weight,bias}           (CausalConv3d)
///   conv_out.conv{1,2}.conv.{w,b}                (ChannelProjection)
///   conv_out.conv_shortcut.conv.{w,b}
///   conv_out.norm3.{weight,bias}
public class DownBlock3D: Module {
    let resnets: [ResnetBlock3D]
    let downsamplers: [CausalConv3d]
    // swiftlint:disable:next identifier_name
    let conv_out: ChannelProjection?

    public init(inChannels: Int, outChannels: Int, numLayers: Int,
                downsample: Bool, causal: Bool = true) {

        // ResNet blocks operate at inChannels
        var blocks: [ResnetBlock3D] = []
        for _ in 0..<numLayers {
            blocks.append(ResnetBlock3D(inChannels: inChannels, outChannels: inChannels,
                                         causal: causal))
        }
        self.resnets = blocks

        // Downsample with stride-2 conv (wrapped in array for key path downsamplers.0)
        if downsample {
            self.downsamplers = [
                CausalConv3d(inChannels: inChannels, outChannels: inChannels,
                             kernelSize: 3, stride: (2, 2, 2), causal: causal)
            ]
        } else {
            self.downsamplers = []
        }

        // Channel projection if in != out
        if inChannels != outChannels {
            self.conv_out = ChannelProjection(inChannels: inChannels, outChannels: outChannels,
                                              causal: causal)
        } else {
            self.conv_out = nil
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h: MLXArray = x
        for resnet in resnets {
            h = resnet(h)
        }
        for ds in downsamplers {
            h = ds(h)
        }
        if let co = conv_out {
            h = co(h)
        }
        return h
    }
}

// MARK: - Mid Block

/// Mid block: N ResNet blocks at the bottleneck.
/// Key path: mid_block.resnets[i].conv{1,2}.conv.{weight,bias}
public class MidBlock3D: Module {
    let resnets: [ResnetBlock3D]

    public init(channels: Int, numLayers: Int, causal: Bool = true) {
        var blocks: [ResnetBlock3D] = []
        for _ in 0..<numLayers {
            blocks.append(ResnetBlock3D(inChannels: channels, outChannels: channels,
                                         causal: causal))
        }
        self.resnets = blocks
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h: MLXArray = x
        for resnet in resnets {
            h = resnet(h)
        }
        return h
    }
}

// MARK: - Video Encoder

/// Complete LTX-Video encoder: patchify → conv_in → down_blocks → mid_block → conv_out
///
/// Property names match safetensors key prefix "encoder.":
///   conv_in.conv.{weight,bias}
///   down_blocks[i].*
///   mid_block.resnets[i].*
///   conv_out.conv.{weight,bias}
///
/// Peak memory (FP16): ~600 MB for 512x320 single-frame encoding.
public class VideoEncoder: Module {
    // swiftlint:disable:next identifier_name
    let conv_in: CausalConv3d
    // swiftlint:disable:next identifier_name
    let down_blocks: [DownBlock3D]
    // swiftlint:disable:next identifier_name
    let mid_block: MidBlock3D
    let normOut: VAERMSNorm
    // swiftlint:disable:next identifier_name
    let conv_out: CausalConv3d

    let patchSize: Int
    let latentChannels: Int  // 128

    public init(
        inChannels: Int = 3,
        latentChannels: Int = 128,
        patchSize: Int = 4
    ) {
        self.patchSize = patchSize
        self.latentChannels = latentChannels

        let patchedChannels: Int = inChannels * patchSize * patchSize  // 3 * 4 * 4 = 48

        // conv_in: patchified channels → 128
        self.conv_in = CausalConv3d(inChannels: patchedChannels, outChannels: 128,
                                     kernelSize: 3, causal: true)

        // Down blocks — structure from safetensors:
        // block 0: 4 resnets @ 128, downsample, project 128→256
        // block 1: 3 resnets @ 256, downsample, project 256→512
        // block 2: 3 resnets @ 512, downsample, no projection
        // block 3: 3 resnets @ 512, no downsample, no projection
        self.down_blocks = [
            DownBlock3D(inChannels: 128, outChannels: 256, numLayers: 4,
                        downsample: true, causal: true),
            DownBlock3D(inChannels: 256, outChannels: 512, numLayers: 3,
                        downsample: true, causal: true),
            DownBlock3D(inChannels: 512, outChannels: 512, numLayers: 3,
                        downsample: true, causal: true),
            DownBlock3D(inChannels: 512, outChannels: 512, numLayers: 3,
                        downsample: false, causal: true),
        ]

        // Mid block: 4 resnets @ 512
        self.mid_block = MidBlock3D(channels: 512, numLayers: 4, causal: true)

        // Output: norm + conv to 129 channels (128 mean + 1 logvar)
        self.normOut = VAERMSNorm()
        self.conv_out = CausalConv3d(inChannels: 512, outChannels: latentChannels + 1,
                                      kernelSize: 3, causal: true)
    }

    /// Encode video/image to latent space.
    /// - Parameter x: [B, C, F, H, W] in channels-first (will be transposed internally)
    /// - Returns: latent mean [B, latentChannels, F', H', W']
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Transpose from [B, C, F, H, W] to MLX channels-last [B, F, H, W, C]
        var h: MLXArray = x.transposed(0, 2, 3, 4, 1)

        // Patchify: [B, F, H, W, C] → [B, F, H/p, W/p, C*p*p]
        h = patchify(h)

        // Encoder forward
        h = conv_in(h)

        for block in down_blocks {
            h = block(h)
        }

        h = mid_block(h)
        h = normOut(h)
        h = silu(h)
        h = conv_out(h)

        // Transpose to channels-first: [B, F', H', W', C] → [B, C, F', H', W']
        h = h.transposed(0, 4, 1, 2, 3)

        // Take mean (first 128 channels), discard logvar (channel 129)
        let mean: MLXArray = h[0..., 0..<latentChannels]
        return mean
    }

    /// Spatial patchification: rearrange pixels into channel dimension.
    /// [B, F, H, W, C] → [B, F, H/p, W/p, C*p*p]
    private func patchify(_ x: MLXArray) -> MLXArray {
        let b: Int = x.dim(0)
        let f: Int = x.dim(1)
        let h: Int = x.dim(2)
        let w: Int = x.dim(3)
        let c: Int = x.dim(4)
        let p: Int = patchSize

        let reshaped: MLXArray = x.reshaped(b, f, h / p, p, w / p, p, c)
        let permuted: MLXArray = reshaped.transposed(0, 1, 2, 4, 3, 5, 6)
        return permuted.reshaped(b, f, h / p, w / p, c * p * p)
    }
}
