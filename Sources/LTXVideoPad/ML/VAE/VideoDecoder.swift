// VideoDecoder.swift
//
// LTX-Video VAE Decoder: latent space → video frames.
// Conv_in → MidBlock → 4 UpBlocks → Conv_out → Unpatchify
//
// Property names match safetensors weight keys exactly.
// Key prefix: "decoder."
//
// Input:  [B, 128, F', H', W'] latent
// Output: [B, 3, F, H, W] video
//
// Memory estimate at FP16: ~478 MB for decoder weights.
// Peak during forward: ~800 MB (weights + activations for full video decode).

import Foundation
import MLX
import MLXNN

// MARK: - Up Block

/// Decoder up block with property names matching safetensors keys:
///   conv_in.conv{1,2}.conv.{w,b}                 (ChannelProjection, optional)
///   conv_in.conv_shortcut.conv.{w,b}
///   conv_in.norm3.{weight,bias}
///   resnets[i].conv{1,2}.conv.{weight,bias}
///   upsamplers[0].conv.conv.{weight,bias}         (Upsampler → CausalConv3d)
public class UpBlock3D: Module {
    // swiftlint:disable:next identifier_name
    let conv_in: ChannelProjection?
    let resnets: [ResnetBlock3D]
    let upsamplers: [Upsampler]

    let outChannels: Int

    public init(inChannels: Int, outChannels: Int, numLayers: Int,
                upsample: Bool, causal: Bool = false) {
        self.outChannels = outChannels

        // Channel projection if needed
        if inChannels != outChannels {
            self.conv_in = ChannelProjection(inChannels: inChannels, outChannels: outChannels,
                                              causal: causal)
        } else {
            self.conv_in = nil
        }

        // ResNet blocks at outChannels
        var blocks: [ResnetBlock3D] = []
        for _ in 0..<numLayers {
            blocks.append(ResnetBlock3D(inChannels: outChannels, outChannels: outChannels,
                                         causal: causal))
        }
        self.resnets = blocks

        // Upsample via depth-to-space (ch → ch*8, then rearrange)
        if upsample {
            self.upsamplers = [
                Upsampler(inChannels: outChannels, outChannels: outChannels * 8,
                          causal: causal)
            ]
        } else {
            self.upsamplers = []
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h: MLXArray = x

        // Channel projection
        if let ci = conv_in {
            h = ci(h)
        }

        // ResNet blocks
        for resnet in resnets {
            h = resnet(h)
        }

        // Upsample via depth-to-space
        for us in upsamplers {
            h = us(h)
            h = depthToSpace3D(h, blockSize: 2)
        }

        return h
    }

    /// Depth-to-space for 3D: rearrange channels into spatial+temporal dimensions.
    /// [B, D, H, W, C*s^3] → [B, D*s, H*s, W*s, C]
    private func depthToSpace3D(_ x: MLXArray, blockSize s: Int) -> MLXArray {
        let b: Int = x.dim(0)
        let d: Int = x.dim(1)
        let h: Int = x.dim(2)
        let w: Int = x.dim(3)
        let c: Int = x.dim(4) / (s * s * s)

        let reshaped: MLXArray = x.reshaped(b, d, h, w, s, s, s, c)
        let permuted: MLXArray = reshaped.transposed(0, 1, 4, 2, 5, 3, 6, 7)
        return permuted.reshaped(b, d * s, h * s, w * s, c)
    }
}

// MARK: - Video Decoder

/// Complete LTX-Video decoder: conv_in → mid_block → up_blocks → norm → conv_out → unpatchify
///
/// Property names match safetensors key prefix "decoder.":
///   conv_in.conv.{weight,bias}
///   mid_block.resnets[i].*
///   up_blocks[i].*
///   conv_out.conv.{weight,bias}
///
/// Peak memory (FP16): ~800 MB for full 49-frame video decode.
public class VideoDecoder: Module {
    // swiftlint:disable:next identifier_name
    let conv_in: CausalConv3d
    // swiftlint:disable:next identifier_name
    let mid_block: MidBlock3D
    // swiftlint:disable:next identifier_name
    let up_blocks: [UpBlock3D]
    let normOut: VAERMSNorm
    // swiftlint:disable:next identifier_name
    let conv_out: CausalConv3d

    let patchSize: Int
    let outChannels: Int  // 3

    public init(
        outChannels: Int = 3,
        latentChannels: Int = 128,
        patchSize: Int = 4
    ) {
        self.patchSize = patchSize
        self.outChannels = outChannels

        // conv_in: 128 latent → 512 bottleneck
        self.conv_in = CausalConv3d(inChannels: latentChannels, outChannels: 512,
                                     kernelSize: 3, causal: false)

        // Mid block: 4 resnets @ 512
        self.mid_block = MidBlock3D(channels: 512, numLayers: 4, causal: false)

        // Up blocks — structure from safetensors:
        // block 0: 3 resnets @ 512, no conv_in, no upsampler
        // block 1: 3 resnets @ 512, no conv_in, upsampler 512→4096
        // block 2: conv_in 512→256, 3 resnets @ 256, upsampler 256→2048
        // block 3: conv_in 256→128, 4 resnets @ 128, upsampler 128→1024
        self.up_blocks = [
            UpBlock3D(inChannels: 512, outChannels: 512, numLayers: 3,
                      upsample: false, causal: false),
            UpBlock3D(inChannels: 512, outChannels: 512, numLayers: 3,
                      upsample: true, causal: false),
            UpBlock3D(inChannels: 512, outChannels: 256, numLayers: 3,
                      upsample: true, causal: false),
            UpBlock3D(inChannels: 256, outChannels: 128, numLayers: 4,
                      upsample: true, causal: false),
        ]

        // Output: norm + conv to patched pixel channels (3 * 4 * 4 = 48)
        let patchedChannels: Int = outChannels * patchSize * patchSize
        self.normOut = VAERMSNorm()
        self.conv_out = CausalConv3d(inChannels: 128, outChannels: patchedChannels,
                                      kernelSize: 3, causal: false)
    }

    /// Decode latent to video.
    /// - Parameter z: [B, C, F', H', W'] latent (channels-first)
    /// - Returns: [B, 3, F, H, W] video
    public func callAsFunction(_ z: MLXArray) -> MLXArray {
        // Transpose to channels-last: [B, F', H', W', C]
        var h: MLXArray = z.transposed(0, 2, 3, 4, 1)

        h = conv_in(h)
        h = mid_block(h)

        for block in up_blocks {
            h = block(h)
        }

        h = normOut(h)
        h = silu(h)
        h = conv_out(h)

        // Unpatchify: [B, F, H/p, W/p, C*p*p] → [B, F, H, W, C]
        h = unpatchify(h)

        // Transpose to channels-first: [B, F, H, W, C] → [B, C, F, H, W]
        return h.transposed(0, 4, 1, 2, 3)
    }

    /// Inverse of patchification: rearrange channel dimension back to spatial pixels.
    /// [B, F, H', W', C*p*p] → [B, F, H'*p, W'*p, C]
    private func unpatchify(_ x: MLXArray) -> MLXArray {
        let b: Int = x.dim(0)
        let f: Int = x.dim(1)
        let hp: Int = x.dim(2)
        let wp: Int = x.dim(3)
        let p: Int = patchSize
        let c: Int = outChannels

        let reshaped: MLXArray = x.reshaped(b, f, hp, wp, p, p, c)
        let permuted: MLXArray = reshaped.transposed(0, 1, 2, 4, 3, 5, 6)
        return permuted.reshaped(b, f, hp * p, wp * p, c)
    }
}
