// VideoDecoder.swift
//
// LTX-Video VAE Decoder: latent space → video frames.
// Conv_in → MidBlock → 4 UpBlocks → Conv_out → Unpatchify
//
// Input:  [B, 128, F', H', W'] latent
// Output: [B, 3, F, H, W] video (F=49, H=320, W=512 at full res)
//
// The decoder uses non-causal (symmetric) temporal padding.
// Upsampling is done via depth-to-space (channel → spatial expansion).
//
// Memory estimate at FP16: ~478 MB for decoder weights.
// Peak during forward: ~800 MB (weights + activations for full video decode).

import Foundation
import MLX
import MLXNN

// MARK: - Up Block

/// Decoder up block: optional channel projection + N ResNet blocks + optional upsample.
/// Upsampling uses depth-to-space: conv to ch*8, then rearrange to 2x2x2 spatial expansion.
public class UpBlock3D: Module {
    let convIn: ResnetBlock3D?       // channel projection (if needed)
    let resnets: [ResnetBlock3D]
    let upsampleConv: CausalConv3d?  // ch → ch*8 for depth-to-space

    let inChannels: Int
    let outChannels: Int

    public init(inChannels: Int, outChannels: Int, numLayers: Int,
                upsample: Bool, causal: Bool = false) {
        self.inChannels = inChannels
        self.outChannels = outChannels

        // Channel projection if needed
        if inChannels != outChannels {
            self.convIn = ResnetBlock3D(inChannels: inChannels, outChannels: outChannels, causal: causal)
        } else {
            self.convIn = nil
        }

        // ResNet blocks at outChannels
        var blocks: [ResnetBlock3D] = []
        for _ in 0..<numLayers {
            blocks.append(ResnetBlock3D(inChannels: outChannels, outChannels: outChannels, causal: causal))
        }
        self.resnets = blocks

        // Upsample via depth-to-space
        if upsample {
            self.upsampleConv = CausalConv3d(
                inChannels: outChannels, outChannels: outChannels * 8,
                kernelSize: 3, causal: causal
            )
        } else {
            self.upsampleConv = nil
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h: MLXArray = x

        // Channel projection
        if let ci = convIn {
            h = ci(h)
        }

        // ResNet blocks
        for resnet in resnets {
            h = resnet(h)
        }

        // Upsample via depth-to-space
        if let usConv = upsampleConv {
            h = usConv(h)
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

        // Reshape: [B, D, H, W, s, s, s, C]
        let reshaped: MLXArray = x.reshaped(b, d, h, w, s, s, s, c)
        // Permute: [B, D, s, H, s, W, s, C]
        let permuted: MLXArray = reshaped.transposed(0, 1, 4, 2, 5, 3, 6, 7)
        // Merge: [B, D*s, H*s, W*s, C]
        return permuted.reshaped(b, d * s, h * s, w * s, c)
    }
}

// MARK: - Video Decoder

/// Complete LTX-Video decoder: conv_in → mid → up_blocks → norm → conv_out → unpatchify
///
/// Peak memory (FP16): ~800 MB for full 49-frame video decode.
public class VideoDecoder: Module {
    let convIn: CausalConv3d
    let midBlock: MidBlock3D
    let upBlocks: [UpBlock3D]
    let normOut: VAERMSNorm
    let convOut: CausalConv3d

    let patchSize: Int
    let outChannels: Int  // 3

    public init(
        outChannels: Int = 3,
        latentChannels: Int = 128,
        blockOutChannels: [Int] = [128, 256, 512, 512],
        layersPerBlock: [Int] = [4, 3, 3, 3, 4],  // first = mid, rest = up blocks
        spatioTemporalScaling: [Bool] = [true, true, true, false],
        patchSize: Int = 4
    ) {
        self.patchSize = patchSize
        self.outChannels = outChannels

        let bottleneckChannels: Int = blockOutChannels.last!  // 512

        // conv_in: latent channels → bottleneck
        self.convIn = CausalConv3d(inChannels: latentChannels, outChannels: bottleneckChannels,
                                    kernelSize: 3, causal: false)  // decoder is non-causal

        // Mid block
        self.midBlock = MidBlock3D(channels: bottleneckChannels,
                                    numLayers: layersPerBlock[0], causal: false)

        // Up blocks (reversed order from encoder)
        let reversedChannels: [Int] = Array(blockOutChannels.reversed())
        let reversedScaling: [Bool] = Array(spatioTemporalScaling.reversed())

        var blocks: [UpBlock3D] = []
        var currentChannels: Int = bottleneckChannels
        for i in 0..<reversedChannels.count {
            let outCh: Int = reversedChannels[i]
            let numLayers: Int = layersPerBlock[i + 1]
            let upsample: Bool = reversedScaling[i]
            blocks.append(UpBlock3D(
                inChannels: currentChannels, outChannels: outCh,
                numLayers: numLayers, upsample: upsample, causal: false
            ))
            currentChannels = outCh
        }
        self.upBlocks = blocks

        // Output: norm + conv to patched pixel channels
        let patchedChannels: Int = outChannels * patchSize * patchSize  // 3 * 4 * 4 = 48
        self.normOut = VAERMSNorm()
        self.convOut = CausalConv3d(inChannels: currentChannels, outChannels: patchedChannels,
                                     kernelSize: 3, causal: false)
    }

    /// Decode latent to video.
    /// - Parameter z: [B, C, F', H', W'] latent (channels-first)
    /// - Returns: [B, 3, F, H, W] video
    public func callAsFunction(_ z: MLXArray) -> MLXArray {
        // Transpose to channels-last: [B, F', H', W', C]
        var h: MLXArray = z.transposed(0, 2, 3, 4, 1)

        h = convIn(h)
        h = midBlock(h)

        for block in upBlocks {
            h = block(h)
        }

        h = normOut(h)
        h = silu(h)
        h = convOut(h)

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
        let hp: Int = x.dim(2)  // H / patchSize
        let wp: Int = x.dim(3)  // W / patchSize
        let p: Int = patchSize
        let c: Int = outChannels  // 3

        // Reshape: [B, F, H', W', p, p, C]
        let reshaped: MLXArray = x.reshaped(b, f, hp, wp, p, p, c)
        // Permute: [B, F, H', p, W', p, C]
        let permuted: MLXArray = reshaped.transposed(0, 1, 2, 4, 3, 5, 6)
        // Merge: [B, F, H'*p, W'*p, C]
        return permuted.reshaped(b, f, hp * p, wp * p, c)
    }
}
