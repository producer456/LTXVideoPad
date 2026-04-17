// VideoEncoder.swift
//
// LTX-Video VAE Encoder: image/video → latent space.
// Patchify → Conv3d → 4 DownBlocks → MidBlock → Conv_out → mean/logvar
//
// Input:  [B, 3, F, H, W] video (normalized to [-1, 1])
// Output: [B, 128, (F-1)/8+1, H/32, W/32] latent
//
// For I2V, input is a single frame: [B, 3, 1, H, W]
// Output: [B, 128, 1, H/32, W/32]
//
// Memory estimate at FP16: ~361 MB for encoder weights.
// Peak during forward: ~600 MB (weights + activations for 512x320).

import Foundation
import MLX
import MLXNN

// MARK: - Down Block

/// Encoder down block: N ResNet blocks + optional strided downsample.
public class DownBlock3D: Module {
    let resnets: [ResnetBlock3D]
    let downsampler: CausalConv3d?
    let convOut: ResnetBlock3D?

    /// - Parameters:
    ///   - inChannels: input channel count
    ///   - outChannels: output channel count (may differ from inChannels)
    ///   - numLayers: number of ResNet blocks
    ///   - downsample: whether to include a stride-2 downsample
    ///   - causal: use causal temporal padding
    public init(inChannels: Int, outChannels: Int, numLayers: Int,
                downsample: Bool, causal: Bool = true) {

        // ResNet blocks operate at inChannels
        var blocks: [ResnetBlock3D] = []
        for _ in 0..<numLayers {
            blocks.append(ResnetBlock3D(inChannels: inChannels, outChannels: inChannels, causal: causal))
        }
        self.resnets = blocks

        // Downsample with stride-2 conv
        if downsample {
            self.downsampler = CausalConv3d(
                inChannels: inChannels, outChannels: inChannels,
                kernelSize: 3, stride: (2, 2, 2), causal: causal
            )
        } else {
            self.downsampler = nil
        }

        // Channel projection if in != out
        if inChannels != outChannels {
            self.convOut = ResnetBlock3D(inChannels: inChannels, outChannels: outChannels, causal: causal)
        } else {
            self.convOut = nil
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h: MLXArray = x
        for resnet in resnets {
            h = resnet(h)
        }
        if let ds = downsampler {
            h = ds(h)
        }
        if let co = convOut {
            h = co(h)
        }
        return h
    }
}

// MARK: - Mid Block

/// Mid block: N ResNet blocks at the bottleneck (512 channels).
public class MidBlock3D: Module {
    let resnets: [ResnetBlock3D]

    public init(channels: Int, numLayers: Int, causal: Bool = true) {
        var blocks: [ResnetBlock3D] = []
        for _ in 0..<numLayers {
            blocks.append(ResnetBlock3D(inChannels: channels, outChannels: channels, causal: causal))
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

/// Complete LTX-Video encoder: patchify → conv_in → down_blocks → mid → conv_out
///
/// Peak memory (FP16): ~600 MB for 512x320 single-frame encoding.
public class VideoEncoder: Module {
    let convIn: CausalConv3d
    let downBlocks: [DownBlock3D]
    let midBlock: MidBlock3D
    let normOut: VAERMSNorm
    let convOut: CausalConv3d

    let patchSize: Int
    let latentChannels: Int  // 128

    public init(
        inChannels: Int = 3,
        latentChannels: Int = 128,
        blockOutChannels: [Int] = [128, 256, 512, 512],
        layersPerBlock: [Int] = [4, 3, 3, 3, 4],  // first = mid, rest = down blocks
        spatioTemporalScaling: [Bool] = [true, true, true, false],
        patchSize: Int = 4
    ) {
        self.patchSize = patchSize
        self.latentChannels = latentChannels

        let patchedChannels: Int = inChannels * patchSize * patchSize  // 3 * 4 * 4 = 48

        // conv_in: patchified channels → first block channels
        self.convIn = CausalConv3d(inChannels: patchedChannels, outChannels: blockOutChannels[0],
                                    kernelSize: 3, causal: true)

        // Down blocks
        var blocks: [DownBlock3D] = []
        var currentChannels: Int = blockOutChannels[0]
        for i in 0..<blockOutChannels.count {
            let outCh: Int = blockOutChannels[i]
            let numLayers: Int = layersPerBlock[i + 1]  // offset by 1 (first is mid)
            let downsample: Bool = spatioTemporalScaling[i]
            blocks.append(DownBlock3D(
                inChannels: currentChannels, outChannels: outCh,
                numLayers: numLayers, downsample: downsample, causal: true
            ))
            currentChannels = outCh
        }
        self.downBlocks = blocks

        // Mid block
        self.midBlock = MidBlock3D(channels: currentChannels,
                                    numLayers: layersPerBlock[0], causal: true)

        // Output: norm + conv to latent channels (128 mean + 1 logvar = 129)
        self.normOut = VAERMSNorm()
        self.convOut = CausalConv3d(inChannels: currentChannels,
                                     outChannels: latentChannels + 1,  // 129
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
        h = convIn(h)

        for block in downBlocks {
            h = block(h)
        }

        h = midBlock(h)
        h = normOut(h)
        h = silu(h)
        h = convOut(h)

        // Split into mean and logvar
        // Output has latentChannels + 1 channels; take first latentChannels as mean
        // Transpose back to channels-first: [B, F', H', W', C] → [B, C, F', H', W']
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

        // Reshape: [B, F, H/p, p, W/p, p, C] then merge patch dims into channels
        let reshaped: MLXArray = x.reshaped(b, f, h / p, p, w / p, p, c)
        // Permute to [B, F, H/p, W/p, p, p, C] then flatten last 3 dims
        let permuted: MLXArray = reshaped.transposed(0, 1, 2, 4, 3, 5, 6)
        return permuted.reshaped(b, f, h / p, w / p, c * p * p)
    }
}
