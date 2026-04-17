// I2VPipeline.swift — Phase 4
//
// End-to-end image-to-video pipeline.
// Memory budget: ~8 GB. Only one large model loaded at a time.

import Foundation
import MLX
import MLXNN
import MLXRandom
import os

public class I2VPipeline {
    private let logger = Logger(subsystem: "com.ltxvideopad", category: "Pipeline")
    private let memMgr: MemoryManager = .shared

    public let t5Dir: URL
    public let vaeDir: URL
    public let ditDir: URL
    public let numFrames: Int
    public let height: Int
    public let width: Int
    public let numSteps: Int

    public init(t5Dir: URL, vaeDir: URL, ditDir: URL,
                numFrames: Int = 49, height: Int = 320, width: Int = 512,
                numSteps: Int = 8) {
        self.t5Dir = t5Dir
        self.vaeDir = vaeDir
        self.ditDir = ditDir
        self.numFrames = numFrames
        self.height = height
        self.width = width
        self.numSteps = numSteps
    }

    /// Run the full I2V pipeline.
    public func generate(prompt: String, image: MLXArray,
                         progress: ((Int, Int, String) -> Void)? = nil) async throws -> MLXArray {
        let nf = self.numFrames
        let h = self.height
        let w = self.width
        let ns = self.numSteps

        // Step 1: T5 encode
        progress?(1, 4, "Encoding prompt...")
        logger.info("Step 1: T5 text encoding")
        memMgr.willLoad(model: "T5-XXL")

        let model = T5EncoderModel()
        try loadT5Weights(model: model, from: t5Dir)
        let tokenIds = MLXArray(Array(repeating: Int32(1), count: 128)).reshaped(1, 128)
        let textEmbeddings = model(tokenIds)
        eval(textEmbeddings)
        logger.info("Text embeddings: \(textEmbeddings.shape)")
        memMgr.didUnload(model: "T5-XXL")

        // Step 2: VAE encode image (placeholder)
        progress?(2, 4, "Encoding image...")
        logger.info("Step 2: VAE encode (placeholder)")
        let latentF = 1
        let latentH = h / 32
        let latentW = w / 32
        let imageLatent = MLXRandom.normal([1, 128, latentF, latentH, latentW])
        eval(imageLatent)

        // Step 3: DiT denoise (placeholder)
        progress?(3, 4, "Generating video...")
        logger.info("Step 3: DiT denoise (\(ns) steps)")
        let fullLatentF = (nf - 1) / 8 + 1
        let seqLen = fullLatentF * latentH * latentW

        let sampler = FlowMatchSampler(numSteps: ns)
        let latents = sampler.sample(shape: [1, seqLen, 128]) { noisy, t in
            return MLXArray.zeros(like: noisy)
        }
        let videoLatent = latents.reshaped(1, 128, fullLatentF, latentH, latentW)
        eval(videoLatent)

        // Step 4: VAE decode (placeholder)
        progress?(4, 4, "Decoding video...")
        logger.info("Step 4: VAE decode (placeholder)")
        let frames = MLXRandom.normal([nf, h, w, 3])
        eval(frames)

        logger.info("Pipeline complete: \(frames.shape)")
        return frames
    }
}
