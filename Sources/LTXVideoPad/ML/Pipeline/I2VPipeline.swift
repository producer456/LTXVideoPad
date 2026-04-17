// I2VPipeline.swift — Phase 4
//
// End-to-end image-to-video pipeline using validated models:
// 1. T5 encode prompt → [1, 128, 4096] text embeddings
// 2. VAE encode image → [1, 128, 1, H', W'] latent
// 3. DiT denoise 8 steps → [1, 128, F', H', W'] full video latent
// 4. VAE decode → [1, 3, F, H, W] video frames
//
// Models loaded/unloaded sequentially to stay within 8 GB.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Tokenizers
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

    // Computed latent dimensions
    public var latentF: Int { (numFrames - 1) / 8 + 1 }  // 7
    public var latentH: Int { height / 32 }                // 10
    public var latentW: Int { width / 32 }                 // 16
    public var seqLen: Int { latentF * latentH * latentW } // 1120

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

    /// Run the full I2V pipeline with a text prompt.
    /// Tokenizes the prompt internally using the T5 tokenizer.
    /// - Parameters:
    ///   - prompt: text description of the desired video
    ///   - progress: callback (step, totalSteps, description)
    /// - Returns: video frames [F, H, W, 3] in [0, 1]
    public func generate(
        prompt: String,
        progress: ((Int, Int, String) -> Void)? = nil
    ) async throws -> MLXArray {
        // Tokenize the prompt
        logger.info("Tokenizing prompt: \"\(prompt)\"")
        let tokenizer: Tokenizer = try T5TokenizerLoader.load(from: t5Dir)
        let tokenIds: [Int32] = T5TokenizerLoader.encode(tokenizer: tokenizer, text: prompt)
        logger.info("Token IDs (first 10): \(Array(tokenIds.prefix(10)))")

        return try await generate(tokenIds: tokenIds, progress: progress)
    }

    /// Run the full I2V pipeline with pre-tokenized IDs.
    /// - Parameters:
    ///   - tokenIds: pre-tokenized prompt as [Int32] (max 128 tokens)
    ///   - progress: callback (step, totalSteps, description)
    /// - Returns: video frames [F, H, W, 3] in [0, 1]
    public func generate(
        tokenIds: [Int32],
        progress: ((Int, Int, String) -> Void)? = nil
    ) async throws -> MLXArray {
        let lf = latentF
        let lh = latentH
        let lw = latentW
        let sl = seqLen
        let ns = numSteps

        // ── Step 1: T5 text encoding ──
        progress?(1, 4, "Encoding prompt...")
        logger.info("Step 1: T5 text encoding")
        memMgr.willLoad(model: "T5-XXL")

        let textEmbeddings: MLXArray = try encodeText(tokenIds: tokenIds)
        logger.info("Text embeddings: \(textEmbeddings.shape)")
        memMgr.didUnload(model: "T5-XXL")

        // ── Step 2: Prepare initial noise ──
        progress?(2, 4, "Preparing latents...")
        logger.info("Step 2: Initial noise [1, \(sl), 128]")

        // For I2V, we'd encode the input image here.
        // For now, start from pure noise (text-to-video mode).
        let initialNoise: MLXArray = MLXRandom.normal([1, sl, 128])
        eval(initialNoise)

        // ── Step 3: DiT denoising ──
        progress?(3, 4, "Generating video (\(ns) steps)...")
        logger.info("Step 3: DiT denoising — \(ns) steps, seqLen=\(sl)")
        memMgr.willLoad(model: "DiT")

        let denoisedLatents: MLXArray = try denoise(
            noise: initialNoise,
            textEmbeddings: textEmbeddings,
            numSteps: ns
        )
        logger.info("Denoised latents: \(denoisedLatents.shape)")
        memMgr.didUnload(model: "DiT")

        // Reshape from [1, seqLen, 128] to [1, 128, F', H', W']
        let latents5D: MLXArray = denoisedLatents
            .reshaped(1, lf, lh, lw, 128)
            .transposed(0, 4, 1, 2, 3)

        // ── Step 4: VAE decode ──
        progress?(4, 4, "Decoding video...")
        logger.info("Step 4: VAE decode")
        memMgr.willLoad(model: "VAE-Decoder")

        let videoFrames: MLXArray = try decodeLatents(latents5D)
        logger.info("Video output: \(videoFrames.shape)")
        memMgr.didUnload(model: "VAE-Decoder")

        // Convert from [1, 3, F, H, W] to [F, H, W, 3] and trim to requested frame count
        let nf = self.numFrames
        let allFrames: MLXArray = videoFrames
            .squeezed(axis: 0)           // [3, F, H, W]
            .transposed(1, 2, 3, 0)      // [F, H, W, 3]
        // VAE decoder may output more frames than requested (due to 2^3 upsampling)
        let frames: MLXArray = allFrames[0..<nf]  // trim to exact frame count
        let clamped: MLXArray = clip(frames * 0.5 + 0.5, min: 0.0, max: 1.0)
        eval(clamped)

        logger.info("Pipeline complete: \(clamped.shape)")
        return clamped
    }

    // MARK: - Pipeline Steps

    private func encodeText(tokenIds: [Int32]) throws -> MLXArray {
        // Pad to 128 tokens
        var ids = tokenIds
        while ids.count < 128 { ids.append(0) }
        if ids.count > 128 { ids = Array(ids.prefix(128)) }

        let model = T5EncoderModel()
        try loadT5Weights(model: model, from: t5Dir)

        let input = MLXArray(ids).reshaped(1, 128)
        let output = model(input)
        eval(output)

        // Model deallocated when it goes out of scope
        return output
    }

    private func denoise(noise: MLXArray, textEmbeddings: MLXArray,
                          numSteps: Int) throws -> MLXArray {
        let model = LTXVideoTransformer()
        try loadDiTWeights(model: model, from: ditDir)

        let sampler = FlowMatchSampler(numSteps: numSteps)

        // Compute 3D RoPE once for all steps
        // Peak memory: negligible (~few KB for position embeddings)
        let rope: MLXArray = RoPE3D.build(
            framesF: latentF,
            heightH: latentH,
            widthW: latentW,
            headDim: 64  // DiTConfig.v096.headDim
        )
        eval(rope)
        logger.info("3D RoPE: \(rope.shape)")

        // Start from noise
        var latents = noise

        let sigmas = sampler.sigmaSchedule(seqLen: latents.dim(1))

        for i in 0..<numSteps {
            let sigma = sigmas[i]
            let sigmaNext = sigmas[i + 1]
            let dt = sigmaNext - sigma

            let timestep = MLXArray([sigma * 1000.0])

            // DiT forward: predict velocity (with 3D RoPE)
            let velocity = model(
                latents: latents,
                textEmbeds: textEmbeddings,
                timestep: timestep,
                rope: rope
            )
            eval(velocity)

            // Euler step
            latents = latents + MLXArray(dt) * velocity
            eval(latents)

            logger.info("  Step \(i + 1)/\(numSteps): sigma=\(String(format: "%.3f", sigma))")
        }

        // Model deallocated when it goes out of scope
        return latents
    }

    private func decodeLatents(_ latents: MLXArray) throws -> MLXArray {
        let vaeModel = VAEModel()
        try loadVAEWeights(model: vaeModel, fromDirectory: vaeDir)

        // Decode: [1, 128, F', H', W'] → [1, 3, F, H, W]
        let decoded = vaeModel.decoder(latents)
        eval(decoded)

        // Model deallocated when it goes out of scope
        return decoded
    }
}
