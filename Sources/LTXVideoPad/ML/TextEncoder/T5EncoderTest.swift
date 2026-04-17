// T5EncoderTest.swift
//
// Phase 1 validation: load T5-XXL encoder, tokenize a prompt, run forward pass,
// verify output shape is [1, 128, 4096].
//
// Usage: Call T5EncoderTest.run(modelDir:) with the path to quantized weights.
//
// Expected output:
//   Tokenized "a cat walking on grass" -> 7 tokens + EOS + padding = 128
//   Forward pass output shape: [1, 128, 4096]
//   First 5 values of output[0,0,:5]: [some floats]
//   PASS

import Foundation
import MLX
import MLXNN
import os

public struct T5EncoderTest {
    private static let logger = Logger(subsystem: "com.ltxvideopad", category: "T5Test")

    /// Run the Phase 1 validation test.
    /// - Parameter modelDir: URL to directory containing quantized weights + tokenizer
    public static func run(modelDir: URL) async {
        logger.info("=== T5 Encoder Phase 1 Test ===")
        logger.info("Model directory: \(modelDir.path)")

        let memMgr: MemoryManager = MemoryManager.shared
        memMgr.logMemory(context: "Before test")

        // Step 1: Load tokenizer
        logger.info("Step 1: Loading tokenizer...")
        let tokenizer: T5TokenizerWrapper
        do {
            tokenizer = try await T5TokenizerWrapper(from: modelDir)
            logger.info("Tokenizer loaded successfully")
        } catch {
            logger.error("Failed to load tokenizer: \(error.localizedDescription)")
            return
        }

        // Step 2: Tokenize test prompt
        let testPrompt: String = "a cat walking on grass"
        logger.info("Step 2: Tokenizing '\(testPrompt)'...")

        let (tokenIds, attentionMask) = tokenizer.encode(testPrompt)
        let realTokenCount: Int = attentionMask.reduce(0, +)
        logger.info("Token IDs (first 10): \(tokenIds.prefix(10))")
        logger.info("Real tokens: \(realTokenCount), padded to: \(tokenIds.count)")

        // Step 3: Create model
        logger.info("Step 3: Creating T5EncoderModel...")
        memMgr.willLoad(model: "T5-XXL")

        let config: T5Config = .xxl
        let model: T5EncoderModel = T5EncoderModel(config: config)
        logger.info("Model created with \(config.numLayers) layers, d_model=\(config.dModel)")

        // Step 4: Load weights
        logger.info("Step 4: Loading weights...")
        do {
            let weights: [String: MLXArray] = try loadT5Weights(from: modelDir)
            applyT5Weights(to: model, weights: weights)
            logger.info("Weights loaded and applied")
        } catch {
            logger.error("Failed to load weights: \(error.localizedDescription)")
            memMgr.didUnload(model: "T5-XXL")
            return
        }

        memMgr.logMemory(context: "After weight load")

        // Step 5: Forward pass
        logger.info("Step 5: Running forward pass...")

        // Convert token IDs to MLXArray: [1, seqLen]
        let inputIds: MLXArray = MLXArray(tokenIds.map { Int32($0) }).reshaped(1, tokenIds.count)

        let startTime: Date = Date()
        let output: MLXArray = model(inputIds)
        // Force evaluation (MLX is lazy)
        eval(output)
        let elapsed: Double = Date().timeIntervalSince(startTime)

        // Step 6: Validate output
        logger.info("Step 6: Validating output...")

        let shape: [Int] = output.shape
        logger.info("Output shape: \(shape)")
        logger.info("Expected:     [1, \(tokenIds.count), \(config.dModel)]")
        logger.info("Forward pass time: \(String(format: "%.2f", elapsed))s")

        let expectedShape: [Int] = [1, tokenIds.count, config.dModel]
        if shape == expectedShape {
            logger.info("PASS: Output shape matches expected")

            // Print first few values as a sanity check
            let firstValues: MLXArray = output[0, 0, 0..<5]
            logger.info("First 5 values of output[0,0,:5]: \(firstValues)")
        } else {
            logger.error("FAIL: Expected \(expectedShape), got \(shape)")
        }

        // Step 7: Cleanup
        memMgr.didUnload(model: "T5-XXL")
        memMgr.logMemory(context: "After test cleanup")

        logger.info("=== Test Complete ===")
    }
}
