// T5EncoderTest.swift
//
// Phase 1 validation: load quantized T5-XXL, tokenize a prompt, forward pass,
// verify output shape is [1, 128, 4096].
//
// Call from the app: await T5EncoderTest.run(modelDir: URL)

import Foundation
import MLX
import MLXNN
import Tokenizers
import Hub
import os

public struct T5EncoderTest {
    private static let logger = Logger(subsystem: "com.ltxvideopad", category: "T5Test")

    public static func run(modelDir: URL) async {
        logger.info("=== T5 Encoder Phase 1 Test ===")
        logger.info("Model directory: \(modelDir.path)")

        let memMgr: MemoryManager = MemoryManager.shared
        memMgr.logMemory(context: "Before test")

        // Step 1: Tokenize test prompt
        logger.info("Step 1: Tokenizing...")
        let testPrompt: String = "a cat walking on grass"

        let tokenizerFile: URL = modelDir.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerFile.path) else {
            logger.error("tokenizer.json not found at \(tokenizerFile.path)")
            return
        }

        let tokenizer: Tokenizer
        do {
            let tokConfig: URL = modelDir.appendingPathComponent("tokenizer_config.json")
            let config = LanguageModelConfigurationFromHub(modelFolder: modelDir)
            tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        } catch {
            logger.error("Failed to load tokenizer: \(error)")
            return
        }

        let encoded = tokenizer.encode(text: testPrompt)
        var tokenIds: [Int32] = encoded.map { Int32($0) }
        tokenIds.append(1) // EOS token
        let maxLen: Int = 128

        // Pad to maxLen
        while tokenIds.count < maxLen {
            tokenIds.append(0) // PAD
        }
        if tokenIds.count > maxLen {
            tokenIds = Array(tokenIds.prefix(maxLen))
        }

        logger.info("Tokenized '\(testPrompt)' -> \(encoded.count) tokens + EOS, padded to \(maxLen)")

        // Step 2: Create model
        logger.info("Step 2: Creating T5EncoderModel...")
        memMgr.willLoad(model: "T5-XXL")

        let model: T5EncoderModel = T5EncoderModel()
        logger.info("Model created with 24 layers")

        // Step 3: Load weights
        logger.info("Step 3: Loading quantized weights...")
        do {
            try loadT5Weights(model: model, from: modelDir)
        } catch {
            logger.error("Failed to load weights: \(error)")
            memMgr.didUnload(model: "T5-XXL")
            return
        }
        memMgr.logMemory(context: "After weight load")

        // Step 4: Forward pass
        logger.info("Step 4: Running forward pass...")
        let inputIds: MLXArray = MLXArray(tokenIds).reshaped(1, maxLen)

        let startTime: Date = Date()
        let output: MLXArray = model(inputIds)
        eval(output)
        let elapsed: Double = Date().timeIntervalSince(startTime)

        // Step 5: Validate
        let shape: [Int] = output.shape
        let expected: [Int] = [1, maxLen, 4096]

        logger.info("Output shape: \(shape)")
        logger.info("Expected:     \(expected)")
        logger.info("Time: \(String(format: "%.2f", elapsed))s")

        if shape == expected {
            logger.info("PASS")
        } else {
            logger.error("FAIL: shape mismatch")
        }

        memMgr.didUnload(model: "T5-XXL")
        logger.info("=== Test Complete ===")
    }
}

