// T5EncoderTest.swift — Phase 1 validation
// Loads quantized T5-XXL, runs forward pass with dummy tokens,
// verifies output shape [1, 128, 4096].

import Foundation
import MLX
import MLXNN
import os

public struct T5EncoderTest {
    private static let logger = Logger(subsystem: "com.ltxvideopad", category: "T5Test")

    public static func run(modelDir: URL) async {
        print("Step 1: Creating T5EncoderModel...")

        let model: T5EncoderModel = T5EncoderModel()
        print("  Model created (24 blocks, d_model=4096)")

        // Step 2: Load weights
        print("Step 2: Loading quantized weights from \(modelDir.path)...")
        do {
            try loadT5Weights(model: model, from: modelDir)
            print("  Weights loaded successfully")
        } catch {
            print("  ERROR loading weights: \(error)")
            return
        }

        print("  Memory after load: \(MemoryManager.shared.currentMemoryMB) MB")

        // Step 3: Forward pass with dummy tokens
        // Using hardcoded token IDs to avoid tokenizer dependency for now
        // "a cat" ≈ tokens [3, 9, 1712, 1] (approximate T5 tokenization)
        print("Step 3: Running forward pass...")

        let maxLen: Int = 128
        var tokenIds: [Int32] = [3, 9, 1712, 1]  // dummy tokens + EOS
        while tokenIds.count < maxLen {
            tokenIds.append(0)  // PAD
        }

        let inputIds: MLXArray = MLXArray(tokenIds).reshaped(1, maxLen)
        print("  Input shape: \(inputIds.shape)")

        let startTime: Date = Date()
        let output: MLXArray = model(inputIds)
        eval(output)
        let elapsed: Double = Date().timeIntervalSince(startTime)

        // Step 4: Validate
        let shape: [Int] = output.shape
        let expected: [Int] = [1, maxLen, 4096]

        print("Step 4: Validation")
        print("  Output shape: \(shape)")
        print("  Expected:     \(expected)")
        print("  Time: \(String(format: "%.2f", elapsed))s")
        print("  Memory: \(MemoryManager.shared.currentMemoryMB) MB")

        if shape == expected {
            print("  ✅ PASS — output shape matches")
        } else {
            print("  ❌ FAIL — shape mismatch")
        }
    }
}
