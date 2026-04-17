// DiTTest.swift — Phase 3 validation
// Loads quantized DiT, runs forward pass with dummy inputs,
// verifies output shape [1, 1120, 128].
//
// Input: [B, seqLen, 128] latents + [B, textLen, 4096] text + [B] timestep
// seqLen = 7*10*16 = 1120, textLen = 128, batch = 1

import Foundation
import MLX
import MLXNN
import os

public struct DiTTest {
    private static let logger = Logger(subsystem: "com.ltxvideopad", category: "DiTTest")

    public static func run(modelDir: URL) async {
        print("Step 1: Creating LTXVideoTransformer...")

        let model: LTXVideoTransformer = LTXVideoTransformer()
        print("  Model created (28 blocks, inner_dim=2048)")

        // Step 2: Load weights
        print("Step 2: Loading quantized weights from \(modelDir.path)...")
        do {
            try loadDiTWeights(model: model, from: modelDir)
            print("  Weights loaded successfully")
        } catch {
            print("  ERROR loading weights: \(error)")
            return
        }

        print("  Memory after load: \(MemoryManager.shared.currentMemoryMB) MB")

        // Step 3: Forward pass with dummy inputs
        print("Step 3: Running forward pass...")

        let batch: Int = 1
        let seqLen: Int = 1120   // 7 * 10 * 16
        let textLen: Int = 128
        let inChannels: Int = 128
        let captionChannels: Int = 4096

        let latents: MLXArray = MLXArray.zeros([batch, seqLen, inChannels]).asType(.float16)
        let textEmbeds: MLXArray = MLXArray.zeros([batch, textLen, captionChannels]).asType(.float16)
        let timestep: MLXArray = MLXArray([Float(0.5)])  // single timestep

        print("  Latents shape: \(latents.shape)")
        print("  Text shape: \(textEmbeds.shape)")
        print("  Timestep shape: \(timestep.shape)")

        let startTime: Date = Date()
        let output: MLXArray = model(latents: latents, textEmbeds: textEmbeds, timestep: timestep)
        eval(output)
        let elapsed: Double = Date().timeIntervalSince(startTime)

        // Step 4: Validate
        let shape: [Int] = output.shape
        let expected: [Int] = [batch, seqLen, inChannels]

        print("Step 4: Validation")
        print("  Output shape: \(shape)")
        print("  Expected:     \(expected)")
        print("  Time: \(String(format: "%.2f", elapsed))s")
        print("  Memory: \(MemoryManager.shared.currentMemoryMB) MB")

        if shape == expected {
            print("  PASS — output shape matches [1, 1120, 128]")
        } else {
            print("  FAIL — shape mismatch")
        }
    }
}
