// VAETest.swift — Phase 2 validation
// Loads VAE weights, runs encoder and decoder forward passes,
// verifies output shapes match expected dimensions.

import Foundation
import MLX
import MLXNN
import MLXRandom
import os

public struct VAETest {
    private static let logger: Logger = Logger(subsystem: "com.ltxvideopad", category: "VAETest")

    public static func run(modelDir: URL) async {
        print("\n=== VAE Encode/Decode Test ===")

        // Step 1: Create model
        print("Step 1: Creating VAEModel...")
        let model: VAEModel = VAEModel()
        print("  Model created (encoder + decoder)")

        // Step 2: Load weights
        print("Step 2: Loading weights from \(modelDir.path)...")
        do {
            try loadVAEWeights(model: model, fromDirectory: modelDir)
            print("  Weights loaded successfully")
        } catch {
            print("  ERROR loading weights: \(error)")
            return
        }

        print("  Memory after load: \(MemoryManager.shared.currentMemoryMB) MB")

        // Step 3: Encoder test
        // Input: [1, 3, 1, 320, 512] (single frame, channels-first)
        // Expected output: [1, 129, 1, 10, 16]
        //   H: 320 / 4 (patch) / 2 / 2 / 2 = 320 / 32 = 10
        //   W: 512 / 4 (patch) / 2 / 2 / 2 = 512 / 32 = 16
        //   F: 1 frame stays 1 (stride-2 on temporal, but ceil division)
        print("Step 3: Encoder forward pass...")

        let encoderInput: MLXArray = MLXRandom.normal([1, 3, 1, 320, 512])
        let startEnc: Date = Date()
        let encoded: MLXArray = model.encoder(encoderInput)
        eval(encoded)
        let encTime: Double = Date().timeIntervalSince(startEnc)

        let encShape: [Int] = encoded.shape
        let expectedEnc: [Int] = [1, 128, 1, 10, 16]

        print("  Input shape:    [1, 3, 1, 320, 512]")
        print("  Output shape:   \(encShape)")
        print("  Expected shape: \(expectedEnc)")
        print("  Time: \(String(format: "%.2f", encTime))s")

        if encShape == expectedEnc {
            print("  PASS — encoder output shape matches")
        } else {
            print("  FAIL — encoder shape mismatch")
        }

        // Step 4: Decoder test
        // Input: [1, 128, 1, 10, 16] (latent, channels-first)
        // Expected output: [1, 3, 8, 320, 512]
        //   3 depth-to-space upsamplers (2x2x2 each): temporal 1→2→4→8
        //   Spatial: 10→20→40→80 (*4 unpatch = 320), 16→32→64→128 (*4 = 512)
        print("Step 4: Decoder forward pass...")

        let decoderInput: MLXArray = MLXRandom.normal([1, 128, 1, 10, 16])
        let startDec: Date = Date()
        let decoded: MLXArray = model.decoder(decoderInput)
        eval(decoded)
        let decTime: Double = Date().timeIntervalSince(startDec)

        let decShape: [Int] = decoded.shape
        let expectedDec: [Int] = [1, 3, 8, 320, 512]

        print("  Input shape:    [1, 128, 1, 10, 16]")
        print("  Output shape:   \(decShape)")
        print("  Expected shape: \(expectedDec)")
        print("  Time: \(String(format: "%.2f", decTime))s")

        if decShape == expectedDec {
            print("  PASS — decoder output shape matches")
        } else {
            print("  FAIL — decoder shape mismatch")
        }

        print("  Memory: \(MemoryManager.shared.currentMemoryMB) MB")

        // Summary
        let encOk: Bool = (encShape == expectedEnc)
        let decOk: Bool = (decShape == expectedDec)
        if encOk && decOk {
            print("\nVAE Test: ALL PASS")
        } else {
            print("\nVAE Test: FAILED")
        }
    }
}
