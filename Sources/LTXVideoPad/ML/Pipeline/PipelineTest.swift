// PipelineTest.swift — Phase 4 validation
// Runs the full T5→DiT→VAE pipeline end-to-end.
// Output is random (noise→denoise→decode) but validates shapes and memory.

import Foundation
import MLX
import os

public struct PipelineTest {
    public static func run(t5Dir: URL, vaeDir: URL, ditDir: URL) async {
        print("=== End-to-End Pipeline Test ===")
        print("  T5 dir:  \(t5Dir.path)")
        print("  VAE dir: \(vaeDir.path)")
        print("  DiT dir: \(ditDir.path)")
        print("  Target: 49 frames @ 320x512, 8 denoise steps")
        print("  Memory: \(MemoryManager.shared.currentMemoryMB) MB")

        let pipeline = I2VPipeline(
            t5Dir: t5Dir, vaeDir: vaeDir, ditDir: ditDir,
            numFrames: 49, height: 320, width: 512, numSteps: 2  // 2 steps for fast test
        )

        // Dummy token IDs (would come from tokenizer in real usage)
        let tokenIds: [Int32] = [3, 9, 1712, 3, 1536, 30, 3084, 1]  // "a cat walking on grass"

        let startTime = Date()

        do {
            let frames = try await pipeline.generate(tokenIds: tokenIds) { step, total, desc in
                print("  [\(step)/\(total)] \(desc)")
            }

            let elapsed = Date().timeIntervalSince(startTime)

            print("\nResults:")
            print("  Output shape: \(frames.shape)")
            print("  Expected:     [49, 320, 512, 3]")
            print("  Total time:   \(String(format: "%.1f", elapsed))s")
            print("  Peak memory:  \(MemoryManager.shared.currentMemoryMB) MB")

            // The actual frame content will be noise since we used 2 steps
            // and random starting point, but shape validation is the goal.
            let expectedShape = [49, 320, 512, 3]
            if frames.shape == expectedShape {
                print("  ✅ PIPELINE PASS — shapes correct through entire flow")
            } else {
                print("  ❌ PIPELINE FAIL — expected \(expectedShape), got \(frames.shape)")
            }
        } catch {
            print("  ❌ PIPELINE ERROR: \(error)")
        }
    }
}
