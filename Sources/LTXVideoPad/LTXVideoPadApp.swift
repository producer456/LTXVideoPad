// LTXVideoPad — Test Runner
// Runs Phase 1 (T5) and Phase 3 (DiT) tests sequentially.

import Foundation
import MLX

@main
struct LTXVideoPadCLI {
    static func main() async {
        let baseDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        let args = CommandLine.arguments
        let testName = args.count > 1 ? args[1] : "all"

        if testName == "t5" || testName == "all" {
            let t5Dir = baseDir.appendingPathComponent("Models/t5xxl-encoder-4bit")
            if FileManager.default.fileExists(atPath: t5Dir.appendingPathComponent("model.safetensors").path) {
                print("\n=== T5 Encoder Test ===")
                await T5EncoderTest.run(modelDir: t5Dir)
            } else {
                print("T5 weights not found — skipping")
            }
        }

        if testName == "dit" || testName == "all" {
            let ditDir = baseDir.appendingPathComponent("Models/dit-4bit")
            if FileManager.default.fileExists(atPath: ditDir.appendingPathComponent("model.safetensors").path) {
                print("\n=== DiT Backbone Test ===")
                await DiTTest.run(modelDir: ditDir)
            } else {
                print("DiT weights not found — skipping")
            }
        }

        if testName == "vae" || testName == "all" {
            let vaeDir = baseDir.appendingPathComponent("Models/vae/vae")
            if FileManager.default.fileExists(atPath: vaeDir.appendingPathComponent("diffusion_pytorch_model.safetensors").path) {
                await VAETest.run(modelDir: vaeDir)
            } else {
                print("VAE weights not found — skipping")
            }
        }

        if testName == "pipeline" || testName == "e2e" {
            let t5Dir = baseDir.appendingPathComponent("Models/t5xxl-encoder-4bit")
            let vaeDir = baseDir.appendingPathComponent("Models/vae/vae")
            let ditDir = baseDir.appendingPathComponent("Models/dit-4bit")

            print("\n=== End-to-End Pipeline Test ===")
            await PipelineTest.run(t5Dir: t5Dir, vaeDir: vaeDir, ditDir: ditDir)
        }

        print("\nMemory: \(MemoryManager.shared.currentMemoryMB) MB")
    }
}
