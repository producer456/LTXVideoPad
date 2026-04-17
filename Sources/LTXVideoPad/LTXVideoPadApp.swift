// LTXVideoPad — Entry point
// CLI: LTXVideoPad t5|dit|vae|pipeline|all
// GUI: LTXVideoPad (no args — launches SwiftUI, requires app bundle)

import Foundation
import MLX
import SwiftUI

@main
struct LTXVideoPadCLI {
    static func main() async {
        let baseDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let args = CommandLine.arguments
        let testName = args.count > 1 ? args[1] : "ui"

        if testName == "ui" {
            // Can't launch SwiftUI from CLI executable — print instructions
            print("LTXVideoPad — On-device video generation")
            print("")
            print("CLI test commands:")
            print("  LTXVideoPad t5       — test T5 encoder")
            print("  LTXVideoPad dit      — test DiT backbone")
            print("  LTXVideoPad vae      — test VAE encode/decode")
            print("  LTXVideoPad pipeline — test full end-to-end")
            print("  LTXVideoPad all      — run all tests")
            return
        }

        if testName == "t5" || testName == "all" {
            let dir = baseDir.appendingPathComponent("Models/t5xxl-encoder-4bit")
            if FileManager.default.fileExists(atPath: dir.appendingPathComponent("model.safetensors").path) {
                print("\n=== T5 Encoder Test ===")
                await T5EncoderTest.run(modelDir: dir)
            } else {
                print("T5 weights not found — skipping")
            }
        }

        if testName == "dit" || testName == "all" {
            let dir = baseDir.appendingPathComponent("Models/dit-4bit")
            if FileManager.default.fileExists(atPath: dir.appendingPathComponent("model.safetensors").path) {
                print("\n=== DiT Backbone Test ===")
                await DiTTest.run(modelDir: dir)
            } else {
                print("DiT weights not found — skipping")
            }
        }

        if testName == "vae" || testName == "all" {
            let dir = baseDir.appendingPathComponent("Models/vae/vae")
            if FileManager.default.fileExists(atPath: dir.appendingPathComponent("diffusion_pytorch_model.safetensors").path) {
                await VAETest.run(modelDir: dir)
            } else {
                print("VAE weights not found — skipping")
            }
        }

        if testName == "pipeline" || testName == "e2e" || testName == "all" {
            let t5Dir = baseDir.appendingPathComponent("Models/t5xxl-encoder-4bit")
            let vaeDir = baseDir.appendingPathComponent("Models/vae/vae")
            let ditDir = baseDir.appendingPathComponent("Models/dit-4bit")

            print("\n=== End-to-End Pipeline Test ===")
            await PipelineTest.run(t5Dir: t5Dir, vaeDir: vaeDir, ditDir: ditDir)
        }

        print("\nMemory: \(MemoryManager.shared.currentMemoryMB) MB")
    }
}
