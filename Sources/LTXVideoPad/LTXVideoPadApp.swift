// LTXVideoPad — Phase 1 Test Runner
// Runs T5 encoder test headless (no UI needed for validation)

import Foundation
import MLX

@main
struct LTXVideoPadCLI {
    static func main() async {
        print("=== LTXVideoPad — Phase 1 T5 Encoder Test ===")
        print("Memory: \(MemoryManager.shared.currentMemoryMB) MB")

        let modelDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("Models/t5xxl-encoder-4bit")

        if !FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("model.safetensors").path) {
            print("ERROR: model.safetensors not found at \(modelDir.path)")
            print("Run: python3 scripts/download_and_quantize_t5.py")
            return
        }

        print("Model directory: \(modelDir.path)")
        await T5EncoderTest.run(modelDir: modelDir)
    }
}
