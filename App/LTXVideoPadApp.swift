import SwiftUI
import MLX

@main
struct LTXVideoPadApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    @State private var status: String = "Ready"
    @State private var isRunning: Bool = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                Image(systemName: "video.badge.waveform")
                    .font(.system(size: 60))
                    .foregroundStyle(.blue)

                Text("LTXVideoPad")
                    .font(.largeTitle)
                    .fontWeight(.bold)

                Text("Phase 1: T5 Encoder Test")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                Text(status)
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
                    .padding(.horizontal)

                Button {
                    runTest()
                } label: {
                    Label("Run T5 Test", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(isRunning ? .gray : .blue, in: RoundedRectangle(cornerRadius: 12))
                        .foregroundStyle(.white)
                }
                .disabled(isRunning)
                .padding(.horizontal)

                Text("Memory: \(MemoryManager.shared.currentMemoryMB) MB")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .navigationTitle("LTXVideoPad")
        }
    }

    private func runTest() {
        isRunning = true
        status = "Running T5 encoder test..."

        Task {
            // Point to the model directory — adjust this path as needed
            let modelDir: URL = URL(fileURLWithPath: "/Users/admin/LTXVideoPad/Models/t5xxl-encoder-4bit")

            if !FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("model.safetensors").path) {
                await MainActor.run {
                    status = "ERROR: model.safetensors not found at \(modelDir.path)"
                    isRunning = false
                }
                return
            }

            await T5EncoderTest.run(modelDir: modelDir)

            await MainActor.run {
                status = "Test complete — check console for results"
                isRunning = false
            }
        }
    }
}
