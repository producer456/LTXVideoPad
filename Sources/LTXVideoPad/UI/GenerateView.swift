// GenerateView.swift — Main generation interface
//
// Layout: Image drop zone (top) → Prompt field → Generate button → Progress/Preview

import SwiftUI
import MLX

public struct GenerateView: View {
    @State private var prompt: String = ""
    @State private var isGenerating: Bool = false
    @State private var progressStep: Int = 0
    @State private var progressTotal: Int = 4
    @State private var progressMessage: String = ""
    @State private var errorMessage: String? = nil
    @State private var generationTime: Double = 0
    @State private var outputFrameCount: Int = 0

    public init() {}

    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // ── Prompt Input ──
                    promptSection

                    // ── Settings ──
                    settingsSection

                    // ── Generate Button ──
                    generateButton

                    // ── Progress ──
                    if isGenerating {
                        progressSection
                    }

                    // ── Result ──
                    if let error = errorMessage {
                        errorSection(error)
                    }

                    if outputFrameCount > 0 && !isGenerating {
                        resultSection
                    }
                }
                .padding()
            }
            .navigationTitle("LTXVideoPad")
        }
    }

    // MARK: - Sections

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Prompt", systemImage: "text.quote")
                .font(.headline)
                .foregroundStyle(.secondary)

            TextField("Describe the video you want to generate...", text: $prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(3...6)
                .disabled(isGenerating)
        }
    }

    private var settingsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Settings", systemImage: "slider.horizontal.3")
                .font(.headline)
                .foregroundStyle(.secondary)

            HStack {
                Label("Resolution", systemImage: "aspectratio")
                    .font(.subheadline)
                Spacer()
                Text("512 × 320")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))

            HStack {
                Label("Duration", systemImage: "clock")
                    .font(.subheadline)
                Spacer()
                Text("2s (49 frames @ 24fps)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))

            HStack {
                Label("Steps", systemImage: "arrow.triangle.2.circlepath")
                    .font(.subheadline)
                Spacer()
                Text("8 (distilled)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))

            HStack {
                Label("Memory", systemImage: "memorychip")
                    .font(.subheadline)
                Spacer()
                Text("\(MemoryManager.shared.currentMemoryMB) MB")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private var generateButton: some View {
        Button {
            startGeneration()
        } label: {
            HStack {
                if isGenerating {
                    ProgressView()
                        .controlSize(.small)
                    Text("Generating...")
                } else {
                    Image(systemName: "video.badge.waveform")
                    Text("Generate Video")
                }
            }
            .font(.headline)
            .frame(maxWidth: .infinity)
            .padding()
            .background(
                isGenerating || prompt.isEmpty ? AnyShapeStyle(.gray) : AnyShapeStyle(.blue),
                in: RoundedRectangle(cornerRadius: 12)
            )
            .foregroundStyle(.white)
        }
        .disabled(isGenerating || prompt.isEmpty)
    }

    private var progressSection: some View {
        VStack(spacing: 12) {
            ProgressView(value: Double(progressStep), total: Double(progressTotal)) {
                Text(progressMessage)
                    .font(.subheadline)
            }
            .tint(.blue)

            Text("Step \(progressStep) of \(progressTotal)")
                .font(.caption)
                .foregroundStyle(.secondary)

            Text("Memory: \(MemoryManager.shared.currentMemoryMB) MB")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding()
        .background(.blue.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
    }

    private func errorSection(_ error: String) -> some View {
        VStack(spacing: 8) {
            Label("Error", systemImage: "exclamationmark.triangle")
                .font(.headline)
                .foregroundStyle(.red)
            Text(error)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(.red.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
    }

    private var resultSection: some View {
        VStack(spacing: 8) {
            Label("Generation Complete", systemImage: "checkmark.circle")
                .font(.headline)
                .foregroundStyle(.green)

            Text("\(outputFrameCount) frames generated in \(String(format: "%.1f", generationTime))s")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Text("Memory peak: \(MemoryManager.shared.currentMemoryMB) MB")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(.green.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Generation

    private func startGeneration() {
        isGenerating = true
        errorMessage = nil
        outputFrameCount = 0

        Task {
            let startTime = Date()

            // Find model directories
            let baseDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            let t5Dir = baseDir.appendingPathComponent("Models/t5xxl-encoder-4bit")
            let vaeDir = baseDir.appendingPathComponent("Models/vae/vae")
            let ditDir = baseDir.appendingPathComponent("Models/dit-4bit")

            let pipeline = I2VPipeline(
                t5Dir: t5Dir, vaeDir: vaeDir, ditDir: ditDir,
                numFrames: 49, height: 320, width: 512, numSteps: 8
            )

            // Dummy tokenization (TODO: real tokenizer)
            let tokenIds: [Int32] = Array(repeating: Int32(1), count: 10) + [1]

            do {
                let frames = try await pipeline.generate(tokenIds: tokenIds, progress: nil)

                let elapsed = Date().timeIntervalSince(startTime)

                await MainActor.run {
                    generationTime = elapsed
                    outputFrameCount = frames.dim(0)
                    isGenerating = false
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    isGenerating = false
                }
            }
        }
    }
}
