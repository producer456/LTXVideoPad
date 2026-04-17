// GenerateView.swift — Main generation interface
// Prompt input → Generate → Progress → Video preview/export

import SwiftUI
import MLX

#if canImport(AVFoundation)
import AVFoundation
#endif

public struct GenerateView: View {
    @State private var prompt: String = ""
    @State private var isGenerating: Bool = false
    @State private var progressStep: Int = 0
    @State private var progressTotal: Int = 4
    @State private var progressMessage: String = "Ready"
    @State private var errorMessage: String? = nil
    @State private var generationTime: Double = 0
    @State private var outputFrameCount: Int = 0
    @State private var exportedVideoURL: URL? = nil
    @State private var modelBaseDir: String = ""

    public init() {}

    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    headerSection
                    promptSection
                    settingsSection
                    generateButton

                    if isGenerating {
                        progressSection
                    }

                    if let error = errorMessage {
                        errorSection(error)
                    }

                    if outputFrameCount > 0 && !isGenerating {
                        resultSection
                    }

                    memorySection
                }
                .padding()
            }
            .navigationTitle("LTXVideoPad")
            .onAppear {
                // Default model directory
                #if targetEnvironment(simulator) || os(macOS)
                modelBaseDir = FileManager.default.currentDirectoryPath
                #else
                modelBaseDir = Bundle.main.bundlePath
                #endif
            }
        }
    }

    // MARK: - Sections

    private var headerSection: some View {
        VStack(spacing: 4) {
            Image(systemName: "video.badge.waveform")
                .font(.system(size: 40))
                .foregroundStyle(.blue)
            Text("LTX-Video 2B Distilled")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text("On-device • 4-bit • 8-step")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(.top, 8)
    }

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Prompt", systemImage: "text.quote")
                .font(.headline)

            TextField("Describe the video...", text: $prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(2...5)
                .disabled(isGenerating)
        }
    }

    private var settingsSection: some View {
        VStack(spacing: 6) {
            settingRow(icon: "aspectratio", label: "Resolution", value: "512 x 320")
            settingRow(icon: "clock", label: "Duration", value: "2s @ 24fps")
            settingRow(icon: "arrow.triangle.2.circlepath", label: "Denoise", value: "8 steps")
            settingRow(icon: "cube", label: "Models", value: "T5 + DiT + VAE (4.7 GB)")
        }
    }

    private func settingRow(icon: String, label: String, value: String) -> some View {
        HStack {
            Label(label, systemImage: icon)
                .font(.subheadline)
            Spacer()
            Text(value)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
    }

    private var generateButton: some View {
        Button {
            startGeneration()
        } label: {
            HStack {
                if isGenerating {
                    ProgressView().controlSize(.small)
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
        VStack(spacing: 10) {
            ProgressView(value: Double(progressStep), total: Double(progressTotal))
                .tint(.blue)
            Text(progressMessage)
                .font(.subheadline)
            Text("\(progressStep)/\(progressTotal)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
        .background(.blue.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
    }

    private func errorSection(_ error: String) -> some View {
        VStack(spacing: 6) {
            Label("Error", systemImage: "exclamationmark.triangle")
                .font(.headline)
                .foregroundStyle(.red)
            Text(error)
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(.red.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
    }

    private var resultSection: some View {
        VStack(spacing: 8) {
            Label("Complete", systemImage: "checkmark.circle.fill")
                .font(.headline)
                .foregroundStyle(.green)

            Text("\(outputFrameCount) frames in \(String(format: "%.1f", generationTime))s")
                .font(.subheadline)

            if let url = exportedVideoURL {
                Text(url.lastPathComponent)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(.green.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
    }

    private var memorySection: some View {
        HStack {
            Image(systemName: "memorychip")
                .foregroundStyle(.secondary)
            Text("\(MemoryManager.shared.currentMemoryMB) MB")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
        }
    }

    // MARK: - Generation

    private func startGeneration() {
        isGenerating = true
        errorMessage = nil
        outputFrameCount = 0
        exportedVideoURL = nil
        progressStep = 0
        progressMessage = "Starting..."

        Task {
            let startTime = Date()
            let base = URL(fileURLWithPath: modelBaseDir)

            let pipeline = I2VPipeline(
                t5Dir: base.appendingPathComponent("Models/t5xxl-encoder-4bit"),
                vaeDir: base.appendingPathComponent("Models/vae/vae"),
                ditDir: base.appendingPathComponent("Models/dit-4bit"),
                numFrames: 49, height: 320, width: 512, numSteps: 8
            )

            do {
                let frames = try await pipeline.generate(prompt: prompt, progress: nil)
                let elapsed = Date().timeIntervalSince(startTime)

                // Export to MP4
                #if canImport(AVFoundation)
                let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                let videoFile = docsDir.appendingPathComponent("ltx_output_\(Int(Date().timeIntervalSince1970)).mp4")

                await MainActor.run {
                    progressMessage = "Exporting MP4..."
                }

                try await exportFramesToMP4(frames: frames, outputURL: videoFile, fps: 24)
                #endif

                await MainActor.run {
                    generationTime = elapsed
                    outputFrameCount = frames.dim(0)
                    #if canImport(AVFoundation)
                    exportedVideoURL = videoFile
                    #endif
                    isGenerating = false
                    progressMessage = "Done"
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
