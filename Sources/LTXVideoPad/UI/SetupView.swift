// SetupView.swift — Model setup & download screen
// Shown on first launch when model weights aren't present.

import SwiftUI

public struct SetupView: View {
    @StateObject private var downloader = ModelDownloader()
    @State private var showingFilePicker = false
    let onComplete: () -> Void

    public init(onComplete: @escaping () -> Void) {
        self.onComplete = onComplete
    }

    public var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                Spacer()

                // Header
                Image(systemName: "square.and.arrow.down.on.square")
                    .font(.system(size: 50))
                    .foregroundStyle(.blue)

                Text("Model Setup Required")
                    .font(.title2)
                    .fontWeight(.bold)

                Text("LTXVideoPad needs ~5.4 GB of model weights to generate video. These are stored locally on your device.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)

                // Missing models
                VStack(alignment: .leading, spacing: 8) {
                    let missing = downloader.missingModels()
                    if missing.isEmpty {
                        Label("All models present!", systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    } else {
                        ForEach(missing, id: \.self) { model in
                            Label(model, systemImage: "xmark.circle")
                                .foregroundStyle(.red)
                                .font(.subheadline)
                        }
                    }
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
                .padding(.horizontal)

                // Status
                if downloader.downloadedSizeMB > 0 {
                    Text("Downloaded: \(downloader.downloadedSizeMB) MB")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if let error = downloader.error {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .padding(.horizontal)
                }

                Spacer()

                // Actions
                VStack(spacing: 12) {
                    // Option 1: Copy from Files app
                    Button {
                        showingFilePicker = true
                    } label: {
                        Label("Import from Files", systemImage: "folder")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(.blue, in: RoundedRectangle(cornerRadius: 12))
                            .foregroundStyle(.white)
                    }

                    // Option 2: Check if models appeared (user may have copied via Finder/AirDrop)
                    Button {
                        if downloader.allModelsPresent {
                            onComplete()
                        }
                    } label: {
                        Label("Check for Models", systemImage: "arrow.clockwise")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(.secondary.opacity(0.2), in: RoundedRectangle(cornerRadius: 12))
                            .foregroundStyle(.primary)
                    }

                    Text("Copy the Models folder to this app's Documents directory via Files, Finder, or AirDrop.")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                }
                .padding(.horizontal)
                .padding(.bottom)
            }
            .navigationTitle("Setup")
            .onAppear {
                if downloader.allModelsPresent {
                    onComplete()
                }
            }
        }
    }
}

// MARK: - Root App View

/// Root view that shows SetupView if models are missing, GenerateView if ready.
public struct RootView: View {
    @State private var modelsReady: Bool = false
    @StateObject private var downloader = ModelDownloader()

    public init() {}

    public var body: some View {
        if modelsReady {
            GenerateView()
        } else {
            SetupView {
                modelsReady = true
            }
            .onAppear {
                modelsReady = downloader.allModelsPresent
            }
        }
    }
}
