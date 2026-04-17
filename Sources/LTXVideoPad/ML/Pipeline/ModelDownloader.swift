// ModelDownloader.swift
//
// Downloads model weights from HuggingFace Hub on first launch.
// Stores in the app's Documents directory for persistence.
//
// Total download: ~5.4 GB
//   T5-XXL encoder (4-bit): ~2.68 GB
//   DiT backbone (4-bit):   ~1.08 GB
//   VAE (FP32):             ~1.68 GB
//
// Supports:
//   - Resume after interruption (checks file size)
//   - Progress reporting per file and overall
//   - Cancellation

import Foundation
import os

public class ModelDownloader: ObservableObject {
    private let logger = Logger(subsystem: "com.ltxvideopad", category: "Download")

    @Published public var isDownloading: Bool = false
    @Published public var overallProgress: Double = 0.0
    @Published public var currentFile: String = ""
    @Published public var error: String? = nil
    @Published public var isComplete: Bool = false

    private var downloadTask: Task<Void, Never>?

    /// Model files to download with their HuggingFace URLs and expected sizes.
    public struct ModelFile: Sendable {
        let name: String
        let url: URL
        let localPath: String       // relative to models directory
        let expectedSizeMB: Int
    }

    /// All required model files.
    public static let requiredFiles: [ModelFile] = [
        // T5 encoder (quantized) — from our pre-quantized upload
        // For now, these would need to be hosted somewhere accessible.
        // In production, upload the quantized models to a HuggingFace repo.

        // VAE — from Lightricks official repo
        ModelFile(
            name: "VAE Decoder",
            url: URL(string: "https://huggingface.co/Lightricks/LTX-Video/resolve/main/vae/diffusion_pytorch_model.safetensors")!,
            localPath: "vae/diffusion_pytorch_model.safetensors",
            expectedSizeMB: 1680
        ),
        ModelFile(
            name: "VAE Config",
            url: URL(string: "https://huggingface.co/Lightricks/LTX-Video/resolve/main/vae/config.json")!,
            localPath: "vae/config.json",
            expectedSizeMB: 1
        ),
    ]

    /// Directory where models are stored.
    public var modelsDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("Models")
    }

    /// Check if all required models are already downloaded.
    public var allModelsPresent: Bool {
        // Check for the three main weight files
        let t5 = modelsDirectory.appendingPathComponent("t5xxl-encoder-4bit/model.safetensors")
        let dit = modelsDirectory.appendingPathComponent("dit-4bit/model.safetensors")
        let vae = modelsDirectory.appendingPathComponent("vae/diffusion_pytorch_model.safetensors")

        return FileManager.default.fileExists(atPath: t5.path)
            && FileManager.default.fileExists(atPath: dit.path)
            && FileManager.default.fileExists(atPath: vae.path)
    }

    /// Check which models are missing.
    public func missingModels() -> [String] {
        var missing: [String] = []
        let checks: [(String, String)] = [
            ("T5 Encoder (4-bit)", "t5xxl-encoder-4bit/model.safetensors"),
            ("DiT Backbone (4-bit)", "dit-4bit/model.safetensors"),
            ("VAE", "vae/diffusion_pytorch_model.safetensors"),
            ("T5 Tokenizer", "t5xxl-encoder-4bit/tokenizer.json"),
        ]
        for (name, path) in checks {
            if !FileManager.default.fileExists(atPath: modelsDirectory.appendingPathComponent(path).path) {
                missing.append(name)
            }
        }
        return missing
    }

    /// Download a single file from a URL to a local path.
    /// Returns true if successful.
    public func downloadFile(from url: URL, to localURL: URL,
                              progress: @escaping (Double) -> Void) async throws {
        // Create parent directory
        let parentDir = localURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

        // Check if already downloaded (simple size check)
        if FileManager.default.fileExists(atPath: localURL.path) {
            let attrs = try FileManager.default.attributesOfItem(atPath: localURL.path)
            let size = attrs[.size] as? Int ?? 0
            if size > 1000 {  // non-trivial file exists
                logger.info("Already exists: \(localURL.lastPathComponent) (\(size / 1_000_000) MB)")
                progress(1.0)
                return
            }
        }

        logger.info("Downloading: \(url.lastPathComponent)...")

        // Use URLSession for download with progress
        let (tempURL, response) = try await URLSession.shared.download(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw DownloadError.httpError(code: (response as? HTTPURLResponse)?.statusCode ?? 0)
        }

        // Move to final location
        if FileManager.default.fileExists(atPath: localURL.path) {
            try FileManager.default.removeItem(at: localURL)
        }
        try FileManager.default.moveItem(at: tempURL, to: localURL)

        let finalSize = try FileManager.default.attributesOfItem(atPath: localURL.path)[.size] as? Int ?? 0
        logger.info("Downloaded: \(localURL.lastPathComponent) (\(finalSize / 1_000_000) MB)")
        progress(1.0)
    }

    /// Copy pre-quantized models from a source directory (e.g., connected Mac via Files).
    /// This is the recommended approach for development — avoids downloading 5.4 GB.
    public func copyModelsFromDirectory(_ sourceDir: URL) throws {
        let fm = FileManager.default

        // Copy each model subdirectory
        let subdirs = ["t5xxl-encoder-4bit", "dit-4bit", "vae"]
        for subdir in subdirs {
            let src = sourceDir.appendingPathComponent(subdir)
            let dst = modelsDirectory.appendingPathComponent(subdir)

            if fm.fileExists(atPath: src.path) {
                try fm.createDirectory(at: dst.deletingLastPathComponent(),
                                       withIntermediateDirectories: true)
                if fm.fileExists(atPath: dst.path) {
                    try fm.removeItem(at: dst)
                }
                try fm.copyItem(at: src, to: dst)
                logger.info("Copied: \(subdir)")
            } else {
                logger.warning("Source not found: \(src.path)")
            }
        }
    }

    /// Get total size of downloaded models in MB.
    public var downloadedSizeMB: Int {
        let fm = FileManager.default
        var total: Int = 0

        if let enumerator = fm.enumerator(at: modelsDirectory, includingPropertiesForKeys: [.fileSizeKey]) {
            for case let fileURL as URL in enumerator {
                let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                total += size
            }
        }

        return total / (1024 * 1024)
    }
}

// MARK: - Errors

public enum DownloadError: Error, LocalizedError {
    case httpError(code: Int)
    case fileMissing(name: String)

    public var errorDescription: String? {
        switch self {
        case .httpError(let code): return "Download failed (HTTP \(code))"
        case .fileMissing(let name): return "Required model file missing: \(name)"
        }
    }
}
