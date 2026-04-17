// VAEWeightLoader.swift
//
// Loads LTX-Video VAE weights from safetensors.
// CRITICAL: Transposes all 5D conv weights from PyTorch layout
// (out_ch, in_ch, D, H, W) to MLX Conv3d layout (out_ch, D, H, W, in_ch).
//
// Expected file: Models/vae/vae/diffusion_pytorch_model.safetensors (~1.67 GB)

import Foundation
import MLX
import MLXNN
import os

// MARK: - VAE Model wrapper

/// Top-level VAE model with encoder + decoder + latent stats.
/// Property names match safetensors root keys: encoder.*, decoder.*, latents_mean, latents_std.
public class VAEModel: Module {
    let encoder: VideoEncoder
    let decoder: VideoDecoder
    // swiftlint:disable:next identifier_name
    var latents_mean: MLXArray
    // swiftlint:disable:next identifier_name
    var latents_std: MLXArray

    public override init() {
        self.encoder = VideoEncoder()
        self.decoder = VideoDecoder()
        self.latents_mean = MLXArray.zeros([128])
        self.latents_std = MLXArray.ones([128])
    }
}

// MARK: - Weight Loading

public enum VAELoadError: Error, LocalizedError {
    case fileNotFound(path: String)
    case transposeFailed(key: String, shape: [Int])

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "VAE safetensors not found at \(path)"
        case .transposeFailed(let key, let shape):
            return "Failed to transpose conv weight \(key) with shape \(shape)"
        }
    }
}

/// Load VAE weights from safetensors and apply to model.
///
/// Transposes all 5D conv weights from PyTorch (out, in, D, H, W)
/// to MLX (out, D, H, W, in) format before applying.
///
/// - Parameters:
///   - model: The VAEModel to load weights into
///   - url: URL to the safetensors file
/// - Throws: VAELoadError if file not found or transpose fails
///
/// Peak memory: ~1.67 GB (full FP32 weights) + transient transpose copies
public func loadVAEWeights(model: VAEModel, from url: URL) throws {
    let logger: Logger = Logger(subsystem: "com.ltxvideopad", category: "VAEWeights")

    guard FileManager.default.fileExists(atPath: url.path) else {
        throw VAELoadError.fileNotFound(path: url.path)
    }

    logger.info("Loading VAE weights from \(url.lastPathComponent)...")

    // Load raw arrays from safetensors
    let rawWeights: [String: MLXArray] = try loadArrays(url: url)
    logger.info("Loaded \(rawWeights.count) arrays from safetensors")

    // Transpose 5D conv weights: PyTorch (O, I, D, H, W) → MLX (O, D, H, W, I)
    var weights: [String: MLXArray] = [:]
    var transposedCount: Int = 0

    for (key, array) in rawWeights {
        if array.ndim == 5 {
            // Conv3d weight: swap axis 1 (in_channels) with axis 4 (W → becomes in_channels)
            weights[key] = array.transposed(0, 2, 3, 4, 1)
            transposedCount += 1
        } else {
            weights[key] = array
        }
    }

    logger.info("Transposed \(transposedCount) conv3d weights from PyTorch to MLX layout")

    // Convert flat keys to nested parameter structure and apply
    let parameters: ModuleParameters = ModuleParameters.unflattened(weights)
    model.update(parameters: parameters)

    logger.info("VAE weights applied successfully")

    // Log verification info
    if let mean = rawWeights["latents_mean"] {
        logger.info("latents_mean shape: \(mean.shape)")
    }
    if let std = rawWeights["latents_std"] {
        logger.info("latents_std shape: \(std.shape)")
    }
}

/// Convenience: load from a directory containing diffusion_pytorch_model.safetensors
public func loadVAEWeights(model: VAEModel, fromDirectory dir: URL) throws {
    let file: URL = dir.appendingPathComponent("diffusion_pytorch_model.safetensors")
    try loadVAEWeights(model: model, from: file)
}
