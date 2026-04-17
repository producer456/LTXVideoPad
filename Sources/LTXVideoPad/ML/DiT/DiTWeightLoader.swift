// DiTWeightLoader.swift
//
// Loads quantized DiT weights from safetensors.
// The model's property names match the safetensors keys exactly,
// so we just load and apply via Module.update(parameters:).
//
// Expected file: Models/dit-4bit/model.safetensors (~0.96 GB)

import Foundation
import MLX
import MLXNN
import os

/// Load quantized DiT weights and apply to model.
///
/// - Parameters:
///   - model: The LTXVideoTransformer to load weights into
///   - directory: URL to directory containing model.safetensors
/// - Throws: If safetensors file cannot be found or loaded
///
/// Peak memory: ~0.96 GB (quantized weights)
public func loadDiTWeights(model: LTXVideoTransformer, from directory: URL) throws {
    let logger = Logger(subsystem: "com.ltxvideopad", category: "DiTWeights")

    let safetensorsFile: URL = directory.appendingPathComponent("model.safetensors")

    guard FileManager.default.fileExists(atPath: safetensorsFile.path) else {
        throw DiTLoadError.noSafetensorsFiles(directory: directory.path)
    }

    logger.info("Loading weights from \(safetensorsFile.lastPathComponent)...")

    // MLX's loadArrays handles quantized weights (weight + scales + biases triplets)
    let weights: [String: MLXArray] = try loadArrays(url: safetensorsFile)
    logger.info("Loaded \(weights.count) arrays from safetensors")

    // Convert flat keys to nested parameter structure and apply
    let parameters = ModuleParameters.unflattened(weights)
    model.update(parameters: parameters)

    logger.info("Weights applied to LTXVideoTransformer")

    // Verify a few key shapes
    if let w = weights["model.diffusion_model.patchify_proj.weight"] {
        logger.info("patchify_proj.weight shape: \(w.shape)")
    }
    if let w = weights["model.diffusion_model.scale_shift_table"] {
        logger.info("scale_shift_table shape: \(w.shape)")
    }
}

// MARK: - Errors

public enum DiTLoadError: Error, LocalizedError {
    case noSafetensorsFiles(directory: String)

    public var errorDescription: String? {
        switch self {
        case .noSafetensorsFiles(let dir):
            return "No model.safetensors found in \(dir)"
        }
    }
}
