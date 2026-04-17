// T5WeightLoader.swift
//
// Loads quantized T5-v1.1-XXL weights from safetensors.
// The model's property names match the safetensors keys exactly,
// so we just load and apply via Module.update(parameters:).
//
// Expected file: Models/t5xxl-encoder-4bit/model.safetensors (~2.68 GB)

import Foundation
import MLX
import MLXNN
import os

/// Load quantized T5 weights and apply to model.
///
/// - Parameters:
///   - model: The T5EncoderModel to load weights into
///   - directory: URL to directory containing model.safetensors
/// - Throws: If safetensors file cannot be found or loaded
///
/// Peak memory: ~2.68 GB (quantized weights)
public func loadT5Weights(model: T5EncoderModel, from directory: URL) throws {
    let logger = Logger(subsystem: "com.ltxvideopad", category: "T5Weights")

    let safetensorsFile: URL = directory.appendingPathComponent("model.safetensors")

    guard FileManager.default.fileExists(atPath: safetensorsFile.path) else {
        throw T5LoadError.noSafetensorsFiles(directory: directory.path)
    }

    logger.info("Loading weights from \(safetensorsFile.lastPathComponent)...")

    // MLX's loadArrays handles quantized weights (weight + scales + biases triplets)
    let weights: [String: MLXArray] = try loadArrays(url: safetensorsFile)
    logger.info("Loaded \(weights.count) arrays from safetensors")

    // Convert flat keys to nested parameter structure and apply
    let parameters = ModuleParameters.unflattened(weights)
    model.update(parameters: parameters)

    logger.info("Weights applied to T5EncoderModel")

    // Verify a few key shapes
    if let embW = weights["encoder.embed_tokens.weight"] {
        logger.info("embed_tokens.weight shape: \(embW.shape)")
    }
    if let finalW = weights["encoder.final_layer_norm.weight"] {
        logger.info("final_layer_norm.weight shape: \(finalW.shape)")
    }
}

// MARK: - Errors

public enum T5LoadError: Error, LocalizedError {
    case noSafetensorsFiles(directory: String)

    public var errorDescription: String? {
        switch self {
        case .noSafetensorsFiles(let dir):
            return "No model.safetensors found in \(dir)"
        }
    }
}
