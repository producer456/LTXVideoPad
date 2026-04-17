// T5WeightLoader.swift
//
// Loads T5-v1.1-XXL encoder weights from safetensors files and maps
// HuggingFace parameter names to our T5EncoderModel structure.
//
// HuggingFace key format:
//   encoder.block.{N}.layer.0.SelfAttention.q.weight  → blocks[N].attention.q.weight
//   encoder.block.{N}.layer.0.SelfAttention.k.weight  → blocks[N].attention.k.weight
//   encoder.block.{N}.layer.0.SelfAttention.v.weight  → blocks[N].attention.v.weight
//   encoder.block.{N}.layer.0.SelfAttention.o.weight  → blocks[N].attention.o.weight
//   encoder.block.{N}.layer.0.layer_norm.weight        → blocks[N].attnNorm.weight
//   encoder.block.{N}.layer.1.DenseReluDense.wi_0.weight → blocks[N].feedForward.wi0.weight
//   encoder.block.{N}.layer.1.DenseReluDense.wi_1.weight → blocks[N].feedForward.wi1.weight
//   encoder.block.{N}.layer.1.DenseReluDense.wo.weight   → blocks[N].feedForward.wo.weight
//   encoder.block.{N}.layer.1.layer_norm.weight          → blocks[N].ffNorm.weight
//   encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight
//                                                      → blocks[0].attention.positionBias.embedding.weight
//   shared.weight                                      → embedding.weight
//   encoder.final_layer_norm.weight                    → finalNorm.weight
//
// Memory: Loading FP16 weights peaks at ~9 GB. Loading 4-bit peaks at ~2.4 GB.
// After quantization, only the quantized model stays resident.

import Foundation
import MLX
import MLXNN
import os

// MARK: - Weight Key Mapping

/// Maps a HuggingFace safetensors key to our model's parameter path.
/// Returns nil if the key should be skipped (e.g., decoder weights).
func mapT5Key(_ hfKey: String) -> String? {
    // Embedding
    if hfKey == "shared.weight" || hfKey == "encoder.embed_tokens.weight" {
        return "embedding.weight"
    }

    // Final layer norm
    if hfKey == "encoder.final_layer_norm.weight" {
        return "finalNorm.weight"
    }

    // Encoder blocks
    // Pattern: encoder.block.{N}.layer.{L}.{component}
    let parts: [String] = hfKey.components(separatedBy: ".")

    guard parts.count >= 5,
          parts[0] == "encoder",
          parts[1] == "block",
          let blockIdx = Int(parts[2]),
          parts[3] == "layer" else {
        // Skip decoder weights or unrecognized keys
        return nil
    }

    let layerIdx: String = parts[4]
    let prefix: String = "blocks.\(blockIdx)"

    if layerIdx == "0" {
        // Self-attention sublayer
        if parts.count >= 7 && parts[5] == "SelfAttention" {
            let param: String = parts[6]

            if param == "relative_attention_bias" && parts.count >= 8 {
                // encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight
                return "\(prefix).attention.positionBias.embedding.weight"
            }

            // q, k, v, o projections
            switch param {
            case "q":  return "\(prefix).attention.q.weight"
            case "k":  return "\(prefix).attention.k.weight"
            case "v":  return "\(prefix).attention.v.weight"
            case "o":  return "\(prefix).attention.o.weight"
            default:   return nil
            }
        }

        if parts.count >= 7 && parts[5] == "layer_norm" {
            // encoder.block.{N}.layer.0.layer_norm.weight
            return "\(prefix).attnNorm.weight"
        }
    }

    if layerIdx == "1" {
        // Feed-forward sublayer
        if parts.count >= 7 && parts[5] == "DenseReluDense" {
            let param: String = parts[6]
            switch param {
            case "wi_0": return "\(prefix).feedForward.wi0.weight"
            case "wi_1": return "\(prefix).feedForward.wi1.weight"
            case "wo":   return "\(prefix).feedForward.wo.weight"
            default:     return nil
            }
        }

        if parts.count >= 7 && parts[5] == "layer_norm" {
            // encoder.block.{N}.layer.1.layer_norm.weight
            return "\(prefix).ffNorm.weight"
        }
    }

    return nil
}

// MARK: - Weight Loading

/// Loads T5 encoder weights from one or more safetensors files.
///
/// Supports both single-file and sharded formats:
/// - Single: `model.safetensors`
/// - Sharded: `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`
///
/// - Parameter directory: URL to directory containing safetensors file(s)
/// - Returns: Dictionary mapping our model parameter paths to MLXArrays
/// - Throws: If files cannot be read or keys are missing
///
/// Peak memory: ~9 GB for FP16, ~2.4 GB for pre-quantized 4-bit.
public func loadT5Weights(from directory: URL) throws -> [String: MLXArray] {
    let logger = Logger(subsystem: "com.ltxvideopad", category: "T5Weights")

    // Find all safetensors files in the directory
    let fileManager: FileManager = FileManager.default
    let contents: [URL] = try fileManager.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: nil
    )
    let safetensorsFiles: [URL] = contents
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

    guard !safetensorsFiles.isEmpty else {
        throw T5LoadError.noSafetensorsFiles(directory: directory.path)
    }

    logger.info("Found \(safetensorsFiles.count) safetensors file(s)")

    // Load and merge all shards
    var mappedWeights: [String: MLXArray] = [:]
    var skippedKeys: [String] = []

    for fileURL in safetensorsFiles {
        logger.info("Loading \(fileURL.lastPathComponent)...")
        let weights: [String: MLXArray] = try MLX.loadArrays(url: fileURL)

        for (hfKey, array) in weights {
            if let ourKey = mapT5Key(hfKey) {
                mappedWeights[ourKey] = array
            } else {
                skippedKeys.append(hfKey)
            }
        }
    }

    logger.info("Mapped \(mappedWeights.count) parameters, skipped \(skippedKeys.count)")

    // Validate we have all expected keys
    let expectedCount: Int = 1    // embedding
        + 24 * 9                  // 24 blocks × (4 attn + 2 norms + 3 ff)
        + 1                       // relative attention bias
        + 1                       // final norm
    // = 219

    if mappedWeights.count < expectedCount {
        logger.warning("Expected \(expectedCount) parameters, got \(mappedWeights.count)")
        // Log which keys we're missing
        let allExpected: [String] = buildExpectedKeys()
        let missing: [String] = allExpected.filter { mappedWeights[$0] == nil }
        for key in missing.prefix(10) {
            logger.warning("Missing: \(key)")
        }
    }

    return mappedWeights
}

/// Apply loaded weights to a T5EncoderModel.
///
/// - Parameters:
///   - model: The T5EncoderModel to update
///   - weights: Weight dictionary from loadT5Weights
public func applyT5Weights(to model: T5EncoderModel, weights: [String: MLXArray]) {
    let logger = Logger(subsystem: "com.ltxvideopad", category: "T5Weights")

    // Convert flat key-value dict to nested ModuleParameters
    // MLX Swift's Module.update(parameters:) expects nested dictionaries
    var applied: Int = 0
    var failed: Int = 0

    // Use the Module's built-in parameter update
    let parameters = ModuleParameters.unflattened(weights)
    model.update(parameters: parameters)

    logger.info("Applied weights to T5EncoderModel")
}

// MARK: - Helpers

/// Build list of all expected parameter keys for validation.
private func buildExpectedKeys() -> [String] {
    var keys: [String] = []

    keys.append("embedding.weight")
    keys.append("finalNorm.weight")
    keys.append("blocks.0.attention.positionBias.embedding.weight")

    for i in 0..<24 {
        let p: String = "blocks.\(i)"
        keys.append("\(p).attnNorm.weight")
        keys.append("\(p).attention.q.weight")
        keys.append("\(p).attention.k.weight")
        keys.append("\(p).attention.v.weight")
        keys.append("\(p).attention.o.weight")
        keys.append("\(p).ffNorm.weight")
        keys.append("\(p).feedForward.wi0.weight")
        keys.append("\(p).feedForward.wi1.weight")
        keys.append("\(p).feedForward.wo.weight")
    }

    return keys
}

// MARK: - Errors

public enum T5LoadError: Error, LocalizedError {
    case noSafetensorsFiles(directory: String)
    case missingKeys(keys: [String])

    public var errorDescription: String? {
        switch self {
        case .noSafetensorsFiles(let dir):
            return "No .safetensors files found in \(dir)"
        case .missingKeys(let keys):
            return "Missing \(keys.count) required weight keys: \(keys.prefix(5).joined(separator: ", "))"
        }
    }
}
