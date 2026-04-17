// LoRALoader.swift — Phase 6
//
// Loads LoRA weights from safetensors and merges into DiT base weights.
// Supports CivitAI and Diffusers formats.
//
// Merge: W_merged = W_base + (alpha / rank) * scale * (lora_up @ lora_down)
// For quantized weights: dequantize → add delta → re-quantize

import Foundation
import MLX
import MLXNN
import os

// MARK: - LoRA Weight Pair

public struct LoRAPair {
    public let targetKey: String     // e.g., "transformer_blocks.0.attn1.to_q"
    public let down: MLXArray        // [rank, in_features]
    public let up: MLXArray          // [out_features, rank]
    public let alpha: Float
    public var rank: Int { down.dim(0) }
}

// MARK: - LoRA Loader

public enum LoRALoader {
    private static let logger = Logger(subsystem: "com.ltxvideopad", category: "LoRA")

    /// Load LoRA weight pairs from a safetensors file.
    public static func load(from url: URL) throws -> [LoRAPair] {
        logger.info("Loading LoRA from \(url.lastPathComponent)...")

        let weights: [String: MLXArray] = try loadArrays(url: url)
        logger.info("Loaded \(weights.count) tensors")

        // Detect format
        let isCivitAI: Bool = weights.keys.contains { $0.hasPrefix("diffusion_model.") }

        // Parse into down/up pairs
        var downWeights: [String: MLXArray] = [:]
        var upWeights: [String: MLXArray] = [:]
        var alphas: [String: Float] = [:]

        for (key, value) in weights {
            var basePath: String = key
            if isCivitAI {
                basePath = String(key.dropFirst("diffusion_model.".count))
            } else if key.hasPrefix("transformer.") {
                basePath = String(key.dropFirst("transformer.".count))
            }

            if basePath.hasSuffix(".lora_down.weight") || basePath.hasSuffix(".lora_A.weight") {
                let target: String = basePath
                    .replacingOccurrences(of: ".lora_down.weight", with: "")
                    .replacingOccurrences(of: ".lora_A.weight", with: "")
                downWeights[target] = value
            } else if basePath.hasSuffix(".lora_up.weight") || basePath.hasSuffix(".lora_B.weight") {
                let target: String = basePath
                    .replacingOccurrences(of: ".lora_up.weight", with: "")
                    .replacingOccurrences(of: ".lora_B.weight", with: "")
                upWeights[target] = value
            } else if basePath.hasSuffix(".alpha") {
                let target: String = basePath.replacingOccurrences(of: ".alpha", with: "")
                alphas[target] = value.item(Float.self)
            }
        }

        var pairs: [LoRAPair] = []
        for (target, down) in downWeights {
            guard let up = upWeights[target] else { continue }
            let alpha: Float = alphas[target] ?? Float(down.dim(0))
            pairs.append(LoRAPair(targetKey: target, down: down, up: up, alpha: alpha))
        }

        logger.info("Found \(pairs.count) LoRA pairs, rank=\(pairs.first?.rank ?? 0)")
        return pairs
    }

    /// Merge LoRA deltas into the model's weight arrays.
    ///
    /// Loads the base model weights, computes deltas, and applies them.
    /// For quantized weights: dequantize → merge → re-quantize.
    ///
    /// - Parameters:
    ///   - pairs: LoRA weight pairs
    ///   - model: target transformer model
    ///   - scale: LoRA strength (0.0 = none, 1.0 = full)
    public static func merge(
        pairs: [LoRAPair],
        into model: LTXVideoTransformer,
        scale: Float = 1.0
    ) {
        logger.info("Merging \(pairs.count) LoRA pairs (scale=\(scale))...")

        // Get model weights as flat dictionary
        let allParams: [(String, MLXArray)] = model.parameters().flattened()
        var paramDict: [String: MLXArray] = [:]
        for (key, val) in allParams {
            paramDict[key] = val
        }

        var updates: [String: MLXArray] = [:]
        var mergedCount: Int = 0

        for pair in pairs {
            // Map LoRA key to model key prefix
            // LoRA: "transformer_blocks.0.attn1.to_q"
            // Model: "model.diffusion_model.transformer_blocks.0.attn1.to_q"
            let prefix: String = "model.diffusion_model.\(pair.targetKey)"
            let weightKey: String = "\(prefix).weight"
            let scalesKey: String = "\(prefix).scales"
            let biasesKey: String = "\(prefix).biases"

            guard let baseWeight = paramDict[weightKey] else {
                logger.warning("Target not found: \(weightKey)")
                continue
            }

            // Compute delta: (alpha/rank) * scale * (up @ down)
            let loraScale: Float = (pair.alpha / Float(pair.rank)) * scale
            let delta: MLXArray = matmul(pair.up.asType(.float16), pair.down.asType(.float16)) * loraScale

            // Check if quantized
            if let scales = paramDict[scalesKey], let biases = paramDict[biasesKey] {
                // Dequantize → merge → re-quantize
                let deq: MLXArray = dequantized(baseWeight, scales: scales, biases: biases,
                                                 groupSize: 64, bits: 4)
                let merged: MLXArray = deq + delta.asType(deq.dtype)
                let (newW, newS, newB) = quantized(merged, groupSize: 64, bits: 4)

                updates[weightKey] = newW
                updates[scalesKey] = newS
                if let nb = newB { updates[biasesKey] = nb }
            } else {
                // Direct merge for non-quantized weights
                updates[weightKey] = baseWeight + delta.asType(baseWeight.dtype)
            }

            mergedCount += 1
        }

        // Apply updates
        if !updates.isEmpty {
            let nested = ModuleParameters.unflattened(updates)
            model.update(parameters: nested)
        }

        logger.info("Merged \(mergedCount)/\(pairs.count) LoRA pairs")
    }
}
