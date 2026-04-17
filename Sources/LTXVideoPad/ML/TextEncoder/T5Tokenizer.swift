// T5Tokenizer.swift
//
// Loads T5 tokenizer from HuggingFace tokenizer.json format.
// Uses the swift-transformers Tokenizers library.
//
// T5 tokenizer details:
// - SentencePiece unigram model
// - Vocab size: 32128
// - Special tokens: <pad> = 0, </s> = 1, <unk> = 2
// - For encoder-only usage, we append </s> at the end of the sequence
// - Default max length for LTX-Video: 128 tokens
//
// The tokenizer.json file should be in the model directory alongside weights.

import Foundation
import Tokenizers
import Hub

// MARK: - T5 Tokenizer Wrapper

/// Wraps HuggingFace Tokenizer for T5-specific encoding.
public class T5TokenizerWrapper {
    private let tokenizer: Tokenizer
    public let maxLength: Int

    // T5 special token IDs
    public static let padTokenId: Int = 0
    public static let eosTokenId: Int = 1
    public static let unkTokenId: Int = 2

    /// Load tokenizer from a directory containing tokenizer.json
    /// - Parameters:
    ///   - directory: URL to directory with tokenizer.json (and optionally tokenizer_config.json)
    ///   - maxLength: maximum sequence length (LTX-Video default: 128)
    public init(from directory: URL, maxLength: Int = 128) async throws {
        self.maxLength = maxLength

        let tokenizerFile: URL = directory.appendingPathComponent("tokenizer.json")

        guard FileManager.default.fileExists(atPath: tokenizerFile.path) else {
            throw T5TokenizerError.missingFile(path: tokenizerFile.path)
        }

        // Load tokenizer from the JSON config
        let tokenizerData: Data = try Data(contentsOf: tokenizerFile)
        self.tokenizer = try JSONDecoder().decode(Tokenizer.self, from: tokenizerData)
    }

    /// Encode a text prompt to token IDs with padding and attention mask.
    ///
    /// - Parameter text: The input prompt string
    /// - Returns: Tuple of (inputIds, attentionMask) as [Int] arrays
    ///
    /// The returned arrays are padded/truncated to maxLength.
    /// T5 convention: append </s> token, pad with 0s, attention mask is 1 for real tokens.
    public func encode(_ text: String) -> (inputIds: [Int], attentionMask: [Int]) {
        // Tokenize
        let encoded = tokenizer.encode(text: text)
        var tokenIds: [Int] = encoded.map { Int($0) }

        // Append EOS token (</s> = 1)
        tokenIds.append(Self.eosTokenId)

        // Truncate if too long
        if tokenIds.count > maxLength {
            tokenIds = Array(tokenIds.prefix(maxLength))
            // Ensure EOS is still at the end
            tokenIds[maxLength - 1] = Self.eosTokenId
        }

        // Build attention mask (1 for real tokens, 0 for padding)
        let realLength: Int = tokenIds.count
        var attentionMask: [Int] = Array(repeating: 1, count: realLength)

        // Pad to maxLength
        while tokenIds.count < maxLength {
            tokenIds.append(Self.padTokenId)
            attentionMask.append(0)
        }

        return (tokenIds, attentionMask)
    }

    /// Encode a batch of prompts.
    /// - Parameter texts: Array of prompt strings
    /// - Returns: Tuple of (inputIds, attentionMasks) as [[Int]] arrays
    public func encodeBatch(_ texts: [String]) -> (inputIds: [[Int]], attentionMasks: [[Int]]) {
        var allIds: [[Int]] = []
        var allMasks: [[Int]] = []

        for text in texts {
            let (ids, mask) = encode(text)
            allIds.append(ids)
            allMasks.append(mask)
        }

        return (allIds, allMasks)
    }
}

// MARK: - Errors

public enum T5TokenizerError: Error, LocalizedError {
    case missingFile(path: String)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let path):
            return "Tokenizer file not found at \(path)"
        }
    }
}
