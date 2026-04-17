// T5TokenizerLoader.swift
//
// Loads the T5 tokenizer from a local tokenizer.json + tokenizer_config.json
// using swift-transformers' PreTrainedTokenizer, without going through
// AutoTokenizer (which tries to download from HuggingFace).
//
// Usage:
//   let tokenizer = try T5TokenizerLoader.load(from: modelDir)
//   let ids = T5TokenizerLoader.encode(tokenizer: tokenizer, text: prompt)
//   // ids is [Int32], padded to 128 with EOS appended

import Foundation
import Hub
import Tokenizers

public enum T5TokenizerLoader {

    public enum TokenizerLoadError: Error, CustomStringConvertible {
        case missingFile(String)
        case parseError(String)

        public var description: String {
            switch self {
            case .missingFile(let path):
                return "Tokenizer file not found: \(path)"
            case .parseError(let msg):
                return "Tokenizer parse error: \(msg)"
            }
        }
    }

    /// Load a Tokenizer from a local directory containing tokenizer.json
    /// and tokenizer_config.json. Does NOT contact HuggingFace.
    ///
    /// - Parameter modelDir: directory containing tokenizer.json and tokenizer_config.json
    /// - Returns: a ready-to-use Tokenizer
    public static func load(from modelDir: URL) throws -> Tokenizer {
        let tokenizerDataURL = modelDir.appendingPathComponent("tokenizer.json")
        let tokenizerConfigURL = modelDir.appendingPathComponent("tokenizer_config.json")

        guard FileManager.default.fileExists(atPath: tokenizerDataURL.path) else {
            throw TokenizerLoadError.missingFile(tokenizerDataURL.path)
        }
        guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
            throw TokenizerLoadError.missingFile(tokenizerConfigURL.path)
        }

        // Load JSON files into swift-transformers Config objects
        let tokenizerData: Config = try loadConfig(from: tokenizerDataURL)
        let tokenizerConfig: Config = try loadConfig(from: tokenizerConfigURL)

        // Build the tokenizer using the same path as AutoTokenizer.from(tokenizerConfig:tokenizerData:)
        return try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }

    /// Encode text into token IDs suitable for T5.
    /// Appends EOS (id=1), pads to maxLength with PAD (id=0).
    ///
    /// - Parameters:
    ///   - tokenizer: a loaded Tokenizer
    ///   - text: the prompt string
    ///   - maxLength: maximum sequence length (default 128)
    /// - Returns: [Int32] of length maxLength
    public static func encode(
        tokenizer: Tokenizer,
        text: String,
        maxLength: Int = 128
    ) -> [Int32] {
        // Encode without special tokens — we'll add EOS ourselves
        var ids: [Int] = tokenizer.encode(text: text, addSpecialTokens: false)

        // Append EOS token (id=1 for T5)
        let eosId: Int = tokenizer.eosTokenId ?? 1
        ids.append(eosId)

        // Truncate if too long
        if ids.count > maxLength {
            ids = Array(ids.prefix(maxLength))
            // Ensure last token is EOS
            ids[maxLength - 1] = eosId
        }

        // Pad to maxLength with PAD (id=0)
        let padId: Int = 0
        while ids.count < maxLength {
            ids.append(padId)
        }

        return ids.map { Int32($0) }
    }

    // MARK: - Private

    private static func loadConfig(from url: URL) throws -> Config {
        let data = try Data(contentsOf: url)
        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dictionary = parsed as? [NSString: Any] else {
            throw TokenizerLoadError.parseError("Failed to parse \(url.lastPathComponent) as JSON dictionary")
        }
        return Config(dictionary)
    }
}
