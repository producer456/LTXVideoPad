// ImageLoader.swift
// Loads an image file and converts it to an MLXArray suitable for VAE encoding.

import Foundation
import MLX

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif
import CoreGraphics

/// Loads an image from disk, resizes to the target dimensions, and returns an
/// MLXArray of shape [1, 3, 1, H, W] with Float32 values normalized to [-1, 1].
///
/// The extra dims are: batch (1) and temporal (1) — matching the VAE encoder's
/// expected input for a single-frame conditioning image.
///
/// - Parameters:
///   - url: File URL pointing to a JPEG or PNG image.
///   - width: Target width in pixels (default 512).
///   - height: Target height in pixels (default 320).
/// - Returns: MLXArray of shape [1, 3, 1, H, W], Float32, range [-1, 1].
func loadImageAsMLXArray(
    url: URL,
    width: Int = 512,
    height: Int = 320
) throws -> MLXArray {
    let cgImage: CGImage = try loadCGImage(from: url)
    let resized: CGImage = try resizeCGImage(cgImage, width: width, height: height)
    let rgbFloats: [Float] = try extractRGBFloats(from: resized, width: width, height: height)

    // rgbFloats is [H, W, 3] in row-major order, values in [0, 1].
    // Rearrange to [1, 3, 1, H, W] and normalize to [-1, 1].
    let hwc: MLXArray = MLXArray(rgbFloats, [height, width, 3])

    // Transpose [H, W, 3] → [3, H, W]
    let chw: MLXArray = hwc.transposed(2, 0, 1)

    // Normalize: [0, 1] → [-1, 1]
    let normalized: MLXArray = chw * 2.0 - 1.0

    // Reshape [3, H, W] → [1, 3, 1, H, W]  (batch=1, temporal=1)
    let result: MLXArray = normalized.reshaped([1, 3, 1, height, width])

    return result
}

// MARK: - Platform-Specific Image Loading

/// Loads a CGImage from a file URL using the platform-appropriate API.
private func loadCGImage(from url: URL) throws -> CGImage {
    #if canImport(UIKit)
    guard let uiImage = UIImage(contentsOfFile: url.path) else {
        throw ImageLoaderError.failedToLoadImage(url)
    }
    guard let cgImage = uiImage.cgImage else {
        throw ImageLoaderError.failedToGetCGImage
    }
    return cgImage
    #elseif canImport(AppKit)
    guard let nsImage = NSImage(contentsOf: url) else {
        throw ImageLoaderError.failedToLoadImage(url)
    }
    guard let cgImage = nsImage.cgImage(
        forProposedRect: nil, context: nil, hints: nil
    ) else {
        throw ImageLoaderError.failedToGetCGImage
    }
    return cgImage
    #else
    throw ImageLoaderError.unsupportedPlatform
    #endif
}

// MARK: - Resize

/// Resizes a CGImage to the given dimensions using CoreGraphics.
private func resizeCGImage(
    _ image: CGImage,
    width: Int,
    height: Int
) throws -> CGImage {
    let colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()
    guard let context = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    ) else {
        throw ImageLoaderError.contextCreationFailed
    }
    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

    guard let resized = context.makeImage() else {
        throw ImageLoaderError.contextCreationFailed
    }
    return resized
}

// MARK: - Pixel Extraction

/// Extracts RGB float values in [0, 1] from a CGImage.
/// Returns a flat array of length H * W * 3 in row-major [H, W, 3] order.
private func extractRGBFloats(
    from image: CGImage,
    width: Int,
    height: Int
) throws -> [Float] {
    let colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()
    let bytesPerRow: Int = width * 4
    var rawPixels: [UInt8] = [UInt8](repeating: 0, count: height * bytesPerRow)

    guard let context = CGContext(
        data: &rawPixels,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    ) else {
        throw ImageLoaderError.contextCreationFailed
    }
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

    // rawPixels is RGBX (noneSkipLast) — 4 bytes per pixel, we take R, G, B.
    let totalPixels: Int = width * height
    var rgbFloats: [Float] = [Float](repeating: 0, count: totalPixels * 3)

    for i in 0..<totalPixels {
        let srcIdx: Int = i * 4
        rgbFloats[i * 3 + 0] = Float(rawPixels[srcIdx + 0]) / 255.0  // R
        rgbFloats[i * 3 + 1] = Float(rawPixels[srcIdx + 1]) / 255.0  // G
        rgbFloats[i * 3 + 2] = Float(rawPixels[srcIdx + 2]) / 255.0  // B
    }

    return rgbFloats
}

// MARK: - Errors

enum ImageLoaderError: Error, LocalizedError {
    case failedToLoadImage(URL)
    case failedToGetCGImage
    case contextCreationFailed
    case unsupportedPlatform

    var errorDescription: String? {
        switch self {
        case .failedToLoadImage(let url):
            return "Failed to load image from \(url.path)"
        case .failedToGetCGImage:
            return "Failed to obtain CGImage from loaded image"
        case .contextCreationFailed:
            return "Failed to create CGContext for image processing"
        case .unsupportedPlatform:
            return "Image loading not supported on this platform"
        }
    }
}
