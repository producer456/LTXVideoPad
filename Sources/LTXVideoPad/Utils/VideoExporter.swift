// VideoExporter.swift
// Exports MLXArray frames to an MP4 video file using AVFoundation.

import Foundation
import MLX

#if canImport(AVFoundation)
import AVFoundation
import CoreVideo

/// Exports a batch of frames stored as an MLXArray to an H.264 MP4 file.
///
/// - Parameters:
///   - frames: MLXArray of shape [F, H, W, 3] with Float32 values in [0, 1].
///   - outputURL: File URL for the output .mp4 file. Will overwrite if exists.
///   - fps: Frames per second (default 24).
///
/// Peak RAM estimate: ~(H * W * 4) bytes per frame for the CVPixelBuffer copy,
/// plus the MLXArray itself.  At 512x320 that is ~0.6 MB per buffer — negligible.
func exportFramesToMP4(
    frames: MLXArray,
    outputURL: URL,
    fps: Int32 = 24
) async throws {
    // --- Validate shape ---------------------------------------------------
    let shape: [Int] = frames.shape  // [F, H, W, 3]
    guard shape.count == 4, shape[3] == 3 else {
        throw VideoExporterError.invalidShape(shape)
    }
    let frameCount: Int = shape[0]
    let height: Int = shape[1]
    let width: Int = shape[2]

    // --- Remove existing file if present ----------------------------------
    if FileManager.default.fileExists(atPath: outputURL.path) {
        try FileManager.default.removeItem(at: outputURL)
    }

    // --- AVAssetWriter setup ----------------------------------------------
    let writer: AVAssetWriter = try AVAssetWriter(
        outputURL: outputURL,
        fileType: .mp4
    )

    let videoSettings: [String: Any] = [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: width,
        AVVideoHeightKey: height,
    ]

    let writerInput: AVAssetWriterInput = AVAssetWriterInput(
        mediaType: .video,
        outputSettings: videoSettings
    )
    writerInput.expectsMediaDataInRealTime = false

    let pixelBufferAttributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
    ]

    let adaptor: AVAssetWriterInputPixelBufferAdaptor =
        AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: pixelBufferAttributes
        )

    guard writer.canAdd(writerInput) else {
        throw VideoExporterError.cannotAddInput
    }
    writer.add(writerInput)

    guard writer.startWriting() else {
        throw VideoExporterError.writerFailed(
            writer.error?.localizedDescription ?? "unknown"
        )
    }
    writer.startSession(atSourceTime: .zero)

    let frameDuration: CMTime = CMTime(value: 1, timescale: fps)

    // --- Write frames -----------------------------------------------------
    // Force the full array to be evaluated once so per-frame slicing is cheap.
    let evaluatedFrames: MLXArray = frames.asType(.float32)
    eval(evaluatedFrames)

    for i in 0..<frameCount {
        // Wait until the input is ready for more data.
        while !writerInput.isReadyForMoreMediaData {
            try await Task.sleep(nanoseconds: 10_000_000) // 10 ms
        }

        // Slice single frame: [H, W, 3]
        let frameSlice: MLXArray = evaluatedFrames[i]
        eval(frameSlice)

        let pixelBuffer: CVPixelBuffer = try createPixelBuffer(
            from: frameSlice, width: width, height: height
        )

        let presentationTime: CMTime = CMTimeMultiply(frameDuration, multiplier: Int32(i))

        guard adaptor.append(pixelBuffer, withPresentationTime: presentationTime) else {
            throw VideoExporterError.appendFailed(frame: i)
        }
    }

    // --- Finalize ---------------------------------------------------------
    writerInput.markAsFinished()
    await writer.finishWriting()

    if writer.status == .failed {
        throw VideoExporterError.writerFailed(
            writer.error?.localizedDescription ?? "unknown"
        )
    }
}

// MARK: - Pixel Buffer Conversion

/// Converts an MLXArray of shape [H, W, 3] (float 0-1) to a BGRA CVPixelBuffer.
private func createPixelBuffer(
    from frame: MLXArray,
    width: Int,
    height: Int
) throws -> CVPixelBuffer {
    var pixelBuffer: CVPixelBuffer?
    let status: CVReturn = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width,
        height,
        kCVPixelFormatType_32BGRA,
        nil,
        &pixelBuffer
    )
    guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
        throw VideoExporterError.pixelBufferCreationFailed
    }

    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

    guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
        throw VideoExporterError.pixelBufferCreationFailed
    }

    let bytesPerRow: Int = CVPixelBufferGetBytesPerRow(buffer)
    let dst: UnsafeMutablePointer<UInt8> = baseAddress
        .assumingMemoryBound(to: UInt8.self)

    // Clamp to [0,1] and convert to UInt8 [0,255].
    let clamped: MLXArray = clip(frame, min: 0.0, max: 1.0)
    let uint8Frame: MLXArray = (clamped * 255).asType(.uint8)
    eval(uint8Frame)

    // Read raw bytes — contiguous [H, W, 3] in row-major order (RGB).
    let totalPixels: Int = height * width
    let flatRGB: [UInt8] = uint8Frame.reshaped([totalPixels * 3]).asArray(UInt8.self)

    for row in 0..<height {
        for col in 0..<width {
            let srcIdx: Int = (row * width + col) * 3
            let dstIdx: Int = row * bytesPerRow + col * 4

            let r: UInt8 = flatRGB[srcIdx]
            let g: UInt8 = flatRGB[srcIdx + 1]
            let b: UInt8 = flatRGB[srcIdx + 2]

            dst[dstIdx + 0] = b  // B
            dst[dstIdx + 1] = g  // G
            dst[dstIdx + 2] = r  // R
            dst[dstIdx + 3] = 255 // A
        }
    }

    return buffer
}

// MARK: - Errors

enum VideoExporterError: Error, LocalizedError {
    case invalidShape([Int])
    case cannotAddInput
    case writerFailed(String)
    case appendFailed(frame: Int)
    case pixelBufferCreationFailed

    var errorDescription: String? {
        switch self {
        case .invalidShape(let s):
            return "Expected MLXArray shape [F, H, W, 3], got \(s)"
        case .cannotAddInput:
            return "AVAssetWriter cannot add video input"
        case .writerFailed(let msg):
            return "AVAssetWriter failed: \(msg)"
        case .appendFailed(let f):
            return "Failed to append pixel buffer for frame \(f)"
        case .pixelBufferCreationFailed:
            return "Failed to create CVPixelBuffer"
        }
    }
}

#endif // canImport(AVFoundation)
