// MemoryManager.swift
//
// Manages sequential loading/unloading of large ML models to stay within
// the 8GB app memory budget on iPad Pro M5 (12GB total).
//
// Pipeline flow:
// 1. Load T5 encoder → encode prompt → unload T5 (~2.5 GB freed)
// 2. Load VAE encoder → encode input image → unload VAE encoder (~0.3 GB freed)
// 3. Load DiT → run 8 denoising steps → unload DiT (~1.5 GB freed)
// 4. Load VAE decoder → decode latents to frames → unload VAE decoder (~0.3 GB freed)
// 5. Export video
//
// At no point should more than one large model be resident.

import Foundation
import os

/// Tracks memory usage and enforces sequential model loading.
final class MemoryManager {
    static let shared = MemoryManager()

    private let logger = Logger(subsystem: "com.ltxvideopad", category: "Memory")
    private(set) var currentlyLoaded: String? = nil

    /// Returns current app memory usage in MB.
    var currentMemoryMB: Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return -1 }
        return Int(info.resident_size / (1024 * 1024))
    }

    /// Log current memory state.
    func logMemory(context: String) {
        logger.info("\(context): \(self.currentMemoryMB) MB resident, loaded=\(self.currentlyLoaded ?? "none")")
    }

    /// Call before loading a model. Verifies nothing else is loaded.
    func willLoad(model: String) {
        if let existing = currentlyLoaded {
            logger.warning("Loading \(model) while \(existing) is still loaded — potential OOM")
        }
        currentlyLoaded = model
        logMemory(context: "willLoad(\(model))")
    }

    /// Call after unloading a model.
    func didUnload(model: String) {
        if currentlyLoaded == model {
            currentlyLoaded = nil
        }
        logMemory(context: "didUnload(\(model))")
    }

    private init() {}
}
