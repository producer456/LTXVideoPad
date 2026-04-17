// ThermalMonitor.swift
//
// Monitors device thermal state and pauses inference if overheating.
// iPad Pro M5 can throttle under sustained ML load — this prevents crashes.
//
// Thermal states: nominal → fair → serious → critical
// We pause at "serious" and abort at "critical".

import Foundation
import os

public class ThermalMonitor: @unchecked Sendable {
    public static let shared = ThermalMonitor()

    private let logger = Logger(subsystem: "com.ltxvideopad", category: "Thermal")
    private var observation: NSObjectProtocol?

    public enum Action {
        case proceed       // safe to continue
        case throttle      // slow down (add delay between steps)
        case pause         // pause until temp drops
        case abort         // too hot, stop immediately
    }

    /// Current thermal recommendation.
    public var currentAction: Action {
        let state = ProcessInfo.processInfo.thermalState
        switch state {
        case .nominal:  return .proceed
        case .fair:     return .proceed
        case .serious:  return .throttle
        case .critical: return .abort
        @unknown default: return .proceed
        }
    }

    /// Current thermal state as a string.
    public var stateString: String {
        let state = ProcessInfo.processInfo.thermalState
        switch state {
        case .nominal:  return "nominal"
        case .fair:     return "fair"
        case .serious:  return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    /// Start observing thermal state changes.
    public func startMonitoring() {
        observation = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil, queue: .main
        ) { [weak self] _ in
            guard let self else { return }
            let state = self.stateString
            self.logger.info("Thermal state changed: \(state)")
        }
        let s = self.stateString
        logger.info("Thermal monitoring started: \(s)")
    }

    /// Stop observing.
    public func stopMonitoring() {
        if let obs = observation {
            NotificationCenter.default.removeObserver(obs)
            observation = nil
        }
    }

    /// Check thermal state and wait if needed. Call between denoise steps.
    /// Returns false if generation should be aborted.
    public func checkAndWait() async -> Bool {
        switch currentAction {
        case .proceed:
            return true
        case .throttle:
            logger.warning("Thermal throttle — adding 2s cooldown")
            try? await Task.sleep(for: .seconds(2))
            return true
        case .pause:
            logger.warning("Thermal pause — waiting for cooldown")
            for _ in 0..<30 {  // wait up to 60 seconds
                try? await Task.sleep(for: .seconds(2))
                if currentAction == .proceed || currentAction == .throttle {
                    logger.info("Thermal recovered — resuming")
                    return true
                }
                if currentAction == .abort {
                    return false
                }
            }
            logger.error("Thermal pause timeout — aborting")
            return false
        case .abort:
            logger.error("Thermal critical — aborting generation")
            return false
        }
    }

    private init() {}
}
