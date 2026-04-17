// FlowMatchSampler.swift
//
// Rectified flow / flow matching sampler for LTX-Video.
// Uses Euler method to integrate the ODE from noise to data in 8 steps.
//
// The DiT predicts velocity v, and the update rule is:
//   x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * v(x_t, t)
//
// Sigma schedule uses dynamic shifting based on sequence length.
//
// Reference: diffusers FlowMatchEulerDiscreteScheduler

import Foundation
import MLX
import MLXRandom

public struct FlowMatchSampler {
    public let numSteps: Int
    public let baseShift: Float
    public let maxShift: Float
    public let shiftTerminal: Float

    public init(numSteps: Int = 8, baseShift: Float = 0.95,
                maxShift: Float = 2.05, shiftTerminal: Float = 0.1) {
        self.numSteps = numSteps
        self.baseShift = baseShift
        self.maxShift = maxShift
        self.shiftTerminal = shiftTerminal
    }

    /// Compute the sigma schedule for the given number of steps.
    /// Uses dynamic shifting based on sequence length.
    /// - Parameter seqLen: number of latent tokens (F' * H' * W')
    /// - Returns: Array of sigma values from 1.0 (noise) to ~0 (clean)
    public func sigmaSchedule(seqLen: Int) -> [Float] {
        // Dynamic shift based on resolution
        let baseSeqLen: Float = 1024.0
        let maxSeqLen: Float = 4096.0
        let normalizedLen: Float = Float(seqLen).clamped(to: baseSeqLen...maxSeqLen)
        let mu: Float = baseShift + (maxShift - baseShift) *
            ((normalizedLen - baseSeqLen) / (maxSeqLen - baseSeqLen))

        // Linear timesteps from 1 to ~0
        var sigmas: [Float] = []
        for i in 0...numSteps {
            let t: Float = 1.0 - Float(i) / Float(numSteps)
            // Apply shift: sigma = shift * t / (1 + (shift - 1) * t)
            let shifted: Float = mu * t / (1.0 + (mu - 1.0) * t)
            sigmas.append(shifted)
        }

        // Terminal shift
        sigmas[numSteps] = shiftTerminal / 1000.0

        return sigmas
    }

    /// Run the full denoising loop.
    /// - Parameters:
    ///   - shape: latent shape [B, seqLen, channels]
    ///   - denoiseFn: closure that takes (noisyLatents, timestep) and returns velocity
    /// - Returns: denoised latents
    ///
    /// Peak memory: depends on DiT size (~1 GB for 4-bit DiT + ~0.5 GB activations)
    public func sample(
        shape: [Int],
        denoiseFn: (MLXArray, MLXArray) -> MLXArray
    ) -> MLXArray {
        let seqLen: Int = shape[1]
        let sigmas: [Float] = sigmaSchedule(seqLen: seqLen)

        // Start from pure noise
        var latents: MLXArray = MLXRandom.normal(shape)

        for i in 0..<numSteps {
            let sigma: Float = sigmas[i]
            let sigmaNext: Float = sigmas[i + 1]
            let dt: Float = sigmaNext - sigma

            // Current timestep (sigma * 1000 for the model)
            let timestep: MLXArray = MLXArray([sigma * 1000.0])

            // Predict velocity
            let velocity: MLXArray = denoiseFn(latents, timestep)

            // Euler step: x_{t+1} = x_t + dt * v
            latents = latents + MLXArray(dt) * velocity

            // Force evaluation to free intermediate memory
            eval(latents)
        }

        return latents
    }
}

// Helper extension
extension Float {
    func clamped(to range: ClosedRange<Float>) -> Float {
        return min(max(self, range.lowerBound), range.upperBound)
    }
}
