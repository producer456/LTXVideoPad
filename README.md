# LTXVideoPad

On-device image-to-video generation for iPad Pro using LTX-Video 2B distilled and MLX Swift.

## What is this?

A native iPadOS app that runs the LTX-Video 2B distilled model (v0.9.6) entirely on-device — no cloud, no API calls. Drop an image, type a prompt, get a 2-second video.

## Target Hardware

- iPad Pro 11" M5 (12GB unified memory)
- iOS 18+

## Architecture

- **T5-XXL** text encoder (4-bit quantized) — prompt → embeddings
- **DiT** transformer backbone (4-bit quantized) — denoises latents
- **VAE3D** encoder/decoder (FP16) — image ↔ latent space
- **8-step rectified flow** sampler — fast inference
- **LoRA** support — runtime style injection

## Memory Budget

| Component | Estimated RAM |
|-----------|--------------|
| T5-XXL (4-bit) | ~2.5 GB |
| DiT (4-bit) | ~1.5 GB |
| VAE3D (FP16) | ~0.5 GB |
| Latents + intermediates | ~1.0 GB |
| App + OS overhead | ~2.5 GB |
| **Total** | **~8.0 GB** |

Components are loaded/unloaded sequentially — only one large model in memory at a time.

## Build

Requires Xcode 16+, Swift 5.10+, MLX Swift framework.

## Status

Phase 1: T5 encoder port in progress.
