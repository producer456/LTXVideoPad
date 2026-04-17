# LTXVideoPad — Claude Code Project Context

## Goal
Build a native iPadOS app that runs LTX-Video 2B distilled (v0.9.6) on-device
via MLX Swift for image-to-video generation with text-descriptor prompts and
LoRA support. Target device: iPad Pro 11" M5 with 12GB unified memory.

## Hard constraints
- 12GB unified RAM — app budget ~8GB. MUST unload unused modules during pipeline.
- 4-bit quantization for all weights (T5, DiT). VAE stays FP16.
- Output max 512x320 @ 49 frames (~2s at 24fps) for v1.
- 8-step distilled sampler only. No full LTX-Video-2B-dev.
- Everything on-device. No cloud fallback. No telemetry.
- Swift 5.10+, iOS 18+, MLX Swift main branch.

## Reference repos (treat as read-only)
- https://github.com/Blaizzy/mlx-video — Python MLX impl, primary reference
- https://github.com/Lightricks/LTX-Video — original PyTorch, architectural truth
- https://github.com/ml-explore/mlx-swift-examples — Swift patterns for weights/quant/tokenizer

## User profile
- Developer new to Swift, strong Python/C/Arduino background.
- Communication: direct, skeptical, no filler. Flag design flaws explicitly.
- Prefers programs that prompt for file paths rather than hardcoding.
- Breaks complex problems into small, testable steps.

## Workflow rules
1. Before changing architecture or adding a dep, ASK.
2. After each file change, state in one sentence what was changed and why.
3. If a Python MLX function in mlx-video is ambiguous, cite the file/line and
   propose 2 Swift translations with tradeoffs rather than guessing.
4. Memory-sensitive code (ML/Pipeline/*, ML/Quantization/*): include estimated
   peak RAM in a comment above each function that allocates >50MB.
5. When writing Swift for the user to read: prefer explicit types over inference
   in ML-layer code (he's learning — readability > idiomatic).
6. Any MLX Swift API uncertainty: check mlx-swift-examples first, don't invent.

## Phase status
- [x] Phase 0: mlx-swift-examples LLMEval running on iPad
- [ ] Phase 1: T5-XXL encoder port + prompt embedding test
- [ ] Phase 2: VAE3D encode/decode validated on static image
- [ ] Phase 3: DiT backbone + rectified flow sampler
- [ ] Phase 4: End-to-end I2V, hardcoded prompt + image
- [ ] Phase 5: SwiftUI shell (ImageDropZone, PromptEditor, progress, export)
- [ ] Phase 6: LoRA loader with runtime merge
- [ ] Phase 7: Polish — queue, thermal throttle, MetalFX upscale

## Current phase
Phase 1 — porting T5Encoder.swift from mlx_video/ltx_2/text_encoder.py
