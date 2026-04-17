#!/usr/bin/env python3
"""
Download T5-v1.1-XXL encoder (FP16) from HuggingFace and quantize to 4-bit.

Downloads:  mlx-community/DeepFloyd-t5-v1_1-xxl (~8.9 GB)
Outputs:    t5xxl-encoder-4bit/ (~2.4 GB safetensors + config + tokenizer)

Requirements:
    pip install mlx huggingface_hub safetensors sentencepiece

Usage:
    python3 scripts/download_and_quantize_t5.py
    python3 scripts/download_and_quantize_t5.py --output-dir /path/to/output
"""

import argparse
import json
import os
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download and quantize T5-XXL encoder to 4-bit")
    parser.add_argument("--output-dir", type=str, default="Models/t5xxl-encoder-4bit",
                        help="Output directory for quantized model")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (default: 4)")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size (default: 64)")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, use cached files")
    args = parser.parse_args()

    try:
        import mlx.core as mx
        import mlx.nn as nn
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install mlx huggingface_hub safetensors sentencepiece")
        return

    model_id = "mlx-community/DeepFloyd-t5-v1_1-xxl"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download from HuggingFace
    print(f"{'=' * 60}")
    print(f"Step 1: Download T5-XXL encoder from HuggingFace")
    print(f"{'=' * 60}")

    if args.skip_download:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        # Find cached model
        print("Skipping download, looking for cached files...")
        local_dir = None
        for d in cache_dir.glob("models--mlx-community--DeepFloyd*"):
            snap = d / "snapshots"
            if snap.exists():
                for s in snap.iterdir():
                    local_dir = str(s)
                    break
        if not local_dir:
            print("ERROR: No cached model found. Run without --skip-download first.")
            return
    else:
        print(f"Downloading {model_id}...")
        print("This is ~8.9 GB and may take a while.")
        local_dir = snapshot_download(
            model_id,
            allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
        )

    print(f"Model files at: {local_dir}")

    # Step 2: Load weights
    print(f"\n{'=' * 60}")
    print(f"Step 2: Load FP16 weights")
    print(f"{'=' * 60}")

    local_path = Path(local_dir)
    safetensors_files = sorted(local_path.glob("*.safetensors"))
    print(f"Found {len(safetensors_files)} safetensors file(s)")

    all_weights = {}
    for f in safetensors_files:
        print(f"  Loading {f.name}...")
        # Use MLX's native safetensors loader (handles bfloat16 correctly)
        shard = mx.load(str(f))
        all_weights.update(shard)

    print(f"Loaded {len(all_weights)} tensors")

    # Filter to encoder-only weights
    encoder_weights = {}
    skipped = []
    for key, val in all_weights.items():
        if key.startswith("encoder.") or key == "shared.weight":
            encoder_weights[key] = val
        else:
            skipped.append(key)

    print(f"Encoder weights: {len(encoder_weights)}")
    if skipped:
        print(f"Skipped {len(skipped)} non-encoder weights: {skipped[:5]}...")

    # Step 3: Quantize
    print(f"\n{'=' * 60}")
    print(f"Step 3: Quantize to {args.bits}-bit (group_size={args.group_size})")
    print(f"{'=' * 60}")

    quantized_weights = {}
    total_original = 0
    total_quantized = 0

    for key, weight in encoder_weights.items():
        original_size = weight.nbytes
        total_original += original_size

        # Only quantize 2D weight matrices (not biases, not norms, not embeddings with special shapes)
        if weight.ndim == 2 and weight.shape[0] >= args.group_size and weight.shape[1] >= args.group_size:
            # Quantize: returns (quantized_weight, scales, biases)
            qw, scales, biases = mx.quantize(weight, group_size=args.group_size, bits=args.bits)
            quantized_weights[key] = qw
            quantized_weights[key.replace(".weight", ".scales")] = scales
            quantized_weights[key.replace(".weight", ".biases")] = biases

            q_size = qw.nbytes + scales.nbytes + biases.nbytes
            total_quantized += q_size
            ratio = original_size / q_size if q_size > 0 else 0
            print(f"  {key}: {weight.shape} -> {args.bits}-bit ({original_size / 1e6:.1f} MB -> {q_size / 1e6:.1f} MB, {ratio:.1f}x)")
        else:
            # Keep as-is (layer norms, small tensors)
            quantized_weights[key] = weight
            total_quantized += weight.nbytes
            print(f"  {key}: {weight.shape} -> kept FP16 ({original_size / 1e6:.1f} MB)")

    print(f"\nTotal: {total_original / 1e9:.2f} GB -> {total_quantized / 1e9:.2f} GB "
          f"({total_original / total_quantized:.1f}x compression)")

    # Step 4: Save quantized weights
    print(f"\n{'=' * 60}")
    print(f"Step 4: Save to {output_dir}")
    print(f"{'=' * 60}")

    # Save as safetensors using MLX's built-in save
    output_file = output_dir / "model.safetensors"
    mx.save_safetensors(str(output_file), quantized_weights)
    print(f"Saved safetensors: {output_file}")

    # Copy tokenizer files
    for pattern in ["*.json", "*.model", "*.txt", "tokenizer*"]:
        for f in local_path.glob(pattern):
            dest = output_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
                print(f"Copied {f.name}")

    # Save quantization config
    quant_config = {
        "quantization": {
            "bits": args.bits,
            "group_size": args.group_size,
        },
        "model_type": "t5-encoder",
        "source": model_id,
        "num_parameters": len(encoder_weights),
        "original_size_gb": round(total_original / 1e9, 2),
        "quantized_size_gb": round(total_quantized / 1e9, 2),
    }
    with open(output_dir / "quantize_config.json", "w") as f:
        json.dump(quant_config, f, indent=2)
    print(f"Saved quantize_config.json")

    print(f"\n{'=' * 60}")
    print(f"Done! Quantized model saved to: {output_dir}")
    print(f"Size: {total_quantized / 1e9:.2f} GB")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
