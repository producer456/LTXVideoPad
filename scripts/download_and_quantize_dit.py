#!/usr/bin/env python3
"""
Download LTX-Video distilled DiT weights (BF16) from HuggingFace and quantize to 4-bit.

Downloads:  Lightricks/LTX-Video  ltxv-2b-0.9.6-distilled-04-25.safetensors (~6.34 GB BF16)
Outputs:    Models/dit-4bit/ (~1 GB safetensors + quantize_config.json)

Only 2D weight matrices (Linear layers) are quantized. Biases, norms, embeddings,
and small tensors are kept in FP16.

Requirements:
    pip install mlx huggingface_hub safetensors

Usage:
    python3 scripts/download_and_quantize_dit.py
    python3 scripts/download_and_quantize_dit.py --output-dir /path/to/output
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download and quantize LTX-Video DiT to 4-bit")
    parser.add_argument("--output-dir", type=str, default="Models/dit-4bit",
                        help="Output directory for quantized model")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (default: 4)")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size (default: 64)")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, use cached file")
    args = parser.parse_args()

    try:
        import mlx.core as mx
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install mlx huggingface_hub safetensors")
        return

    repo_id = "Lightricks/LTX-Video"
    filename = "ltxv-2b-0.9.6-distilled-04-25.safetensors"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download from HuggingFace
    print(f"{'=' * 60}")
    print(f"Step 1: Download DiT weights from HuggingFace")
    print(f"{'=' * 60}")

    if args.skip_download:
        # Look for cached file
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        local_path = None
        for d in cache_dir.glob("models--Lightricks--LTX-Video"):
            blobs = d / "blobs"
            snapshots = d / "snapshots"
            if snapshots.exists():
                for snap in snapshots.iterdir():
                    candidate = snap / filename
                    if candidate.exists():
                        local_path = str(candidate)
                        break
        if not local_path:
            print("ERROR: No cached file found. Run without --skip-download first.")
            return
    else:
        print(f"Downloading {repo_id}/{filename}...")
        print("This is ~6.34 GB and may take a while.")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )

    print(f"Model file at: {local_path}")

    # Step 2: Load weights
    print(f"\n{'=' * 60}")
    print(f"Step 2: Load BF16 weights")
    print(f"{'=' * 60}")

    raw_weights = mx.load(local_path)
    print(f"Loaded {len(raw_weights)} tensors total")

    # The distilled checkpoint bundles DiT + VAE. Strip VAE since we have it separately.
    all_weights = {k: v for k, v in raw_weights.items() if k.startswith("model.diffusion_model")}
    vae_count = sum(1 for k in raw_weights if k.startswith("vae."))
    if vae_count:
        print(f"Stripped {vae_count} bundled VAE tensors (use separate VAE weights)")
    print(f"DiT tensors to quantize: {len(all_weights)}")

    # Show some sample keys for debugging
    sample_keys = sorted(all_weights.keys())[:10]
    print(f"Sample keys: {sample_keys}")

    # Step 3: Quantize
    print(f"\n{'=' * 60}")
    print(f"Step 3: Quantize to {args.bits}-bit (group_size={args.group_size})")
    print(f"{'=' * 60}")

    quantized_weights = {}
    total_original = 0
    total_quantized = 0
    quantized_count = 0
    kept_count = 0

    for key in sorted(all_weights.keys()):
        weight = all_weights[key]
        original_size = weight.nbytes
        total_original += original_size

        # Only quantize 2D weight matrices (Linear layers)
        # Skip biases (1D), norms, embeddings, and small tensors
        if (weight.ndim == 2
            and weight.shape[0] >= args.group_size
            and weight.shape[1] >= args.group_size):

            # Cast to float16 first (MLX quantize needs float, not bfloat16)
            w_f16 = weight.astype(mx.float16)
            qw, scales, biases = mx.quantize(w_f16, group_size=args.group_size, bits=args.bits)
            quantized_weights[key] = qw
            quantized_weights[key.replace(".weight", ".scales")] = scales
            quantized_weights[key.replace(".weight", ".biases")] = biases

            q_size = qw.nbytes + scales.nbytes + biases.nbytes
            total_quantized += q_size
            quantized_count += 1
            ratio = original_size / q_size if q_size > 0 else 0
            print(f"  Q {key}: {weight.shape} {weight.dtype} -> {args.bits}-bit "
                  f"({original_size / 1e6:.1f} MB -> {q_size / 1e6:.1f} MB, {ratio:.1f}x)")
        else:
            # Keep as FP16
            quantized_weights[key] = weight.astype(mx.float16)
            total_quantized += quantized_weights[key].nbytes
            kept_count += 1
            if weight.nbytes > 100_000:  # Only log larger kept tensors
                print(f"  K {key}: {weight.shape} {weight.dtype} -> kept FP16 ({original_size / 1e6:.1f} MB)")

    print(f"\nQuantized {quantized_count} tensors, kept {kept_count} as FP16")
    print(f"Total: {total_original / 1e9:.2f} GB -> {total_quantized / 1e9:.2f} GB "
          f"({total_original / total_quantized:.1f}x compression)")

    # Step 4: Save quantized weights
    print(f"\n{'=' * 60}")
    print(f"Step 4: Save to {output_dir}")
    print(f"{'=' * 60}")

    # Evaluate all arrays before saving (force computation)
    mx.eval(*quantized_weights.values())

    output_file = output_dir / "model.safetensors"
    mx.save_safetensors(str(output_file), quantized_weights)
    actual_size = output_file.stat().st_size
    print(f"Saved safetensors: {output_file} ({actual_size / 1e9:.2f} GB)")

    # Save quantization config
    quant_config = {
        "quantization": {
            "bits": args.bits,
            "group_size": args.group_size,
        },
        "model_type": "ltx-video-dit-distilled",
        "source": f"{repo_id}/{filename}",
        "num_parameters_total": len(all_weights),
        "num_quantized": quantized_count,
        "num_kept_fp16": kept_count,
        "original_size_gb": round(total_original / 1e9, 2),
        "quantized_size_gb": round(total_quantized / 1e9, 2),
    }
    with open(output_dir / "quantize_config.json", "w") as f:
        json.dump(quant_config, f, indent=2)
    print(f"Saved quantize_config.json")

    print(f"\n{'=' * 60}")
    print(f"Done! Quantized DiT model saved to: {output_dir}")
    print(f"Size on disk: {actual_size / 1e9:.2f} GB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
