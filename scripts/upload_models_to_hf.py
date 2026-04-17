#!/usr/bin/env python3
"""
Upload quantized LTXVideoPad models to HuggingFace Hub.

This creates a repo with all pre-quantized model weights so the iPad app
can download them on first launch.

Usage:
    # First, login to HuggingFace:
    huggingface-cli login

    # Then upload:
    python3 scripts/upload_models_to_hf.py

Repo structure:
    producer456/LTXVideoPad-models/
    ├── t5xxl-encoder-4bit/
    │   ├── model.safetensors      (~2.68 GB)
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   └── config.json
    ├── dit-4bit/
    │   └── model.safetensors      (~1.08 GB)
    └── vae/
        ├── diffusion_pytorch_model.safetensors  (~1.68 GB)
        └── config.json
"""

import os
from pathlib import Path

def main():
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("pip install huggingface_hub")
        return

    repo_id = "producer456/LTXVideoPad-models"
    models_dir = Path("Models")

    # Create repo if it doesn't exist
    api = HfApi()
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"Repo: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload each model directory
    uploads = [
        ("t5xxl-encoder-4bit", ["model.safetensors", "tokenizer.json",
                                 "tokenizer_config.json", "config.json",
                                 "spiece.model", "special_tokens_map.json"]),
        ("dit-4bit", ["model.safetensors"]),
        ("vae/vae", ["diffusion_pytorch_model.safetensors", "config.json"]),
    ]

    for subdir, files in uploads:
        local_dir = models_dir / subdir
        for filename in files:
            local_path = local_dir / filename
            if not local_path.exists():
                print(f"  SKIP (not found): {local_path}")
                continue

            size_mb = local_path.stat().st_size / (1024 * 1024)
            # Flatten the upload path (remove nested vae/vae)
            remote_subdir = subdir.replace("vae/vae", "vae")
            remote_path = f"{remote_subdir}/{filename}"

            print(f"  Uploading {remote_path} ({size_mb:.0f} MB)...")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  Done: {remote_path}")

    print(f"\nAll uploads complete!")
    print(f"https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
