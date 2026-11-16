#!/usr/bin/env python3
"""Upload Haru-4B GGUF Quantized model and audio components to Hugging Face Hub.

Sets base_model: higurashi-ai/Haru-4B in the model card metadata so it links
automatically to the official upstream model page.
"""

import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import HfApi

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set.")
        sys.exit(1)

    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/Haru-4B-GGUF"

    print("=" * 80)
    print(f"  UPLOADING HARU-4B GGUF TO HUGGING FACE HUB")
    print(f"  Target Repository: https://huggingface.co/{repo_id}")
    print(f"  Base Model Linked: higurashi-ai/Haru-4B")
    print("=" * 80)

    # 1. Create public repository
    print(f"Creating public repository {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

    # 2. Prepare staging folder
    staging = Path("/root/hf_upload_stage")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    # 3. Model Card (README.md) with base_model metadata
    readme_content = """---
base_model: higurashi-ai/Haru-4B
language:
- en
- es
- pt
- de
- fr
- zh
- ja
- ru
- ko
pipeline_tag: audio-to-audio
tags:
- speech
- full-duplex
- speech-to-speech
- haru
- gguf
- quantized
- qwen3_5
license: apache-2.0
---

# Haru-4B GGUF (Quantized)

This repository provides **GGUF quantized weights** and extracted FP16 audio modules for **[higurashi-ai/Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B)**, the open-weight continuous full-duplex spoken dialogue foundation model.

## 🚀 Model Architecture & Quantization Strategy
Haru-4B is a full-duplex Listening-while-Speaking Language Model (LSLM) operating at **7.5 Hz (133.33 ms frame budget)**. 

To enable CPU and low-VRAM deployment without compromising voice fidelity, this release applies a **hybrid quantization approach**:
- **Language Model Backbone:** `Qwen3.5-4B` extracted and quantized to **GGUF Q4_K_M** (compressed from 8.4 GB down to **2.6 GB**).
- **Audio Codec (VibeVoice):** Preserved in **FP16** (~1.3 GB) to maintain causal convolution quality and avoid metallic static artifacts.
- **Audio Generation Head (MeanFlow):** Preserved in **FP16** (~694 MB) for 1-step continuous ODE flow-matching sampling.

## 📦 Repository Contents

| File | Format | Size | Description |
| :--- | :--- | :--- | :--- |
| `qwen-haru-q4_k_m.gguf` | GGUF Q4_K_M | **2.6 GB** | 4-bit medium quantized Qwen3.5 language model backbone |
| `vibevoice-codec-f16.safetensors` | Safetensors FP16 | **1.3 GB** | Causal 1D convolution VibeVoice acoustic tokenizer |
| `meanflow-head-f16.safetensors` | Safetensors FP16 | **694 MB** | 4-layer 1-step MeanFlow ODE audio generation head |
| `config.json` | JSON | ~2 KB | Model hyperparameter configuration |
| `tokenizer.json` | JSON | ~20 MB | Qwen tokenizer vocabulary (248,320 tokens) |

## 🛠️ Usage with llama.cpp

The quantized language model backbone can be run with [llama.cpp](https://github.com/ggml-org/llama.cpp):

```bash
# Run inference on CPU
./llama-cli -m qwen-haru-q4_k_m.gguf -p "Hello! Let's talk." -n 64 -t 4
```

## 📜 Citation & Credits
Original base model and architecture developed by **higurashi-ai**:
- Official Repository: [github.com/higurashi-ai/Haru](https://github.com/higurashi-ai/Haru)
- Base Model: [higurashi-ai/Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B)
- License: Apache-2.0
"""
    with open(staging / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("Prepared README.md with base_model metadata.")

    # 4. Copy GGUF
    gguf_src = Path("/root/qwen-haru-q4_k_m.gguf")
    if gguf_src.exists():
        print(f"Staging {gguf_src.name} (size: {gguf_src.stat().st_size / 1e9:.2f} GB)...")
        shutil.copy(gguf_src, staging / "qwen-haru-q4_k_m.gguf")

    # 5. Copy Codec
    codec_src = Path("/root/extracted_codec/model.safetensors")
    if codec_src.exists():
        print(f"Staging vibevoice-codec-f16.safetensors...")
        shutil.copy(codec_src, staging / "vibevoice-codec-f16.safetensors")

    # 6. Copy Head
    head_src = Path("/root/extracted_head/model.safetensors")
    if head_src.exists():
        print(f"Staging meanflow-head-f16.safetensors...")
        shutil.copy(head_src, staging / "meanflow-head-f16.safetensors")

    # 7. Copy Config & Tokenizer
    backbone_dir = Path("/root/extracted_backbone")
    if (backbone_dir / "config.json").exists():
        shutil.copy(backbone_dir / "config.json", staging / "config.json")
    if (backbone_dir / "tokenizer.json").exists():
        shutil.copy(backbone_dir / "tokenizer.json", staging / "tokenizer.json")
    if (backbone_dir / "tokenizer_config.json").exists():
        shutil.copy(backbone_dir / "tokenizer_config.json", staging / "tokenizer_config.json")

    # 8. Upload folder to Hugging Face
    print("Uploading all files to Hugging Face Hub (this will upload fast from Linode datacenter)...")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload Haru-4B GGUF Q4_K_M quantized backbone and FP16 audio modules"
    )

    print("\n" + "=" * 80)
    print(f"  SUCCESSFULLY PUBLISHED TO HUGGING FACE!")
    print(f"  URL: https://huggingface.co/{repo_id}")
    print("=" * 80)

if __name__ == "__main__":
    main()
