#!/usr/bin/env python3
"""Inspect Haru-4B model weights and configuration.

Supports inspecting both local directories and remote Hugging Face Hub repositories.
When querying Hugging Face, uses HTTP Range requests on model.safetensors to fetch
only the 111 KB header, avoiding downloading the 10.5 GB weight file.
"""

import sys
import json
import struct
import math
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def fetch_remote_safetensors_header(repo_id: str) -> Tuple[Dict[str, Any], int]:
    url = f"https://huggingface.co/{repo_id}/resolve/main/model.safetensors"
    # 1. Fetch first 8 bytes to get header length
    req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(req) as resp:
        header_len = struct.unpack("<Q", resp.read())[0]

    # 2. Fetch the JSON header bytes
    req2 = urllib.request.Request(url, headers={"Range": f"bytes=8-{8 + header_len - 1}"})
    with urllib.request.urlopen(req2) as resp2:
        header_json = json.loads(resp2.read().decode("utf-8"))

    return header_json, header_len


def read_local_safetensors_header(file_path: Path) -> Tuple[Dict[str, Any], int]:
    with open(file_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_json = json.loads(f.read(header_len).decode("utf-8"))
    return header_json, header_len


def analyze_tensors(header: Dict[str, Any]):
    counts: Dict[str, int] = {}
    dtypes: Dict[str, int] = {}
    tensors_by_prefix: Dict[str, list] = {}

    for k, v in header.items():
        if k == "__metadata__":
            continue
        prefix = k.split(".")[0]
        shape = v.get("shape", [])
        dtype = v.get("dtype", "unknown")
        param_count = math.prod(shape) if shape else 1

        counts[prefix] = counts.get(prefix, 0) + param_count
        dtypes[dtype] = dtypes.get(dtype, 0) + param_count
        tensors_by_prefix.setdefault(prefix, []).append((k, shape, dtype, param_count))

    total_params = sum(counts.values())

    print("\n" + "=" * 80)
    print("  HARU-4B ARCHITECTURE & WEIGHT INSPECTOR")
    print("=" * 80)
    print(f"Total Tensors in Checkpoint : {len(header) - 1:,d}")
    print(f"Total Parameter Count       : {total_params:,d} ({total_params / 1e9:.2f} Billion)")
    print(f"Primary Tensor Data Type    : {', '.join(f'{dt} ({cnt / total_params * 100:.1f}%)' for dt, cnt in dtypes.items())}")
    print("-" * 80)

    print(f"{'Module Prefix':<20} | {'Parameters':<16} | {'% of Total':<12} | {'Tensor Count'}")
    print("-" * 80)
    for prefix, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = (count / total_params) * 100
        num_tensors = len(tensors_by_prefix[prefix])
        print(f"{prefix:<20} | {count:>14,d} | {pct:>10.2f}% | {num_tensors:>10d}")
    print("=" * 80)

    # Detailed component commentary
    print("\n[Quantization & Porting Strategy by Component]")
    print(f"1. Language Model Backbone (lm.*):")
    print(f"   - Size: {counts.get('lm', 0):,d} params (~{counts.get('lm', 0) * 2 / 1e9:.2f} GB in BF16)")
    print(f"   - Strategy: Extract and quantize to GGUF Q4_K_M (~{counts.get('lm', 0) * 0.56 / 1e9:.2f} GB).")
    print(f"   - Target Engine: llama.cpp / ggml-based Qwen runtime with inputs_embeds support.")

    print(f"\n2. VibeVoice Audio Codec (codec.*):")
    print(f"   - Size: {counts.get('codec', 0):,d} params (~{counts.get('codec', 0) * 2 / 1e9:.2f} GB in BF16)")
    print(f"   - Strategy: Maintain in FP16 / BF16. Do NOT quantize to 4-bit to prevent audio distortion.")
    print(f"   - Target Engine: C++ causal Conv1d & ConvTranspose1d ggml implementation.")

    print(f"\n3. MeanFlow Continuous Head (head.*):")
    print(f"   - Size: {counts.get('head', 0):,d} params (~{counts.get('head', 0) * 2 / 1e9:.2f} GB in BF16)")
    print(f"   - Strategy: Maintain in FP16. 1-step ODE solver runs in < 2 ms.")

    print(f"\n4. Audio Projectors (model_audio_proj.*, user_audio_proj.*):")
    print(f"   - Size: {counts.get('model_audio_proj', 0) + counts.get('user_audio_proj', 0):,d} params (< 30 MB in FP16)")
    print(f"   - Strategy: Keep FP16; projects 64-dim latents into 2560-dim LM embedding space.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect Haru-4B weights and tensor dimensions.")
    parser.add_argument("--repo", type=str, default="higurashi-ai/Haru-4B", help="Hugging Face repo ID")
    parser.add_argument("--local-file", type=str, default=None, help="Path to local model.safetensors file")

    args = parser.parse_args()

    if args.local_file:
        file_path = Path(args.local_file)
        if not file_path.exists():
            print(f"Error: Local file {file_path} not found.")
            sys.exit(1)
        print(f"Reading local safetensors header from {file_path}...")
        header, _ = read_local_safetensors_header(file_path)
    else:
        print(f"Streaming safetensors header via HTTP Range request from https://huggingface.co/{args.repo}...")
        try:
            header, length = fetch_remote_safetensors_header(args.repo)
            print(f"Successfully fetched {length:,d} bytes header from remote repository.")
        except Exception as e:
            print(f"Failed to fetch remote header: {e}")
            sys.exit(1)

    analyze_tensors(header)


if __name__ == "__main__":
    main()
