#!/usr/bin/env python3
"""Qwen3.5-4B Backbone Extraction & GGUF Quantization Pipeline for Haru-4B.

This script:
1. Extracts the language model sub-module ('lm.*') from Haru-4B checkpoints.
2. Formats it into a clean, standalone Qwen3.5 Hugging Face model directory.
3. Generates the exact conversion command for llama.cpp's convert_hf_to_gguf.py.
4. Generates llama-quantize commands for producing Q4_K_M, Q8_0, and Q5_K_M GGUFs.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def generate_qwen_config(haru_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and format Qwen3.5 text model configuration from Haru config."""
    lm_cfg = haru_config.get("lm", {})
    return {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_text",
        "vocab_size": lm_cfg.get("vocab_size", 248320),
        "hidden_size": lm_cfg.get("hidden_size", 2560),
        "intermediate_size": lm_cfg.get("intermediate_size", 9216),
        "num_hidden_layers": lm_cfg.get("num_hidden_layers", 32),
        "num_attention_heads": lm_cfg.get("num_attention_heads", 16),
        "num_key_value_heads": lm_cfg.get("num_key_value_heads", 4),
        "head_dim": lm_cfg.get("head_dim", 256),
        "hidden_act": lm_cfg.get("hidden_act", "silu"),
        "max_position_embeddings": lm_cfg.get("max_position_embeddings", 262144),
        "rms_norm_eps": lm_cfg.get("rms_norm_eps", 1e-6),
        "attention_bias": lm_cfg.get("attention_bias", False),
        "attention_dropout": lm_cfg.get("attention_dropout", 0.0),
        "attn_output_gate": lm_cfg.get("attn_output_gate", True),
        "full_attention_interval": lm_cfg.get("full_attention_interval", 4),
        "layer_types": lm_cfg.get("layer_types", []),
        "linear_conv_kernel_dim": lm_cfg.get("linear_conv_kernel_dim", 4),
        "linear_key_head_dim": lm_cfg.get("linear_key_head_dim", 128),
        "linear_value_head_dim": lm_cfg.get("linear_value_head_dim", 128),
        "linear_num_key_heads": lm_cfg.get("linear_num_key_heads", 16),
        "linear_num_value_heads": lm_cfg.get("linear_num_value_heads", 32),
        "mamba_ssm_dtype": lm_cfg.get("mamba_ssm_dtype", "float32"),
        "partial_rotary_factor": lm_cfg.get("partial_rotary_factor", 0.25),
        "rope_parameters": lm_cfg.get("rope_parameters", {}),
        "tie_word_embeddings": lm_cfg.get("tie_word_embeddings", True),
        "eos_token_id": lm_cfg.get("eos_token_id", 248044),
        "pad_token_id": haru_config.get("pad_token_id", 248076),
        "torch_dtype": "bfloat16"
    }


def extract_and_save_backbone(
    model_safetensors_path: Path,
    output_dir: Path,
    haru_config_path: Path,
):
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
        import torch
    except ImportError:
        print("Error: torch and safetensors required for weight extraction.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write standalone Qwen config
    with open(haru_config_path, "r", encoding="utf-8") as f:
        haru_config = json.load(f)
    qwen_config = generate_qwen_config(haru_config)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(qwen_config, f, indent=2)
    print(f"✅ Generated standalone Qwen config at {output_dir / 'config.json'}")

    # 2. Extract tensors with prefix 'lm.'
    print(f"Reading weights from {model_safetensors_path}...")
    extracted_tensors = {}
    with safe_open(model_safetensors_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("lm."):
                # Rename 'lm.model.*' to 'model.*' or keep as Qwen standard
                new_key = k[3:]  # strip 'lm.'
                extracted_tensors[new_key] = f.get_tensor(k)

    print(f"Extracted {len(extracted_tensors)} backbone tensors ({sum(t.numel() for t in extracted_tensors.values()):,d} parameters).")
    save_path = output_dir / "model.safetensors"
    save_file(extracted_tensors, save_path)
    print(f"✅ Saved extracted Qwen backbone to {save_path}")


def print_gguf_instructions(output_dir: Path):
    print("\n" + "=" * 80)
    print("  NEXT STEPS: CONVERT & QUANTIZE TO GGUF VIA LLAMA.CPP")
    print("=" * 80)
    print("1. Clone llama.cpp if not already present:")
    print("   git clone https://github.com/ggml-org/llama.cpp.git\n")

    print("2. Convert the extracted Qwen backbone to FP16 GGUF:")
    print(f"   python llama.cpp/convert_hf_to_gguf.py {output_dir} --outfile {output_dir}/qwen-haru-f16.gguf --outtype f16\n")

    print("3. Quantize to recommended target formats:")
    print("   # Recommended balanced quant (approx 2.5 GB, runs on CPU < 40ms):")
    print(f"   ./llama.cpp/llama-quantize {output_dir}/qwen-haru-f16.gguf {output_dir}/qwen-haru-q4_k_m.gguf Q4_K_M\n")
    print("   # High fidelity 8-bit quant (approx 4.3 GB):")
    print(f"   ./llama.cpp/llama-quantize {output_dir}/qwen-haru-f16.gguf {output_dir}/qwen-haru-q8_0.gguf Q8_0\n")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Extract and prepare Haru-4B Qwen backbone for GGUF quantization.")
    parser.add_argument("--config", type=str, default=str(root_dir / "models" / "haru-4b" / "configs" / "config.json"), help="Path to config.json")
    parser.add_argument("--weights", type=str, default=None, help="Path to local model.safetensors")
    parser.add_argument("--outdir", type=str, default=str(root_dir / "models" / "haru-4b" / "extracted_backbone"), help="Output directory")

    args = parser.parse_args()
    config_path = Path(args.config)
    out_path = Path(args.outdir)

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            print(f"Error: Weights file not found at {weights_path}")
            sys.exit(1)
        extract_and_save_backbone(weights_path, out_path, config_path)
    else:
        print("Note: No local weights provided (--weights). Running in scaffold/config-generation mode.")
        out_path.mkdir(parents=True, exist_ok=True)
        with open(config_path, "r", encoding="utf-8") as f:
            haru_config = json.load(f)
        qwen_config = generate_qwen_config(haru_config)
        with open(out_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(qwen_config, f, indent=2)
        print(f"✅ Generated standalone Qwen backbone config at: {out_path / 'config.json'}")

    print_gguf_instructions(out_path)


if __name__ == "__main__":
    main()
