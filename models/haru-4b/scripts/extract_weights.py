import os
import json
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

src_dir = Path("/root/.cache/huggingface/hub/models--higurashi-ai--Haru-4B/snapshots/8d6e83bcd3701a76fc27a764965665da1df7da78")
src_weights = src_dir / "model.safetensors"
out_dir = Path("/root/extracted_backbone")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Config
with open(src_dir / "config.json") as f:
    haru_cfg = json.load(f)

lm_cfg = haru_cfg["lm"]
qwen_cfg = {
    "architectures": ["Qwen3_5ForCausalLM"],
    "model_type": "qwen3_5_text",
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": True,
    **lm_cfg
}

with open(out_dir / "config.json", "w") as f:
    json.dump(qwen_cfg, f, indent=2)

print("Generated config.json for Qwen3.5 backbone.")

# 2. Tokenizer
import shutil
if (src_dir / "tokenizer.json").exists():
    shutil.copy(src_dir / "tokenizer.json", out_dir / "tokenizer.json")
if (src_dir / "tokenizer_config.json").exists():
    shutil.copy(src_dir / "tokenizer_config.json", out_dir / "tokenizer_config.json")

# 3. Extract tensors
print("Extracting LM backbone tensors from model.safetensors...")
extracted_tensors = {}
codec_tensors = {}
head_tensors = {}

with safe_open(src_weights, framework="pt", device="cpu") as f:
    for k in f.keys():
        if k.startswith("lm."):
            new_k = "model." + k[3:]
            extracted_tensors[new_k] = f.get_tensor(k)
        elif k.startswith("codec."):
            codec_tensors[k] = f.get_tensor(k)
        elif k.startswith("head."):
            head_tensors[k] = f.get_tensor(k)

# Tie word embeddings is handled via config.json ("tie_word_embeddings": true)
print(f"Extracted {len(extracted_tensors)} LM tensors.")
save_file(extracted_tensors, out_dir / "model.safetensors")
print(f"Saved Qwen backbone weights to {out_dir / 'model.safetensors'}")

# Save codec & head
out_codec = Path("/root/extracted_codec")
out_codec.mkdir(parents=True, exist_ok=True)
save_file(codec_tensors, out_codec / "model.safetensors")
print(f"Saved VibeVoice codec weights ({len(codec_tensors)} tensors) to {out_codec}")

out_head = Path("/root/extracted_head")
out_head.mkdir(parents=True, exist_ok=True)
save_file(head_tensors, out_head / "model.safetensors")
print(f"Saved MeanFlow head weights ({len(head_tensors)} tensors) to {out_head}")
