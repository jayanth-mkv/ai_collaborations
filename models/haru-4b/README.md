# Haru-4B: Experimentation, Quantization & Porting Guide

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jayanth-mkv/ai_collaborations/blob/main/models/haru-4b/notebooks/Haru_4B_Colab_Kaggle.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model%20Hub-Jayanth--MKV%2FHaru--4B--GGUF-yellow)](https://huggingface.co/Jayanth-MKV/Haru-4B-GGUF)

> **Model Overview:** Haru-4B is an open-weight, multilingual, full-duplex continuous spoken dialogue foundation model created by **higurashi-ai**. It listens and speaks simultaneously at **7.5 Hz (133.33 ms audio frames)** using a unified embedding space.

---

## 🌟 Official Hugging Face Quantized Release
The quantized model weights and FP16 modular artifacts are published and linked directly to the upstream model:
- **Hugging Face Model Hub:** [Jayanth-MKV/Haru-4B-GGUF](https://huggingface.co/Jayanth-MKV/Haru-4B-GGUF)
- **Base Model Linked:** [`higurashi-ai/Haru-4B`](https://huggingface.co/higurashi-ai/Haru-4B) (specifying `base_model: higurashi-ai/Haru-4B` and `pipeline_tag: audio-to-audio`)
- **Backbone Format:** GGUF `Q4_K_M` (**2.6 GB**, compressed from 8.4 GB BF16)
- **Audio Codec:** VibeVoice causal 1D conv tokenizer (`vibevoice-codec-f16.safetensors`, 1.3 GB)
- **Acoustic Head:** MeanFlow 1-step ODE velocity predictor (`meanflow-head-f16.safetensors`, 694 MB)

```bash
# Download the GGUF model directly via huggingface-cli
huggingface-cli download Jayanth-MKV/Haru-4B-GGUF qwen-haru-q4_k_m.gguf --local-dir .
```

---

## 🔬 Core Architecture Breakdown

| Component | Upstream Implementation | Architecture Details | Quantization Strategy |
| :--- | :--- | :--- | :--- |
| **Language Model Backbone** | `Qwen3.5-4B` | 32 layers, hidden size 2560, hybrid linear SSM (Mamba) + full attention every 4th layer | Quantize to **GGUF Q4_K_M** (~2.6 GB) via `llama.cpp` |
| **Audio Codec (Tokenizer)** | `VibeVoice` | Causal 1D convolution stack, 24 kHz, downsampling ratio 3200 (8×5×5×4×2×2) -> 64-dim continuous latent | **Keep FP16** (~1.3 GB). Quantizing audio vocoders to 4-bit causes severe distortion & metallic artifacts |
| **Continuous Audio Head** | `MeanFlow Head` | 4-layer MLP, 1-step ODE/flow-matching velocity prediction with CFG guidance | **Keep FP16** (~694 MB) for smooth acoustic trajectory modeling |
| **Streaming Loop** | `DuplexSession` | 7.5 Hz (133.33 ms frame budget), 3200 audio samples in / 3200 out, simultaneous listening & speaking | Multi-threaded lock-free ring buffer in C++ or Python |

---

## ⚡ The Quantization & Porting Challenge

### Why `convert_hf_to_gguf.py` alone doesn't work:
1. **Multimodal Continuous Audio:** Standard `llama.cpp` models take discrete text tokens in and generate discrete text tokens out. Haru-4B sums three continuous embeddings at each timestep:
   $$\mathbf{x}_t = \text{Embed}(w_t) + \text{Proj}_{\text{model}}(\mathbf{z}_t^{\text{model}}) + \text{Proj}_{\text{user}}(\mathbf{z}_t^{\text{user}})$$
2. **Missing C++ Kernels in llama.cpp:** Neither the causal convolution layers of VibeVoice nor the 1-step MeanFlow ODE velocity solver exist in vanilla `llama.cpp`.
3. **Strict 133 ms Deadline:** Full-duplex audio generation requires finishing the encoder, backbone forward pass, MeanFlow sampling, and decoder in under 133 ms. If a CPU forward pass takes 200 ms, the audio stream immediately suffers underrun and drops out.

### The Solution:
1. **Split the model:** Quantize the heavy 4B text backbone to GGUF `Q4_K_M` (reducing memory from 10.5 GB to 2.6 GB).
2. **Keep audio components in FP16:** Keep VibeVoice and MeanFlow in FP16 for clean acoustic synthesis.
3. **Build a dedicated ggml runtime (`haru_cpp`):** Use ggml tensor operators to implement the audio causal convolutions and MeanFlow head, feeding embeddings via `llama_decode(inputs_embeds)`.

---

## 🛠️ How We Quantized Haru-4B: The Step-by-Step Engineering Process

A simplified breakdown of the exact process and tools used to create our published release:

### 1. Zero-Download Inspection ([`scripts/inspect_model.py`](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/scripts/inspect_model.py))
* **Tool:** `urllib` HTTP Range requests + `safetensors.safe_open`.
* **Action:** Inspected layer shapes, parameter counts, and tensor names directly from Hugging Face without downloading 10.5 GB. Confirmed that the language backbone represents **79.8% of the weights (4.21B parameters)**.

### 2. Weight Slicing & Extraction ([`scripts/extract_weights.py`](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/scripts/extract_weights.py))
* **Tool:** Python `safetensors` memory-mapped I/O.
* **Action:**
  - Sliced all `lm.*` tensors into `model.*` to construct a standalone `Qwen3_5ForCausalLM` checkpoint.
  - Handled memory sharing: Omitted duplicate `lm_head.weight` and set `"tie_word_embeddings": true` to avoid `RuntimeError: Some tensors share memory`.
  - Extracted `codec.*` into `vibevoice-codec-f16.safetensors` (1.3 GB) and `head.*` into `meanflow-head-f16.safetensors` (694 MB).

### 3. GGUF Conversion (`convert_hf_to_gguf.py`)
* **Tool:** `llama.cpp` conversion script.
* **Action:** Ran `python convert_hf_to_gguf.py /root/extracted_backbone --outfile qwen-haru-bf16.gguf --no-mtp`.
* **Result:** Generated a clean 7.94 GB BF16 base GGUF file containing all 291 language model tensors.

### 4. 4-Bit Medium Quantization (`llama-quantize`)
* **Tool:** `llama.cpp/build/bin/llama-quantize`.
* **Action:** Ran `llama-quantize qwen-haru-bf16.gguf qwen-haru-q4_k_m.gguf Q4_K_M`.
* **Result:** Compressed the model from **8.02 GB down to 2.60 GB** (3.08× smaller, 67.5% memory reduction).

### 5. Multi-Hardware Benchmark (`llama-cli`)
* **AMD EPYC CPU (4 Cores):** Verified CPU inference at **8.4 tokens/sec**.
* **NVIDIA Tesla T4 GPU:** Verified GPU inference at **68.2 tokens/sec** (`-ngl 99` offloads all 32 layers to VRAM).

### 6. Automated Publishing to Hugging Face ([`scripts/upload_to_hf.py`](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/scripts/upload_to_hf.py))
* **Tool:** `huggingface_hub.HfApi`.
* **Action:** Created the public model repository [Jayanth-MKV/Haru-4B-GGUF](https://huggingface.co/Jayanth-MKV/Haru-4B-GGUF) and linked it to `base_model: higurashi-ai/Haru-4B` in the YAML frontmatter.

👉 **For the full technical guide, reproducing commands, and tensor charts, see [QUANTIZATION_WALKTHROUGH.md](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/docs/QUANTIZATION_WALKTHROUGH.md).**

---

## 🚀 Quickstart & Notebooks

* **1-Click Cloud Deployment:** [Open in Google Colab](https://colab.research.google.com/github/jayanth-mkv/ai_collaborations/blob/main/models/haru-4b/notebooks/Haru_4B_Colab_Kaggle.ipynb)
* **Inspect Upstream Safetensors (Zero-Download):** `python scripts/inspect_model.py --repo higurashi-ai/Haru-4B`
* **Latency & Memory Bandwidth Calculator:** `python scripts/latency_calculator.py --quant Q4_K_M`
* **Full-Duplex 133ms Simulator:** `python scripts/test_duplex_loop.py --frames 30`
* **C++/ggml Engine Scaffolding:** Build via `cd haru_cpp && cmake -B build && cmake --build build`

---

## 📖 Deep Dive Documentation
- [Complete Quantization Walkthrough](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/docs/QUANTIZATION_WALKTHROUGH.md) — Detailed engineering guide, pitfalls, tools, and reproduction cheat sheet.
- [Architecture Deep Dive](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/docs/ARCHITECTURE_DEEP_DIVE.md) — Tensor math, causal convolution downsampling, and 133 ms streaming loop.
- [Cascaded Turn-Based CPU Fallback](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/docs/CASCADED_CPU_FALLBACK.md) — Whisper + Llama + Piper fallback for low-power CPUs.
