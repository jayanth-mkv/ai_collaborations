# AI Collaborations & Model Quantization Lab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jayanth-mkv/ai_collaborations/blob/main/models/haru-4b/notebooks/Haru_4B_Colab_Kaggle.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Jayanth--MKV-yellow)](https://huggingface.co/Jayanth-MKV)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> A standardized experimentation lab for evaluating foundation models, benchmarking hardware feasibility, developing custom C++/ggml runtimes, and releasing production-ready quantizations.

---

## 🔬 Model Catalog

| Model | Domain | Architecture | Quantization & Size | Target Hardware | 1-Click Demo | Model Hub |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Haru-4B** | Full-Duplex Spoken Dialogue | Qwen3.5-4B + VibeVoice + MeanFlow | **GGUF Q4_K_M** (2.6 GB) + FP16 Audio Codec & Head | 4-Core CPU / Low-VRAM GPU | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jayanth-mkv/ai_collaborations/blob/main/models/haru-4b/notebooks/Haru_4B_Colab_Kaggle.ipynb) | [Jayanth-MKV/Haru-4B-GGUF](https://huggingface.co/Jayanth-MKV/Haru-4B-GGUF) |
| *More Models* | *Vision / Speech / Reasoning* | *Expanding catalog* | *GGUF / AWQ / bitsandbytes* | *CPU / GPU / Edge* | *In Queue* | [`models/`](file:///D:/PROJECTS/jay/ai_collaborations/models/) |

---

## ⚡ Quickstart: Try Haru-4B (1-Click Deployment)

You can launch and test the full-duplex live voice portal for free on a Google Colab T4 GPU:

👉 **[Launch Interactive Voice Portal in Google Colab](https://colab.research.google.com/github/jayanth-mkv/ai_collaborations/blob/main/models/haru-4b/notebooks/Haru_4B_Colab_Kaggle.ipynb)**

Or run our quantized GGUF backbone locally via `llama.cpp`:
```bash
# 1. Download our 2.6 GB quantized model from Hugging Face
huggingface-cli download Jayanth-MKV/Haru-4B-GGUF qwen-haru-q4_k_m.gguf --local-dir .

# 2. Run inference on CPU (or pass -ngl 99 for GPU acceleration)
./llama-cli -m qwen-haru-q4_k_m.gguf -p "Hello! Let's talk." -n 128 -t 4
```

---

## 🛠️ Adding New Models (Scalable Framework)

Every model in this lab follows an isolated, standardized workflow under `models/<model-name>/`:

1. **Hardware Feasibility & Memory Profiling:** Use [`shared/quant_utils.py`](file:///D:/PROJECTS/jay/ai_collaborations/shared/quant_utils.py) to calculate theoretical memory bandwidth demands against target hardware before downloading weights.
2. **Selective Quantization:** Split composite multimodal architectures—compress heavy transformer backbones to 4-bit (GGUF / AWQ) while preserving sensitive acoustic and visual decoders in FP16 to eliminate distortion.
3. **Turnkey Deployment & Upstream Release:** Package 1-click cloud notebooks, link published artifacts directly to the upstream model on Hugging Face (`base_model: ...`), and provide C++/ggml scaffolds for open-source PRs.

---

## 📖 Deep Dives & Documentation
* [Haru-4B Quantization Walkthrough](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/docs/QUANTIZATION_WALKTHROUGH.md) — Complete engineering steps, tools used, and reproduction cheat sheet.
* [Haru-4B Architecture Deep Dive](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/docs/ARCHITECTURE_DEEP_DIVE.md) — Mathematical tensor breakdown & 133 ms streaming loop.
* [Cascaded Turn-Based CPU Fallback](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/docs/CASCADED_CPU_FALLBACK.md) — Turn-based fallback pipeline for low-power CPUs.
* [Shared Quantization Utilities](file:///D:/PROJECTS/jay/ai_collaborations/shared/quant_utils.py) — Bandwidth and memory calculators for LLM and speech architectures.
