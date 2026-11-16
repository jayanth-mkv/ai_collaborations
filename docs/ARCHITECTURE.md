# Repository Architecture: AI Collaborations & Model Quantization Lab

This repository is dedicated to **evaluating, experimenting with, and quantizing modern AI foundation models** for production hosting, cost-effective cloud deployments, and edge/CPU execution.

---

## 🏛 Repository Layout & Design Pattern

To keep experiments clean, reproducible, and extensible across diverse model families (speech, multimodal, LLMs, vision, diffusion), the repository adopts an **isolated per-model folder structure**:

```text
ai_collaborations/
├── README.md                      # Global documentation, catalog of tested models, quantization guide
├── requirements.txt               # Unified project dependencies
├── .gitignore                     # Model weight & binary exclusion rules
├── docs/                          # Cross-cutting architectural documentation & raw input logs
│   ├── raw.txt                    # Original research input notes (Haru-4B, Linode CPU, PR strategy)
│   ├── ARCHITECTURE.md            # System layout & design principles
│   └── CONTRIBUTING.md            # Guidelines for adding new models & submitting upstream PRs
├── shared/                        # Shared mathematical, audio, and quantization utilities
│   ├── __init__.py
│   ├── quant_utils.py             # Memory bandwidth, parameter bits, and latency calculators
│   └── audio_utils.py             # Streaming audio ring buffers, resampling, framing tools
└── models/                        # Self-contained directories for each experimented model
    ├── haru-4b/                   # Model 1: Higurashi Haru-4B Full-Duplex Spoken Dialogue Model
    │   ├── README.md              # Model-specific executive summary & quickstart
    │   ├── configs/               # Upstream Hugging Face configuration files
    │   ├── docs/                  # Deep dives (hardware feasibility, latency budgets, PR roadmap)
    │   ├── scripts/               # Inspection, GGUF conversion, latency verification, loop testing
    │   ├── haru_cpp/              # C++/ggml engine scaffold for upstream PR
    │   └── cascaded_pipeline/     # Low-cost CPU alternative (Whisper + Llama + Piper)
    └── <future-model>/            # Model 2, Model 3...
```

---

## 📐 Standardization Checklist for Each Model Folder

Whenever experimenting with a new model in `models/<model-name>/`, adhere to this 5-point checklist:

1. **`configs/config.json`:** Mirror upstream configuration and hyperparameter specifications.
2. **`docs/`:**
   * Architectural deep-dive: layer types, context length, memory requirements.
   * Hosting and hardware feasibility: CPU bandwidth vs GPU VRAM constraints.
3. **`scripts/`:**
   * Remote/local inspector: analyze tensor counts, shapes, parameter distributions without forcing large weight downloads.
   * Latency & bandwidth calculator: calculate required RAM/VRAM throughput.
   * Quantization script: export to GGUF, AWQ, GPTQ, or custom ggml formats.
4. **C++/ggml / Edge Runtime Scaffold:**
   * Scaffolding for porting model layers to ggml / ONNX Runtime / TensorRT-LLM.
5. **Low-Resource Fallback Pipeline:**
   * Alternative architecture for budget deployments when the primary model exceeds target hardware constraints.
