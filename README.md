# AI Collaborations & Model Quantization Lab

> A specialized experimentation lab and porting toolkit for evaluating cutting-edge AI models, testing quantization techniques (GGUF, ggml, AWQ, GPTQ), and building high-performance deployment runtimes across CPU, GPU, and Edge devices.

---

## 🗂 Repository Structure

Each model tested in this repository receives its own dedicated workspace under [`models/`](file:///D:/PROJECTS/jay/ai_collaborations/models/), isolating its configuration, quantization scripts, hardware feasibility benchmarks, C++ runtime scaffolds, and fallback pipelines:

```text
ai_collaborations/
├── README.md                      # Global documentation and model catalog
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git ignore rules (weights, caches, build artifacts)
├── docs/                          # Architectural documentation & raw research inputs
│   ├── raw.txt                    # Starting research notes (Haru-4B, Linode CPU, PR strategy)
│   ├── ARCHITECTURE.md            # System layout, directory design & standardization checklist
│   └── CONTRIBUTING.md            # Guidelines for adding models & upstream PR etiquette
├── shared/                        # Shared mathematical, audio & quantization utilities
│   ├── __init__.py
│   ├── quant_utils.py             # Memory bandwidth, parameter bits, and latency calculators
│   └── audio_utils.py             # Ring buffers, audio resampling, and 133ms chunking
└── models/                        # Isolated directories for each model under test
    └── haru-4b/                   # [Model 1] Haru-4B Full-Duplex Spoken Dialogue Model
        ├── README.md              # Model-specific guide & quickstart
        ├── configs/               # Model hyperparameter & architecture definitions
        ├── docs/                  # Architecture deep dive, Linode CPU analysis, PR strategy
        ├── scripts/               # Inspect, latency calculator, GGUF quantizer, duplex loop
        ├── haru_cpp/              # Standalone C++/ggml engine scaffold for upstream PR
        └── cascaded_pipeline/     # Turn-based low-cost CPU fallback (Whisper + Llama + Piper)
```

---

## 🔬 Model Catalog

| Model Name | Category | Base Architecture | Param Count | Quantization Strategy | Target Hardware | Documentation & Model Hub |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Haru-4B** (`higurashi-ai/Haru-4B`) | Spoken Dialogue (LSLM) | Qwen3.5-4B + VibeVoice + MeanFlow | 5.27B Total (4.2B LM + 687M Codec + 363M Head) | Hybrid: GGUF Q4_K_M for LM, FP16 for Codec & Head | 16c Dedicated CPU (EPYC), Apple Silicon, or GPU | [Haru-4B README](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/README.md) · [HF Hub: Jayanth-MKV/Haru-4B-GGUF](https://huggingface.co/Jayanth-MKV/Haru-4B-GGUF) |

---

## 🎯 Getting Started with Haru-4B

Originating from research notes in [`docs/raw.txt`](file:///D:/PROJECTS/jay/ai_collaborations/docs/raw.txt), **Haru-4B** is an open-source, full-duplex speech foundation model that listens and speaks simultaneously at **7.5 Hz (133.33 ms frame budget)**.

### 1. Inspect Model Architecture (Without Downloading 10.5 GB)
Inspect tensor counts, parameter distributions, and data types via HTTP Range requests directly from Hugging Face:
```bash
python models/haru-4b/scripts/inspect_model.py --repo higurashi-ai/Haru-4B
```

### 2. Calculate Memory Bandwidth & Latency Feasibility
Verify whether a given CPU or GPU instance can sustain the 133.33 ms deadline:
```bash
python models/haru-4b/scripts/latency_calculator.py --params 4.08 --fps 7.5
```

### 3. Test the Full-Duplex 7.5 Hz Audio Loop
Run the high-resolution simulation harness measuring encode, backbone forward pass, MeanFlow sampling, and audio playback:
```bash
python models/haru-4b/scripts/test_duplex_loop.py --frames 20
```

### 4. Verify Numerical Parity for ggml Tensor Ops
Run the mathematical verification suite ensuring C++/ggml operators match PyTorch outputs:
```bash
python models/haru-4b/scripts/verify_tensors.py
```

### 5. Low-Cost Linode CPU Workaround (Cascaded Pipeline)
For shared Linode CPU servers ($12–$24/month) where full-duplex inference cannot hit the 133ms budget:
```bash
python models/haru-4b/cascaded_pipeline/run_pipeline.py
```

---

## 📚 Key Research & Architectural Documents
* **[Haru-4B Architecture Deep Dive](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/docs/ARCHITECTURE_DEEP_DIVE.md):** Mathematical formulations of VibeVoice downsampling ($3200\times$), MeanFlow 1-step ODE velocity, and embedding fusion.
* **[Linode CPU Feasibility Analysis](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/docs/LINODE_CPU_FEASIBILITY.md):** DDR4/DDR5 memory bandwidth physics, why shared CPU Linodes fail, and why 16-core Dedicated EPYC CPUs succeed.
* **[Open Source PR Roadmap](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/docs/PR_STRATEGY.md):** Step-by-step roadmap for upstream PRs to `higurashi-ai/Haru`, `ggml-org/llama.cpp`, and Hugging Face Hub.
* **[Cascaded Turn-Based CPU Alternative](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/docs/CASCADED_CPU_FALLBACK.md):** Whisper.cpp + Llama.cpp + Piper TTS fallback for low-cost servers.
