# Haru-4B: Experimentation, Quantization & Porting Guide

> **Model Overview:** Haru-4B is an open-weight, multilingual, full-duplex continuous spoken dialogue foundation model created by **higurashi-ai**. It listens and speaks simultaneously at **7.5 Hz (133.33 ms audio frames)** using a unified embedding space.

---

## 📑 Directory Structure

```text
models/haru-4b/
├── README.md                      # This guide
├── configs/
│   └── config.json                # Upstream Haru-4B model configuration
├── docs/
│   ├── ARCHITECTURE_DEEP_DIVE.md  # Detailed tensor shapes, math, and streaming loop
│   ├── LINODE_CPU_FEASIBILITY.md  # Memory bandwidth & Linode CPU hardware analysis
│   ├── PR_STRATEGY.md             # Open source pull request roadmap (Haru, llama.cpp, HF)
│   └── CASCADED_CPU_FALLBACK.md   # Whisper.cpp + Llama.cpp + Piper TTS alternative
├── scripts/
│   ├── inspect_model.py           # Remote/local safetensors inspector
│   ├── latency_calculator.py      # Bandwidth & latency budget calculator
│   ├── quantize_backbone.py       # Qwen3.5-4B extraction & GGUF conversion script
│   ├── test_duplex_loop.py        # 133 ms full-duplex loop simulator & harness
│   └── verify_tensors.py          # PyTorch vs ggml tensor parity verification
├── haru_cpp/                      # Standalone C++/ggml port scaffold for upstream PR
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── haru_ggml.h
│   │   └── audio_ring_buffer.h
│   ├── src/
│   │   ├── haru_ggml.cpp
│   │   └── main.cpp
│   └── README.md
└── cascaded_pipeline/             # 3-part CPU voice fallback pipeline
    ├── run_pipeline.py
    └── README.md
```

---

## 🔬 Core Architecture Breakdown

| Component | Upstream Implementation | Architecture Details | Quantization Strategy |
| :--- | :--- | :--- | :--- |
| **Language Model Backbone** | `Qwen3.5-4B` | 32 layers, hidden size 2560, hybrid linear SSM (Mamba) + full attention every 4th layer | Quantize to **GGUF Q4_K_M** (~2.5 GB) or **Q8_0** (~4.3 GB) via `llama.cpp` |
| **Audio Codec (Tokenizer)** | `VibeVoice` | Causal 1D convolution stack, 24 kHz, downsampling ratio 3200 (8×5×5×4×2×2) -> 64-dim continuous latent | **Keep FP16** (~150 MB). Quantizing audio vocoders to 4-bit causes severe distortion & metallic artifacts |
| **Continuous Audio Head** | `MeanFlow Head` | 4-layer MLP, 1-step ODE/flow-matching velocity prediction with CFG guidance | **Keep FP16** (~45 MB) or Q8_0 |
| **Streaming Loop** | `DuplexSession` | 7.5 Hz (133.33 ms frame budget), 3200 audio samples in / 3200 out, simultaneous listening & speaking | Multi-threaded lock-free ring buffer in C++ or Python |

---

## ⚡ The Quantization & Porting Challenge

### Why `convert_hf_to_gguf.py` alone doesn't work:
1. **Multimodal Continuous Audio:** Standard `llama.cpp` models take discrete text tokens in and generate discrete text tokens out. Haru-4B sums three continuous embeddings at each timestep:
   $$\mathbf{x}_t = \text{Embed}(w_t) + \text{Proj}_{\text{model}}(\mathbf{z}_t^{\text{model}}) + \text{Proj}_{\text{user}}(\mathbf{z}_t^{\text{user}})$$
2. **Missing C++ Kernels in llama.cpp:** Neither the causal convolution layers of VibeVoice nor the 1-step MeanFlow ODE velocity solver exist in vanilla `llama.cpp`.
3. **Strict 133 ms Deadline:** Full-duplex audio generation requires finishing the encoder, backbone forward pass, MeanFlow sampling, and decoder in under 133 ms. If a CPU forward pass takes 200 ms, the audio stream immediately suffers underrun and drops out.

### The Solution:
1. **Split the model:** Quantize the heavy 4B text backbone to GGUF `Q4_K_M` (reducing memory from 10.5 GB to 2.5 GB).
2. **Keep audio components in FP16:** Keep VibeVoice and MeanFlow in FP16 for clean acoustic synthesis.
3. **Build a dedicated ggml runtime (`haru_cpp`):** Use ggml tensor operators to implement the audio causal convolutions and MeanFlow head, feeding embeddings via `llama_decode(inputs_embeds)`.

---

## 🚀 Quickstart

### 1. Run the Latency & Bandwidth Calculator
Calculate whether your hardware can achieve real-time full-duplex inference:
```bash
python scripts/latency_calculator.py --quant Q4_K_M
```

### 2. Inspect the Upstream Model Safetensors
Inspect layer shapes, parameter distribution, and tensor names without downloading 10.5 GB:
```bash
python scripts/inspect_model.py --repo higurashi-ai/Haru-4B
```

### 3. Simulate the 133 ms Duplex Loop
Test the audio frame ring buffer and duplex pipeline simulation:
```bash
python scripts/test_duplex_loop.py --frames 30
```

### 4. Build the C++/ggml Engine Scaffolding
```bash
cd haru_cpp
mkdir build && cd build
cmake ..
cmake --build .
```

---

## 📖 Deep Dive Documentation
- [Architecture Deep Dive](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/docs/ARCHITECTURE_DEEP_DIVE.md)
- [Linode CPU Feasibility & Bandwidth Analysis](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/docs/LINODE_CPU_FEASIBILITY.md)
- [Open Source Pull Request Roadmap](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/docs/PR_STRATEGY.md)
- [Cascaded Turn-Based CPU Fallback](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/docs/CASCADED_CPU_FALLBACK.md)
