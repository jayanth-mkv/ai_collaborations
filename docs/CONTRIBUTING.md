# Contributing & Open-Source Collaboration Guide

This repository serves as both an internal experimentation sandbox and an incubator for upstream contributions to the open-source AI ecosystem (e.g., `higurashi-ai/Haru`, `ggml-org/llama.cpp`, Hugging Face Hub).

---

## 1. Adding a New Model

1. Create a new directory under `models/<model-name>/`.
2. Populate the standard directories:
   * `configs/`
   * `docs/`
   * `scripts/`
3. Implement `latency_calculator.py` and `inspect_model.py` adapted to the model's architecture.
4. Update the root [README.md](file:///D:/PROJECTS/jay/ai_collaborations/README.md) model catalog.

---

## 2. Upstream Pull Request Etiquette

When porting architectures to upstream repositories:
1. **RFC First:** Open an issue or RFC discussion with maintainers before opening a massive code PR.
2. **Minimize Dependencies:** Upstream projects value zero or header-only dependencies (e.g. `dr_wav.h` for audio, `ggml` for tensors).
3. **Parity Verification:** Always provide a verification test script demonstrating numerical equivalence with the PyTorch baseline.
