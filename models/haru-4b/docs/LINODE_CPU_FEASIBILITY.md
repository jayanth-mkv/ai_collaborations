# Hosting Haru-4B on Linode CPU: Hardware & Feasibility Analysis

This report analyzes the technical feasibility, memory bandwidth physics, and cost economics of hosting **Haru-4B** on a CPU-only Linode (Akamai) server without a dedicated GPU.

---

## 1. Executive Summary

| Environment | Quantization | Effective RAM Bandwidth | Est. Frame Latency | Feasible (< 133 ms)? | Monthly Cost | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linode Shared 2–4 vCPU** | FP16 (10.5 GB) | ~8–12 GB/s | ~1,100 ms | ❌ **No** | $12–$24/mo | Unusable |
| **Linode Shared 8 vCPU** | Q4_K_M (2.5 GB) | ~15–20 GB/s | ~180–220 ms | ❌ **No** | $48/mo | Audio stutter / dropped frames |
| **Linode Dedicated 8 vCPU (EPYC)** | Q4_K_M (2.5 GB) | ~40–50 GB/s | ~70–90 ms | ⚠️ **Marginal** | $144/mo | Feasible under low load |
| **Linode Dedicated 16 vCPU (EPYC)** | Q4_K_M (2.5 GB) | ~70–85 GB/s | ~40–60 ms | ✅ **Yes** | $288/mo | Smooth real-time streaming |
| **On-Demand Cloud GPU (RTX 4000 Ada / L4)** | FP16 Native | 300–360 GB/s | ~20–35 ms | ✅ **Yes** | ~$0.30–$0.50/hr | **Best price/performance** |

---

## 2. The Physics: Why Memory Bandwidth Dictates Feasibility

Full-duplex spoken dialogue models are autoregressive: in every 133.33 ms frame, the model must read all active weights from system memory into processor cache/registers to complete a single forward pass.

$$\text{Minimum Required Bandwidth (GB/s)} = \text{Model Size (GB)} \times f_{\text{frame}}$$

where $f_{\text{frame}} = 7.5 \text{ frames/second}$.

### Bandwidth Required Across Quantization Levels:
* **FP16 Native (10.5 GB):**
  $$10.5 \text{ GB} \times 7.5 \text{ s}^{-1} = \mathbf{78.75 \text{ GB/s}}$$
  *No commodity single-socket CPU can stream 78.75 GB/s sustained from standard DDR4 RAM.*
* **Q8_0 Quantization (4.3 GB):**
  $$4.3 \text{ GB} \times 7.5 \text{ s}^{-1} = \mathbf{32.25 \text{ GB/s}}$$
  *Requires quad-channel DDR4 or dual-channel DDR5 server memory.*
* **Q4_K_M Quantization (2.5 GB):**
  $$2.5 \text{ GB} \times 7.5 \text{ s}^{-1} = \mathbf{18.75 \text{ GB/s}}$$
  *Mathematically viable on high-end server CPUs with high memory throughput.*

---

## 3. Linode CPU Architecture Realities

### A. Shared CPU Instances (Nanode / Standard Linode)
1. **Noisy Neighbor Contention:** Memory buses and L3 cache slices are shared across multiple tenants on the same physical host.
2. **Limited SIMD / Vector Throttling:** Cloud hypervisors often throttle heavy AVX-512 matrix operations on shared cores to prevent thermal hotspots.
3. **Effective Throughput:** Sustained memory bandwidth rarely exceeds **10–18 GB/s**.
4. **Outcome:** Forward pass takes 180–300 ms. Since each audio frame is 133 ms, the server falls behind after 2 seconds, the audio buffer starves, and output audio breaks into distorted static.

### B. Dedicated CPU Instances (AMD EPYC)
1. **Isolated Cores & Dedicated Channels:** 8 to 16 dedicated EPYC cores provide direct access to multi-channel DDR4/DDR5 memory controllers.
2. **AVX-512 & VNNI:** EPYC processors support AVX-512 and VNNI instructions, accelerating ggml 4-bit dot products.
3. **Effective Throughput:** Sustained memory bandwidth of **45–80+ GB/s**.
4. **Outcome:** Frame generation time drops to **45–70 ms**, comfortably beating the 133 ms deadline.

---

## 4. Cost Comparison: Dedicated CPU vs Cloud GPU

| Deployment Option | Compute Spec | Model Setup | Latency | Monthly Cost | Efficiency Rating |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linode Dedicated 16 vCPU** | 16 EPYC Cores, 32 GB RAM | Custom C++/ggml Q4_K_M port required | ~50 ms | **$288 / month** | Low (Heavy engineering + high hosting cost) |
| **Linode GPU (RTX 4000 Ada)** | 4 vCPU, 20GB VRAM | Official PyTorch / vLLM-Omni | ~25 ms | **$190 / month** | High |
| **RunPod / Vast.ai (On-Demand)** | 1x RTX 4090 or L4 | Official PyTorch / vLLM-Omni | ~20 ms | **~$0.35 / hour** (~$25/mo for 70 hrs testing) | **Highest for experimentation** |

---

## 5. Decision Matrix

* **If you have a dedicated budget ($250+/mo) and need 24/7 uptime on Linode without GPU:**
  Choose **Dedicated 16 vCPU Linode**, build the **`haru_cpp`** ggml engine, and quantize the backbone to **Q4_K_M**.
* **If you are on a budget ($10–$30/mo) and locked to a CPU Linode:**
  Do **not** run Haru-4B directly. Instead, deploy the **Cascaded CPU Voice Pipeline** ([CASCADED_CPU_FALLBACK.md](file:///D:/PROJECTS/jay/ai_collaborations/models/haru-4b/docs/CASCADED_CPU_FALLBACK.md)) using `whisper.cpp` + `llama.cpp` + `Piper TTS`.
* **If you want true full-duplex Haru-4B with zero engineering friction:**
  Rent an on-demand cloud GPU via RunPod or Linode GPU to run the official repository directly.
