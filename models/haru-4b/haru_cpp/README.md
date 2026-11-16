# Haru C++ / ggml Inference Engine (`haru_cpp`)

A lightweight, cross-platform C++ runtime designed to run **Haru-4B** in real time on modern CPUs and edge devices.

---

## 🏗 Architecture

* **Backbone:** Quantized `Qwen3.5-4B` in GGUF format (`Q4_K_M` or `Q8_0`), utilizing `inputs_embeds` injection.
* **VibeVoice Causal Codec:** Written using ggml 1D causal convolution and transposed convolution operators.
* **MeanFlow Head:** 1-step ODE sampling with guidance scale $w=1.5$.
* **Streaming Audio Pipeline:** Lock-free circular buffer operating on 24 kHz audio at 7.5 Hz (3200 samples / 133.33 ms frame).

---

## 🛠 Building the C++ Engine

### Prerequisites
* CMake 3.16 or later
* C++17 compatible compiler:
  * **Windows:** Visual Studio 2019/2022 (MSVC) or Clang
  * **Linux:** GCC 9+ or Clang 10+
  * **macOS:** Apple Clang / Xcode

### Build Instructions
```bash
# From the models/haru-4b/haru_cpp directory
mkdir build
cd build

# Configure build with CMake
cmake ..

# Compile the target
cmake --build . --config Release
```

### Running the Benchmark Runner
```bash
# On Windows
.\Release\haru_runner.exe
# Or if single-config:
.\haru_runner.exe

# On Linux/macOS
./haru_runner
```
