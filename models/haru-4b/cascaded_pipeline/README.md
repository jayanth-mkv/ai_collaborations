# Cascaded CPU Voice Pipeline: Whisper.cpp + llama.cpp + Piper TTS

A turn-based voice pipeline designed to run on low-cost CPU-only cloud instances (such as Linode Shared 2–4 vCPU, $12–$24/month) where full-duplex LSLMs like Haru-4B exceed the 133ms real-time frame budget.

---

## 🚀 Quickstart

Run the simulation runner:
```bash
python run_pipeline.py --prompt "How does this voice pipeline compare to Haru full duplex?"
```

---

## 🛠 Deploying on a Linode Server

### 1. Install Dependencies
```bash
sudo apt update && sudo apt install -y build-essential cmake ffmpeg
```

### 2. Whisper.cpp (STT)
```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
cmake -B build
cmake --build build --config Release
bash ./models/download-ggml-model.sh base.en
```

### 3. llama.cpp (LLM)
```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release
# Download quantized Qwen 3B
wget -O qwen2.5-3b-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
```

### 4. Piper (TTS)
```bash
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_linux_x86_64.tar.gz
tar -xzf piper_linux_x86_64.tar.gz
# Download voice model
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```
