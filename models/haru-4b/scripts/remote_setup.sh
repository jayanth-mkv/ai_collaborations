#!/usr/bin/env bash
set -e

echo "===================================================================="
echo "  SETTING UP HARU-4B RUNNER ON LINODE DEDICATED AMD EPYC CPU"
echo "===================================================================="

# 1. Update and install system dependencies
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y build-essential cmake git python3 python3-pip python3-venv ffmpeg htop libomp-dev

# 2. Create workspace directory
mkdir -p /root/ai_collaborations
cd /root/ai_collaborations

# 3. Create Python virtual environment
if [ ! -d "/root/ai_collaborations/venv" ]; then
    python3 -m venv /root/ai_collaborations/venv
fi

source /root/ai_collaborations/venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers safetensors huggingface_hub numpy scipy rich

echo "===================================================================="
echo "  SYSTEM & PYTHON ENVIRONMENT INITIALIZED"
echo "===================================================================="
