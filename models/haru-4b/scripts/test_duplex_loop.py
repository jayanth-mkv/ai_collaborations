#!/usr/bin/env python3
"""Simulation & benchmark harness for Haru-4B's 7.5 Hz (133.33 ms) full-duplex loop.

Simulates the complete audio streaming lifecycle:
1. Producer: Ingests 24 kHz PCM audio into 3200-sample chunks.
2. VibeVoice Encoder: Maps 3200 samples -> 64-dimensional continuous latent.
3. Multimodal Embedding Fusion: Sums user audio + model audio + text token into LM space (2560).
4. LM Backbone: Forward pass over single LM timestep.
5. MeanFlow Head: 1-step ODE velocity sampling producing next speech latent.
6. VibeVoice Decoder: Synthesizes 64-dim latent back into 3200 PCM samples.
7. Playout Buffer: Measures underruns, latency jitter, and real-time factor (RTF).
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from shared.audio_utils import AudioFrameBuffer


class MockVibeVoiceCodec:
    """Mock implementation of VibeVoice causal acoustic tokenizer."""
    def __init__(self, latent_dim: int = 64, frame_samples: int = 3200):
        self.latent_dim = latent_dim
        self.frame_samples = frame_samples

    def encode(self, pcm_chunk: np.ndarray) -> np.ndarray:
        # Simulate ~3 ms causal conv compute
        time.sleep(0.003)
        # Compute real feature proxy (energy/spectral envelope) to simulate continuous latent
        latent = np.zeros(self.latent_dim, dtype=np.float32)
        stride = len(pcm_chunk) // self.latent_dim
        for i in range(self.latent_dim):
            latent[i] = np.mean(np.abs(pcm_chunk[i * stride : (i + 1) * stride]))
        return latent

    def decode(self, latent: np.ndarray) -> np.ndarray:
        # Simulate ~4 ms causal transposed conv compute
        time.sleep(0.004)
        # Reconstruct carrier waveform shaped by latent amplitudes
        t = np.linspace(0, 1.0, self.frame_samples, dtype=np.float32)
        carrier = np.sin(2 * np.pi * 440 * t)
        expanded = np.repeat(latent, self.frame_samples // self.latent_dim)
        return (carrier * expanded).astype(np.float32)


class MockMeanFlowHead:
    """Mock implementation of 1-step MeanFlow ODE audio head."""
    def __init__(self, hidden_size: int = 2560, latent_dim: int = 64):
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

    def sample(self, hidden_state: np.ndarray, noise: np.ndarray, guidance: float = 1.5) -> np.ndarray:
        # Simulate ~1.5 ms MLP forward pass
        time.sleep(0.0015)
        # 1-step Euler step: z_next = z_0 + h * velocity
        velocity = np.tanh(hidden_state[:self.latent_dim]) * guidance
        return (noise + 1.0 * velocity).astype(np.float32)


def run_duplex_benchmark(total_frames: int = 30, simulated_backbone_latency_ms: float = 40.0):
    print("=" * 80)
    print(f"  HARU-4B FULL-DUPLEX 7.5 Hz STREAMING HARNESS ({total_frames} Frames / {total_frames * 0.1333:.1f}s)")
    print(f"  Simulated Backbone Forward Pass Latency: {simulated_backbone_latency_ms:.1f} ms")
    print("=" * 80)

    mic_buffer = AudioFrameBuffer(frame_size=3200)
    speaker_buffer = AudioFrameBuffer(frame_size=3200)
    codec = MockVibeVoiceCodec()
    head = MockMeanFlowHead()

    # Pre-populate microphone with synthetic test speech (white noise + tones)
    rng = np.random.default_rng(42)
    synthetic_speech = rng.standard_normal(3200 * total_frames).astype(np.float32) * 0.1
    mic_buffer.write(synthetic_speech)

    model_audio_latent = np.zeros(64, dtype=np.float32)
    frame_latencies = []
    underruns = 0

    print(f"{'Frame':<8} | {'Encode (ms)':<12} | {'Backbone (ms)':<14} | {'Head (ms)':<10} | {'Decode (ms)':<12} | {'Total (ms)':<12} | {'Status'}")
    print("-" * 80)

    for i in range(1, total_frames + 1):
        t0 = time.perf_counter()

        # 1. Pop incoming 133ms chunk
        user_pcm = mic_buffer.read_frame()
        if user_pcm is None:
            break

        # 2. VibeVoice Encode
        t_enc0 = time.perf_counter()
        user_latent = codec.encode(user_pcm)
        t_enc = (time.perf_counter() - t_enc0) * 1000.0

        # 3. Simulate Backbone Forward Pass (inputs_embeds)
        t_bb0 = time.perf_counter()
        # Simulated Q4_K_M forward pass time on target CPU
        time.sleep(simulated_backbone_latency_ms / 1000.0)
        hidden_state = rng.standard_normal(2560).astype(np.float32)
        t_bb = (time.perf_counter() - t_bb0) * 1000.0

        # 4. MeanFlow Head 1-Step Sampling
        t_hd0 = time.perf_counter()
        noise = rng.standard_normal(64).astype(np.float32)
        model_audio_latent = head.sample(hidden_state, noise)
        t_hd = (time.perf_counter() - t_hd0) * 1000.0

        # 5. VibeVoice Decode
        t_dec0 = time.perf_counter()
        out_pcm = codec.decode(model_audio_latent)
        t_dec = (time.perf_counter() - t_dec0) * 1000.0

        # 6. Push to speaker ring buffer
        speaker_buffer.write(out_pcm)

        total_frame_ms = (time.perf_counter() - t0) * 1000.0
        frame_latencies.append(total_frame_ms)

        is_ok = total_frame_ms <= 133.33
        if not is_ok:
            underruns += 1
        status = "✅ OK" if is_ok else "❌ DROP"

        if i <= 10 or i % 5 == 0 or not is_ok:
            print(f"{i:<8} | {t_enc:<12.2f} | {t_bb:<14.2f} | {t_hd:<10.2f} | {t_dec:<12.2f} | {total_frame_ms:<12.2f} | {status}")

    print("-" * 80)
    avg_latency = np.mean(frame_latencies)
    p95_latency = np.percentile(frame_latencies, 95)
    rtf = avg_latency / 133.33

    print(f"\nBenchmark Summary:")
    print(f"  Frames Tested         : {len(frame_latencies)}")
    print(f"  Average Frame Latency : {avg_latency:.2f} ms")
    print(f"  P95 Frame Latency     : {p95_latency:.2f} ms")
    print(f"  Max Latency Window    : 133.33 ms")
    print(f"  Real-Time Factor (RTF): {rtf:.3f} (Values < 1.0 are real-time)")
    print(f"  Buffer Underruns      : {underruns} ({underruns / len(frame_latencies) * 100:.1f}%)")
    print(f"  Playout Samples Ready : {speaker_buffer.samples_available:,d}")

    if rtf < 1.0:
        print("  VERDICT: ✅ Hardware configuration meets real-time full-duplex requirements.")
    else:
        print("  VERDICT: ❌ Hardware configuration is too slow. Real-time audio stream will stutter.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Haru-4B Full-Duplex Loop Simulation Harness.")
    parser.add_argument("--frames", type=int, default=30, help="Number of 133ms frames to simulate")
    parser.add_argument("--backbone-ms", type=float, default=40.0, help="Simulated LM backbone latency in ms (e.g. 40ms for Q4_K_M on 16c EPYC)")

    args = parser.parse_args()
    run_duplex_benchmark(args.frames, args.backbone_ms)


if __name__ == "__main__":
    main()
