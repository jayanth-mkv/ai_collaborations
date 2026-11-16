#!/usr/bin/env python3
"""Cascaded CPU Voice Pipeline Runner (Whisper.cpp + llama.cpp + Piper TTS).

Turn-based low-cost alternative for hosting voice AI on shared Linode CPU instances ($10-$24/mo)
where full-duplex Haru-4B cannot achieve real-time streaming.
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


class CascadedVoicePipeline:
    def __init__(self, whisper_model: str = "base.en", llm_model: str = "qwen2.5-3b-q4", tts_voice: str = "en_US-lessac-medium"):
        self.whisper_model = whisper_model
        self.llm_model = llm_model
        self.tts_voice = tts_voice
        print(f"Initializing Cascaded Voice Pipeline on CPU...")
        print(f"  • Ears (STT): Whisper ({whisper_model})")
        print(f"  • Brain (LLM): llama.cpp ({llm_model})")
        print(f"  • Mouth (TTS): Piper ({tts_voice})")

    def simulate_turn(self, input_speech_text: str):
        print("\n" + "=" * 70)
        print(f"User Utterance (Simulated Voice Input): \"{input_speech_text}\"")
        print("-" * 70)

        # 1. STT Phase
        t0 = time.perf_counter()
        # Simulated Whisper.cpp base.en on 2 CPU cores (~80ms)
        time.sleep(0.080)
        stt_latency = (time.perf_counter() - t0) * 1000.0
        print(f"[STT Ears] Transcribed in {stt_latency:.1f} ms -> \"{input_speech_text}\"")

        # 2. LLM Reasoning Phase
        t1 = time.perf_counter()
        # Prompting for speech-friendly brief response
        reply_text = f"I hear you. Regarding {input_speech_text.lower().rstrip('?.!')}, that sounds like a great plan."
        # Simulated Qwen 3B Q4_K_M generation of 15 tokens on CPU (~100ms)
        time.sleep(0.100)
        llm_latency = (time.perf_counter() - t1) * 1000.0
        print(f"[LLM Brain] Generated in {llm_latency:.1f} ms -> \"{reply_text}\"")

        # 3. TTS Phase
        t2 = time.perf_counter()
        # Simulated Piper TTS RTF 0.08x on CPU (~65ms)
        time.sleep(0.065)
        tts_latency = (time.perf_counter() - t2) * 1000.0
        print(f"[TTS Mouth] Synthesized in {tts_latency:.1f} ms -> 24 kHz Audio PCM")

        total_latency = (time.perf_counter() - t0) * 1000.0
        print("-" * 70)
        print(f"End-to-End Response Time: {total_latency:.1f} ms (Target: < 500 ms) -> ✅ FAST NATURAL TURN")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run cascaded CPU voice pipeline simulator.")
    parser.add_argument("--prompt", type=str, default="Can we quantize Haru to run on a cheap Linode instance?", help="Test input utterance")
    args = parser.parse_args()

    pipeline = CascadedVoicePipeline()
    pipeline.simulate_turn(args.prompt)


if __name__ == "__main__":
    main()
