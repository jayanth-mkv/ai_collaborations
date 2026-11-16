#!/usr/bin/env python3
"""Haru-4B Latency & Memory Bandwidth Calculator.

Calculates model footprint, required memory bandwidth, and frame latency
across quantization types (FP16, Q8_0, Q4_K_M, etc.) and evaluates feasibility
against the 133.33 ms real-time frame budget.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from shared.quant_utils import (
    QUANT_BITS_MAP,
    HARDWARE_BANDWIDTH_PROFILES,
    estimate_model_memory,
    calculate_required_bandwidth,
    estimate_frame_latency,
)


def print_banner():
    print("=" * 80)
    print("  HARU-4B REAL-TIME LATENCY & BANDWIDTH CALCULATOR (7.5 Hz / 133.33 ms)")
    print("=" * 80)


def evaluate_all(param_count: int = 4_080_000_000, frame_rate: float = 7.5, budget_ms: float = 133.33):
    print(f"Target Model: Haru-4B (~{param_count / 1e9:.2f} Billion Parameters)")
    print(f"Streaming Target: {frame_rate} Hz ({budget_ms:.2f} ms per frame window)\n")

    print(f"{'Quant Format':<10} | {'Bits/Param':<10} | {'Model Size':<12} | {'Min Bandwidth':<15} | {'Feasible on Linode CPU?'}")
    print("-" * 80)

    for quant, bits in QUANT_BITS_MAP.items():
        size_gb = estimate_model_memory(param_count, quant)
        req_bw = calculate_required_bandwidth(size_gb, frame_rate)
        
        # Check against dedicated epyc (~75 GB/s) and shared (~15 GB/s)
        shared_latency = estimate_frame_latency(size_gb, HARDWARE_BANDWIDTH_PROFILES["linode_shared_8vcpu"])
        dedicated_latency = estimate_frame_latency(size_gb, HARDWARE_BANDWIDTH_PROFILES["linode_dedicated_16vcpu_epyc"])
        
        if dedicated_latency <= budget_ms:
            status = f"✅ YES (Dedicated 16c: {dedicated_latency:.1f}ms)"
        else:
            status = f"❌ NO (Dedicated: {dedicated_latency:.1f}ms)"

        print(f"{quant:<10} | {bits:<10.2f} | {size_gb:<10.2f} GB | {req_bw:<13.2f} GB/s | {status}")

    print("-" * 80)
    print("\nHardware Bandwidth Latency Matrix for Q4_K_M (~2.52 GB):")
    q4_size = estimate_model_memory(param_count, "Q4_K_M")
    print(f"{'Hardware Tier':<32} | {'Bandwidth':<12} | {'Est. Frame Time':<18} | {'Feasible?'}")
    print("-" * 80)

    for hw, bw in HARDWARE_BANDWIDTH_PROFILES.items():
        est_ms = estimate_frame_latency(q4_size, bw)
        is_ok = est_ms <= budget_ms
        verdict = "✅ REAL-TIME" if is_ok else "❌ BUFFER UNDERRUN"
        print(f"{hw:<32} | {bw:<8.1f} GB/s | {est_ms:<14.1f} ms | {verdict}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Calculate Haru-4B quantization requirements.")
    parser.add_argument("--params", type=float, default=4.08, help="Model parameters in billions (default: 4.08)")
    parser.add_argument("--fps", type=float, default=7.5, help="Target frames per second (default: 7.5)")
    parser.add_argument("--budget", type=float, default=133.33, help="Max latency budget in ms (default: 133.33)")
    parser.add_argument("--custom-bw", type=float, default=None, help="Custom system bandwidth in GB/s to test")

    args = parser.parse_args()
    print_banner()
    evaluate_all(int(args.params * 1e9), args.fps, args.budget)

    if args.custom_bw:
        size = estimate_model_memory(int(args.params * 1e9), "Q4_K_M")
        lat = estimate_frame_latency(size, args.custom_bw)
        print(f"\n[Custom Check] With {args.custom_bw} GB/s bandwidth:")
        print(f"  Q4_K_M forward pass: {lat:.1f} ms (Budget: {args.budget} ms) -> {'PASS' if lat <= args.budget else 'FAIL'}")


if __name__ == "__main__":
    main()
