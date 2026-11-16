"""Quantization, memory footprint, and bandwidth calculation utilities.

Helps determine whether a model of N parameters quantized to format Q
can run within a given latency budget (e.g., 133ms frame time for LSLMs)
on specified hardware architectures (CPUs, GPUs, Apple Silicon).
"""

from typing import Dict, Any, Optional

# Bits per parameter for common quantization formats (including metadata/scales overhead)
QUANT_BITS_MAP: Dict[str, float] = {
    "FP32": 32.0,
    "FP16": 16.0,
    "BF16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.56,
    "Q5_K_M": 5.5,
    "Q5_0": 5.5,
    "Q4_K_M": 4.5,
    "Q4_0": 4.5,
    "Q3_K_M": 3.44,
    "Q3_K_S": 3.44,
    "Q2_K": 2.62,
}

# Typical effective memory bandwidths (GB/s) across common platforms
HARDWARE_BANDWIDTH_PROFILES: Dict[str, float] = {
    "linode_shared_2vcpu": 10.0,         # Shared channel, noisy neighbor, ~10 GB/s
    "linode_shared_8vcpu": 25.0,         # Shared multi-core, ~25 GB/s
    "linode_dedicated_8vcpu_epyc": 45.0, # Dedicated AMD EPYC DDR4, ~45 GB/s
    "linode_dedicated_16vcpu_epyc": 75.0,# Dedicated AMD EPYC DDR4/DDR5 multi-channel, ~75 GB/s
    "server_dual_ddr5": 85.0,            # Modern dual-channel DDR5 5600, ~85 GB/s
    "server_octa_ddr5": 300.0,           # 8-channel server memory (EPYC Genoa / Xeon Sapphire), ~300 GB/s
    "apple_m1_m2_base": 100.0,           # Unified memory, ~100 GB/s
    "apple_m1_m2_pro": 200.0,            # Unified memory, ~200 GB/s
    "apple_m1_m2_m3_max": 400.0,         # Unified memory, ~400 GB/s
    "nvidia_rtx_3060": 360.0,            # 12GB GDDR6, ~360 GB/s
    "nvidia_rtx_4090": 1008.0,           # 24GB GDDR6X, ~1008 GB/s
    "nvidia_l4": 300.0,                  # 24GB GDDR6, ~300 GB/s
    "nvidia_a100_80gb": 2039.0,          # 80GB HBM2e, ~2039 GB/s
}


def estimate_model_memory(param_count: int, quant_type: str = "FP16") -> float:
    """Calculate the estimated memory footprint in Gigabytes (GB) for a given parameter count and quant type.
    
    Args:
        param_count: Total model parameters (e.g., 4_000_000_000 for 4B)
        quant_type: Quantization format (e.g., 'FP16', 'Q4_K_M', 'Q8_0')
        
    Returns:
        Estimated model size in GB (1 GB = 1e9 bytes)
    """
    bits_per_param = QUANT_BITS_MAP.get(quant_type.upper(), 16.0)
    bytes_per_param = bits_per_param / 8.0
    total_bytes = param_count * bytes_per_param
    return total_bytes / 1e9


def calculate_required_bandwidth(model_size_gb: float, frame_rate: float) -> float:
    """Calculate the theoretical minimum memory bandwidth (GB/s) needed to run full model inference at a target frame rate.
    
    In autoregressive / streaming generation, weights must be read from RAM/VRAM once per forward pass.
    
    Args:
        model_size_gb: Memory footprint in Gigabytes
        frame_rate: Forward passes per second (e.g. 7.5 Hz for 133ms audio frame)
        
    Returns:
        Required bandwidth in GB/s
    """
    return model_size_gb * frame_rate


def estimate_frame_latency(model_size_gb: float, available_bandwidth_gb_s: float, compute_overhead_factor: float = 1.25) -> float:
    """Estimate the time in milliseconds to complete one forward pass.
    
    Args:
        model_size_gb: Model size in GB
        available_bandwidth_gb_s: Memory bandwidth of the machine in GB/s
        compute_overhead_factor: Multiplier for compute latency, cache misses, and pipeline overhead (typically 1.2 - 1.5)
        
    Returns:
        Estimated frame latency in milliseconds (ms)
    """
    if available_bandwidth_gb_s <= 0:
        return float("inf")
    transfer_seconds = model_size_gb / available_bandwidth_gb_s
    total_seconds = transfer_seconds * compute_overhead_factor
    return total_seconds * 1000.0


def analyze_quantization_viability(
    param_count: int,
    frame_rate: float = 7.5,
    budget_ms: float = 133.33,
    custom_bandwidth_gb_s: Optional[float] = None
) -> Dict[str, Any]:
    """Generate a comprehensive feasibility report across quantization formats and hardware profiles."""
    results = {}
    for quant, bits in QUANT_BITS_MAP.items():
        size_gb = estimate_model_memory(param_count, quant)
        req_bw = calculate_required_bandwidth(size_gb, frame_rate)
        
        hardware_status = {}
        for hw_name, bw in HARDWARE_BANDWIDTH_PROFILES.items():
            est_ms = estimate_frame_latency(size_gb, bw)
            feasible = est_ms <= budget_ms
            hardware_status[hw_name] = {
                "latency_ms": round(est_ms, 2),
                "feasible": feasible,
                "margin_ms": round(budget_ms - est_ms, 2)
            }
            
        custom_status = None
        if custom_bandwidth_gb_s:
            custom_ms = estimate_frame_latency(size_gb, custom_bandwidth_gb_s)
            custom_status = {
                "latency_ms": round(custom_ms, 2),
                "feasible": custom_ms <= budget_ms,
                "margin_ms": round(budget_ms - custom_ms, 2)
            }
            
        results[quant] = {
            "bits_per_param": bits,
            "model_size_gb": round(size_gb, 2),
            "required_bandwidth_gb_s": round(req_bw, 2),
            "hardware_status": hardware_status,
            "custom_status": custom_status
        }
    return results
