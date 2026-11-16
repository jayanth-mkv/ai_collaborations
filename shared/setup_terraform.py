#!/usr/bin/env python3
"""Terraform Configuration Generator & Availability Checker for Linode.

Scans live Linode capacity for GPUs (RTX 4000 Ada) and Dedicated CPUs across regions,
selects the optimal available target, and generates terraform.tfvars.
"""

import os
import sys
import json
import urllib.request
from pathlib import Path
import dotenv

root_dir = Path(__file__).resolve().parents[1]
dotenv.load_dotenv(root_dir / ".env")

API_BASE = "https://api.linode.com/v4"


def get_token() -> str:
    token = os.getenv("LINODE_CLI_TOKEN") or os.getenv("LINODE_TOKEN")
    if not token:
        raise ValueError("LINODE_CLI_TOKEN not found in .env")
    return token.strip()


def check_availability(region_id: str, plan_id: str) -> bool:
    token = get_token()
    url = f"{API_BASE}/regions/{region_id}/availability"
    headers = {"Authorization": f"Bearer {token}"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for item in data:
                if item.get("plan") == plan_id:
                    return bool(item.get("available", False))
    except Exception:
        pass
    return False


def find_best_target(target_preference: str = "gpu"):
    token = get_token()

    # Preferred regions in order of proximity & latency
    candidate_regions = ["in-bom-2", "in-maa", "sg-sin-2", "de-fra-2", "gb-lon", "us-iad-2", "us-ord"]
    gpu_plans = ["g2-gpu-rtx4000a1-s", "g2-gpu-rtx4000a1-m", "g1-gpu-rtx6000-1"]
    cpu_plans = ["g6-dedicated-4", "g6-dedicated-2", "g6-standard-4", "g6-standard-2"]

    if target_preference == "gpu":
        print("Checking GPU availability across candidate regions...")
        for reg in candidate_regions:
            for plan in gpu_plans:
                if check_availability(reg, plan):
                    print(f"✅ Found AVAILABLE GPU: region='{reg}', plan='{plan}'")
                    return reg, plan, "gpu"

        print("⚠️ No GPU instances currently available in candidate regions. Falling back to Dedicated CPU...")

    # CPU check
    print("Checking Dedicated CPU availability...")
    for reg in candidate_regions:
        for plan in cpu_plans:
            if check_availability(reg, plan):
                print(f"✅ Found AVAILABLE Dedicated CPU: region='{reg}', plan='{plan}'")
                return reg, plan, "cpu"

    # Default fallback
    return "in-bom-2", "g6-standard-4", "shared"


def generate_tfvars(preference: str = "gpu"):
    token = get_token()
    region, plan, category = find_best_target(preference)

    ssh_key = ""
    ssh_key_path = Path(os.path.expanduser("~/.ssh/id_rsa.pub"))
    if ssh_key_path.exists():
        ssh_key = ssh_key_path.read_text(encoding="utf-8").strip()
        print(f"Loaded SSH key from {ssh_key_path}")

    tfvars_path = root_dir / "terraform" / "terraform.tfvars"
    with open(tfvars_path, "w", encoding="utf-8") as f:
        f.write(f'linode_token   = "{token}"\n')
        f.write(f'region         = "{region}"\n')
        f.write(f'instance_type  = "{plan}"\n')
        f.write(f'label          = "haru-4b-{category}-runner"\n')
        f.write(f'image          = "linode/ubuntu24.04"\n')
        if ssh_key:
            f.write(f'ssh_public_key = "{ssh_key}"\n')

    print(f"\n✅ Successfully generated {tfvars_path}")
    print(f"Target Instance: {plan} ({category.upper()}) in {region}")
    return region, plan, category


if __name__ == "__main__":
    pref = "gpu"
    if len(sys.argv) > 1 and sys.argv[1] in ["gpu", "cpu"]:
        pref = sys.argv[1]
    generate_tfvars(pref)
