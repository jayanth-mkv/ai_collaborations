#!/usr/bin/env python3
"""Linode Account & Infrastructure Manager.

Utilities for querying Linode account status, available GPU & Dedicated CPU instance
types, regional availability, and provisioning/deprovisioning test instances.
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Optional
import dotenv

# Load environment variables
root_dir = Path(__file__).resolve().parents[1]
dotenv.load_dotenv(root_dir / ".env")

API_BASE = "https://api.linode.com/v4"


def get_token() -> str:
    token = os.getenv("LINODE_CLI_TOKEN") or os.getenv("LINODE_TOKEN")
    if not token:
        raise ValueError("LINODE_CLI_TOKEN not found in .env or environment variables.")
    return token.strip()


def api_request(endpoint: str, method: str = "GET", data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    token = get_token()
    url = f"{API_BASE}/{endpoint.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "ai_collaborations-manager/1.0"
    }

    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_msg)
            errors = err_json.get("errors", [{"reason": err_msg}])
            reasons = "; ".join(err.get("reason", str(err)) for err in errors)
            raise RuntimeError(f"Linode API Error ({e.code}): {reasons}") from e
        except Exception:
            raise RuntimeError(f"Linode API Error ({e.code}): {err_msg}") from e


def get_account_summary() -> Dict[str, Any]:
    acc = api_request("account")
    instances = api_request("linode/instances")
    return {
        "active_since": acc.get("active_since"),
        "balance": acc.get("balance", 0.0),
        "balance_uninvoiced": acc.get("balance_uninvoiced", 0.0),
        "capabilities": acc.get("capabilities", []),
        "active_instances_count": instances.get("results", 0),
        "instances": instances.get("data", [])
    }


def get_instance_types(category: Optional[str] = None) -> List[Dict[str, Any]]:
    resp = api_request("linode/types?page_size=100")
    all_types = resp.get("data", [])

    if category == "gpu":
        return [t for t in all_types if "gpu" in t["id"].lower() or (t.get("gpus") and t.get("gpus") > 0)]
    elif category == "dedicated":
        return [t for t in all_types if "dedicated" in t["id"].lower()]
    elif category == "standard":
        return [t for t in all_types if "standard" in t["id"].lower()]
    return all_types


def get_gpu_regions() -> List[Dict[str, Any]]:
    resp = api_request("regions")
    regions = resp.get("data", [])
    gpu_regions = []
    for r in regions:
        caps = r.get("capabilities", [])
        if "GPU Linodes" in caps or any("gpu" in c.lower() for c in caps):
            gpu_regions.append({
                "id": r["id"],
                "label": r.get("label"),
                "country": r.get("country"),
                "capabilities": caps
            })
    return gpu_regions


def create_instance(
    label: str,
    region: str = "in-bom-2",
    instance_type: str = "g2-gpu-rtx4000a1-s",
    image: str = "linode/ubuntu24.04",
    root_pass: Optional[str] = None,
    authorized_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "label": label,
        "region": region,
        "type": instance_type,
        "image": image,
        "booted": True,
    }
    if root_pass:
        payload["root_pass"] = root_pass
    if authorized_keys:
        payload["authorized_keys"] = authorized_keys

    return api_request("linode/instances", method="POST", data=payload)


def delete_instance(linode_id: int) -> Dict[str, Any]:
    return api_request(f"linode/instances/{linode_id}", method="DELETE")


def print_account_overview():
    summary = get_account_summary()
    print("=" * 80)
    print("  LINODE ACCOUNT STATUS & CREDITS OVERVIEW")
    print("=" * 80)
    print(f"Account Active Since      : {summary['active_since']}")
    print(f"Current Account Balance   : ${summary['balance']:.2f}")
    print(f"Uninvoiced Usage          : ${summary['balance_uninvoiced']:.2f}")
    print(f"Active Running Linodes    : {summary['active_instances_count']}")

    if summary['instances']:
        print("\nActive Instances:")
        for inst in summary['instances']:
            print(f"  • {inst.get('label')} (ID: {inst.get('id')}): status={inst.get('status')} | type={inst.get('type')} | IPs={inst.get('ipv4')}")
    else:
        print("  • No active instances running currently.")

    print("\n" + "=" * 80)
    print("  AVAILABLE GPU INSTANCE TIERS (For Full-Duplex Haru-4B)")
    print("=" * 80)
    gpu_types = get_instance_types("gpu")
    print(f"{'Type ID':<22} | {'VCPUs':<6} | {'RAM (GB)':<9} | {'VRAM / GPUs':<15} | {'Hourly':<10} | {'Monthly'}")
    print("-" * 80)
    for g in gpu_types:
        price_h = f"${float(g.get('price', {}).get('hourly') or 0.0):.3f}/hr"
        price_m = f"${float(g.get('price', {}).get('monthly') or 0.0):.1f}/mo"
        ram_gb = g.get('memory', 0) // 1024
        gpu_count = f"{g.get('gpus', 1)}x GPU"
        print(f"{g['id']:<22} | {g.get('vcpus', 0):<6} | {ram_gb:<9} | {gpu_count:<15} | {price_h:<10} | {price_m}")

    print("\n" + "=" * 80)
    print("  AVAILABLE DEDICATED CPU TIERS (For Quantized Q4_K_M Haru-4B)")
    print("=" * 80)
    ded_types = get_instance_types("dedicated")
    print(f"{'Type ID':<22} | {'VCPUs':<6} | {'RAM (GB)':<9} | {'Hourly':<10} | {'Monthly'}")
    print("-" * 80)
    for d in ded_types[:6]:
        price_h = f"${float(d.get('price', {}).get('hourly') or 0.0):.3f}/hr"
        price_m = f"${float(d.get('price', {}).get('monthly') or 0.0):.1f}/mo"
        ram_gb = d.get('memory', 0) // 1024
        print(f"{d['id']:<22} | {d.get('vcpus', 0):<6} | {ram_gb:<9} | {price_h:<10} | {price_m}")

    print("\n" + "=" * 80)
    print("  REGIONS WITH GPU AVAILABILITY")
    print("=" * 80)
    gpu_regions = get_gpu_regions()
    for r in gpu_regions:
        print(f"  • {r['id']:<15} | {r.get('label'):<25} | {r.get('country')}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Manage Linode instances and account resources.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Print account overview, balance, and instance types.")
    subparsers.add_parser("regions", help="List GPU-capable regions.")

    create_parser = subparsers.add_parser("create", help="Spin up a new Linode instance.")
    create_parser.add_argument("--label", type=str, required=True, help="Instance label")
    create_parser.add_argument("--type", type=str, default="g2-gpu-rtx4000a1-s", help="Linode instance type")
    create_parser.add_argument("--region", type=str, default="in-bom-2", help="Target region ID (e.g. in-bom-2, us-iad-2, gb-lon)")
    create_parser.add_argument("--image", type=str, default="linode/ubuntu24.04", help="OS distribution image")
    create_parser.add_argument("--root-pass", type=str, default=None, help="Root password (minimum 12 chars, upper/lower/symbols)")

    del_parser = subparsers.add_parser("delete", help="Delete a Linode instance.")
    del_parser.add_argument("--id", type=int, required=True, help="Linode instance ID to destroy")

    args = parser.parse_args()

    if args.command == "create":
        print(f"Creating Linode instance: label='{args.label}', type='{args.type}', region='{args.region}'...")
        res = create_instance(args.label, args.region, args.type, args.image, args.root_pass)
        print(f"✅ Instance created! ID: {res.get('id')} | Status: {res.get('status')} | IPv4: {res.get('ipv4')}")
    elif args.command == "delete":
        print(f"Deleting Linode instance ID: {args.id}...")
        delete_instance(args.id)
        print(f"✅ Instance {args.id} deleted successfully.")
    elif args.command == "regions":
        for r in get_gpu_regions():
            print(f"{r['id']:<15} | {r['label']:<25} | {r['country']}")
    else:
        print_account_overview()


if __name__ == "__main__":
    main()
