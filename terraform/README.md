# Terraform Infrastructure for Linode Model Runners

Automated provisioning of GPU and Dedicated CPU instances on Linode (Akamai) for testing AI models (such as **Haru-4B**).

---

## 📋 Available Hardware Scanned

* **GPU:** `g2-gpu-rtx4000a1-s` (1x NVIDIA RTX 4000 Ada 20GB VRAM, 4 vCPUs, 16GB RAM) at **$0.520/hr** in `in-bom-2` (Mumbai), `sg-sin-2` (Singapore), `de-fra-2` (Frankfurt).
* **Dedicated CPU:** `g6-dedicated-16` (16 dedicated AMD EPYC cores, 32GB RAM) at **$0.432/hr** in `in-bom-2`.

---

## 🚀 Quickstart

### 1. Select Target & Generate `terraform.tfvars`
To select the best available **GPU** instance:
```bash
python ../shared/setup_terraform.py gpu
```

To switch to a **Dedicated CPU** instance:
```bash
python ../shared/setup_terraform.py cpu
```

### 2. Initialize & Review Plan
```bash
terraform init
terraform plan
```

### 3. Deploy Instance
```bash
terraform apply -auto-approve
```

### 4. Connect via SSH
```bash
ssh root@<PUBLIC_IP>
```

### 5. Destroy Instance (Prevent Unneeded Credit Usage)
```bash
terraform destroy -auto-approve
```
