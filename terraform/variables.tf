variable "linode_token" {
  description = "Linode Personal Access Token"
  type        = string
  sensitive   = true
}

variable "region" {
  description = "Target Linode region ID (e.g. in-bom-2, sg-sin-2, us-ord, de-fra-2)"
  type        = string
  default     = "in-bom-2"
}

variable "instance_type" {
  description = "Linode instance plan ID (e.g. g2-gpu-rtx4000a1-s for GPU, g6-dedicated-16 for CPU)"
  type        = string
  default     = "g2-gpu-rtx4000a1-s"
}

variable "label" {
  description = "Hostname and label for the instance"
  type        = string
  default     = "haru-4b-runner"
}

variable "image" {
  description = "Distribution image"
  type        = string
  default     = "linode/ubuntu24.04"
}

variable "ssh_public_key" {
  description = "Public SSH key string to install on the instance"
  type        = string
  default     = ""
}

variable "root_password" {
  description = "Root password (if left empty, a secure password is generated automatically)"
  type        = string
  default     = ""
  sensitive   = true
}
