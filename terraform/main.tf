provider "linode" {
  token = var.linode_token
}

resource "random_password" "root_pass" {
  length           = 24
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

locals {
  effective_root_password = var.root_password != "" ? var.root_password : random_password.root_pass.result
  authorized_keys         = var.ssh_public_key != "" ? [trimspace(var.ssh_public_key)] : []
}

resource "linode_instance" "ai_runner" {
  label           = var.label
  image           = var.image
  region          = var.region
  type            = var.instance_type
  authorized_keys = local.authorized_keys
  root_pass       = local.effective_root_password
  booted          = true

  tags = ["ai-collaborations", "haru-4b"]
}
