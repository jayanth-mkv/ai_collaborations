terraform {
  required_version = ">= 1.5.0"
  required_providers {
    linode = {
      source  = "linode/linode"
      version = ">= 2.16.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.5.0"
    }
  }
}
