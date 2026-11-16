output "instance_id" {
  description = "Linode Instance ID"
  value       = linode_instance.ai_runner.id
}

output "instance_label" {
  description = "Linode Instance Label"
  value       = linode_instance.ai_runner.label
}

output "instance_type" {
  description = "Plan Type"
  value       = linode_instance.ai_runner.type
}

output "region" {
  description = "Deployed Region"
  value       = linode_instance.ai_runner.region
}

output "public_ip" {
  description = "Public IPv4 Address"
  value       = one(linode_instance.ai_runner.ipv4)
}

output "status" {
  description = "Instance Status"
  value       = linode_instance.ai_runner.status
}

output "ssh_command" {
  description = "Direct SSH Command"
  value       = "ssh root@${one(linode_instance.ai_runner.ipv4)}"
}

output "root_password" {
  description = "Generated Root Password (if no SSH key used)"
  value       = local.effective_root_password
  sensitive   = true
}
