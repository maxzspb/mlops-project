output "eks_cluster_name" {
  value       = aws_eks_cluster.main.name
  description = "EKS cluster name"
}

output "eks_cluster_endpoint" {
  value       = aws_eks_cluster.main.endpoint
  description = "Endpoint for your EKS Kubernetes API"
}

output "eks_cluster_version" {
  value       = aws_eks_cluster.main.version
  description = "The Kubernetes server version"
}

output "rds_postgres_endpoint" {
  value       = aws_db_instance.postgres.endpoint
  description = "RDS PostgreSQL endpoint"
}

output "redis_endpoint" {
  value       = try(aws_elasticache_cluster.redis.cache_nodes[0].address, "not-created")
  description = "Redis cluster endpoint"
}

output "s3_bucket_name" {
  value       = aws_s3_bucket.data_lake.id
  description = "S3 data lake bucket name"
}

output "kube_config_command" {
  value = "aws eks update-kubeconfig --name ${aws_eks_cluster.main.name} --region ${var.aws_region}"
  description = "Command to update kubeconfig"
}

output "rds_password_secret_name" {
  value       = aws_secretsmanager_secret.rds_password.name
  description = "Secrets Manager secret for RDS password"
}

output "redis_password_secret_name" {
  value       = aws_secretsmanager_secret.redis_password.name
  description = "Secrets Manager secret for Redis password"
}