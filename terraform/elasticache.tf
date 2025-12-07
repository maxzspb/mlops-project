resource "aws_elasticache_subnet_group" "main" {
  name       = "mlops-redis-subnet-group"
  subnet_ids = aws_subnet.private[*].id
}

resource "random_password" "redis" {
  length  = 32
  special = true
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "mlops-redis"
  engine               = "redis"
  engine_version       = "7.0"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  
  port               = 6379
  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]
  
  snapshot_retention_limit = 30
  snapshot_window    = "03:00-04:00"
  maintenance_window = "sun:04:00-sun:05:00"  # Другой день, не пересекаются

  
  tags = { Name = "mlops-redis" }
}

resource "aws_cloudwatch_log_group" "redis" {
  name              = "/aws/elasticache/mlops-redis"
  retention_in_days = 30
}

resource "aws_secretsmanager_secret" "redis_password" {
  name                    = "mlops/redis/auth-token"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "redis_password" {
  secret_id     = aws_secretsmanager_secret.redis_password.id
  secret_string = random_password.redis.result
}
