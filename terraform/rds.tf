resource "aws_db_subnet_group" "main" {
  name       = "mlops-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id
  
  tags = { Name = "mlops-db-subnet-group" }
}

resource "random_password" "rds" {
  length  = 32
  special = true
}

resource "aws_kms_key" "rds" {
  description             = "KMS key for RDS encryption"
  deletion_window_in_days = 10
  enable_key_rotation     = true
}

resource "aws_kms_alias" "rds" {
  name          = "alias/mlops-rds"
  target_key_id = aws_kms_key.rds.key_id
}

# resource "aws_rds_cluster" "main" {
#   cluster_identifier              = "mlops-postgres"
#   engine                          = "aurora-postgresql"
#   engine_version                  = "15.3"
#   database_name                   = "mlops"
#   master_username                 = "postgres"
#   master_password                 = random_password.rds.result
#   db_subnet_group_name            = aws_db_subnet_group.main.name
#   vpc_security_group_ids          = [aws_security_group.rds.id]

#   backup_retention_period         = 1
#   preferred_backup_window         = "03:00-04:00"
#   preferred_maintenance_window    = "sun:04:00-sun:05:00"
#   storage_encrypted               = true
#   kms_key_id                      = aws_kms_key.rds.arn
  
#   enabled_cloudwatch_logs_exports = ["postgresql"]
#   monitoring_interval             = 60
#   monitoring_role_arn             = aws_iam_role.rds_monitoring.arn
  
#   skip_final_snapshot       = false
#   final_snapshot_identifier = "mlops-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
#   copy_tags_to_snapshot     = true
  
#   tags = { Name = "mlops-postgres-cluster" }
# }

# resource "aws_rds_cluster_instance" "main" {
#   count              = 1
#   identifier         = "mlops-postgres-${count.index + 1}"
#   cluster_identifier = aws_rds_cluster.main.id
#   instance_class     = "db.t3.micro"
#   engine              = aws_rds_cluster.main.engine
#   engine_version      = aws_rds_cluster.main.engine_version
#   publicly_accessible        = false
#   auto_minor_version_upgrade = true
  
#   tags = { Name = "mlops-postgres-instance-${count.index + 1}" }
# }

resource "aws_cloudwatch_log_group" "rds" {
  name              = "/aws/rds/mlops-postgres"
  retention_in_days = 30
}

resource "aws_secretsmanager_secret" "rds_password" {
  name                    = "mlops/rds/password"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "rds_password" {
  secret_id     = aws_secretsmanager_secret.rds_password.id
  secret_string = random_password.rds.result
}

resource "aws_db_instance" "postgres" {
  identifier            = var.db_name
  engine                = "postgres"
  engine_version        = "15.3"
  instance_class        = "db.t3.micro"
  allocated_storage     = 20
  storage_type          = "gp2"
  
  db_name  = var.db_name
  username = var.db_user
  password = var.db_password
  
  backup_retention_period = 7
  skip_final_snapshot     = true
  
  publicly_accessible   = false
  
  parameter_group_name = "default.postgres15"

  tags = { Name = var.db_name }
}

output "rds_endpoint" {
  value = aws_db_instance.postgres.endpoint
}
