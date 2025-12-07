resource "aws_eks_cluster" "main" {
  name            = var.cluster_name
  role_arn        = aws_iam_role.eks_cluster.arn
  version         = var.eks_version
  
  vpc_config {
    subnet_ids              = concat(aws_subnet.public[*].id, aws_subnet.private[*].id)
    security_group_ids      = [aws_security_group.eks_cluster.id]
    endpoint_private_access = true
    endpoint_public_access  = true
  }
  
  enabled_cluster_log_types = [
    "api",
    "audit",
    "authenticator",
    "controllerManager",
    "scheduler"
  ]
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster,
    aws_cloudwatch_log_group.eks_cluster
  ]
  
  tags = { Name = var.cluster_name }
}

resource "aws_cloudwatch_log_group" "eks_cluster" {
  name              = "/aws/eks/${var.cluster_name}/cluster"
  retention_in_days = 30
  
  tags = { Name = "mlops-eks-logs" }
}

data "tls_certificate" "cluster" {
  url = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "cluster" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.cluster.certificates[0].sha1_fingerprint]
  url = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

resource "aws_eks_node_group" "general" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "general-${var.environment}"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = aws_subnet.private[*].id
  ami_type        = "AL2023_x86_64_STANDARD"

  scaling_config {
    desired_size = 1
    max_size     = 3
    min_size     = 1
  }
  
  instance_types = ["t3.micro"]
  capacity_type  = "SPOT"
  disk_size      = 20
  
  labels = {
    Environment = var.environment
    NodeType    = "general"
  }
  
  tags = { Name = "mlops-general-nodes" }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_nodes_basic,
    aws_iam_role_policy_attachment.eks_nodes_cni,
    aws_iam_role_policy_attachment.eks_nodes_registry,
    aws_iam_role_policy_attachment.eks_nodes_s3
  ]
}

# resource "aws_eks_node_group" "gpu" {
#   cluster_name    = aws_eks_cluster.main.name
#   node_group_name = "gpu-${var.environment}"
#   node_role_arn   = aws_iam_role.eks_nodes.arn
#   subnet_ids      = aws_subnet.private[*].id
  
#   scaling_config {
#     desired_size = 0
#     max_size     = 3
#     min_size     = 0
#   }
  
#   instance_types = ["g4dn.xlarge"]
#   capacity_type  = "SPOT"
#   disk_size      = 100
  
#   labels = {
#     Environment = var.environment
#     NodeType    = "gpu"
#   }
  
#   taint {
#     key    = "nvidia.com/gpu"
#     value  = "true"
#     effect = "NO_SCHEDULE"
#   }
  
#   tags = { Name = "mlops-gpu-nodes" }
  
#   depends_on = [
#     aws_iam_role_policy_attachment.eks_nodes_basic,
#     aws_iam_role_policy_attachment.eks_nodes_cni,
#     aws_iam_role_policy_attachment.eks_nodes_registry,
#     aws_iam_role_policy_attachment.eks_nodes_s3
#   ]
# }

resource "aws_eks_addon" "vpc_cni" {
  cluster_name                = aws_eks_cluster.main.name
  addon_name                  = "vpc-cni"
  addon_version               = "v1.14.1-eksbuild.1"
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"
  service_account_role_arn    = aws_iam_role.eks_nodes.arn
}

resource "aws_eks_addon" "coredns" {
  cluster_name                = aws_eks_cluster.main.name
  addon_name                  = "coredns"
  addon_version               = "v1.10.1-eksbuild.2"
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"
}

resource "aws_eks_addon" "kube_proxy" {
  cluster_name                = aws_eks_cluster.main.name
  addon_name                  = "kube-proxy"
  addon_version               = "v1.28.1-eksbuild.1"
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"
}
