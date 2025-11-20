# Terraform Infrastructure as Code

This directory contains Terraform configurations for deploying the ML Portfolio infrastructure on AWS and GCP.

## 📁 Structure

```
terraform/
├── aws/                    # AWS infrastructure
│   ├── main.tf            # Main AWS resources (EKS, VPC, S3, RDS, ECR)
│   ├── variables.tf       # Input variables
│   └── terraform.tfvars   # Variable values (gitignored)
│
└── gcp/                    # GCP infrastructure
    ├── main.tf            # Main GCP resources (GKE, GCS, Cloud SQL)
    ├── variables.tf       # Input variables
    └── terraform.tfvars   # Variable values (gitignored)
```

## 🚀 AWS Deployment

### Prerequisites

```bash
# Install Terraform
brew install terraform  # macOS
# or
sudo apt-get install terraform  # Linux

# Configure AWS credentials
aws configure
```

### Resources Created

- **EKS Cluster**: Managed Kubernetes cluster (v1.28)
- **VPC**: Networking with public/private subnets
- **S3 Buckets**: 
  - ML models storage
  - MLflow artifacts storage
- **RDS PostgreSQL**: Backend for MLflow tracking server
- **ECR Repositories**: Docker image registries for all 7 ML services
- **CloudWatch**: Centralized logging

### Deploy

```bash
cd infra/terraform/aws

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
project_name = "ml-portfolio"
environment  = "production"
aws_region   = "us-east-1"

db_username = "mlflow"
db_password = "your-secure-password"
EOF

# Initialize Terraform
terraform init

# Review plan
terraform plan

# Apply changes
terraform apply
```

### Outputs

After successful deployment:

```bash
# Get cluster credentials
aws eks update-kubeconfig --name ml-portfolio-eks-production --region us-east-1

# Verify connection
kubectl get nodes
```

## ☁️ GCP Deployment

### Prerequisites

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### Resources Created

- **GKE Cluster**: Managed Kubernetes cluster (auto-upgrade enabled)
- **VPC & Subnets**: Networking with secondary ranges for pods/services
- **Cloud Storage**: 
  - ML models bucket
  - MLflow artifacts bucket
- **Cloud SQL PostgreSQL**: Backend for MLflow
- **Artifact Registry**: Container image registry
- **Workload Identity**: Secure service account bindings

### Deploy

```bash
cd infra/terraform/gcp

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
project_id   = "your-gcp-project-id"
project_name = "ml-portfolio"
environment  = "production"
region       = "us-central1"

db_password = "your-secure-password"
EOF

# Initialize Terraform
terraform init

# Review plan
terraform plan

# Apply changes
terraform apply
```

### Outputs

After successful deployment:

```bash
# Get cluster credentials
gcloud container clusters get-credentials ml-portfolio-gke-production --region us-central1

# Verify connection
kubectl get nodes
```

## 📊 Cost Estimation

### AWS (Production)

| Resource | Type | Monthly Cost |
|----------|------|-------------|
| EKS Cluster | Control plane | $72 |
| EC2 Instances | 3x t3.large | ~$190 |
| RDS PostgreSQL | db.t3.micro | ~$15 |
| S3 Storage | 100GB | ~$3 |
| Data Transfer | 100GB | ~$9 |
| **Total** | | **~$289/month** |

### GCP (Production)

| Resource | Type | Monthly Cost |
|----------|------|-------------|
| GKE Cluster | Control plane | Free (one per billing account) |
| Compute Engine | 3x e2-standard-4 | ~$270 |
| Cloud SQL | db-f1-micro | ~$7 |
| Cloud Storage | 100GB | ~$2 |
| Data Transfer | 100GB | ~$12 |
| **Total** | | **~$291/month** |

### Development Environment

For dev/staging, set `environment = "dev"` in `terraform.tfvars`:
- Uses smaller instance types
- Single NAT gateway (AWS)
- Preemptible VMs (GCP)
- **Estimated cost: ~$100-150/month**

## 🔒 Security Best Practices

### Secrets Management

Never commit sensitive values. Use environment variables or secret management tools:

```bash
# AWS
export TF_VAR_db_password="your-secure-password"

# GCP
export TF_VAR_db_password="your-secure-password"

# Or use AWS Secrets Manager / GCP Secret Manager
```

### State Management

Both configurations use remote state storage:
- **AWS**: S3 + DynamoDB for locking
- **GCP**: Cloud Storage bucket

Create state backends before running terraform:

```bash
# AWS
aws s3api create-bucket \
  --bucket ml-portfolio-terraform-state \
  --region us-east-1

aws dynamodb create-table \
  --table-name terraform-state-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

# GCP
gsutil mb -l us-central1 gs://ml-portfolio-terraform-state/
```

## 🧹 Cleanup

To destroy all infrastructure:

```bash
# AWS
cd infra/terraform/aws
terraform destroy

# GCP
cd infra/terraform/gcp
terraform destroy
```

## 📚 Additional Resources

- [Terraform AWS Provider Docs](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Terraform GCP Provider Docs](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [EKS Best Practices](https://aws.github.io/aws-eks-best-practices/)
- [GKE Best Practices](https://cloud.google.com/kubernetes-engine/docs/best-practices)

## 🔧 Customization

### Change Instance Types

Edit `variables.tf` and set:

```hcl
# AWS
variable "eks_node_instance_type" {
  default = "t3.xlarge"  # More powerful
}

# GCP
variable "machine_type" {
  default = "e2-standard-8"  # More powerful
}
```

### Add More Services

Add to ECR/Artifact Registry in `main.tf`:

```hcl
resource "aws_ecr_repository" "new_service" {
  name = "${var.project_name}/new-service"
  # ... rest of config
}
```

## 📞 Support

For issues or questions:
- Check Terraform logs: `TF_LOG=DEBUG terraform apply`
- Validate syntax: `terraform validate`
- Format code: `terraform fmt -recursive`
