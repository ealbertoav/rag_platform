# EKS Cluster Setup — rag-platform

End-to-end guide from zero to a running production cluster. Assumes AWS CLI credentials are already configured. One session (~45–60 min, most of it waiting on AWS provisioning).

---

## Prerequisites

```bash
# AWS CLI v2
aws --version

# eksctl
brew tap weaveworks/tap && brew install weaveworks/tap/eksctl

# kubectl
brew install kubectl

# Helm
brew install helm

# Verify AWS identity
aws sts get-caller-identity
```

---

## 1. Create the EKS cluster

```bash
eksctl create cluster \
  --name rag-platform-prod \
  --region us-east-1 \
  --nodegroup-name standard \
  --node-type m7g.2xlarge \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 10 \
  --managed \
  --with-oidc
```

`--with-oidc` creates the OIDC provider required by the AWS Load Balancer Controller.  
`m7g.2xlarge` is Graviton3 (arm64, cost-efficient). For NVIDIA GPU inference swap to `g4dn.xlarge` and add `--node-labels accelerator=nvidia-gpu`.

**Estimated time:** 15–20 min.

Update kubeconfig once the cluster is up:

```bash
aws eks update-kubeconfig --name rag-platform-prod --region us-east-1
kubectl get nodes   # should show 3 Ready nodes
```

---

## 2. Install required add-ons

### 2a. EBS CSI driver (for `gp3` PVCs — Qdrant storage)

```bash
eksctl create addon \
  --name aws-ebs-csi-driver \
  --cluster rag-platform-prod \
  --region us-east-1 \
  --force
```

Create the `gp3` StorageClass and set it as default:

```bash
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  encrypted: "true"
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Retain
EOF
```

### 2b. EFS CSI driver (for `ReadOnlyMany` models PVC)

Required for `persistence.models.accessMode: ReadOnlyMany`. Skip if you're using `ReadWriteOnce` on a single-replica deployment.

```bash
# Install the EFS CSI driver add-on
eksctl create addon \
  --name aws-efs-csi-driver \
  --cluster rag-platform-prod \
  --region us-east-1

# Create an EFS file system (note the FileSystemId in the output)
aws efs create-file-system \
  --region us-east-1 \
  --tags Key=Name,Value=rag-platform-models

# Create mount targets in each subnet (replace with your subnet IDs)
VPC_ID=$(aws eks describe-cluster --name rag-platform-prod \
  --query "cluster.resourcesVpcConfig.vpcId" --output text)
SUBNETS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query "Subnets[*].SubnetId" --output text)
SG=$(aws eks describe-cluster --name rag-platform-prod \
  --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text)
EFS_ID=<FileSystemId from above>
for subnet in $SUBNETS; do
  aws efs create-mount-target --file-system-id $EFS_ID \
    --subnet-id $subnet --security-groups $SG
done

# Create the StorageClass
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: efs-sc
provisioner: efs.csi.aws.com
parameters:
  fileSystemId: ${EFS_ID}
  directoryPerms: "700"
EOF
```

### 2c. AWS Load Balancer Controller (for ALB Ingress)

```bash
# Create IAM policy
curl -fsSL https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/main/docs/install/iam_policy.json \
  -o /tmp/alb-iam-policy.json

aws iam create-policy \
  --policy-name AWSLoadBalancerControllerIAMPolicy \
  --policy-document file:///tmp/alb-iam-policy.json

# Create IAM service account (OIDC must be enabled — done in step 1)
eksctl create iamserviceaccount \
  --cluster rag-platform-prod \
  --namespace kube-system \
  --name aws-load-balancer-controller \
  --attach-policy-arn arn:aws:iam::$(aws sts get-caller-identity \
      --query Account --output text):policy/AWSLoadBalancerControllerIAMPolicy \
  --approve

# Install via Helm
helm repo add eks https://aws.github.io/eks-charts
helm repo update
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  --namespace kube-system \
  --set clusterName=rag-platform-prod \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller

kubectl -n kube-system rollout status deployment/aws-load-balancer-controller
```

### 2d. metrics-server (for HPA)

```bash
helm repo add metrics-server https://kubernetes-sigs.github.io/metrics-server/
helm install metrics-server metrics-server/metrics-server --namespace kube-system
kubectl -n kube-system rollout status deployment/metrics-server
```

---

## 3. Push images to ECR

```bash
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=us-east-1
ECR_BASE=${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Authenticate Docker
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $ECR_BASE

# Create repositories (once)
aws ecr create-repository --repository-name rag-platform-api  --region $AWS_REGION
aws ecr create-repository --repository-name rag-platform-worker --region $AWS_REGION

# Build and push (run from repo root)
docker build -f docker/Dockerfile.api    -t $ECR_BASE/rag-platform-api:latest .
docker build -f docker/Dockerfile.worker -t $ECR_BASE/rag-platform-worker:latest .
docker push $ECR_BASE/rag-platform-api:latest
docker push $ECR_BASE/rag-platform-worker:latest
```

---

## 4. Deploy the Helm chart

```bash
# Get your ACM certificate ARN
ACM_ARN=$(aws acm list-certificates \
  --query "CertificateSummaryList[?DomainName=='api.yourdomain.com'].CertificateArn" \
  --output text)

AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ECR_BASE=${AWS_ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com

helm install rag-platform helm/rag-platform \
  --namespace rag-platform \
  --create-namespace \
  --set image.api.repository=$ECR_BASE/rag-platform-api \
  --set image.worker.repository=$ECR_BASE/rag-platform-worker \
  --set ingress.enabled=true \
  --set ingress.host=api.yourdomain.com \
  --set ingress.certificateArn=$ACM_ARN \
  --set persistence.models.storageClass=efs-sc \
  --set env.embeddingsDevice=cpu

# Watch rollout
kubectl -n rag-platform rollout status deployment/rag-platform-api
kubectl -n rag-platform get ingress   # note the ALB DNS name
```

Point your DNS record at the ALB DNS name (CNAME).

### Upgrade

```bash
helm upgrade rag-platform helm/rag-platform \
  --namespace rag-platform \
  --reuse-values \
  --set image.api.tag=<new-tag>
```

---

## 5. Connect Lens

**Install Lens Desktop** (if not already): https://k8slens.dev

1. Open Lens → **File → Add Cluster**
2. Select **Paste as text** and paste the output of:
   ```bash
   aws eks update-kubeconfig --name rag-platform-prod --region us-east-1 --dry-run
   ```
   Or point Lens at your kubeconfig file directly:  
   **Settings → Kubernetes → Kubeconfig** → `~/.kube/config`
3. Lens auto-discovers the cluster. Click **Connect**.

**Key views for this cluster:**

| What to check | Lens path |
|---|---|
| Pod health / restarts | Workloads → Pods → namespace `rag-platform` |
| HPA scaling events | Workloads → HPA |
| PVC binding status | Storage → Persistent Volume Claims |
| ALB Ingress address | Network → Ingresses |
| Live logs | Click any pod → Logs tab |
| Shell into api pod | Click any pod → Terminal tab |
| Prometheus metrics | Install the Lens Metrics extension or use port-forward |

Port-forward Prometheus if you haven't set up external access:

```bash
kubectl -n rag-platform port-forward svc/prometheus 9090:9090
# open http://localhost:9090
```

---

## 6. Common operations

```bash
# Scale api manually (while autoscaling is disabled)
kubectl -n rag-platform scale deployment/rag-platform-api --replicas=4

# Run a one-shot ingestion job
kubectl -n rag-platform run ingest \
  --image=$ECR_BASE/rag-platform-worker:latest \
  --restart=Never \
  --env-file=.env \
  -- python scripts/ingest.py --source /app/data/raw

# Tail api logs
kubectl -n rag-platform logs -f -l app.kubernetes.io/component=api

# Open a shell in an api pod
kubectl -n rag-platform exec -it \
  $(kubectl -n rag-platform get pod -l app.kubernetes.io/component=api \
    -o jsonpath='{.items[0].metadata.name}') -- bash
```

---

## 7. Teardown

```bash
# Remove the Helm release (keeps PVCs — Qdrant data is preserved)
helm uninstall rag-platform --namespace rag-platform

# Delete PVCs explicitly if you want to destroy data
kubectl -n rag-platform delete pvc --all

# Delete the cluster (also removes node groups and the VPC if eksctl created it)
eksctl delete cluster --name rag-platform-prod --region us-east-1
```

> **Warning:** `eksctl delete cluster` removes the EKS control plane, node groups, and the VPC. EFS file systems and ECR repositories are **not** deleted — remove them manually if needed.
