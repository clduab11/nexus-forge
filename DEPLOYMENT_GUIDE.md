# Nexus Forge Production Deployment Guide

This guide provides complete instructions for deploying Nexus Forge to Google Cloud Platform using Cloud Run, with optional Kubernetes deployment support.

## 🚀 Quick Start

1. **Prerequisites Setup**
   ```bash
   # Install required tools
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   
   # Set environment variables
   export GCP_PROJECT_ID="your-project-id"
   export GCP_REGION="us-central1"
   export NOTIFICATION_EMAIL="alerts@yourcompany.com"
   ```

2. **Deploy Infrastructure**
   ```bash
   ./scripts/setup_gcp_resources.sh
   ```

3. **Deploy Monitoring**
   ```bash
   ./scripts/deploy_monitoring.sh
   ```

4. **Deploy Application**
   ```bash
   ./scripts/deploy_cloud_run.sh
   ```

## 📁 Deployment Structure

```
parallax-pal/
├── scripts/                          # Deployment automation
│   ├── deploy_cloud_run.sh          # Cloud Run deployment
│   ├── setup_gcp_resources.sh       # Infrastructure setup
│   ├── deploy_monitoring.sh         # Monitoring stack
│   └── validate_deployment.sh       # Configuration validation
├── k8s/                              # Kubernetes manifests
│   ├── deployment.yml               # Application deployment
│   ├── service.yml                  # Service configuration
│   ├── ingress.yml                  # External access
│   └── configmap.yml                # Environment configuration
├── config/                           # Application configuration
│   └── production.yml               # Production settings
├── terraform/                       # Infrastructure as Code
│   └── main.tf                      # Complete GCP infrastructure
├── .github/workflows/               # CI/CD Pipeline
│   └── deploy.yml                   # GitHub Actions workflow
└── Dockerfile                       # Multi-stage production build
```

## 🛠 Deployment Components

### 1. Cloud Run Deployment (`scripts/deploy_cloud_run.sh`)

**Features:**
- Multi-stage Docker build with security optimizations
- Automatic service account creation with minimal permissions
- Gradual traffic rollout (10% → 50% → 100%)
- Health checks and smoke tests
- Automatic cleanup of old revisions
- Comprehensive error handling and logging

**Configuration:**
- Memory: 2Gi
- CPU: 2 cores
- Min instances: 1
- Max instances: 100
- Timeout: 300s
- Concurrency: 1000

### 2. Infrastructure Setup (`scripts/setup_gcp_resources.sh`)

**Creates:**
- Service accounts (runtime, build, monitoring)
- IAM roles and permissions
- Storage buckets (app storage, build artifacts)
- Secret Manager secrets
- Static IP reservation
- Firestore database
- Pub/Sub topics
- Cloud Scheduler jobs
- Log sinks and error alerting

### 3. Monitoring Stack (`scripts/deploy_monitoring.sh`)

**Includes:**
- Alert policies (error rate, memory, CPU, latency, uptime)
- Notification channels (email, Pub/Sub)
- Uptime checks from multiple regions
- Custom dashboards with 6 key metrics
- Log-based metrics
- Scheduled monitoring jobs

### 4. Kubernetes Support (`k8s/`)

**Components:**
- Deployment with HPA (3-100 replicas)
- Service with backend configuration
- Ingress with SSL and Cloud Armor
- ConfigMap with comprehensive settings
- Security policies and network policies
- Pod disruption budgets

### 5. CI/CD Pipeline (`.github/workflows/deploy.yml`)

**Pipeline Stages:**
1. **Quality Gates:** Testing, linting, security scanning
2. **Build:** Multi-stage Docker build with SBOM generation
3. **Security:** Trivy vulnerability scanning
4. **Infrastructure:** Terraform deployment
5. **Deployment:** Cloud Run with integration tests
6. **Post-Deploy:** Traffic management and cleanup

## 🔧 Configuration Management

### Production Configuration (`config/production.yml`)

Comprehensive configuration covering:
- Application settings (server, security, CORS)
- Database configuration (PostgreSQL, Redis)
- Google Cloud Platform integration
- AI service configuration (Gemini, Imagen, Veo)
- Caching strategies
- Monitoring and logging
- File upload handling
- API settings and authentication
- Background task processing
- Feature flags
- Performance tuning
- Alerts and notifications

### Environment Variables

Required environment variables:
```bash
# Core Configuration
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
SERVICE_NAME=nexus-forge

# Secrets (stored in Secret Manager)
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
GEMINI_API_KEY=your-gemini-key
JWT_SECRET=your-jwt-secret

# Optional
NOTIFICATION_EMAIL=alerts@yourcompany.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

## 🏗 Infrastructure as Code (`terraform/main.tf`)

**Resources Created:**
- **Networking:** VPC, subnets, private service connection
- **Compute:** Cloud Run service, service accounts
- **Storage:** Cloud Storage buckets, Cloud SQL PostgreSQL
- **Caching:** Redis (Memorystore)
- **Security:** KMS keys, Secret Manager, Cloud Armor
- **Monitoring:** Dashboards, alert policies, uptime checks
- **Database:** Firestore, Cloud SQL with HA configuration
- **Messaging:** Pub/Sub topics for events and alerts

**State Management:**
- Remote state in Google Cloud Storage
- State locking with Cloud Storage
- Modular resource organization

## 🐳 Docker Configuration

**Multi-stage Build:**
1. **Base:** System dependencies and security updates
2. **Builder:** Build dependencies and package installation
3. **Production:** Minimal runtime with security hardening
4. **Development:** Optional development tools

**Security Features:**
- Non-root user execution
- Read-only root filesystem
- Minimal attack surface
- Multi-architecture support
- Comprehensive health checks

## 🔒 Security Features

### Authentication & Authorization
- JWT-based API authentication
- Service account with minimal permissions
- Secret Manager for credential storage
- CORS configuration for web security

### Network Security
- Cloud Armor protection
- Private networking for databases
- SSL/TLS encryption everywhere
- Network policies in Kubernetes

### Container Security
- Non-root container execution
- Read-only filesystem
- Security context enforcement
- Regular vulnerability scanning

### Data Protection
- Encryption at rest and in transit
- Secure secret management
- Audit logging
- Access controls

## 📊 Monitoring & Observability

### Metrics
- Request rate and latency
- Error rates and types
- Resource utilization (CPU, memory)
- Custom business metrics

### Alerting
- High error rate (>5%)
- High memory usage (>80%)
- High CPU usage (>80%)
- Response time degradation (>2s)
- Service downtime

### Logging
- Structured JSON logging
- Centralized log aggregation
- Error tracking and analysis
- Request tracing

### Dashboards
- Real-time performance metrics
- Historical trend analysis
- Service health overview
- Custom KPI tracking

## 🚦 Deployment Process

### 1. Pre-deployment Validation
```bash
./scripts/validate_deployment.sh
```

### 2. Infrastructure Deployment
```bash
# Option A: Using scripts
./scripts/setup_gcp_resources.sh

# Option B: Using Terraform
cd terraform
terraform init
terraform plan
terraform apply
```

### 3. Application Deployment
```bash
# Cloud Run (recommended)
./scripts/deploy_cloud_run.sh

# Kubernetes (alternative)
kubectl apply -f k8s/
```

### 4. Monitoring Setup
```bash
./scripts/deploy_monitoring.sh
```

### 5. Post-deployment Verification
- Health check: `curl https://your-service-url/health`
- API check: `curl https://your-service-url/api/v1/health`
- Monitor dashboards and alerts

## 🔄 CI/CD Integration

### GitHub Actions Workflow
- **Triggers:** Push to main/production, pull requests
- **Quality Gates:** Tests, linting, security scans
- **Deployment:** Automated with manual approval gates
- **Rollback:** Automated rollback on failure

### Manual Deployment
```bash
# Build and push image
docker build -t gcr.io/$PROJECT_ID/nexus-forge:$TAG .
docker push gcr.io/$PROJECT_ID/nexus-forge:$TAG

# Deploy to Cloud Run
gcloud run deploy nexus-forge \
  --image gcr.io/$PROJECT_ID/nexus-forge:$TAG \
  --region us-central1
```

## 🚨 Troubleshooting

### Common Issues

1. **Service Account Permissions**
   ```bash
   # Check service account roles
   gcloud projects get-iam-policy $PROJECT_ID
   ```

2. **Container Startup Issues**
   ```bash
   # Check logs
   gcloud logs read "resource.type=cloud_run_revision"
   ```

3. **Database Connection**
   ```bash
   # Test database connectivity
   gcloud sql connect nexus-forge-postgres
   ```

### Health Checks
- **Service Health:** `/health`
- **Detailed Health:** `/health/detailed`
- **Readiness:** `/ready`
- **Metrics:** `/metrics`

## 📈 Scaling & Performance

### Horizontal Scaling
- Cloud Run: Auto-scaling 1-100 instances
- Kubernetes: HPA based on CPU/memory
- Database: Read replicas for scaling

### Performance Optimization
- Connection pooling
- Redis caching
- CDN for static assets
- Async processing for heavy tasks

## 🔄 Maintenance

### Regular Tasks
- Monitor resource usage and costs
- Update dependencies and base images
- Review and rotate secrets
- Analyze performance metrics
- Backup critical data

### Updates
- Blue-green deployments for zero downtime
- Gradual traffic shifting
- Automated rollback on failure
- Database migration handling

## 📞 Support

### Monitoring Access
- **Dashboards:** [Google Cloud Console Monitoring](https://console.cloud.google.com/monitoring)
- **Logs:** [Cloud Logging](https://console.cloud.google.com/logs)
- **Alerts:** Configure email/Slack notifications

### Emergency Procedures
1. **Rollback:** Use GitHub Actions rollback workflow
2. **Scale Down:** Reduce Cloud Run max instances
3. **Circuit Breaker:** Disable problematic features via feature flags
4. **Emergency Contact:** Use configured alert channels

---

## 🎯 Next Steps

1. **Customize Configuration:** Update placeholder values in configs
2. **Set Up Secrets:** Populate Secret Manager with real credentials
3. **Configure DNS:** Point your domain to the static IP
4. **Test Deployment:** Run end-to-end tests
5. **Monitor Performance:** Set up dashboards and alerts
6. **Plan Scaling:** Configure auto-scaling based on usage patterns

This deployment setup provides a production-ready, scalable, and secure foundation for the Nexus Forge application on Google Cloud Platform.