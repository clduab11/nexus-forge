# Nexus Forge Deployment Guide

## 🚀 Production Deployment

This guide covers deploying Nexus Forge to Google Cloud Platform for the Multi-Agent Hackathon submission.

## 📋 Prerequisites

### Required Accounts & Services
- Google Cloud Platform account with billing enabled
- Google Cloud Project with the following APIs enabled:
  - Vertex AI API
  - Cloud Run API
  - Cloud SQL API
  - Cloud Storage API
  - Identity and Access Management (IAM) API
  - Artifact Registry API

### Required Tools
```bash
# Google Cloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Docker
# Install from https://docs.docker.com/get-docker/

# Python 3.11+
python --version  # Should be 3.11 or higher

# Node.js 18+
node --version    # Should be 18 or higher
```

## 🔧 Environment Setup

### 1. Google Cloud Configuration

```bash
# Set your project ID
export PROJECT_ID="your-nexus-forge-project"
export REGION="us-central1"

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable \
  aiplatform.googleapis.com \
  run.googleapis.com \
  sql-component.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com
```

### 2. Service Account Creation

```bash
# Create service account for Nexus Forge
gcloud iam service-accounts create nexus-forge-sa \
  --description="Service account for Nexus Forge application" \
  --display-name="Nexus Forge SA"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:nexus-forge-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:nexus-forge-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:nexus-forge-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client"

# Create and download service account key
gcloud iam service-accounts keys create nexus-forge-key.json \
  --iam-account=nexus-forge-sa@$PROJECT_ID.iam.gserviceaccount.com
```

### 3. Database Setup

```bash
# Create Cloud SQL instance
gcloud sql instances create nexus-forge-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=$REGION \
  --storage-type=SSD \
  --storage-size=20GB \
  --backup-start-time=03:00

# Create database
gcloud sql databases create nexusforge \
  --instance=nexus-forge-db

# Create database user
gcloud sql users create nexusforge \
  --instance=nexus-forge-db \
  --password=GENERATE_SECURE_PASSWORD_HERE
```

### 4. Storage Bucket Setup

```bash
# Create bucket for generated content
gsutil mb gs://$PROJECT_ID-nexus-forge-content

# Set bucket permissions
gsutil iam ch serviceAccount:nexus-forge-sa@$PROJECT_ID.iam.gserviceaccount.com:objectAdmin gs://$PROJECT_ID-nexus-forge-content
```

## 🔑 Environment Variables

Create a `.env.production` file:

```bash
# Application
APP_NAME=Nexus Forge
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=generate-strong-secret-key-here

# Database
DATABASE_URL=postgresql+asyncpg://nexusforge:PASSWORD@/nexusforge?host=/cloudsql/PROJECT_ID:REGION:nexus-forge-db

# Redis (Cloud Memorystore)
REDIS_URL=redis://redis-instance-ip:6379/0

# Google Cloud
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/app/nexus-forge-key.json

# AI Models Configuration
GEMINI_MODEL_NAME=gemini-2.5-pro
GEMINI_FLASH_MODEL_NAME=gemini-2.5-flash
VEO_MODEL_NAME=veo-3
IMAGEN_MODEL_NAME=imagen-4

# Storage
STORAGE_BUCKET=your-project-id-nexus-forge-content

# Frontend
FRONTEND_URL=https://nexusforge.example.com
CORS_ORIGINS=https://nexusforge.example.com

# Security
ALLOWED_HOSTS=nexusforge.example.com
SECURE_HEADERS=true
RATE_LIMIT_PER_MINUTE=120

# OAuth (configure with your providers)
GOOGLE_CLIENT_ID=your-google-oauth-client-id
GOOGLE_CLIENT_SECRET=your-google-oauth-client-secret
GOOGLE_REDIRECT_URI=https://nexusforge.example.com/auth/google/callback

# Monitoring
SENTRY_DSN=your-sentry-dsn
ENABLE_METRICS=true
METRICS_PORT=9090
```

## 🐳 Docker Configuration

### Backend Dockerfile
```dockerfile
# /Dockerfile.backend
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY nexus-forge-key.json ./

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile
```dockerfile
# /frontend/Dockerfile
FROM node:18-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## 🚢 Cloud Run Deployment

### 1. Build and Push Images

```bash
# Configure Artifact Registry
gcloud artifacts repositories create nexus-forge \
  --repository-format=docker \
  --location=$REGION

# Configure Docker authentication
gcloud auth configure-docker $REGION-docker.pkg.dev

# Build and push backend
docker build -f Dockerfile.backend -t $REGION-docker.pkg.dev/$PROJECT_ID/nexus-forge/backend:latest .
docker push $REGION-docker.pkg.dev/$PROJECT_ID/nexus-forge/backend:latest

# Build and push frontend
cd frontend
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/nexus-forge/frontend:latest .
docker push $REGION-docker.pkg.dev/$PROJECT_ID/nexus-forge/frontend:latest
cd ..
```

### 2. Deploy Backend to Cloud Run

```bash
# Deploy backend
gcloud run deploy nexus-forge-backend \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/nexus-forge/backend:latest \
  --platform=managed \
  --region=$REGION \
  --service-account=nexus-forge-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=1 \
  --max-instances=10 \
  --timeout=300 \
  --add-cloudsql-instances=$PROJECT_ID:$REGION:nexus-forge-db \
  --set-env-vars="ENVIRONMENT=production,GOOGLE_CLOUD_PROJECT=$PROJECT_ID"
```

### 3. Deploy Frontend to Cloud Run

```bash
# Deploy frontend
gcloud run deploy nexus-forge-frontend \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/nexus-forge/frontend:latest \
  --platform=managed \
  --region=$REGION \
  --allow-unauthenticated \
  --memory=512Mi \
  --cpu=1 \
  --min-instances=1 \
  --max-instances=5
```

## 🔧 Database Migration

```bash
# Run database migrations
gcloud run jobs create nexus-forge-migrate \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/nexus-forge/backend:latest \
  --command="python" \
  --args="-m,alembic,upgrade,head" \
  --service-account=nexus-forge-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --add-cloudsql-instances=$PROJECT_ID:$REGION:nexus-forge-db \
  --set-env-vars="ENVIRONMENT=production,GOOGLE_CLOUD_PROJECT=$PROJECT_ID"

# Execute migration
gcloud run jobs execute nexus-forge-migrate --region=$REGION
```

## 🌐 Domain and SSL Setup

### 1. Domain Mapping

```bash
# Map custom domain to backend
gcloud run domain-mappings create \
  --service=nexus-forge-backend \
  --domain=api.nexusforge.example.com \
  --region=$REGION

# Map custom domain to frontend
gcloud run domain-mappings create \
  --service=nexus-forge-frontend \
  --domain=nexusforge.example.com \
  --region=$REGION
```

### 2. DNS Configuration

Add the following DNS records to your domain:

```
# For frontend
nexusforge.example.com    CNAME    ghs.googlehosted.com.

# For API
api.nexusforge.example.com    CNAME    ghs.googlehosted.com.
```

## 📊 Monitoring Setup

### 1. Enable Cloud Monitoring

```bash
# Install monitoring agent (if using custom metrics)
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com
```

### 2. Create Alerting Policies

```bash
# Create uptime check
gcloud alpha monitoring uptime create nexus-forge-uptime \
  --hostname=nexusforge.example.com \
  --path=/health \
  --check-interval=60s

# Create error rate alert
gcloud alpha monitoring policies create \
  --policy-from-file=monitoring/error-rate-policy.yaml
```

## 🔒 Security Configuration

### 1. Cloud Armor (WAF)

```bash
# Create security policy
gcloud compute security-policies create nexus-forge-security-policy \
  --description="Security policy for Nexus Forge"

# Add rate limiting rule
gcloud compute security-policies rules create 1000 \
  --security-policy=nexus-forge-security-policy \
  --action=rate-based-ban \
  --rate-limit-threshold-count=100 \
  --rate-limit-threshold-interval-sec=60 \
  --ban-duration-sec=600
```

### 2. Secret Manager

```bash
# Store sensitive configuration
echo -n "your-secret-key" | gcloud secrets create nexus-forge-secret-key --data-file=-
echo -n "your-db-password" | gcloud secrets create nexus-forge-db-password --data-file=-

# Grant access to service account
gcloud secrets add-iam-policy-binding nexus-forge-secret-key \
  --member="serviceAccount:nexus-forge-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## 🧪 Testing Deployment

### 1. Health Checks

```bash
# Test backend health
curl https://api.nexusforge.example.com/health

# Expected response:
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:00:00Z",
  "environment": "production"
}
```

### 2. End-to-End Test

```bash
# Test app building workflow
curl -X POST https://api.nexusforge.example.com/api/nexus-forge/build \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Build a simple todo app with React and FastAPI",
    "config": {
      "useAdaptiveThinking": true,
      "enableVideoDemo": true,
      "deployToCloudRun": true
    }
  }'
```

## 📈 Performance Optimization

### 1. Auto-scaling Configuration

```bash
# Update backend with optimized scaling
gcloud run services update nexus-forge-backend \
  --region=$REGION \
  --cpu-throttling \
  --concurrency=100 \
  --min-instances=2 \
  --max-instances=20
```

### 2. CDN Setup

```bash
# Create Cloud CDN for static assets
gcloud compute backend-services create nexus-forge-cdn-backend \
  --global \
  --enable-cdn

# Configure cache settings
gcloud compute backend-services update nexus-forge-cdn-backend \
  --global \
  --cache-mode=CACHE_ALL_STATIC \
  --default-ttl=3600
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - id: 'auth'
      uses: 'google-github-actions/auth@v1'
      with:
        credentials_json: '${{ secrets.GCP_SA_KEY }}'
    
    - name: 'Set up Cloud SDK'
      uses: 'google-github-actions/setup-gcloud@v1'
    
    - name: 'Docker auth'
      run: gcloud auth configure-docker ${{ vars.REGION }}-docker.pkg.dev
    
    - name: 'Build and push backend'
      run: |
        docker build -f Dockerfile.backend -t ${{ vars.REGION }}-docker.pkg.dev/${{ vars.PROJECT_ID }}/nexus-forge/backend:${{ github.sha }} .
        docker push ${{ vars.REGION }}-docker.pkg.dev/${{ vars.PROJECT_ID }}/nexus-forge/backend:${{ github.sha }}
    
    - name: 'Deploy to Cloud Run'
      run: |
        gcloud run deploy nexus-forge-backend \
          --image=${{ vars.REGION }}-docker.pkg.dev/${{ vars.PROJECT_ID }}/nexus-forge/backend:${{ github.sha }} \
          --region=${{ vars.REGION }} \
          --platform=managed
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Build Timeouts
```bash
# Increase Cloud Run timeout
gcloud run services update nexus-forge-backend \
  --region=$REGION \
  --timeout=900  # 15 minutes
```

#### 2. Memory Issues
```bash
# Increase memory allocation
gcloud run services update nexus-forge-backend \
  --region=$REGION \
  --memory=4Gi
```

#### 3. Database Connection Issues
```bash
# Check Cloud SQL connectivity
gcloud sql instances describe nexus-forge-db

# Test connection
gcloud sql connect nexus-forge-db --user=nexusforge
```

### Logs and Debugging

```bash
# View Cloud Run logs
gcloud run services logs tail nexus-forge-backend --region=$REGION

# Check specific deployment
gcloud run revisions list --service=nexus-forge-backend --region=$REGION

# Debug specific revision
gcloud run revisions describe REVISION_NAME --region=$REGION
```

## 📊 Cost Optimization

### 1. Resource Right-sizing

```bash
# Monitor usage and adjust resources
gcloud monitoring dashboards list

# Optimize based on metrics
gcloud run services update nexus-forge-backend \
  --region=$REGION \
  --cpu=1 \
  --memory=1Gi \
  --min-instances=0  # Scale to zero when not in use
```

### 2. Budget Alerts

```bash
# Create budget alert
gcloud alpha billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="Nexus Forge Budget" \
  --budget-amount=100USD \
  --threshold-rule=percent:50 \
  --threshold-rule=percent:90
```

---

## 🎯 Production Checklist

- [ ] All APIs enabled in Google Cloud
- [ ] Service account created with proper permissions
- [ ] Database instance created and accessible
- [ ] Storage bucket configured
- [ ] Environment variables set
- [ ] Docker images built and pushed
- [ ] Cloud Run services deployed
- [ ] Domain mapping configured
- [ ] SSL certificates provisioned
- [ ] Monitoring and alerting set up
- [ ] Security policies configured
- [ ] CI/CD pipeline operational
- [ ] Load testing completed
- [ ] Backup strategy implemented
- [ ] Documentation updated

*This deployment guide ensures Nexus Forge is production-ready for the Google Cloud Multi-Agent Hackathon demonstration.*