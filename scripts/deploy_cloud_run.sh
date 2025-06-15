#!/bin/bash
set -euo pipefail

# Cloud Run Deployment Script for Nexus Forge
# This script handles the complete deployment process to Google Cloud Run

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
SERVICE_NAME="${SERVICE_NAME:-nexus-forge}"
REGION="${GCP_REGION:-us-central1}"
IMAGE_NAME="${IMAGE_NAME:-nexus-forge}"
REGISTRY="${REGISTRY:-gcr.io}"
MIN_INSTANCES="${MIN_INSTANCES:-1}"
MAX_INSTANCES="${MAX_INSTANCES:-100}"
MEMORY="${MEMORY:-2Gi}"
CPU="${CPU:-2}"
TIMEOUT="${TIMEOUT:-300}"
CONCURRENCY="${CONCURRENCY:-1000}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check if PROJECT_ID is set
    if [ -z "$PROJECT_ID" ]; then
        log_error "GCP_PROJECT_ID environment variable is not set."
        exit 1
    fi
    
    # Check if authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        log_error "Not authenticated with gcloud. Run 'gcloud auth login' first."
        exit 1
    fi
    
    log_info "Prerequisites check passed."
}

configure_gcloud() {
    log_info "Configuring gcloud..."
    gcloud config set project "$PROJECT_ID"
    gcloud config set run/region "$REGION"
    
    # Enable required APIs
    log_info "Enabling required APIs..."
    gcloud services enable \
        run.googleapis.com \
        containerregistry.googleapis.com \
        cloudbuild.googleapis.com \
        secretmanager.googleapis.com \
        cloudresourcemanager.googleapis.com
}

build_and_push_image() {
    log_info "Building Docker image..."
    
    # Build with cache and multi-stage optimization
    docker build \
        --cache-from "${REGISTRY}/${PROJECT_ID}/${IMAGE_NAME}:latest" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        -t "${IMAGE_NAME}:latest" \
        -t "${IMAGE_NAME}:${GITHUB_SHA:-latest}" \
        -f Dockerfile \
        .
    
    # Tag for registry
    docker tag "${IMAGE_NAME}:latest" "${REGISTRY}/${PROJECT_ID}/${IMAGE_NAME}:latest"
    docker tag "${IMAGE_NAME}:latest" "${REGISTRY}/${PROJECT_ID}/${IMAGE_NAME}:${GITHUB_SHA:-latest}"
    
    log_info "Pushing image to Container Registry..."
    
    # Configure Docker for GCR
    gcloud auth configure-docker "${REGISTRY}"
    
    # Push images
    docker push "${REGISTRY}/${PROJECT_ID}/${IMAGE_NAME}:latest"
    docker push "${REGISTRY}/${PROJECT_ID}/${IMAGE_NAME}:${GITHUB_SHA:-latest}"
}

create_service_account() {
    log_info "Creating service account if not exists..."
    
    SA_NAME="${SERVICE_NAME}-sa"
    SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
    
    # Create service account if it doesn't exist
    if ! gcloud iam service-accounts describe "$SA_EMAIL" &> /dev/null; then
        gcloud iam service-accounts create "$SA_NAME" \
            --display-name="Cloud Run Service Account for ${SERVICE_NAME}"
    fi
    
    # Grant necessary permissions
    log_info "Granting permissions to service account..."
    
    # Grant roles
    for role in \
        "roles/logging.logWriter" \
        "roles/monitoring.metricWriter" \
        "roles/cloudtrace.agent" \
        "roles/secretmanager.secretAccessor" \
        "roles/storage.objectViewer" \
        "roles/aiplatform.user"
    do
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:${SA_EMAIL}" \
            --role="$role" \
            --quiet
    done
    
    echo "$SA_EMAIL"
}

deploy_to_cloud_run() {
    log_info "Deploying to Cloud Run..."
    
    SA_EMAIL=$(create_service_account)
    
    # Deploy with all production settings
    gcloud run deploy "$SERVICE_NAME" \
        --image="${REGISTRY}/${PROJECT_ID}/${IMAGE_NAME}:${GITHUB_SHA:-latest}" \
        --platform=managed \
        --region="$REGION" \
        --memory="$MEMORY" \
        --cpu="$CPU" \
        --timeout="$TIMEOUT" \
        --concurrency="$CONCURRENCY" \
        --min-instances="$MIN_INSTANCES" \
        --max-instances="$MAX_INSTANCES" \
        --service-account="$SA_EMAIL" \
        --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID}" \
        --set-env-vars="ENVIRONMENT=production" \
        --set-env-vars="LOG_LEVEL=info" \
        --set-secrets="GOOGLE_APPLICATION_CREDENTIALS=nexus-forge-credentials:latest" \
        --allow-unauthenticated \
        --cpu-throttling \
        --execution-environment=gen2 \
        --ingress=all \
        --labels="app=nexus-forge,environment=production,version=${GITHUB_SHA:-latest}" \
        --revision-suffix="${GITHUB_SHA:-latest}" \
        --tag="latest" \
        --no-traffic
    
    log_info "Service deployed successfully!"
}

run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
        --region="$REGION" \
        --format="value(status.url)")
    
    if [ -z "$SERVICE_URL" ]; then
        log_error "Failed to get service URL"
        exit 1
    fi
    
    log_info "Service URL: $SERVICE_URL"
    
    # Health check
    log_info "Performing health check..."
    HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/health")
    
    if [ "$HEALTH_RESPONSE" -eq 200 ]; then
        log_info "Health check passed!"
    else
        log_error "Health check failed with status: $HEALTH_RESPONSE"
        exit 1
    fi
    
    # API check
    log_info "Checking API endpoint..."
    API_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/api/v1/health")
    
    if [ "$API_RESPONSE" -eq 200 ]; then
        log_info "API check passed!"
    else
        log_warn "API check returned status: $API_RESPONSE"
    fi
}

promote_traffic() {
    log_info "Promoting traffic to new revision..."
    
    # Gradually migrate traffic
    if [ "${GRADUAL_ROLLOUT:-false}" = "true" ]; then
        # 10% traffic initially
        gcloud run services update-traffic "$SERVICE_NAME" \
            --region="$REGION" \
            --to-revisions="LATEST=10"
        
        log_info "10% traffic migrated. Monitor for issues..."
        sleep 60
        
        # 50% traffic
        gcloud run services update-traffic "$SERVICE_NAME" \
            --region="$REGION" \
            --to-revisions="LATEST=50"
        
        log_info "50% traffic migrated. Monitor for issues..."
        sleep 120
    fi
    
    # Full traffic migration
    gcloud run services update-traffic "$SERVICE_NAME" \
        --region="$REGION" \
        --to-latest
    
    log_info "Traffic migration complete!"
}

cleanup_old_revisions() {
    log_info "Cleaning up old revisions..."
    
    # Keep only the last 3 revisions
    REVISIONS=$(gcloud run revisions list \
        --service="$SERVICE_NAME" \
        --region="$REGION" \
        --format="value(name)" \
        --sort-by="~creationTimestamp" | tail -n +4)
    
    if [ -n "$REVISIONS" ]; then
        for revision in $REVISIONS; do
            log_info "Deleting revision: $revision"
            gcloud run revisions delete "$revision" \
                --region="$REGION" \
                --quiet || true
        done
    fi
}

main() {
    log_info "Starting Cloud Run deployment for ${SERVICE_NAME}..."
    
    check_prerequisites
    configure_gcloud
    build_and_push_image
    deploy_to_cloud_run
    run_smoke_tests
    
    if [ "${SKIP_TRAFFIC_MIGRATION:-false}" != "true" ]; then
        promote_traffic
    fi
    
    cleanup_old_revisions
    
    log_info "Deployment completed successfully!"
    
    # Output service details
    gcloud run services describe "$SERVICE_NAME" \
        --region="$REGION" \
        --format="table(status.url,spec.template.spec.containers[0].image)"
}

# Run main function
main "$@"