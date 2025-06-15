#!/bin/bash
set -euo pipefail

# GCP Infrastructure Setup Script for Nexus Forge
# Creates all necessary GCP resources for production deployment

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-a}"
SERVICE_NAME="${SERVICE_NAME:-nexus-forge}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

log_section() {
    echo -e "\n${BLUE}===== $1 =====${NC}"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed."
        exit 1
    fi
    
    if [ -z "$PROJECT_ID" ]; then
        log_error "GCP_PROJECT_ID environment variable is not set."
        exit 1
    fi
    
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        log_error "Not authenticated with gcloud. Run 'gcloud auth login' first."
        exit 1
    fi
    
    log_info "Prerequisites check passed."
}

setup_project() {
    log_section "Setting up GCP Project"
    
    log_info "Setting project: $PROJECT_ID"
    gcloud config set project "$PROJECT_ID"
    
    log_info "Setting default region: $REGION"
    gcloud config set compute/region "$REGION"
    gcloud config set compute/zone "$ZONE"
    gcloud config set run/region "$REGION"
    
    # Verify project exists and is accessible
    if ! gcloud projects describe "$PROJECT_ID" &> /dev/null; then
        log_error "Project $PROJECT_ID does not exist or is not accessible."
        exit 1
    fi
    
    log_info "Project setup complete."
}

enable_apis() {
    log_section "Enabling Required APIs"
    
    APIS=(
        "compute.googleapis.com"
        "container.googleapis.com"
        "run.googleapis.com"
        "cloudbuild.googleapis.com"
        "containerregistry.googleapis.com"
        "secretmanager.googleapis.com"
        "monitoring.googleapis.com"
        "logging.googleapis.com"
        "cloudresourcemanager.googleapis.com"
        "iam.googleapis.com"
        "aiplatform.googleapis.com"
        "storage.googleapis.com"
        "cloudkms.googleapis.com"
        "cloudscheduler.googleapis.com"
        "pubsub.googleapis.com"
        "firestore.googleapis.com"
        "redis.googleapis.com"
        "sqladmin.googleapis.com"
        "servicenetworking.googleapis.com"
    )
    
    for api in "${APIS[@]}"; do
        log_info "Enabling $api..."
        gcloud services enable "$api"
    done
    
    log_info "All APIs enabled successfully."
}

create_service_accounts() {
    log_section "Creating Service Accounts"
    
    # Cloud Run service account
    SA_NAME="${SERVICE_NAME}-runtime-sa"
    SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
    
    if ! gcloud iam service-accounts describe "$SA_EMAIL" &> /dev/null; then
        log_info "Creating Cloud Run service account: $SA_NAME"
        gcloud iam service-accounts create "$SA_NAME" \
            --display-name="Nexus Forge Cloud Run Runtime Service Account" \
            --description="Service account for Cloud Run runtime operations"
    else
        log_info "Cloud Run service account already exists: $SA_NAME"
    fi
    
    # Cloud Build service account
    BUILD_SA_NAME="${SERVICE_NAME}-build-sa"
    BUILD_SA_EMAIL="${BUILD_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
    
    if ! gcloud iam service-accounts describe "$BUILD_SA_EMAIL" &> /dev/null; then
        log_info "Creating Cloud Build service account: $BUILD_SA_NAME"
        gcloud iam service-accounts create "$BUILD_SA_NAME" \
            --display-name="Nexus Forge Cloud Build Service Account" \
            --description="Service account for Cloud Build operations"
    else
        log_info "Cloud Build service account already exists: $BUILD_SA_NAME"
    fi
    
    # Monitoring service account
    MONITOR_SA_NAME="${SERVICE_NAME}-monitor-sa"
    MONITOR_SA_EMAIL="${MONITOR_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
    
    if ! gcloud iam service-accounts describe "$MONITOR_SA_EMAIL" &> /dev/null; then
        log_info "Creating monitoring service account: $MONITOR_SA_NAME"
        gcloud iam service-accounts create "$MONITOR_SA_NAME" \
            --display-name="Nexus Forge Monitoring Service Account" \
            --description="Service account for monitoring and alerting"
    else
        log_info "Monitoring service account already exists: $MONITOR_SA_NAME"
    fi
}

setup_iam_roles() {
    log_section "Setting up IAM Roles"
    
    SA_EMAIL="${SERVICE_NAME}-runtime-sa@${PROJECT_ID}.iam.gserviceaccount.com"
    BUILD_SA_EMAIL="${SERVICE_NAME}-build-sa@${PROJECT_ID}.iam.gserviceaccount.com"
    MONITOR_SA_EMAIL="${SERVICE_NAME}-monitor-sa@${PROJECT_ID}.iam.gserviceaccount.com"
    
    # Runtime service account roles
    RUNTIME_ROLES=(
        "roles/logging.logWriter"
        "roles/monitoring.metricWriter"
        "roles/cloudtrace.agent"
        "roles/secretmanager.secretAccessor"
        "roles/storage.objectViewer"
        "roles/aiplatform.user"
        "roles/pubsub.publisher"
        "roles/pubsub.subscriber"
        "roles/datastore.user"
    )
    
    for role in "${RUNTIME_ROLES[@]}"; do
        log_info "Granting $role to runtime service account"
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:${SA_EMAIL}" \
            --role="$role" \
            --quiet
    done
    
    # Build service account roles
    BUILD_ROLES=(
        "roles/cloudbuild.builds.builder"
        "roles/storage.admin"
        "roles/containeranalysis.admin"
        "roles/source.reader"
        "roles/run.developer"
    )
    
    for role in "${BUILD_ROLES[@]}"; do
        log_info "Granting $role to build service account"
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:${BUILD_SA_EMAIL}" \
            --role="$role" \
            --quiet
    done
    
    # Monitoring service account roles
    MONITOR_ROLES=(
        "roles/monitoring.metricWriter"
        "roles/monitoring.dashboardEditor"
        "roles/logging.logWriter"
        "roles/cloudscheduler.admin"
        "roles/pubsub.admin"
    )
    
    for role in "${MONITOR_ROLES[@]}"; do
        log_info "Granting $role to monitoring service account"
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:${MONITOR_SA_EMAIL}" \
            --role="$role" \
            --quiet
    done
}

create_storage_buckets() {
    log_section "Creating Storage Buckets"
    
    # Application storage bucket
    APP_BUCKET="${PROJECT_ID}-nexus-forge-storage"
    if ! gsutil ls "gs://${APP_BUCKET}" &> /dev/null; then
        log_info "Creating application storage bucket: $APP_BUCKET"
        gsutil mb -l "$REGION" "gs://${APP_BUCKET}"
        
        # Set bucket permissions
        gsutil iam ch "serviceAccount:${SERVICE_NAME}-runtime-sa@${PROJECT_ID}.iam.gserviceaccount.com:objectViewer" \
            "gs://${APP_BUCKET}"
    else
        log_info "Application storage bucket already exists: $APP_BUCKET"
    fi
    
    # Build artifacts bucket
    BUILD_BUCKET="${PROJECT_ID}-nexus-forge-builds"
    if ! gsutil ls "gs://${BUILD_BUCKET}" &> /dev/null; then
        log_info "Creating build artifacts bucket: $BUILD_BUCKET"
        gsutil mb -l "$REGION" "gs://${BUILD_BUCKET}"
        
        # Set lifecycle policy to delete old builds
        cat > /tmp/lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30}
      }
    ]
  }
}
EOF
        gsutil lifecycle set /tmp/lifecycle.json "gs://${BUILD_BUCKET}"
        rm /tmp/lifecycle.json
    else
        log_info "Build artifacts bucket already exists: $BUILD_BUCKET"
    fi
}

setup_secrets() {
    log_section "Setting up Secret Manager"
    
    # Google Cloud credentials secret
    SECRET_NAME="nexus-forge-credentials"
    if ! gcloud secrets describe "$SECRET_NAME" &> /dev/null; then
        log_info "Creating secret: $SECRET_NAME"
        gcloud secrets create "$SECRET_NAME" \
            --labels="app=nexus-forge,environment=production"
        
        # Create a placeholder service account key
        SA_EMAIL="${SERVICE_NAME}-runtime-sa@${PROJECT_ID}.iam.gserviceaccount.com"
        gcloud iam service-accounts keys create /tmp/sa-key.json \
            --iam-account="$SA_EMAIL"
        
        gcloud secrets versions add "$SECRET_NAME" \
            --data-file="/tmp/sa-key.json"
        
        rm /tmp/sa-key.json
        
        log_info "Secret created and populated with service account key"
    else
        log_info "Secret already exists: $SECRET_NAME"
    fi
    
    # Database connection string secret (if using Cloud SQL)
    DB_SECRET_NAME="nexus-forge-db-connection"
    if ! gcloud secrets describe "$DB_SECRET_NAME" &> /dev/null; then
        log_info "Creating database secret: $DB_SECRET_NAME"
        gcloud secrets create "$DB_SECRET_NAME" \
            --labels="app=nexus-forge,environment=production"
        
        # Add placeholder value
        echo "postgresql://user:password@localhost:5432/nexus_forge" | \
            gcloud secrets versions add "$DB_SECRET_NAME" --data-file=-
    else
        log_info "Database secret already exists: $DB_SECRET_NAME"
    fi
    
    # API keys secret
    API_SECRET_NAME="nexus-forge-api-keys"
    if ! gcloud secrets describe "$API_SECRET_NAME" &> /dev/null; then
        log_info "Creating API keys secret: $API_SECRET_NAME"
        gcloud secrets create "$API_SECRET_NAME" \
            --labels="app=nexus-forge,environment=production"
        
        # Add placeholder JSON
        echo '{"gemini_api_key": "", "other_api_key": ""}' | \
            gcloud secrets versions add "$API_SECRET_NAME" --data-file=-
    else
        log_info "API keys secret already exists: $API_SECRET_NAME"
    fi
}

setup_monitoring() {
    log_section "Setting up Monitoring and Logging"
    
    # Create custom metrics (will be implemented in monitoring script)
    log_info "Monitoring resources will be created by deploy_monitoring.sh"
    
    # Set up log sinks for important logs
    SINK_NAME="nexus-forge-error-sink"
    if ! gcloud logging sinks describe "$SINK_NAME" &> /dev/null; then
        log_info "Creating error log sink: $SINK_NAME"
        
        # Create Pub/Sub topic for error alerts
        ERROR_TOPIC="nexus-forge-errors"
        if ! gcloud pubsub topics describe "$ERROR_TOPIC" &> /dev/null; then
            gcloud pubsub topics create "$ERROR_TOPIC"
        fi
        
        # Create log sink
        gcloud logging sinks create "$SINK_NAME" \
            "pubsub.googleapis.com/projects/${PROJECT_ID}/topics/${ERROR_TOPIC}" \
            --log-filter='severity>=ERROR AND resource.type="cloud_run_revision" AND resource.labels.service_name="nexus-forge"'
        
        # Grant permissions to sink service account
        SINK_SA=$(gcloud logging sinks describe "$SINK_NAME" --format="value(writerIdentity)")
        gcloud pubsub topics add-iam-policy-binding "$ERROR_TOPIC" \
            --member="$SINK_SA" \
            --role="roles/pubsub.publisher"
    else
        log_info "Error log sink already exists: $SINK_NAME"
    fi
}

setup_networking() {
    log_section "Setting up Networking"
    
    # Reserve static IP for Load Balancer (if needed)
    IP_NAME="nexus-forge-ip"
    if ! gcloud compute addresses describe "$IP_NAME" --global &> /dev/null; then
        log_info "Reserving global static IP: $IP_NAME"
        gcloud compute addresses create "$IP_NAME" --global
        
        IP_ADDRESS=$(gcloud compute addresses describe "$IP_NAME" --global --format="value(address)")
        log_info "Reserved IP address: $IP_ADDRESS"
    else
        log_info "Static IP already exists: $IP_NAME"
        IP_ADDRESS=$(gcloud compute addresses describe "$IP_NAME" --global --format="value(address)")
        log_info "Existing IP address: $IP_ADDRESS"
    fi
}

create_firestore_database() {
    log_section "Setting up Firestore Database"
    
    # Check if Firestore database exists
    if ! gcloud firestore databases describe --database="(default)" &> /dev/null; then
        log_info "Creating Firestore database..."
        gcloud firestore databases create \
            --location="$REGION" \
            --type=firestore-native
        
        log_info "Firestore database created successfully"
    else
        log_info "Firestore database already exists"
    fi
}

setup_cloud_scheduler() {
    log_section "Setting up Cloud Scheduler"
    
    # Create App Engine app (required for Cloud Scheduler)
    if ! gcloud app describe &> /dev/null; then
        log_info "Creating App Engine application..."
        gcloud app create --region="$REGION"
    else
        log_info "App Engine application already exists"
    fi
    
    # Scheduled jobs will be created by the monitoring script
    log_info "Scheduled jobs will be configured by deploy_monitoring.sh"
}

print_summary() {
    log_section "Infrastructure Setup Summary"
    
    log_info "Project ID: $PROJECT_ID"
    log_info "Region: $REGION"
    log_info "Service Name: $SERVICE_NAME"
    log_info ""
    log_info "Created Resources:"
    log_info "- Service Accounts: ${SERVICE_NAME}-runtime-sa, ${SERVICE_NAME}-build-sa, ${SERVICE_NAME}-monitor-sa"
    log_info "- Storage Buckets: ${PROJECT_ID}-nexus-forge-storage, ${PROJECT_ID}-nexus-forge-builds"
    log_info "- Secrets: nexus-forge-credentials, nexus-forge-db-connection, nexus-forge-api-keys"
    log_info "- Static IP: nexus-forge-ip"
    log_info "- Firestore Database: (default)"
    log_info "- Log Sink: nexus-forge-error-sink"
    log_info ""
    log_info "Next Steps:"
    log_info "1. Update secrets with actual values"
    log_info "2. Run deploy_monitoring.sh to set up monitoring"
    log_info "3. Run deploy_cloud_run.sh to deploy the application"
}

main() {
    log_info "Starting GCP infrastructure setup for Nexus Forge..."
    
    check_prerequisites
    setup_project
    enable_apis
    create_service_accounts
    setup_iam_roles
    create_storage_buckets
    setup_secrets
    setup_monitoring
    setup_networking
    create_firestore_database
    setup_cloud_scheduler
    print_summary
    
    log_info "Infrastructure setup completed successfully!"
}

# Run main function
main "$@"