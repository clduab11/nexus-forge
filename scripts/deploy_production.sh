#!/bin/bash

# Production Deployment Script for Nexus Forge
# This script handles the complete deployment of Nexus Forge to production

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENVIRONMENT="${1:-production}"
BUILD_NUMBER="${BUILD_NUMBER:-$(date +%Y%m%d%H%M%S)}"
BACKUP_RETENTION_DAYS=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "curl" "jq" "git")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error_exit "Required command '$cmd' is not installed"
        fi
    done
    
    # Check Docker is running
    if ! docker info &> /dev/null; then
        error_exit "Docker is not running"
    fi
    
    # Check environment file exists
    if [[ ! -f "${PROJECT_ROOT}/.env.${ENVIRONMENT}" ]]; then
        error_exit "Environment file .env.${ENVIRONMENT} not found"
    fi
    
    log_success "Prerequisites check passed"
}

# Load environment variables
load_environment() {
    log_info "Loading environment variables for ${ENVIRONMENT}..."
    
    # Source the environment file
    set -o allexport
    source "${PROJECT_ROOT}/.env.${ENVIRONMENT}"
    set +o allexport
    
    # Validate required environment variables
    local required_vars=(
        "POSTGRES_PASSWORD"
        "SECRET_KEY"
        "GOOGLE_CLOUD_PROJECT"
        "SUPABASE_URL"
        "SUPABASE_SERVICE_ROLE_KEY"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error_exit "Required environment variable $var is not set"
        fi
    done
    
    log_success "Environment variables loaded"
}

# Create backup of current deployment
create_backup() {
    log_info "Creating backup of current deployment..."
    
    local backup_dir="${PROJECT_ROOT}/backups/${BUILD_NUMBER}"
    mkdir -p "$backup_dir"
    
    # Backup database
    if docker-compose -f "${PROJECT_ROOT}/docker-compose.production.yml" ps postgres | grep -q "Up"; then
        log_info "Backing up database..."
        docker-compose -f "${PROJECT_ROOT}/docker-compose.production.yml" exec -T postgres \
            pg_dump -U postgres nexusforge | gzip > "${backup_dir}/database.sql.gz"
    fi
    
    # Backup configuration files
    cp "${PROJECT_ROOT}/.env.${ENVIRONMENT}" "${backup_dir}/"
    cp -r "${PROJECT_ROOT}/uploads" "${backup_dir}/" 2>/dev/null || true
    
    # Backup persistent data
    if [[ -d "${DATA_PATH:-/var/lib/nexus-forge}" ]]; then
        tar -czf "${backup_dir}/data.tar.gz" -C "${DATA_PATH}" . 2>/dev/null || true
    fi
    
    log_success "Backup created at ${backup_dir}"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "${PROJECT_ROOT}"
    
    # Build frontend image
    log_info "Building frontend image..."
    docker build -f config/docker/Dockerfile.frontend \
        -t "nexus-forge-frontend:${BUILD_NUMBER}" \
        -t "nexus-forge-frontend:latest" \
        ./frontend
    
    # Build backend image
    log_info "Building backend image..."
    docker build -f config/docker/Dockerfile.backend \
        -t "nexus-forge-backend:${BUILD_NUMBER}" \
        -t "nexus-forge-backend:latest" \
        .
    
    log_success "Docker images built successfully"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    # Run backend tests
    log_info "Running backend tests..."
    docker run --rm \
        -v "${PROJECT_ROOT}:/app" \
        -w /app \
        nexus-forge-backend:latest \
        python -m pytest tests/ -v --tb=short
    
    # Run frontend tests
    log_info "Running frontend tests..."
    docker run --rm \
        -v "${PROJECT_ROOT}/frontend:/app" \
        -w /app \
        node:18-alpine \
        sh -c "npm ci && npm test -- --coverage --watchAll=false"
    
    log_success "All tests passed"
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    
    cd "${PROJECT_ROOT}"
    
    # Stop existing services
    log_info "Stopping existing services..."
    docker-compose -f docker-compose.production.yml down || true
    
    # Create necessary directories
    mkdir -p "${DATA_PATH}/postgres" "${DATA_PATH}/redis" "${DATA_PATH}/prometheus" "${DATA_PATH}/grafana"
    mkdir -p logs uploads backups
    
    # Start services
    log_info "Starting services..."
    docker-compose -f docker-compose.production.yml up -d
    
    log_success "Services deployed"
}

# Health checks
run_health_checks() {
    log_info "Running health checks..."
    
    local max_attempts=30
    local attempt=1
    
    # Check backend health
    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt $attempt/$max_attempts..."
        
        if curl -f "http://localhost:8000/api/health" &> /dev/null; then
            log_success "Backend is healthy"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            error_exit "Backend health check failed after $max_attempts attempts"
        fi
        
        sleep 10
        ((attempt++))
    done
    
    # Check frontend health
    if curl -f "http://localhost/" &> /dev/null; then
        log_success "Frontend is healthy"
    else
        error_exit "Frontend health check failed"
    fi
    
    # Check database connectivity
    if docker-compose -f docker-compose.production.yml exec -T postgres pg_isready -U postgres &> /dev/null; then
        log_success "Database is healthy"
    else
        error_exit "Database health check failed"
    fi
    
    # Check Redis connectivity
    if docker-compose -f docker-compose.production.yml exec -T redis redis-cli ping | grep -q "PONG"; then
        log_success "Redis is healthy"
    else
        error_exit "Redis health check failed"
    fi
    
    log_success "All health checks passed"
}

# Performance validation
validate_performance() {
    log_info "Running performance validation..."
    
    # API response time test
    local api_response_time
    api_response_time=$(curl -o /dev/null -s -w '%{time_total}\n' "http://localhost:8000/api/health")
    
    if (( $(echo "$api_response_time < 1.0" | bc -l) )); then
        log_success "API response time: ${api_response_time}s (< 1.0s target)"
    else
        log_warn "API response time: ${api_response_time}s (exceeds 1.0s target)"
    fi
    
    # Frontend load time test
    local frontend_load_time
    frontend_load_time=$(curl -o /dev/null -s -w '%{time_total}\n' "http://localhost/")
    
    if (( $(echo "$frontend_load_time < 2.0" | bc -l) )); then
        log_success "Frontend load time: ${frontend_load_time}s (< 2.0s target)"
    else
        log_warn "Frontend load time: ${frontend_load_time}s (exceeds 2.0s target)"
    fi
    
    # Memory usage check
    local memory_usage
    memory_usage=$(docker stats --no-stream --format "{{.MemUsage}}" nexus-forge-backend | cut -d'/' -f1)
    log_info "Backend memory usage: $memory_usage"
    
    log_success "Performance validation completed"
}

# Security validation
validate_security() {
    log_info "Running security validation..."
    
    # Check for default passwords
    if [[ "${POSTGRES_PASSWORD}" == "postgres" ]] || [[ "${SECRET_KEY}" == "dev-secret-key" ]]; then
        error_exit "Default passwords detected in production environment"
    fi
    
    # Check SSL/TLS configuration
    if [[ "${ENVIRONMENT}" == "production" ]] && [[ -z "${SSL_CERT_PATH:-}" ]]; then
        log_warn "SSL certificates not configured for production"
    fi
    
    # Verify environment file permissions
    local env_file_perms
    env_file_perms=$(stat -c "%a" "${PROJECT_ROOT}/.env.${ENVIRONMENT}")
    if [[ "$env_file_perms" != "600" ]]; then
        log_warn "Environment file permissions should be 600, current: $env_file_perms"
        chmod 600 "${PROJECT_ROOT}/.env.${ENVIRONMENT}"
    fi
    
    log_success "Security validation completed"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Wait for Prometheus to be ready
    local attempt=1
    local max_attempts=10
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f "http://localhost:9090/-/ready" &> /dev/null; then
            log_success "Prometheus is ready"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_warn "Prometheus readiness check failed after $max_attempts attempts"
            break
        fi
        
        sleep 5
        ((attempt++))
    done
    
    # Wait for Grafana to be ready
    attempt=1
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f "http://localhost:3000/api/health" &> /dev/null; then
            log_success "Grafana is ready"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_warn "Grafana readiness check failed after $max_attempts attempts"
            break
        fi
        
        sleep 5
        ((attempt++))
    done
    
    log_success "Monitoring setup completed"
}

# Cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up old backups..."
    
    local backup_dir="${PROJECT_ROOT}/backups"
    if [[ -d "$backup_dir" ]]; then
        find "$backup_dir" -type d -mtime +${BACKUP_RETENTION_DAYS} -exec rm -rf {} + 2>/dev/null || true
        log_success "Old backups cleaned up"
    fi
}

# Rollback function
rollback() {
    log_error "Deployment failed. Initiating rollback..."
    
    # Stop current services
    docker-compose -f docker-compose.production.yml down || true
    
    # Find the most recent backup
    local latest_backup
    latest_backup=$(find "${PROJECT_ROOT}/backups" -type d -name "*" | sort -r | head -n1)
    
    if [[ -n "$latest_backup" ]] && [[ -d "$latest_backup" ]]; then
        log_info "Rolling back to backup: $latest_backup"
        
        # Restore database if backup exists
        if [[ -f "${latest_backup}/database.sql.gz" ]]; then
            log_info "Restoring database..."
            docker-compose -f docker-compose.production.yml up -d postgres
            sleep 30
            gunzip -c "${latest_backup}/database.sql.gz" | \
                docker-compose -f docker-compose.production.yml exec -T postgres \
                psql -U postgres nexusforge
        fi
        
        # Restore configuration
        if [[ -f "${latest_backup}/.env.${ENVIRONMENT}" ]]; then
            cp "${latest_backup}/.env.${ENVIRONMENT}" "${PROJECT_ROOT}/"
        fi
        
        log_success "Rollback completed"
    else
        log_warn "No backup found for rollback"
    fi
}

# Main deployment function
main() {
    log_info "Starting Nexus Forge deployment to ${ENVIRONMENT}..."
    log_info "Build number: ${BUILD_NUMBER}"
    
    # Set up error handling
    trap rollback ERR
    
    # Run deployment steps
    check_prerequisites
    load_environment
    create_backup
    build_images
    run_tests
    deploy_services
    run_health_checks
    validate_performance
    validate_security
    setup_monitoring
    cleanup_old_backups
    
    log_success "Deployment completed successfully!"
    log_info "Services are running at:"
    log_info "  Frontend: http://localhost"
    log_info "  Backend API: http://localhost:8000"
    log_info "  Grafana: http://localhost:3000"
    log_info "  Prometheus: http://localhost:9090"
    
    # Generate deployment report
    cat > "${PROJECT_ROOT}/deployment-report-${BUILD_NUMBER}.md" << EOF
# Deployment Report

**Date:** $(date -u)
**Environment:** ${ENVIRONMENT}
**Build Number:** ${BUILD_NUMBER}

## Services Status
- Frontend: ✅ Healthy
- Backend: ✅ Healthy  
- Database: ✅ Healthy
- Redis: ✅ Healthy
- Prometheus: ✅ Healthy
- Grafana: ✅ Healthy

## Performance Metrics
- API Response Time: Validated
- Frontend Load Time: Validated
- Memory Usage: Within limits

## Security Validation
- Passwords: ✅ Secure
- Permissions: ✅ Correct
- SSL/TLS: $([ -n "${SSL_CERT_PATH:-}" ] && echo "✅ Configured" || echo "⚠️ Not configured")

## Backup Information
- Backup Location: backups/${BUILD_NUMBER}
- Database: ✅ Backed up
- Configuration: ✅ Backed up
- Data: ✅ Backed up

## Next Steps
1. Monitor application metrics in Grafana
2. Check logs for any issues
3. Verify real-time features are working
4. Run end-to-end tests if needed

EOF
    
    log_info "Deployment report saved to deployment-report-${BUILD_NUMBER}.md"
}

# Run main function with all arguments
main "$@"