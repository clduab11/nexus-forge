#!/bin/bash
set -euo pipefail

# Deployment Validation Script for Nexus Forge
# Validates all deployment configurations and infrastructure setup

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
PASSED=0
FAILED=0

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

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

check_warn() {
    echo -e "${YELLOW}!${NC} $1"
}

# Validation functions
validate_scripts() {
    log_section "Validating Deployment Scripts"
    
    # Check if scripts exist
    SCRIPTS=(
        "scripts/deploy_cloud_run.sh"
        "scripts/setup_gcp_resources.sh"
        "scripts/deploy_monitoring.sh"
    )
    
    for script in "${SCRIPTS[@]}"; do
        if [ -f "$script" ]; then
            check_pass "Script exists: $script"
        else
            check_fail "Script missing: $script"
            continue
        fi
        
        # Check if executable
        if [ -x "$script" ]; then
            check_pass "Script executable: $script"
        else
            check_fail "Script not executable: $script"
        fi
        
        # Check syntax
        if bash -n "$script" 2>/dev/null; then
            check_pass "Script syntax valid: $script"
        else
            check_fail "Script syntax invalid: $script"
        fi
    done
}

validate_docker() {
    log_section "Validating Docker Configuration"
    
    if [ -f "Dockerfile" ]; then
        check_pass "Dockerfile exists"
        
        # Check for multi-stage build
        if grep -q "FROM.*as.*base" Dockerfile; then
            check_pass "Multi-stage build configured"
        else
            check_warn "Multi-stage build not detected"
        fi
        
        # Check for security practices
        if grep -q "USER.*nexus" Dockerfile; then
            check_pass "Non-root user configured"
        else
            check_fail "Root user detected in Dockerfile"
        fi
        
        # Check for health check
        if grep -q "HEALTHCHECK" Dockerfile; then
            check_pass "Health check configured"
        else
            check_warn "Health check not configured"
        fi
        
        # Check for production target
        if grep -q "FROM.*as production" Dockerfile; then
            check_pass "Production target configured"
        else
            check_warn "Production target not found"
        fi
        
    else
        check_fail "Dockerfile missing"
    fi
}

validate_kubernetes() {
    log_section "Validating Kubernetes Manifests"
    
    K8S_FILES=(
        "k8s/deployment.yml"
        "k8s/service.yml"
        "k8s/ingress.yml"
        "k8s/configmap.yml"
    )
    
    for file in "${K8S_FILES[@]}"; do
        if [ -f "$file" ]; then
            check_pass "K8s manifest exists: $file"
            
            # Validate YAML syntax
            if python3 -c "import yaml; yaml.safe_load(open('$file'))" 2>/dev/null; then
                check_pass "YAML syntax valid: $file"
            else
                check_fail "YAML syntax invalid: $file"
            fi
            
        else
            check_fail "K8s manifest missing: $file"
        fi
    done
    
    # Check for security configurations
    if [ -f "k8s/deployment.yml" ]; then
        if grep -q "runAsNonRoot: true" k8s/deployment.yml; then
            check_pass "Security context configured"
        else
            check_warn "Security context not found"
        fi
        
        if grep -q "readOnlyRootFilesystem: true" k8s/deployment.yml; then
            check_pass "Read-only root filesystem configured"
        else
            check_warn "Read-only root filesystem not configured"
        fi
        
        if grep -q "HorizontalPodAutoscaler" k8s/deployment.yml; then
            check_pass "Horizontal Pod Autoscaler configured"
        else
            check_warn "HPA not configured"
        fi
    fi
}

validate_config() {
    log_section "Validating Configuration Files"
    
    if [ -f "config/production.yml" ]; then
        check_pass "Production config exists"
        
        # Validate YAML syntax
        if python3 -c "import yaml; yaml.safe_load(open('config/production.yml'))" 2>/dev/null; then
            check_pass "Production config YAML valid"
        else
            check_fail "Production config YAML invalid"
        fi
        
        # Check for required sections
        if grep -q "app:" config/production.yml; then
            check_pass "App configuration section found"
        else
            check_fail "App configuration section missing"
        fi
        
        if grep -q "database:" config/production.yml; then
            check_pass "Database configuration section found"
        else
            check_fail "Database configuration section missing"
        fi
        
        if grep -q "gcp:" config/production.yml; then
            check_pass "GCP configuration section found"
        else
            check_fail "GCP configuration section missing"
        fi
        
        if grep -q "monitoring:" config/production.yml; then
            check_pass "Monitoring configuration section found"
        else
            check_fail "Monitoring configuration section missing"
        fi
        
    else
        check_fail "Production config missing"
    fi
}

validate_terraform() {
    log_section "Validating Terraform Configuration"
    
    if [ -f "terraform/main.tf" ]; then
        check_pass "Terraform main.tf exists"
        
        # Check for required providers
        if grep -q "provider.*google" terraform/main.tf; then
            check_pass "Google provider configured"
        else
            check_fail "Google provider missing"
        fi
        
        # Check for state backend
        if grep -q "backend.*gcs" terraform/main.tf; then
            check_pass "GCS backend configured"
        else
            check_warn "GCS backend not configured"
        fi
        
        # Check for resource definitions
        if grep -q "resource.*google_service_account" terraform/main.tf; then
            check_pass "Service accounts configured"
        else
            check_fail "Service accounts missing"
        fi
        
        if grep -q "resource.*google_storage_bucket" terraform/main.tf; then
            check_pass "Storage buckets configured"
        else
            check_fail "Storage buckets missing"
        fi
        
        if grep -q "resource.*google_secret_manager_secret" terraform/main.tf; then
            check_pass "Secret Manager configured"
        else
            check_fail "Secret Manager missing"
        fi
        
    else
        check_fail "Terraform main.tf missing"
    fi
}

validate_ci_cd() {
    log_section "Validating CI/CD Pipeline"
    
    if [ -f ".github/workflows/deploy.yml" ]; then
        check_pass "GitHub Actions workflow exists"
        
        # Check for required jobs
        if grep -q "jobs:" .github/workflows/deploy.yml; then
            check_pass "Jobs section found"
        else
            check_fail "Jobs section missing"
        fi
        
        if grep -q "test:" .github/workflows/deploy.yml; then
            check_pass "Test job configured"
        else
            check_fail "Test job missing"
        fi
        
        if grep -q "security:" .github/workflows/deploy.yml; then
            check_pass "Security scanning configured"
        else
            check_warn "Security scanning not configured"
        fi
        
        if grep -q "build:" .github/workflows/deploy.yml; then
            check_pass "Build job configured"
        else
            check_fail "Build job missing"
        fi
        
        if grep -q "deploy-cloudrun:" .github/workflows/deploy.yml; then
            check_pass "Cloud Run deployment configured"
        else
            check_fail "Cloud Run deployment missing"
        fi
        
    else
        check_fail "GitHub Actions workflow missing"
    fi
}

validate_dependencies() {
    log_section "Validating Dependencies"
    
    # Check for pyproject.toml
    if [ -f "pyproject.toml" ]; then
        check_pass "pyproject.toml exists"
        
        # Check for production dependencies
        if grep -q "\[project.optional-dependencies\]" pyproject.toml; then
            check_pass "Optional dependencies configured"
        else
            check_warn "Optional dependencies not found"
        fi
        
    else
        check_fail "pyproject.toml missing"
    fi
    
    # Check for requirements files
    if [ -f "requirements.txt" ] || [ -f "pyproject.toml" ]; then
        check_pass "Dependency management configured"
    else
        check_fail "No dependency management found"
    fi
}

validate_security() {
    log_section "Validating Security Configuration"
    
    # Check for .gitignore
    if [ -f ".gitignore" ]; then
        check_pass ".gitignore exists"
        
        if grep -q "\.env" .gitignore; then
            check_pass "Environment files ignored"
        else
            check_warn "Environment files not ignored"
        fi
        
        if grep -q "secrets" .gitignore; then
            check_pass "Secrets ignored"
        else
            check_warn "Secrets not ignored"
        fi
        
    else
        check_warn ".gitignore missing"
    fi
    
    # Check for hardcoded secrets
    if grep -r "password\|secret\|key" --include="*.py" --include="*.yml" --include="*.yaml" . | grep -v "\${" | grep -v "# " | head -5; then
        check_warn "Potential hardcoded secrets found (review output above)"
    else
        check_pass "No obvious hardcoded secrets"
    fi
}

validate_monitoring() {
    log_section "Validating Monitoring Setup"
    
    # Check monitoring script
    if [ -f "scripts/deploy_monitoring.sh" ]; then
        if grep -q "create_alert_policies" scripts/deploy_monitoring.sh; then
            check_pass "Alert policies configured"
        else
            check_warn "Alert policies not found"
        fi
        
        if grep -q "create_uptime_checks" scripts/deploy_monitoring.sh; then
            check_pass "Uptime checks configured"
        else
            check_warn "Uptime checks not found"
        fi
        
        if grep -q "create_custom_dashboards" scripts/deploy_monitoring.sh; then
            check_pass "Custom dashboards configured"
        else
            check_warn "Custom dashboards not found"
        fi
    fi
}

print_summary() {
    log_section "Validation Summary"
    
    TOTAL=$((PASSED + FAILED))
    PASS_RATE=$(( (PASSED * 100) / TOTAL ))
    
    echo "Total checks: $TOTAL"
    echo -e "Passed: ${GREEN}$PASSED${NC}"
    echo -e "Failed: ${RED}$FAILED${NC}"
    echo -e "Pass rate: ${GREEN}$PASS_RATE%${NC}"
    
    if [ $FAILED -eq 0 ]; then
        echo -e "\n${GREEN}🎉 All validations passed! Deployment configuration is ready.${NC}"
        exit 0
    elif [ $PASS_RATE -ge 80 ]; then
        echo -e "\n${YELLOW}⚠️  Most validations passed. Review failed checks before deployment.${NC}"
        exit 0
    else
        echo -e "\n${RED}❌ Multiple validation failures. Fix issues before deployment.${NC}"
        exit 1
    fi
}

print_recommendations() {
    log_section "Deployment Recommendations"
    
    echo "1. Update placeholder values in configuration files:"
    echo "   - Replace PROJECT_ID in k8s manifests"
    echo "   - Set actual domain names in ingress"
    echo "   - Configure secrets with real values"
    echo ""
    echo "2. Set required environment variables:"
    echo "   - GCP_PROJECT_ID"
    echo "   - GCP_REGION"
    echo "   - NOTIFICATION_EMAIL"
    echo ""
    echo "3. Ensure GCP authentication:"
    echo "   - gcloud auth login"
    echo "   - gcloud config set project PROJECT_ID"
    echo ""
    echo "4. Deploy in order:"
    echo "   - ./scripts/setup_gcp_resources.sh"
    echo "   - ./scripts/deploy_monitoring.sh"
    echo "   - ./scripts/deploy_cloud_run.sh"
}

main() {
    log_info "Starting deployment validation for Nexus Forge..."
    
    validate_scripts
    validate_docker
    validate_kubernetes
    validate_config
    validate_terraform
    validate_ci_cd
    validate_dependencies
    validate_security
    validate_monitoring
    
    print_summary
    print_recommendations
}

# Run main function
main "$@"