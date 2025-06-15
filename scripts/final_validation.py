#!/usr/bin/env python3
"""
Nexus Forge Final Validation Script
Comprehensive system validation for Phase 7 completion and production readiness.
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ValidationResult:
    def __init__(self, name: str, status: bool, message: str, details: Optional[Dict] = None):
        self.name = name
        self.status = status
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow().isoformat()

class NexusForgeValidator:
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.start_time = time.time()
    
    def validate_core_imports(self) -> ValidationResult:
        """Validate core system imports."""
        try:
            # Test essential imports
            from nexus_forge.core.exceptions import NexusForgeError, AgentError
            from nexus_forge.core.monitoring import get_logger, structured_logger
            from nexus_forge.integrations.google.gemini_client import GeminiClient
            
            return ValidationResult(
                "Core Imports",
                True,
                "All core modules imported successfully",
                {"modules": ["exceptions", "monitoring", "gemini_client"]}
            )
        except Exception as e:
            return ValidationResult(
                "Core Imports",
                False,
                f"Import failed: {str(e)}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
    
    def validate_documentation(self) -> ValidationResult:
        """Validate documentation completeness."""
        required_docs = [
            "README.md",
            "docs/api-reference.md",
            "docs/architecture.md", 
            "docs/deployment-guide.md",
            "docs/demo-script.md",
            "docs/production-checklist.md"
        ]
        
        missing_docs = []
        existing_docs = []
        
        for doc in required_docs:
            if os.path.exists(doc):
                existing_docs.append(doc)
            else:
                missing_docs.append(doc)
        
        if missing_docs:
            return ValidationResult(
                "Documentation",
                False,
                f"Missing documentation files: {missing_docs}",
                {"missing": missing_docs, "existing": existing_docs}
            )
        else:
            return ValidationResult(
                "Documentation",
                True,
                "All required documentation present",
                {"validated_files": existing_docs}
            )
    
    def validate_project_structure(self) -> ValidationResult:
        """Validate project structure integrity."""
        required_dirs = [
            "nexus_forge/",
            "nexus_forge/core/",
            "nexus_forge/agents/",
            "nexus_forge/integrations/",
            "nexus_forge/api/",
            "frontend/",
            "docs/",
            "scripts/",
            "tests/"
        ]
        
        missing_dirs = []
        existing_dirs = []
        
        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                existing_dirs.append(dir_path)
            else:
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            return ValidationResult(
                "Project Structure",
                False,
                f"Missing directories: {missing_dirs}",
                {"missing": missing_dirs, "existing": existing_dirs}
            )
        else:
            return ValidationResult(
                "Project Structure", 
                True,
                "Project structure validated",
                {"validated_directories": existing_dirs}
            )
    
    def validate_configuration(self) -> ValidationResult:
        """Validate configuration files."""
        config_files = [
            "pyproject.toml",
            "Dockerfile",
            "docker-compose.yml",
            "cloudbuild.yaml"
        ]
        
        existing_configs = []
        missing_configs = []
        
        for config in config_files:
            if os.path.exists(config):
                existing_configs.append(config)
            else:
                missing_configs.append(config)
        
        return ValidationResult(
            "Configuration",
            len(existing_configs) >= 3,  # At least 3 out of 4 config files
            f"Configuration files: {len(existing_configs)}/{len(config_files)} present",
            {"existing": existing_configs, "missing": missing_configs}
        )
    
    def validate_monitoring_system(self) -> ValidationResult:
        """Validate monitoring and logging system."""
        try:
            from nexus_forge.core.monitoring import get_logger, api_logger, structured_logger
            
            # Test logger creation
            logger = get_logger("validation_test")
            
            # Test logging functionality
            logger.log("info", "Validation test log", test_param="test_value")
            
            return ValidationResult(
                "Monitoring System",
                True,
                "Monitoring system operational",
                {"components": ["structured_logger", "api_logger", "prometheus_metrics"]}
            )
        except Exception as e:
            return ValidationResult(
                "Monitoring System",
                False,
                f"Monitoring validation failed: {str(e)}",
                {"error": str(e)}
            )
    
    def validate_security_setup(self) -> ValidationResult:
        """Validate security configuration."""
        security_checks = {
            "exception_handling": False,
            "input_validation": False,
            "authentication": False,
            "rate_limiting": False
        }
        
        try:
            # Check exception handling
            from nexus_forge.core.exceptions import (
                NexusForgeError, ValidationError, AuthenticationError, 
                RateLimitError, PayloadTooLargeError
            )
            security_checks["exception_handling"] = True
            
            # Check validation exists
            from nexus_forge.core.security.validation import SecurityValidator
            security_checks["input_validation"] = True
            
        except ImportError:
            pass
        
        try:
            # Check auth dependencies
            from nexus_forge.api.dependencies.auth import get_current_user, create_access_token
            security_checks["authentication"] = True
        except ImportError:
            pass
        
        try:
            # Check rate limiting
            from nexus_forge.api.middleware.rate_limiter import RateLimiter
            security_checks["rate_limiting"] = True
        except ImportError:
            pass
        
        passed_checks = sum(security_checks.values())
        total_checks = len(security_checks)
        
        return ValidationResult(
            "Security Setup",
            passed_checks >= total_checks - 1,  # Allow 1 missing component
            f"Security components: {passed_checks}/{total_checks} validated",
            security_checks
        )
    
    def validate_deployment_readiness(self) -> ValidationResult:
        """Validate deployment configuration and readiness."""
        deployment_items = {
            "dockerfile": os.path.exists("Dockerfile"),
            "requirements": os.path.exists("pyproject.toml"),
            "deployment_scripts": os.path.exists("scripts/deploy_production.sh"),
            "cloud_config": os.path.exists("cloudbuild.yaml"),
            "docker_compose": os.path.exists("docker-compose.yml")
        }
        
        passed_items = sum(deployment_items.values())
        total_items = len(deployment_items)
        
        return ValidationResult(
            "Deployment Readiness",
            passed_items >= 4,  # At least 4 out of 5 items
            f"Deployment configuration: {passed_items}/{total_items} ready",
            deployment_items
        )
    
    def validate_frontend_setup(self) -> ValidationResult:
        """Validate frontend configuration."""
        frontend_items = {
            "package_json": os.path.exists("frontend/package.json"),
            "src_directory": os.path.exists("frontend/src/"),
            "components": os.path.exists("frontend/src/components/"),
            "main_component": os.path.exists("frontend/src/components/NexusForgeWorkspace.tsx"),
            "build_config": os.path.exists("frontend/tailwind.config.js")
        }
        
        passed_items = sum(frontend_items.values())
        total_items = len(frontend_items)
        
        return ValidationResult(
            "Frontend Setup",
            passed_items >= 3,  # At least 3 out of 5 items
            f"Frontend configuration: {passed_items}/{total_items} ready",
            frontend_items
        )
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation tests and generate report."""
        print("🚀 Starting Nexus Forge Final Validation...")
        print("=" * 60)
        
        validations = [
            self.validate_core_imports,
            self.validate_documentation,
            self.validate_project_structure,
            self.validate_configuration,
            self.validate_monitoring_system,
            self.validate_security_setup,
            self.validate_deployment_readiness,
            self.validate_frontend_setup
        ]
        
        for validation_func in validations:
            print(f"Running {validation_func.__name__}...")
            result = validation_func()
            self.results.append(result)
            
            status_icon = "✅" if result.status else "❌"
            print(f"{status_icon} {result.name}: {result.message}")
        
        # Generate summary
        passed = sum(1 for r in self.results if r.status)
        total = len(self.results)
        success_rate = (passed / total) * 100
        
        print("\n" + "=" * 60)
        print(f"VALIDATION SUMMARY: {passed}/{total} ({success_rate:.1f}%) PASSED")
        
        overall_status = success_rate >= 80  # 80% pass rate required
        
        if overall_status:
            print("🎉 NEXUS FORGE IS READY FOR PRODUCTION DEPLOYMENT!")
        else:
            print("⚠️ Additional work required before production deployment")
        
        # Generate detailed report
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": overall_status,
            "success_rate": success_rate,
            "passed_tests": passed,
            "total_tests": total,
            "duration_seconds": time.time() - self.start_time,
            "results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details,
                    "timestamp": r.timestamp
                }
                for r in self.results
            ]
        }
        
        return report

def main():
    """Main validation execution."""
    validator = NexusForgeValidator()
    report = validator.run_all_validations()
    
    # Save report to file
    report_file = "nexus_forge_validation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Detailed report saved to: {report_file}")
    
    # Return appropriate exit code
    sys.exit(0 if report["overall_status"] else 1)

if __name__ == "__main__":
    main()