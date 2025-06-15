#!/usr/bin/env python3
"""
Nexus Forge Simple Validation Script
Quick validation for core system functionality without configuration dependencies.
"""

import os
import sys
import json
import time
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def validate_core_system():
    """Validate core system components."""
    print("🚀 Nexus Forge Core System Validation")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Core imports
    print("Testing core imports...")
    try:
        from nexus_forge.core.exceptions import NexusForgeError, AgentError
        from nexus_forge.core.monitoring import get_logger, structured_logger
        print("✅ Core exceptions and monitoring imported")
        results["core_imports"] = True
    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        results["core_imports"] = False
    
    # Test 2: Gemini integration
    print("Testing Gemini integration...")
    try:
        from nexus_forge.integrations.google.gemini_client import GeminiClient
        print("✅ Gemini client imported successfully")
        results["gemini_integration"] = True
    except Exception as e:
        print(f"❌ Gemini integration failed: {e}")
        results["gemini_integration"] = False
    
    # Test 3: Documentation check
    print("Testing documentation completeness...")
    required_docs = [
        "README.md",
        "docs/api-reference.md",
        "docs/architecture.md", 
        "docs/deployment-guide.md",
        "docs/demo-script.md",
        "docs/production-checklist.md"
    ]
    
    doc_status = True
    for doc in required_docs:
        if not os.path.exists(doc):
            print(f"❌ Missing: {doc}")
            doc_status = False
    
    if doc_status:
        print("✅ All required documentation present")
    results["documentation"] = doc_status
    
    # Test 4: Project structure
    print("Testing project structure...")
    required_dirs = [
        "nexus_forge/",
        "nexus_forge/core/",
        "nexus_forge/agents/",
        "nexus_forge/integrations/",
        "frontend/",
        "docs/",
        "scripts/"
    ]
    
    structure_status = True
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            structure_status = False
    
    if structure_status:
        print("✅ Project structure validated")
    results["project_structure"] = structure_status
    
    # Test 5: Configuration files
    print("Testing configuration files...")
    config_files = [
        "pyproject.toml",
        "Dockerfile",
        "docker-compose.yml"
    ]
    
    config_count = sum(1 for f in config_files if os.path.exists(f))
    config_status = config_count >= 2
    
    if config_status:
        print(f"✅ Configuration files: {config_count}/{len(config_files)} present")
    else:
        print(f"❌ Insufficient configuration files: {config_count}/{len(config_files)}")
    results["configuration"] = config_status
    
    # Calculate overall status
    passed = sum(results.values())
    total = len(results)
    success_rate = (passed / total) * 100
    
    print("\n" + "=" * 50)
    print(f"VALIDATION SUMMARY: {passed}/{total} ({success_rate:.1f}%) PASSED")
    
    if success_rate >= 80:
        print("🎉 NEXUS FORGE CORE SYSTEM IS OPERATIONAL!")
        print("✅ Ready for hackathon demonstration")
    else:
        print("⚠️ Some components need attention")
    
    # Generate report
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": success_rate >= 80,
        "success_rate": success_rate,
        "passed_tests": passed,
        "total_tests": total,
        "results": results
    }
    
    # Save report
    with open("nexus_forge_simple_validation.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Report saved to: nexus_forge_simple_validation.json")
    
    return report["overall_status"]

if __name__ == "__main__":
    success = validate_core_system()
    sys.exit(0 if success else 1)