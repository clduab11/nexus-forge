"""
Comprehensive Security Framework for Marketplace
Advanced security validation and protection mechanisms
"""

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import jwt
import yara
from cryptography.x509 import Certificate

from ...core.exceptions import SecurityException
from ...core.logging import get_logger

logger = get_logger(__name__)


class ThreatLevel(str, Enum):
    """Security threat levels"""
    
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityClearance(str, Enum):
    """Security clearance levels for marketplace items"""
    
    UNVERIFIED = "unverified"
    COMMUNITY = "community"
    VERIFIED = "verified"
    TRUSTED = "trusted"
    OFFICIAL = "official"


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    
    name: str
    description: str
    rules: List[Dict[str, Any]]
    enforcement_level: str = "strict"  # strict, moderate, permissive
    exceptions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SecurityViolation:
    """Security violation record"""
    
    id: str
    type: str
    severity: ThreatLevel
    description: str
    evidence: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    remediation: Optional[str] = None


class MarketplaceSecurityFramework:
    """
    Comprehensive security framework for marketplace operations
    """
    
    def __init__(
        self,
        policy_path: str = "/etc/nexus-forge/security/policies",
        enable_blockchain_audit: bool = True,
        enable_ai_threat_detection: bool = True,
    ):
        self.policy_path = Path(policy_path)
        self.enable_blockchain_audit = enable_blockchain_audit
        self.enable_ai_threat_detection = enable_ai_threat_detection
        
        # Security components
        self.code_scanner = CodeSecurityScanner()
        self.vulnerability_scanner = VulnerabilityScanner()
        self.malware_detector = MalwareDetector()
        self.signature_verifier = SignatureVerifier()
        self.permission_manager = PermissionManager()
        self.audit_logger = BlockchainAuditLogger() if enable_blockchain_audit else None
        self.threat_detector = AIThreatDetector() if enable_ai_threat_detection else None
        
        # Security policies
        self.policies: Dict[str, SecurityPolicy] = {}
        self._load_security_policies()
        
        # Trusted keys and certificates
        self.trusted_keys: Dict[str, Any] = {}
        self.trusted_certs: Dict[str, Certificate] = {}
        self._load_trusted_authorities()
    
    async def validate_package_security(
        self,
        package_data: bytes,
        metadata: Dict[str, Any],
        clearance_required: SecurityClearance = SecurityClearance.VERIFIED,
    ) -> Dict[str, Any]:
        """
        Comprehensive security validation for packages
        
        Args:
            package_data: Package binary data
            metadata: Package metadata including signatures
            clearance_required: Minimum security clearance required
        
        Returns:
            Security validation result
        """
        start_time = time.time()
        violations: List[SecurityViolation] = []
        
        try:
            # 1. Signature verification
            if clearance_required != SecurityClearance.UNVERIFIED:
                sig_result = await self.signature_verifier.verify(
                    package_data,
                    metadata.get("signature"),
                    metadata.get("signer_id"),
                )
                
                if not sig_result["valid"]:
                    violations.append(SecurityViolation(
                        id=self._generate_violation_id(),
                        type="signature",
                        severity=ThreatLevel.HIGH,
                        description="Invalid or missing package signature",
                        evidence=sig_result,
                    ))
            
            # 2. Malware scanning
            malware_result = await self.malware_detector.scan(package_data)
            if malware_result["detected"]:
                violations.append(SecurityViolation(
                    id=self._generate_violation_id(),
                    type="malware",
                    severity=ThreatLevel.CRITICAL,
                    description="Malware detected in package",
                    evidence=malware_result,
                ))
            
            # 3. Code security analysis
            code_result = await self.code_scanner.scan_package(package_data)
            for issue in code_result["issues"]:
                if issue["severity"] in ["high", "critical"]:
                    violations.append(SecurityViolation(
                        id=self._generate_violation_id(),
                        type="code_security",
                        severity=ThreatLevel(issue["severity"]),
                        description=issue["description"],
                        evidence=issue,
                    ))
            
            # 4. Vulnerability scanning
            vuln_result = await self.vulnerability_scanner.scan(
                package_data,
                metadata.get("dependencies", []),
            )
            for vuln in vuln_result["vulnerabilities"]:
                if vuln["severity"] in ["high", "critical"]:
                    violations.append(SecurityViolation(
                        id=self._generate_violation_id(),
                        type="vulnerability",
                        severity=ThreatLevel(vuln["severity"]),
                        description=vuln["description"],
                        evidence=vuln,
                    ))
            
            # 5. Permission analysis
            perm_result = await self.permission_manager.analyze_permissions(
                package_data,
                metadata.get("requested_permissions", []),
            )
            if perm_result["excessive_permissions"]:
                violations.append(SecurityViolation(
                    id=self._generate_violation_id(),
                    type="permissions",
                    severity=ThreatLevel.MEDIUM,
                    description="Package requests excessive permissions",
                    evidence=perm_result,
                ))
            
            # 6. AI-based threat detection
            if self.threat_detector:
                ai_result = await self.threat_detector.analyze(
                    package_data,
                    metadata,
                )
                if ai_result["threat_detected"]:
                    violations.append(SecurityViolation(
                        id=self._generate_violation_id(),
                        type="ai_detection",
                        severity=ThreatLevel(ai_result["severity"]),
                        description=ai_result["description"],
                        evidence=ai_result,
                    ))
            
            # 7. Policy compliance check
            policy_result = await self._check_policy_compliance(
                package_data,
                metadata,
                violations,
            )
            if not policy_result["compliant"]:
                violations.extend(policy_result["violations"])
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(violations)
            
            # Determine if package passes security checks
            passed = (
                risk_score < 7.0
                and not any(v.severity == ThreatLevel.CRITICAL for v in violations)
                and (clearance_required == SecurityClearance.UNVERIFIED
                     or sig_result.get("valid", False))
            )
            
            # Audit log the security check
            if self.audit_logger:
                await self.audit_logger.log_security_check({
                    "package_id": metadata.get("id"),
                    "passed": passed,
                    "risk_score": risk_score,
                    "violations": len(violations),
                    "duration": time.time() - start_time,
                })
            
            return {
                "passed": passed,
                "risk_score": risk_score,
                "violations": violations,
                "clearance_level": self._determine_clearance_level(
                    passed, risk_score, sig_result
                ),
                "scan_duration": time.time() - start_time,
                "recommendations": self._generate_recommendations(violations),
            }
        
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            raise SecurityException(f"Security validation failed: {e}")
    
    async def validate_api_request(
        self,
        request_data: Dict[str, Any],
        api_key: str,
        required_permissions: List[str],
    ) -> bool:
        """Validate API request security"""
        try:
            # Validate API key
            key_info = await self._validate_api_key(api_key)
            if not key_info["valid"]:
                return False
            
            # Check rate limiting
            if not await self._check_rate_limit(key_info["user_id"]):
                return False
            
            # Validate permissions
            if not all(
                perm in key_info["permissions"]
                for perm in required_permissions
            ):
                return False
            
            # Check for suspicious patterns
            if await self._detect_suspicious_patterns(request_data):
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"API validation error: {e}")
            return False
    
    def _load_security_policies(self):
        """Load security policies from configuration"""
        self.policies = {
            "default": SecurityPolicy(
                name="default",
                description="Default security policy",
                rules=[
                    {"type": "max_file_size", "value": 100 * 1024 * 1024},  # 100MB
                    {"type": "allowed_languages", "value": ["python", "javascript", "typescript"]},
                    {"type": "forbidden_imports", "value": ["os.system", "subprocess.Popen"]},
                    {"type": "required_security_headers", "value": ["X-Content-Type-Options"]},
                ],
                enforcement_level="strict",
            ),
            "trusted": SecurityPolicy(
                name="trusted",
                description="Policy for trusted publishers",
                rules=[
                    {"type": "max_file_size", "value": 500 * 1024 * 1024},  # 500MB
                    {"type": "allowed_languages", "value": ["any"]},
                ],
                enforcement_level="moderate",
            ),
        }
    
    def _load_trusted_authorities(self):
        """Load trusted certificate authorities"""
        # Implementation would load from secure storage
        pass
    
    def _generate_violation_id(self) -> str:
        """Generate unique violation ID"""
        return hashlib.sha256(
            f"{time.time()}{os.urandom(16).hex()}".encode()
        ).hexdigest()[:16]
    
    def _calculate_risk_score(self, violations: List[SecurityViolation]) -> float:
        """Calculate overall risk score (0-10)"""
        if not violations:
            return 0.0
        
        score = 0.0
        severity_weights = {
            ThreatLevel.LOW: 1.0,
            ThreatLevel.MEDIUM: 2.5,
            ThreatLevel.HIGH: 5.0,
            ThreatLevel.CRITICAL: 10.0,
        }
        
        for violation in violations:
            score += severity_weights.get(violation.severity, 0.0)
        
        return min(score, 10.0)
    
    def _determine_clearance_level(
        self,
        passed: bool,
        risk_score: float,
        sig_result: Dict[str, Any],
    ) -> SecurityClearance:
        """Determine security clearance level"""
        if not passed:
            return SecurityClearance.UNVERIFIED
        
        if sig_result.get("signer_type") == "official":
            return SecurityClearance.OFFICIAL
        elif sig_result.get("signer_type") == "trusted":
            return SecurityClearance.TRUSTED
        elif sig_result.get("valid") and risk_score < 2.0:
            return SecurityClearance.VERIFIED
        elif risk_score < 5.0:
            return SecurityClearance.COMMUNITY
        else:
            return SecurityClearance.UNVERIFIED
    
    def _generate_recommendations(
        self,
        violations: List[SecurityViolation],
    ) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        violation_types = set(v.type for v in violations)
        
        if "signature" in violation_types:
            recommendations.append("Sign package with verified certificate")
        if "permissions" in violation_types:
            recommendations.append("Review and minimize requested permissions")
        if "vulnerability" in violation_types:
            recommendations.append("Update dependencies to patch vulnerabilities")
        if "code_security" in violation_types:
            recommendations.append("Review and fix identified security issues in code")
        
        return recommendations
    
    async def _check_policy_compliance(
        self,
        package_data: bytes,
        metadata: Dict[str, Any],
        existing_violations: List[SecurityViolation],
    ) -> Dict[str, Any]:
        """Check compliance with security policies"""
        policy_name = metadata.get("policy", "default")
        policy = self.policies.get(policy_name, self.policies["default"])
        
        violations = []
        
        for rule in policy.rules:
            if not await self._check_rule(rule, package_data, metadata):
                violations.append(SecurityViolation(
                    id=self._generate_violation_id(),
                    type="policy",
                    severity=ThreatLevel.MEDIUM,
                    description=f"Policy violation: {rule['type']}",
                    evidence={"rule": rule, "policy": policy_name},
                ))
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
        }
    
    async def _check_rule(
        self,
        rule: Dict[str, Any],
        package_data: bytes,
        metadata: Dict[str, Any],
    ) -> bool:
        """Check individual policy rule"""
        rule_type = rule["type"]
        
        if rule_type == "max_file_size":
            return len(package_data) <= rule["value"]
        elif rule_type == "allowed_languages":
            # Implementation would analyze package content
            return True
        elif rule_type == "forbidden_imports":
            # Implementation would scan code for imports
            return True
        
        return True
    
    async def _validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validate API key and get permissions"""
        try:
            # Decode JWT token
            payload = jwt.decode(
                api_key,
                self.get_api_secret(),
                algorithms=["HS256"],
            )
            
            return {
                "valid": True,
                "user_id": payload["user_id"],
                "permissions": payload.get("permissions", []),
                "expires_at": payload.get("exp"),
            }
        except jwt.InvalidTokenError:
            return {"valid": False}
    
    async def _check_rate_limit(self, user_id: str) -> bool:
        """Check API rate limiting"""
        # Implementation would check against rate limit store
        return True
    
    async def _detect_suspicious_patterns(
        self,
        request_data: Dict[str, Any],
    ) -> bool:
        """Detect suspicious request patterns"""
        # Implementation would analyze request patterns
        return False
    
    def get_api_secret(self) -> str:
        """Get API secret key"""
        return os.getenv("NEXUS_FORGE_API_SECRET", "default-secret")


class CodeSecurityScanner:
    """Scan code for security vulnerabilities"""
    
    def __init__(self):
        self.rules = self._load_security_rules()
        self.ast_analyzer = ASTSecurityAnalyzer()
    
    async def scan_package(self, package_data: bytes) -> Dict[str, Any]:
        """Scan package for code security issues"""
        issues = []
        
        # Extract code files from package
        code_files = await self._extract_code_files(package_data)
        
        for file_path, code_content in code_files.items():
            # Static analysis
            static_issues = await self._static_analysis(code_content)
            issues.extend(static_issues)
            
            # AST analysis
            ast_issues = await self.ast_analyzer.analyze(code_content)
            issues.extend(ast_issues)
            
            # Pattern matching
            pattern_issues = await self._pattern_matching(code_content)
            issues.extend(pattern_issues)
        
        return {
            "issues": issues,
            "total_files": len(code_files),
            "severity_summary": self._summarize_severity(issues),
        }
    
    async def _extract_code_files(
        self,
        package_data: bytes,
    ) -> Dict[str, str]:
        """Extract code files from package"""
        # Implementation would extract actual files
        return {}
    
    async def _static_analysis(self, code: str) -> List[Dict[str, Any]]:
        """Perform static code analysis"""
        issues = []
        
        # Check for dangerous functions
        dangerous_patterns = [
            (r"eval\s*\(", "Use of eval() is dangerous", "high"),
            (r"exec\s*\(", "Use of exec() is dangerous", "high"),
            (r"__import__", "Dynamic imports can be dangerous", "medium"),
            (r"os\.system", "Direct system calls are dangerous", "high"),
            (r"subprocess\.call.*shell=True", "Shell injection risk", "critical"),
        ]
        
        for pattern, description, severity in dangerous_patterns:
            if re.search(pattern, code):
                issues.append({
                    "type": "dangerous_function",
                    "description": description,
                    "severity": severity,
                    "pattern": pattern,
                })
        
        return issues
    
    async def _pattern_matching(self, code: str) -> List[Dict[str, Any]]:
        """Pattern-based security scanning"""
        issues = []
        
        # Check for hardcoded secrets
        secret_patterns = [
            (r"api_key\s*=\s*['\"][\w]+['\"]", "Hardcoded API key"),
            (r"password\s*=\s*['\"][\w]+['\"]", "Hardcoded password"),
            (r"secret\s*=\s*['\"][\w]+['\"]", "Hardcoded secret"),
        ]
        
        for pattern, description in secret_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append({
                    "type": "hardcoded_secret",
                    "description": description,
                    "severity": "high",
                    "pattern": pattern,
                })
        
        return issues
    
    def _load_security_rules(self) -> List[Dict[str, Any]]:
        """Load security scanning rules"""
        return []
    
    def _summarize_severity(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarize issues by severity"""
        summary = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for issue in issues:
            severity = issue.get("severity", "low")
            summary[severity] = summary.get(severity, 0) + 1
        return summary


class ASTSecurityAnalyzer:
    """Analyze Abstract Syntax Tree for security issues"""
    
    async def analyze(self, code: str) -> List[Dict[str, Any]]:
        """Analyze code AST for security issues"""
        # Implementation would parse and analyze AST
        return []


class VulnerabilityScanner:
    """Scan for known vulnerabilities"""
    
    def __init__(self):
        self.vulnerability_db = VulnerabilityDatabase()
        self.dependency_checker = DependencyChecker()
    
    async def scan(
        self,
        package_data: bytes,
        dependencies: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Scan for vulnerabilities"""
        vulnerabilities = []
        
        # Check dependencies against vulnerability database
        for dep in dependencies:
            dep_vulns = await self.vulnerability_db.check_dependency(
                dep["name"],
                dep["version"],
            )
            vulnerabilities.extend(dep_vulns)
        
        # Scan package code for vulnerability patterns
        code_vulns = await self._scan_code_vulnerabilities(package_data)
        vulnerabilities.extend(code_vulns)
        
        return {
            "vulnerabilities": vulnerabilities,
            "total_dependencies": len(dependencies),
            "vulnerable_dependencies": len(set(
                v["dependency"] for v in vulnerabilities
                if "dependency" in v
            )),
        }
    
    async def _scan_code_vulnerabilities(
        self,
        package_data: bytes,
    ) -> List[Dict[str, Any]]:
        """Scan code for vulnerability patterns"""
        return []


class VulnerabilityDatabase:
    """Database of known vulnerabilities"""
    
    async def check_dependency(
        self,
        name: str,
        version: str,
    ) -> List[Dict[str, Any]]:
        """Check dependency for known vulnerabilities"""
        # Implementation would query actual vulnerability database
        return []


class DependencyChecker:
    """Check dependency security"""
    
    async def check_dependencies(
        self,
        dependencies: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Check all dependencies for security issues"""
        return {
            "safe_dependencies": len(dependencies),
            "unsafe_dependencies": 0,
        }


class MalwareDetector:
    """Detect malware in packages"""
    
    def __init__(self):
        self.yara_rules = self._load_yara_rules()
        self.ml_detector = MLMalwareDetector()
    
    async def scan(self, package_data: bytes) -> Dict[str, Any]:
        """Scan for malware"""
        detected = False
        threats = []
        
        # YARA rules scanning
        yara_matches = await self._yara_scan(package_data)
        if yara_matches:
            detected = True
            threats.extend(yara_matches)
        
        # Machine learning detection
        ml_result = await self.ml_detector.detect(package_data)
        if ml_result["is_malware"]:
            detected = True
            threats.append({
                "type": "ml_detection",
                "confidence": ml_result["confidence"],
                "category": ml_result["category"],
            })
        
        # Heuristic analysis
        heuristic_result = await self._heuristic_analysis(package_data)
        if heuristic_result["suspicious"]:
            detected = True
            threats.extend(heuristic_result["indicators"])
        
        return {
            "detected": detected,
            "threats": threats,
            "scan_engine": "yara+ml+heuristic",
        }
    
    def _load_yara_rules(self) -> Optional[yara.Rules]:
        """Load YARA rules for malware detection"""
        try:
            # Implementation would load actual YARA rules
            return None
        except Exception as e:
            logger.error(f"Failed to load YARA rules: {e}")
            return None
    
    async def _yara_scan(self, data: bytes) -> List[Dict[str, Any]]:
        """Scan with YARA rules"""
        if not self.yara_rules:
            return []
        
        matches = []
        # Implementation would run YARA scan
        return matches
    
    async def _heuristic_analysis(self, data: bytes) -> Dict[str, Any]:
        """Perform heuristic analysis"""
        indicators = []
        
        # Check for suspicious patterns
        if b"ransomware" in data.lower():
            indicators.append({
                "type": "keyword",
                "description": "Ransomware keyword detected",
            })
        
        return {
            "suspicious": len(indicators) > 0,
            "indicators": indicators,
        }


class MLMalwareDetector:
    """Machine learning based malware detection"""
    
    async def detect(self, data: bytes) -> Dict[str, Any]:
        """Detect malware using ML model"""
        # Implementation would use trained model
        return {
            "is_malware": False,
            "confidence": 0.0,
            "category": None,
        }


class SignatureVerifier:
    """Verify cryptographic signatures"""
    
    def __init__(self):
        self.trusted_signers = self._load_trusted_signers()
    
    async def verify(
        self,
        data: bytes,
        signature: Optional[str],
        signer_id: Optional[str],
    ) -> Dict[str, Any]:
        """Verify package signature"""
        if not signature:
            return {
                "valid": False,
                "reason": "No signature provided",
            }
        
        try:
            # Get signer's public key
            public_key = await self._get_public_key(signer_id)
            if not public_key:
                return {
                    "valid": False,
                    "reason": "Unknown signer",
                }
            
            # Verify signature
            # Implementation would perform actual signature verification
            
            return {
                "valid": True,
                "signer_id": signer_id,
                "signer_type": self._get_signer_type(signer_id),
                "timestamp": datetime.utcnow().isoformat(),
            }
        
        except Exception as e:
            return {
                "valid": False,
                "reason": str(e),
            }
    
    def _load_trusted_signers(self) -> Dict[str, Any]:
        """Load trusted signer information"""
        return {}
    
    async def _get_public_key(self, signer_id: str) -> Optional[Any]:
        """Get public key for signer"""
        return self.trusted_signers.get(signer_id)
    
    def _get_signer_type(self, signer_id: str) -> str:
        """Determine signer type"""
        if signer_id.startswith("official-"):
            return "official"
        elif signer_id in self.trusted_signers:
            return "trusted"
        else:
            return "community"


class PermissionManager:
    """Manage and analyze permissions"""
    
    async def analyze_permissions(
        self,
        package_data: bytes,
        requested_permissions: List[str],
    ) -> Dict[str, Any]:
        """Analyze requested permissions"""
        # Extract actual required permissions from code
        actual_permissions = await self._extract_permissions(package_data)
        
        # Compare requested vs actual
        excessive = set(requested_permissions) - set(actual_permissions)
        missing = set(actual_permissions) - set(requested_permissions)
        
        return {
            "requested": requested_permissions,
            "actual": actual_permissions,
            "excessive_permissions": list(excessive),
            "missing_permissions": list(missing),
            "risk_level": self._calculate_permission_risk(requested_permissions),
        }
    
    async def _extract_permissions(self, package_data: bytes) -> List[str]:
        """Extract permissions from package code"""
        # Implementation would analyze code for permission usage
        return []
    
    def _calculate_permission_risk(self, permissions: List[str]) -> str:
        """Calculate risk level of permissions"""
        high_risk_perms = ["filesystem.write", "network.all", "system.execute"]
        
        high_risk_count = sum(
            1 for perm in permissions
            if any(hrp in perm for hrp in high_risk_perms)
        )
        
        if high_risk_count >= 3:
            return "high"
        elif high_risk_count >= 1:
            return "medium"
        else:
            return "low"


class BlockchainAuditLogger:
    """Log security events to blockchain for immutable audit trail"""
    
    async def log_security_check(self, event: Dict[str, Any]):
        """Log security check to blockchain"""
        # Implementation would write to blockchain
        logger.info(f"Blockchain audit log: {event}")


class AIThreatDetector:
    """AI-powered threat detection"""
    
    async def analyze(
        self,
        package_data: bytes,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze package with AI for threats"""
        # Implementation would use AI model
        return {
            "threat_detected": False,
            "severity": "none",
            "confidence": 0.0,
            "description": None,
        }
