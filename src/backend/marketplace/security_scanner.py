"""
Security scanner for agent packages
Detects vulnerabilities, malware, and suspicious patterns
"""

import ast
import asyncio
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import AgentPackage, SecurityReport


class SecurityScanner:
    """Comprehensive security scanner for agent packages"""

    # Suspicious patterns to detect
    SUSPICIOUS_IMPORTS = [
        "eval",
        "exec",
        "compile",
        "__import__",
        "subprocess",
        "os.system",
        "os.popen",
        "pickle",
        "marshal",
        "shelve",
    ]

    SUSPICIOUS_PATTERNS = [
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__\s*\(",
        r"compile\s*\(",
        r"globals\s*\(\)",
        r"locals\s*\(\)",
        r"vars\s*\(\)",
        r"open\s*\([^,]+,\s*['\"][wa]",  # File write operations
        r"requests\.(get|post|put|delete)",  # Network requests
        r"urllib\.(request|urlopen)",
        r"socket\.",
        r"base64\.b64decode",
        r"\\x[0-9a-fA-F]{2}",  # Hex encoded strings
    ]

    # File patterns to check
    DANGEROUS_FILES = [
        "*.pyc",
        "*.pyo",
        "*.so",
        "*.dll",
        "*.exe",
        "*.sh",
        "*.bat",
        "*.cmd",
        "*.ps1",
    ]

    def __init__(self):
        self.vulnerability_db = self._load_vulnerability_db()

    def _load_vulnerability_db(self) -> Dict[str, Any]:
        """Load known vulnerability database"""
        # In production, this would connect to a real vulnerability database
        return {
            "known_malware_hashes": set(),
            "vulnerable_packages": {},
            "cve_database": {},
        }

    async def scan_package(self, package_path: str) -> SecurityReport:
        """Perform comprehensive security scan on agent package"""
        report = SecurityReport()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract package
            extracted_path = await self._extract_package(package_path, temp_dir)

            # Run all security checks in parallel
            tasks = [
                self._scan_code_patterns(extracted_path),
                self._check_dependencies(extracted_path),
                self._scan_for_malware(extracted_path),
                self._check_permissions(extracted_path),
                self._validate_manifest(extracted_path),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            code_issues, dep_issues, malware_found, perm_issues, manifest_issues = (
                results
            )

            # Compile report
            if code_issues:
                report.suspicious_patterns.extend(code_issues)

            if dep_issues:
                report.dependency_vulnerabilities.extend(dep_issues)

            if malware_found:
                report.malware_detected = True
                report.vulnerabilities.append(
                    {
                        "type": "malware",
                        "severity": "critical",
                        "description": "Malware signature detected",
                    }
                )

            if perm_issues:
                report.vulnerabilities.extend(perm_issues)

            if manifest_issues:
                report.vulnerabilities.extend(manifest_issues)

            # Calculate risk score
            report.risk_score = self._calculate_risk_score(report)

        return report

    async def _extract_package(self, package_path: str, extract_to: str) -> str:
        """Safely extract package archive"""
        try:
            with zipfile.ZipFile(package_path, "r") as zip_file:
                # Check for path traversal vulnerabilities
                for member in zip_file.namelist():
                    if os.path.isabs(member) or ".." in member:
                        raise SecurityError(
                            f"Path traversal attempt detected: {member}"
                        )

                zip_file.extractall(extract_to)

            return extract_to
        except Exception as e:
            raise SecurityError(f"Failed to extract package: {str(e)}")

    async def _scan_code_patterns(self, path: str) -> List[str]:
        """Scan Python code for suspicious patterns"""
        suspicious_findings = []

        for py_file in Path(path).rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check for suspicious imports
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name in self.SUSPICIOUS_IMPORTS:
                                suspicious_findings.append(
                                    f"Suspicious import '{alias.name}' in {py_file}"
                                )

                    elif isinstance(node, ast.ImportFrom):
                        if node.module in self.SUSPICIOUS_IMPORTS:
                            suspicious_findings.append(
                                f"Suspicious import from '{node.module}' in {py_file}"
                            )

                # Check for pattern matches
                for pattern in self.SUSPICIOUS_PATTERNS:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        suspicious_findings.append(
                            f"Suspicious pattern '{pattern}' found in {py_file}: {matches[0][:50]}..."
                        )

            except Exception as e:
                suspicious_findings.append(f"Failed to analyze {py_file}: {str(e)}")

        return suspicious_findings

    async def _check_dependencies(self, path: str) -> List[Dict[str, Any]]:
        """Check dependencies for known vulnerabilities"""
        vulnerabilities = []

        # Check requirements.txt
        req_file = Path(path) / "requirements.txt"
        if req_file.exists():
            with open(req_file, "r") as f:
                requirements = f.readlines()

            for req in requirements:
                req = req.strip()
                if not req or req.startswith("#"):
                    continue

                # Parse package name and version
                package_name, version = self._parse_requirement(req)

                # Check against vulnerability database
                if package_name in self.vulnerability_db.get("vulnerable_packages", {}):
                    vuln_info = self.vulnerability_db["vulnerable_packages"][
                        package_name
                    ]
                    vulnerabilities.append(
                        {
                            "type": "dependency",
                            "severity": vuln_info.get("severity", "medium"),
                            "package": package_name,
                            "version": version,
                            "description": vuln_info.get(
                                "description", "Known vulnerability"
                            ),
                            "cve": vuln_info.get("cve"),
                        }
                    )

        # Check for pip audit if available
        try:
            result = await asyncio.create_subprocess_exec(
                "pip-audit",
                "--format",
                "json",
                "--requirement",
                str(req_file),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, _ = await result.communicate()

            if result.returncode == 0 and stdout:
                audit_results = json.loads(stdout)
                for vuln in audit_results.get("vulnerabilities", []):
                    vulnerabilities.append(
                        {
                            "type": "dependency",
                            "severity": "high",
                            "package": vuln["name"],
                            "version": vuln["version"],
                            "description": vuln["description"],
                            "cve": vuln.get("aliases", []),
                        }
                    )
        except Exception:
            pass  # pip-audit not available

        return vulnerabilities

    async def _scan_for_malware(self, path: str) -> bool:
        """Scan for known malware signatures"""
        # Calculate file hashes and check against malware database
        for file_path in Path(path).rglob("*"):
            if file_path.is_file():
                file_hash = self._calculate_file_hash(file_path)
                if file_hash in self.vulnerability_db.get(
                    "known_malware_hashes", set()
                ):
                    return True

        # Check for dangerous file types
        for pattern in self.DANGEROUS_FILES:
            dangerous_files = list(Path(path).rglob(pattern))
            if dangerous_files:
                # Additional checks for executable files
                for file in dangerous_files:
                    if self._is_executable(file):
                        return True

        return False

    async def _check_permissions(self, path: str) -> List[Dict[str, Any]]:
        """Check for excessive permission requests"""
        issues = []
        manifest_path = Path(path) / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            # Check for dangerous permissions
            dangerous_perms = [
                "filesystem.write",
                "network.full",
                "system.execute",
                "admin.access",
            ]

            requested_perms = manifest.get("permissions", [])
            for perm in requested_perms:
                if perm in dangerous_perms:
                    issues.append(
                        {
                            "type": "permission",
                            "severity": "high",
                            "permission": perm,
                            "description": f"Dangerous permission requested: {perm}",
                        }
                    )

        return issues

    async def _validate_manifest(self, path: str) -> List[Dict[str, Any]]:
        """Validate agent manifest for security issues"""
        issues = []
        manifest_path = Path(path) / "agent.json"

        if not manifest_path.exists():
            issues.append(
                {
                    "type": "manifest",
                    "severity": "high",
                    "description": "Missing agent manifest file",
                }
            )
            return issues

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            # Check for required fields
            required_fields = ["name", "version", "main_class", "author"]
            for field in required_fields:
                if field not in manifest:
                    issues.append(
                        {
                            "type": "manifest",
                            "severity": "medium",
                            "field": field,
                            "description": f"Missing required field: {field}",
                        }
                    )

            # Validate main class path
            if "main_class" in manifest:
                main_class = manifest["main_class"]
                if ".." in main_class or main_class.startswith("/"):
                    issues.append(
                        {
                            "type": "manifest",
                            "severity": "critical",
                            "field": "main_class",
                            "description": "Invalid main class path",
                        }
                    )

        except Exception as e:
            issues.append(
                {
                    "type": "manifest",
                    "severity": "critical",
                    "description": f"Invalid manifest file: {str(e)}",
                }
            )

        return issues

    def _calculate_risk_score(self, report: SecurityReport) -> float:
        """Calculate overall risk score based on findings"""
        score = 0.0

        # Critical findings
        if report.malware_detected:
            score += 10.0

        critical_vulns = [
            v for v in report.vulnerabilities if v.get("severity") == "critical"
        ]
        score += len(critical_vulns) * 3.0

        # High severity findings
        high_vulns = [v for v in report.vulnerabilities if v.get("severity") == "high"]
        score += len(high_vulns) * 2.0

        # Medium severity findings
        medium_vulns = [
            v for v in report.vulnerabilities if v.get("severity") == "medium"
        ]
        score += len(medium_vulns) * 1.0

        # Suspicious patterns
        score += len(report.suspicious_patterns) * 0.5

        # Dependency vulnerabilities
        score += len(report.dependency_vulnerabilities) * 1.5

        return min(score, 10.0)

    def _parse_requirement(self, requirement: str) -> Tuple[str, Optional[str]]:
        """Parse package name and version from requirement string"""
        # Simple parser - in production use packaging library
        match = re.match(r"^([a-zA-Z0-9\-_]+)([><=!]+)?(.+)?$", requirement.strip())
        if match:
            return match.group(1), match.group(3)
        return requirement.strip(), None

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _is_executable(self, file_path: Path) -> bool:
        """Check if file is executable"""
        return os.access(file_path, os.X_OK)


class SecurityError(Exception):
    """Security-related errors"""

    pass
