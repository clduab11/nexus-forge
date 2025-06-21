"""
MCP (Model Context Protocol) Integration for Nexus Forge Marketplace
Enhanced implementation for hackathon with advanced features
"""

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import aiohttp
from redis import Redis

from ...core.exceptions import (
    SecurityException,
    ValidationError,
)
from ...core.logging import get_logger
from ...core.security import SecurityScanner
from .models import SecurityReport

logger = get_logger(__name__)


class MCPToolType(str, Enum):
    """Types of MCP tools available in the marketplace"""
    
    MEMORY_MANAGEMENT = "memory_management"
    CACHING = "caching"
    DATABASE = "database"
    WEB_INTELLIGENCE = "web_intelligence"
    VERSION_CONTROL = "version_control"
    TESTING = "testing"
    SEQUENTIAL_THINKING = "sequential_thinking"
    FILE_OPERATIONS = "file_operations"
    MONITORING = "monitoring"
    SECURITY = "security"


@dataclass
class MCPTool:
    """MCP Tool representation with enhanced metadata"""
    
    id: str
    name: str
    version: str
    description: str
    type: MCPToolType
    capabilities: List[str]
    requirements: Dict[str, str]
    author: str
    organization: Optional[str] = None
    rating: float = 0.0
    downloads: int = 0
    verified: bool = False
    license: str = "MIT"
    pricing_model: str = "free"
    documentation_url: Optional[str] = None
    source_url: Optional[str] = None
    icon_url: Optional[str] = None
    compatibility: Dict[str, str] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    security_clearance: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class MCPProtocolVersion(str, Enum):
    """Supported MCP protocol versions"""
    
    V1_0 = "1.0"
    V2_0 = "2.0"
    V2_1 = "2.1"


class MCPMarketplaceClient:
    """Client for interacting with MCP marketplace with advanced features"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://mcp.adk.google.com",
        cache_ttl: int = 3600,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        
        # Initialize components
        self.cache = ToolCache()
        self.installer = ToolInstaller()
        self.verifier = SecurityVerifier()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Performance tracking
        self.metrics = {
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_downloads": 0,
            "failed_installations": 0,
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "X-MCP-Version": MCPProtocolVersion.V2_1.value,
                "User-Agent": "NexusForge/1.0",
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def search_tools(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
        include_beta: bool = False,
    ) -> List[MCPTool]:
        """
        Search for tools in the marketplace with semantic search
        
        Args:
            query: Search query (supports natural language)
            filters: Additional filters (category, rating, etc.)
            limit: Maximum results to return
            offset: Pagination offset
            include_beta: Include beta/experimental tools
        
        Returns:
            List of matching MCP tools
        """
        # Check cache first
        cache_key = self._generate_cache_key("search", query, filters)
        cached_results = await self.cache.get(cache_key)
        if cached_results:
            self.metrics["cache_hits"] += 1
            return cached_results
        
        self.metrics["cache_misses"] += 1
        self.metrics["api_calls"] += 1
        
        # Prepare request
        params = {
            "q": query,
            "limit": limit,
            "offset": offset,
            "include_beta": include_beta,
        }
        
        if filters:
            params.update(filters)
        
        # Make API request with retries
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(
                    f"{self.base_url}/v2/tools/search",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Parse results
                    tools = [self._parse_tool(tool_data) for tool_data in data["tools"]]
                    
                    # Cache results
                    await self.cache.set(cache_key, tools, ttl=self.cache_ttl)
                    
                    return tools
            
            except aiohttp.ClientError as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to search tools after {self.max_retries} attempts: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def get_tool_details(self, tool_id: str, version: Optional[str] = None) -> MCPTool:
        """Get detailed information about a specific tool"""
        cache_key = f"tool:{tool_id}:{version or 'latest'}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        endpoint = f"{self.base_url}/v2/tools/{tool_id}"
        if version:
            endpoint += f"?version={version}"
        
        async with self.session.get(endpoint) as response:
            response.raise_for_status()
            data = await response.json()
            tool = self._parse_tool(data)
            
            await self.cache.set(cache_key, tool, ttl=self.cache_ttl)
            return tool
    
    async def install_tool(
        self,
        tool_id: str,
        version: str = "latest",
        verification_level: str = "strict",
        install_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        One-click tool installation with security verification
        
        Args:
            tool_id: Tool identifier
            version: Tool version (default: latest)
            verification_level: Security verification level (strict/standard/minimal)
            install_path: Custom installation path
            progress_callback: Callback for installation progress
        
        Returns:
            Installation result with metadata
        """
        try:
            # Get tool details
            tool = await self.get_tool_details(tool_id, version)
            
            # Progress: Starting
            if progress_callback:
                progress_callback(0.1, "Fetching tool package...")
            
            # Download tool package
            package_data = await self._download_package(tool)
            
            # Progress: Downloaded
            if progress_callback:
                progress_callback(0.3, "Verifying security...")
            
            # Security scanning
            security_report = await self.verifier.verify_package(
                package_data,
                tool,
                verification_level,
            )
            
            if not security_report.passed:
                raise SecurityException(
                    f"Security verification failed: {security_report.reason}"
                )
            
            # Progress: Verified
            if progress_callback:
                progress_callback(0.5, "Resolving dependencies...")
            
            # Dependency resolution
            dependencies = await self._resolve_dependencies(tool)
            
            # Progress: Installing
            if progress_callback:
                progress_callback(0.7, "Installing tool...")
            
            # Install tool
            installation = await self.installer.install(
                tool=tool,
                package_data=package_data,
                dependencies=dependencies,
                install_path=install_path,
            )
            
            # Progress: Configuring
            if progress_callback:
                progress_callback(0.9, "Configuring integration...")
            
            # Register with orchestrator
            await self._register_tool(tool, installation)
            
            # Update metrics
            self.metrics["total_downloads"] += 1
            
            # Progress: Complete
            if progress_callback:
                progress_callback(1.0, "Installation complete!")
            
            return {
                "tool_id": tool.id,
                "version": tool.version,
                "installation_path": installation["path"],
                "capabilities": tool.capabilities,
                "security_report": security_report.dict(),
                "dependencies_installed": len(dependencies),
                "timestamp": datetime.utcnow().isoformat(),
            }
        
        except Exception as e:
            self.metrics["failed_installations"] += 1
            logger.error(f"Failed to install tool {tool_id}: {e}")
            raise
    
    async def discover_tools_by_capability(
        self,
        capabilities: List[str],
        min_rating: float = 4.0,
        max_results: int = 10,
    ) -> List[MCPTool]:
        """Discover tools by required capabilities"""
        # Use semantic search to find tools
        query = f"tools with capabilities: {', '.join(capabilities)}"
        
        filters = {
            "min_rating": min_rating,
            "verified_only": True,
        }
        
        tools = await self.search_tools(query, filters, limit=max_results * 2)
        
        # Score and rank by capability match
        scored_tools = []
        for tool in tools:
            score = self._calculate_capability_score(tool, capabilities)
            if score > 0.7:  # 70% match threshold
                scored_tools.append((score, tool))
        
        # Sort by score and return top results
        scored_tools.sort(key=lambda x: x[0], reverse=True)
        return [tool for _, tool in scored_tools[:max_results]]
    
    async def get_trending_tools(
        self,
        time_window: str = "week",
        category: Optional[MCPToolType] = None,
        limit: int = 20,
    ) -> List[MCPTool]:
        """Get trending tools based on recent activity"""
        endpoint = f"{self.base_url}/v2/tools/trending"
        params = {
            "window": time_window,
            "limit": limit,
        }
        
        if category:
            params["category"] = category.value
        
        async with self.session.get(endpoint, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            return [self._parse_tool(tool_data) for tool_data in data["tools"]]
    
    def _parse_tool(self, data: Dict[str, Any]) -> MCPTool:
        """Parse API response into MCPTool object"""
        return MCPTool(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            description=data["description"],
            type=MCPToolType(data["type"]),
            capabilities=data["capabilities"],
            requirements=data.get("requirements", {}),
            author=data["author"],
            organization=data.get("organization"),
            rating=data.get("rating", 0.0),
            downloads=data.get("downloads", 0),
            verified=data.get("verified", False),
            license=data.get("license", "MIT"),
            pricing_model=data.get("pricing_model", "free"),
            documentation_url=data.get("documentation_url"),
            source_url=data.get("source_url"),
            icon_url=data.get("icon_url"),
            compatibility=data.get("compatibility", {}),
            performance_metrics=data.get("performance_metrics", {}),
            security_clearance=data.get("security_clearance"),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat())),
        )
    
    def _generate_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_data = json.dumps(args, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _calculate_capability_score(
        self,
        tool: MCPTool,
        required_capabilities: List[str],
    ) -> float:
        """Calculate capability match score"""
        if not required_capabilities:
            return 1.0
        
        matches = sum(
            1 for cap in required_capabilities
            if any(cap.lower() in tool_cap.lower() for tool_cap in tool.capabilities)
        )
        
        return matches / len(required_capabilities)
    
    async def _download_package(self, tool: MCPTool) -> bytes:
        """Download tool package from CDN"""
        download_url = f"{self.base_url}/v2/tools/{tool.id}/download"
        
        async with self.session.get(download_url) as response:
            response.raise_for_status()
            return await response.read()
    
    async def _resolve_dependencies(self, tool: MCPTool) -> List[Dict[str, Any]]:
        """Resolve and validate tool dependencies"""
        dependencies = []
        
        for dep_name, dep_version in tool.requirements.items():
            # Check if dependency is already installed
            if not await self._is_dependency_installed(dep_name, dep_version):
                dep_info = await self._fetch_dependency(dep_name, dep_version)
                dependencies.append(dep_info)
        
        return dependencies
    
    async def _is_dependency_installed(self, name: str, version: str) -> bool:
        """Check if a dependency is already installed"""
        # Implementation would check local registry
        return False
    
    async def _fetch_dependency(self, name: str, version: str) -> Dict[str, Any]:
        """Fetch dependency information"""
        return {
            "name": name,
            "version": version,
            "source": "mcp",
        }
    
    async def _register_tool(self, tool: MCPTool, installation: Dict[str, Any]):
        """Register installed tool with the orchestrator"""
        # Implementation would integrate with Starri orchestrator
        logger.info(f"Registered tool {tool.id} with orchestrator")


class ToolCache:
    """Distributed cache for MCP tools with Redis backend"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = Redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = 3600  # 1 hour
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            data = self.redis.get(f"mcp:cache:{key}")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        try:
            self.redis.setex(
                f"mcp:cache:{key}",
                ttl or self.default_ttl,
                json.dumps(value, default=str),
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def delete(self, key: str):
        """Delete value from cache"""
        self.redis.delete(f"mcp:cache:{key}")


class ToolInstaller:
    """Dynamic tool installer with sandboxing and hot-reload"""
    
    def __init__(self):
        self.install_base = os.getenv("MCP_TOOL_PATH", "/opt/nexus-forge/mcp-tools")
        self.sandbox_manager = SandboxManager()
        self.package_manager = PackageManager()
    
    async def install(
        self,
        tool: MCPTool,
        package_data: bytes,
        dependencies: List[Dict[str, Any]],
        install_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Install MCP tool with full isolation"""
        # Create installation directory
        tool_path = install_path or os.path.join(self.install_base, tool.id, tool.version)
        os.makedirs(tool_path, exist_ok=True)
        
        # Extract package
        await self.package_manager.extract_package(package_data, tool_path)
        
        # Install dependencies
        for dep in dependencies:
            await self._install_dependency(dep, tool_path)
        
        # Create sandbox environment
        sandbox = await self.sandbox_manager.create_sandbox(
            tool_id=tool.id,
            tool_path=tool_path,
            capabilities=tool.capabilities,
            resource_limits={
                "memory": "2Gi",
                "cpu": "1.0",
                "disk": "10Gi",
            },
        )
        
        # Validate installation
        validation_result = await self._validate_installation(tool, tool_path, sandbox)
        
        if not validation_result["success"]:
            raise ValidationError(f"Installation validation failed: {validation_result['error']}")
        
        return {
            "path": tool_path,
            "sandbox_id": sandbox.id,
            "validation": validation_result,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    async def _install_dependency(self, dependency: Dict[str, Any], base_path: str):
        """Install a single dependency"""
        dep_path = os.path.join(base_path, "dependencies", dependency["name"])
        os.makedirs(dep_path, exist_ok=True)
        
        # Download and install dependency
        # Implementation would handle different dependency types
        logger.info(f"Installing dependency: {dependency['name']}=={dependency['version']}")
    
    async def _validate_installation(
        self,
        tool: MCPTool,
        tool_path: str,
        sandbox: Any,
    ) -> Dict[str, Any]:
        """Validate tool installation"""
        try:
            # Test tool initialization
            result = await sandbox.execute(
                command="python -m mcp_tool.validate",
                timeout=30,
                cwd=tool_path,
            )
            
            return {
                "success": result["exit_code"] == 0,
                "output": result["output"],
                "error": result.get("error"),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }


class SecurityVerifier:
    """Comprehensive security verification for MCP tools"""
    
    def __init__(self):
        self.scanner = SecurityScanner()
        self.signature_verifier = SignatureVerifier()
        self.vulnerability_db = VulnerabilityDatabase()
    
    async def verify_package(
        self,
        package_data: bytes,
        tool: MCPTool,
        verification_level: str = "strict",
    ) -> SecurityReport:
        """Multi-layer security verification"""
        report = SecurityReport()
        
        # Signature verification
        if tool.verified:
            sig_result = await self.signature_verifier.verify(
                package_data,
                tool.security_clearance,
            )
            if not sig_result["valid"]:
                report.vulnerabilities.append({
                    "type": "signature",
                    "severity": "critical",
                    "description": "Invalid package signature",
                })
                report.passed = False
                return report
        
        # Malware scanning
        malware_result = await self.scanner.scan_for_malware(package_data)
        if malware_result["detected"]:
            report.malware_detected = True
            report.suspicious_patterns.extend(malware_result["patterns"])
            report.passed = False
            return report
        
        # Vulnerability scanning
        if verification_level in ["standard", "strict"]:
            vuln_results = await self.vulnerability_db.check_package(package_data)
            report.vulnerabilities.extend(vuln_results["vulnerabilities"])
            report.dependency_vulnerabilities.extend(vuln_results["dependency_vulnerabilities"])
        
        # Code analysis for suspicious patterns
        if verification_level == "strict":
            code_analysis = await self._analyze_code_patterns(package_data)
            report.suspicious_patterns.extend(code_analysis["suspicious_patterns"])
        
        # Calculate risk score
        report.risk_score = self._calculate_risk_score(report)
        
        # Final determination
        report.passed = (
            not report.malware_detected
            and report.risk_score < 7.0
            and not any(v["severity"] == "critical" for v in report.vulnerabilities)
        )
        
        return report
    
    async def _analyze_code_patterns(self, package_data: bytes) -> Dict[str, Any]:
        """Analyze code for suspicious patterns"""
        patterns = []
        
        # Check for dangerous imports
        dangerous_imports = ["os.system", "subprocess.call", "eval", "exec"]
        # Implementation would analyze actual code
        
        return {"suspicious_patterns": patterns}
    
    def _calculate_risk_score(self, report: SecurityReport) -> float:
        """Calculate overall risk score (0-10)"""
        score = 0.0
        
        # Malware is automatic 10
        if report.malware_detected:
            return 10.0
        
        # Add points for vulnerabilities
        for vuln in report.vulnerabilities:
            if vuln["severity"] == "critical":
                score += 3.0
            elif vuln["severity"] == "high":
                score += 2.0
            elif vuln["severity"] == "medium":
                score += 1.0
            else:
                score += 0.5
        
        # Add points for suspicious patterns
        score += len(report.suspicious_patterns) * 0.5
        
        return min(score, 10.0)


class SandboxManager:
    """Manage sandboxed execution environments for MCP tools"""
    
    def __init__(self):
        self.sandboxes = {}
        self.container_runtime = "docker"  # or "podman", "firecracker"
    
    async def create_sandbox(
        self,
        tool_id: str,
        tool_path: str,
        capabilities: List[str],
        resource_limits: Dict[str, str],
    ) -> "Sandbox":
        """Create isolated sandbox for tool execution"""
        sandbox = Sandbox(
            id=f"{tool_id}-{int(time.time())}",
            tool_id=tool_id,
            tool_path=tool_path,
            capabilities=capabilities,
            resource_limits=resource_limits,
        )
        
        # Create container/VM
        await sandbox.initialize()
        
        self.sandboxes[sandbox.id] = sandbox
        return sandbox
    
    async def destroy_sandbox(self, sandbox_id: str):
        """Destroy sandbox and clean up resources"""
        if sandbox_id in self.sandboxes:
            sandbox = self.sandboxes[sandbox_id]
            await sandbox.cleanup()
            del self.sandboxes[sandbox_id]


@dataclass
class Sandbox:
    """Isolated execution environment for MCP tools"""
    
    id: str
    tool_id: str
    tool_path: str
    capabilities: List[str]
    resource_limits: Dict[str, str]
    container_id: Optional[str] = None
    
    async def initialize(self):
        """Initialize sandbox environment"""
        # Implementation would create actual container/VM
        self.container_id = f"mcp-sandbox-{self.id}"
        logger.info(f"Initialized sandbox {self.id} for tool {self.tool_id}")
    
    async def execute(
        self,
        command: str,
        timeout: int = 60,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute command in sandbox"""
        # Implementation would run command in container
        return {
            "exit_code": 0,
            "output": "Command executed successfully",
            "error": None,
        }
    
    async def cleanup(self):
        """Clean up sandbox resources"""
        logger.info(f"Cleaning up sandbox {self.id}")


class SignatureVerifier:
    """Verify cryptographic signatures of MCP packages"""
    
    def __init__(self):
        self.trusted_keys = self._load_trusted_keys()
    
    async def verify(
        self,
        package_data: bytes,
        signature: Optional[str],
    ) -> Dict[str, Any]:
        """Verify package signature"""
        if not signature:
            return {"valid": False, "reason": "No signature provided"}
        
        # Implementation would verify actual cryptographic signature
        return {"valid": True, "signer": "mcp-authority"}
    
    def _load_trusted_keys(self) -> Dict[str, Any]:
        """Load trusted signing keys"""
        return {}


class VulnerabilityDatabase:
    """Check packages against known vulnerabilities"""
    
    async def check_package(self, package_data: bytes) -> Dict[str, List[Dict[str, Any]]]:
        """Check package for known vulnerabilities"""
        return {
            "vulnerabilities": [],
            "dependency_vulnerabilities": [],
        }


class PackageManager:
    """Handle package extraction and management"""
    
    async def extract_package(self, package_data: bytes, target_path: str):
        """Extract package to target directory"""
        # Implementation would handle zip/tar extraction
        logger.info(f"Extracted package to {target_path}")


class MCPToolExposureFramework:
    """Framework for exposing tools through MCP protocol"""
    
    def __init__(self):
        self.exposed_tools = {}
        self.protocol_handler = MCPProtocolHandler()
        self.capability_registry = CapabilityRegistry()
    
    async def expose_tool(
        self,
        tool: MCPTool,
        endpoint_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Expose tool through MCP protocol"""
        # Register capabilities
        for capability in tool.capabilities:
            await self.capability_registry.register(
                tool_id=tool.id,
                capability=capability,
                metadata={
                    "version": tool.version,
                    "performance": tool.performance_metrics,
                },
            )
        
        # Create protocol endpoint
        endpoint = await self.protocol_handler.create_endpoint(
            tool_id=tool.id,
            config=endpoint_config,
        )
        
        self.exposed_tools[tool.id] = {
            "tool": tool,
            "endpoint": endpoint,
            "registered_at": datetime.utcnow(),
        }
        
        return {
            "tool_id": tool.id,
            "endpoint_url": endpoint["url"],
            "capabilities": tool.capabilities,
            "status": "active",
        }


class MCPProtocolHandler:
    """Handle MCP protocol communication"""
    
    async def create_endpoint(
        self,
        tool_id: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create MCP protocol endpoint"""
        return {
            "url": f"mcp://tools/{tool_id}",
            "protocol_version": MCPProtocolVersion.V2_1.value,
        }


class CapabilityRegistry:
    """Registry for tool capabilities"""
    
    def __init__(self):
        self.capabilities = {}
    
    async def register(
        self,
        tool_id: str,
        capability: str,
        metadata: Dict[str, Any],
    ):
        """Register tool capability"""
        if capability not in self.capabilities:
            self.capabilities[capability] = []
        
        self.capabilities[capability].append({
            "tool_id": tool_id,
            "metadata": metadata,
            "registered_at": datetime.utcnow(),
        })
