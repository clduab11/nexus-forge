"""
Marketplace data models and schemas
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, constr, validator


class AgentStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"


class AgentCategory(str, Enum):
    DATA_PROCESSING = "data_processing"
    NATURAL_LANGUAGE = "natural_language"
    COMPUTER_VISION = "computer_vision"
    AUDIO_PROCESSING = "audio_processing"
    CODE_GENERATION = "code_generation"
    WORKFLOW_AUTOMATION = "workflow_automation"
    ANALYTICS = "analytics"
    INTEGRATION = "integration"
    UTILITY = "utility"


class AgentManifest(BaseModel):
    """Agent manifest schema for marketplace submissions"""

    name: constr(min_length=3, max_length=100, regex="^[a-z0-9-]+$") = Field(
        ..., description="Unique agent name (lowercase, alphanumeric, hyphens)"
    )
    version: constr(regex="^\\d+\\.\\d+\\.\\d+$") = Field(
        ..., description="Semantic version (e.g., 1.0.0)"
    )
    display_name: str = Field(..., description="Human-readable agent name")
    description: str = Field(..., description="Agent description")
    author: str = Field(..., description="Agent author/organization")
    license: str = Field(..., description="License type (e.g., MIT, Apache-2.0)")
    category: AgentCategory = Field(..., description="Primary agent category")
    tags: List[str] = Field(default_factory=list, description="Search tags")

    # Technical specifications
    base_model: Optional[str] = Field(None, description="Base AI model used")
    capabilities: List[str] = Field(..., description="List of agent capabilities")
    requirements: Dict[str, str] = Field(
        default_factory=dict, description="Runtime requirements (e.g., python>=3.8)"
    )
    dependencies: List[Dict[str, str]] = Field(
        default_factory=list, description="Other agent dependencies"
    )

    # Entry points
    main_class: str = Field(
        ..., description="Main agent class (e.g., my_agent.MyAgent)"
    )
    config_schema: Optional[Dict[str, Any]] = Field(
        None, description="JSON schema for agent configuration"
    )

    # Resources
    documentation_url: Optional[HttpUrl] = None
    repository_url: Optional[HttpUrl] = None
    icon_url: Optional[HttpUrl] = None

    # Compatibility
    nexus_forge_version: str = Field(
        ..., description="Compatible Nexus Forge version (e.g., >=1.0.0)"
    )
    supported_platforms: List[str] = Field(
        default_factory=lambda: ["linux", "darwin", "win32"],
        description="Supported platforms",
    )

    @validator("tags")
    def validate_tags(cls, v):
        if len(v) > 10:
            raise ValueError("Maximum 10 tags allowed")
        return [tag.lower() for tag in v]


class SecurityReport(BaseModel):
    """Security scan results for an agent package"""

    scan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scan_timestamp: datetime = Field(default_factory=datetime.utcnow)
    scanner_version: str = "1.0.0"

    vulnerabilities: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of detected vulnerabilities"
    )
    malware_detected: bool = False
    suspicious_patterns: List[str] = Field(default_factory=list)

    dependency_vulnerabilities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Vulnerabilities in dependencies"
    )

    risk_score: float = Field(
        0.0, ge=0.0, le=10.0, description="Overall risk score (0=safe, 10=critical)"
    )
    passed: bool = True

    @validator("passed", always=True)
    def validate_passed(cls, v, values):
        if values.get("malware_detected"):
            return False
        if values.get("risk_score", 0) > 7.0:
            return False
        critical_vulns = [
            vuln
            for vuln in values.get("vulnerabilities", [])
            if vuln.get("severity") == "critical"
        ]
        if critical_vulns:
            return False
        return True


class PerformanceMetrics(BaseModel):
    """Performance benchmarking results"""

    benchmark_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    benchmark_timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Latency metrics (milliseconds)
    avg_response_time: float = Field(..., description="Average response time in ms")
    p50_response_time: float = Field(..., description="50th percentile response time")
    p95_response_time: float = Field(..., description="95th percentile response time")
    p99_response_time: float = Field(..., description="99th percentile response time")

    # Resource usage
    avg_memory_mb: float = Field(..., description="Average memory usage in MB")
    peak_memory_mb: float = Field(..., description="Peak memory usage in MB")
    avg_cpu_percent: float = Field(..., description="Average CPU usage percentage")

    # Throughput
    requests_per_second: float = Field(..., description="Sustained RPS")

    # Quality metrics
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate (0-1)")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Error rate (0-1)")

    # Overall score
    performance_score: float = Field(
        ..., ge=0.0, le=100.0, description="Overall performance score (0-100)"
    )


class AgentPackage(BaseModel):
    """Complete agent package for marketplace"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    manifest: AgentManifest
    status: AgentStatus = AgentStatus.PENDING

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None

    # Author info
    author_id: str
    author_email: str
    organization: Optional[str] = None

    # Package files
    package_url: Optional[str] = Field(None, description="URL to package archive")
    package_size_bytes: Optional[int] = None
    package_checksum: Optional[str] = None

    # Validation results
    security_report: Optional[SecurityReport] = None
    performance_metrics: Optional[PerformanceMetrics] = None

    # Marketplace metadata
    downloads: int = 0
    stars: int = 0
    rating: Optional[float] = Field(None, ge=0.0, le=5.0)
    rating_count: int = 0

    # Review process
    review_status: Optional[str] = None
    review_notes: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class AgentRating(BaseModel):
    """User rating for an agent"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    user_id: str
    rating: int = Field(..., ge=1, le=5)
    review: Optional[str] = Field(None, max_length=1000)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    helpful_count: int = 0

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class AgentSearchResult(BaseModel):
    """Search result for marketplace discovery"""

    agent: AgentPackage
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    matched_fields: List[str] = Field(default_factory=list)
    highlight: Optional[Dict[str, str]] = None


class MarketplaceStats(BaseModel):
    """Marketplace statistics"""

    total_agents: int = 0
    total_downloads: int = 0
    total_authors: int = 0
    agents_by_category: Dict[str, int] = Field(default_factory=dict)
    popular_tags: List[Dict[str, Any]] = Field(default_factory=list)
    trending_agents: List[str] = Field(default_factory=list)
    recent_updates: List[str] = Field(default_factory=list)
