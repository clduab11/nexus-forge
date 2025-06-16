"""
Multi-region data models and schemas
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import uuid


class RegionStatus(str, Enum):
    """Region operational status"""
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    GEOGRAPHIC = "geographic"
    WEIGHTED = "weighted"
    LATENCY_BASED = "latency_based"


class ReplicationMode(str, Enum):
    """Data replication modes"""
    ASYNC = "async"
    SYNC = "sync"
    EVENTUAL = "eventual"
    MASTER_SLAVE = "master_slave"
    MULTI_MASTER = "multi_master"


class EdgeTier(str, Enum):
    """Edge location tiers"""
    TIER_1 = "tier_1"  # Primary data centers
    TIER_2 = "tier_2"  # Regional edge locations
    TIER_3 = "tier_3"  # Local edge nodes


class HealthStatus(str, Enum):
    """Health check status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ResourceMetrics(BaseModel):
    """Resource utilization metrics"""
    cpu_usage_percent: float = Field(0.0, ge=0, le=100)
    memory_usage_percent: float = Field(0.0, ge=0, le=100)
    storage_usage_percent: float = Field(0.0, ge=0, le=100)
    network_bandwidth_mbps: float = Field(0.0, ge=0)
    active_connections: int = Field(0, ge=0)
    requests_per_second: float = Field(0.0, ge=0)
    
    # Latency metrics
    avg_response_time_ms: float = Field(0.0, ge=0)
    p95_response_time_ms: float = Field(0.0, ge=0)
    p99_response_time_ms: float = Field(0.0, ge=0)
    
    # Error metrics
    error_rate_percent: float = Field(0.0, ge=0, le=100)
    
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class GeographicLocation(BaseModel):
    """Geographic coordinates and metadata"""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    country: str = Field(..., min_length=2, max_length=2)  # ISO country code
    country_name: str = Field(..., min_length=1)
    region: str = Field(..., min_length=1)  # Geographic region
    city: str = Field(..., min_length=1)
    timezone: str = Field(..., min_length=1)  # IANA timezone


class NetworkConfig(BaseModel):
    """Network configuration for region/edge"""
    vpc_id: Optional[str] = Field(None)
    subnet_ids: List[str] = Field(default_factory=list)
    security_group_ids: List[str] = Field(default_factory=list)
    load_balancer_arn: Optional[str] = Field(None)
    cdn_distribution_id: Optional[str] = Field(None)
    
    # DNS configuration
    dns_zone_id: Optional[str] = Field(None)
    custom_domain: Optional[str] = Field(None)
    ssl_certificate_arn: Optional[str] = Field(None)
    
    # Network policies
    allowed_cidr_blocks: List[str] = Field(default_factory=list)
    port_ranges: List[Dict[str, int]] = Field(default_factory=list)


class StorageConfig(BaseModel):
    """Storage configuration for region/edge"""
    primary_storage_class: str = Field("standard")
    backup_storage_class: str = Field("glacier")
    replication_enabled: bool = Field(True)
    encryption_enabled: bool = Field(True)
    encryption_key_id: Optional[str] = Field(None)
    
    # Retention policies
    backup_retention_days: int = Field(30, ge=1)
    log_retention_days: int = Field(90, ge=1)
    
    # Capacity
    allocated_storage_gb: float = Field(0.0, ge=0)
    used_storage_gb: float = Field(0.0, ge=0)


class ComputeConfig(BaseModel):
    """Compute configuration for region/edge"""
    cluster_name: str = Field(...)
    node_type: str = Field("e2-standard-4")
    min_nodes: int = Field(1, ge=1)
    max_nodes: int = Field(10, ge=1)
    
    # Auto-scaling
    auto_scaling_enabled: bool = Field(True)
    cpu_target_percent: float = Field(70.0, ge=0, le=100)
    memory_target_percent: float = Field(80.0, ge=0, le=100)
    
    # Node configuration
    node_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Current state
    current_nodes: int = Field(0, ge=0)
    available_capacity: Dict[str, float] = Field(default_factory=dict)


class Region(BaseModel):
    """Primary region configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1)
    code: str = Field(..., min_length=2, max_length=10)  # e.g., "us-east-1"
    
    # Geographic information
    location: GeographicLocation = Field(...)
    
    # Infrastructure
    provider: str = Field(...)  # aws, gcp, azure, etc.
    status: RegionStatus = Field(default=RegionStatus.ACTIVE)
    
    # Configurations
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    compute: ComputeConfig = Field(...)
    
    # Capabilities
    supports_gpu: bool = Field(default=False)
    supports_edge_functions: bool = Field(default=True)
    supports_real_time: bool = Field(default=True)
    
    # Priority and routing
    priority: int = Field(1, ge=1)  # Lower number = higher priority
    weight: float = Field(1.0, ge=0)  # For weighted routing
    
    # Health and metrics
    health_status: HealthStatus = Field(default=HealthStatus.UNKNOWN)
    metrics: ResourceMetrics = Field(default_factory=ResourceMetrics)
    
    # Cost tracking
    hourly_cost: float = Field(0.0, ge=0)
    monthly_cost: float = Field(0.0, ge=0)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_health_check: Optional[datetime] = Field(None)
    
    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("code")
    def validate_region_code(cls, v):
        # Ensure region code follows standard format
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Region code must be alphanumeric with hyphens/underscores")
        return v.lower()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EdgeLocation(BaseModel):
    """Edge location/CDN node configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1)
    code: str = Field(..., min_length=2, max_length=10)
    
    # Edge tier and capabilities
    tier: EdgeTier = Field(...)
    parent_region_id: str = Field(...)  # References primary region
    
    # Geographic information
    location: GeographicLocation = Field(...)
    
    # Infrastructure
    provider: str = Field(...)
    status: RegionStatus = Field(default=RegionStatus.ACTIVE)
    
    # Edge-specific config
    cache_size_gb: float = Field(10.0, ge=0)
    max_cache_ttl_seconds: int = Field(3600, ge=0)
    edge_functions_enabled: bool = Field(True)
    real_time_enabled: bool = Field(False)
    
    # Network configuration
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    
    # Routing configuration
    routing_rules: List[Dict[str, Any]] = Field(default_factory=list)
    failover_enabled: bool = Field(True)
    
    # Health and metrics
    health_status: HealthStatus = Field(default=HealthStatus.UNKNOWN)
    metrics: ResourceMetrics = Field(default_factory=ResourceMetrics)
    
    # Cost tracking
    hourly_cost: float = Field(0.0, ge=0)
    monthly_cost: float = Field(0.0, ge=0)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_health_check: Optional[datetime] = Field(None)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LoadBalancerConfig(BaseModel):
    """Load balancer configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1)
    
    # Strategy
    strategy: LoadBalancingStrategy = Field(default=LoadBalancingStrategy.GEOGRAPHIC)
    
    # Health check configuration
    health_check_path: str = Field("/health")
    health_check_interval_seconds: int = Field(30, ge=5)
    health_check_timeout_seconds: int = Field(5, ge=1)
    healthy_threshold: int = Field(2, ge=1)
    unhealthy_threshold: int = Field(3, ge=1)
    
    # Routing rules
    geographic_routing: bool = Field(True)
    latency_routing: bool = Field(True)
    failover_enabled: bool = Field(True)
    
    # Session affinity
    session_affinity: bool = Field(False)
    session_cookie_name: str = Field("nexus-forge-session")
    session_ttl_seconds: int = Field(3600, ge=0)
    
    # Rate limiting
    rate_limiting_enabled: bool = Field(True)
    requests_per_second_limit: int = Field(1000, ge=0)
    burst_limit: int = Field(2000, ge=0)
    
    # SSL/TLS
    ssl_enabled: bool = Field(True)
    ssl_certificate_arn: Optional[str] = Field(None)
    ssl_redirect: bool = Field(True)
    
    # Monitoring
    access_logs_enabled: bool = Field(True)
    metrics_enabled: bool = Field(True)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DataSyncConfig(BaseModel):
    """Data synchronization configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1)
    
    # Sync strategy
    replication_mode: ReplicationMode = Field(default=ReplicationMode.ASYNC)
    
    # Source and targets
    source_region_id: str = Field(...)
    target_region_ids: List[str] = Field(...)
    
    # Sync configuration
    sync_interval_seconds: int = Field(300, ge=60)  # 5 minutes minimum
    batch_size: int = Field(1000, ge=1)
    max_retry_attempts: int = Field(3, ge=1)
    
    # Data filters
    table_patterns: List[str] = Field(default_factory=list)  # Tables to sync
    exclude_patterns: List[str] = Field(default_factory=list)  # Tables to exclude
    
    # Conflict resolution
    conflict_resolution: str = Field("timestamp")  # timestamp, source_wins, manual
    
    # Monitoring
    sync_monitoring_enabled: bool = Field(True)
    alert_on_failure: bool = Field(True)
    alert_on_lag_seconds: int = Field(600, ge=0)  # Alert if sync lags > 10 minutes
    
    # Status
    last_sync: Optional[datetime] = Field(None)
    sync_lag_seconds: int = Field(0, ge=0)
    failed_attempts: int = Field(0, ge=0)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class RegionConfig(BaseModel):
    """Global region configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Primary regions
    primary_regions: List[str] = Field(...)  # Region IDs
    
    # Edge locations
    edge_locations: List[str] = Field(default_factory=list)  # Edge location IDs
    
    # Global load balancer
    load_balancer: LoadBalancerConfig = Field(...)
    
    # Data synchronization
    data_sync_configs: List[DataSyncConfig] = Field(default_factory=list)
    
    # Failover configuration
    auto_failover_enabled: bool = Field(True)
    failover_threshold_seconds: int = Field(60, ge=30)
    manual_failover_enabled: bool = Field(True)
    
    # Global settings
    default_region_id: str = Field(...)  # Default region for new deployments
    maintenance_mode: bool = Field(default=False)
    
    # Monitoring
    global_monitoring_enabled: bool = Field(True)
    cross_region_health_checks: bool = Field(True)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TrafficRouting(BaseModel):
    """Traffic routing configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Routing rules
    geographic_rules: List[Dict[str, Any]] = Field(default_factory=list)
    latency_rules: List[Dict[str, Any]] = Field(default_factory=list)
    weighted_rules: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Failover configuration
    failover_regions: List[str] = Field(default_factory=list)
    automatic_failover: bool = Field(True)
    
    # A/B testing
    ab_testing_enabled: bool = Field(False)
    ab_test_rules: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Canary deployment
    canary_enabled: bool = Field(False)
    canary_percentage: float = Field(5.0, ge=0, le=100)
    canary_regions: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class RegionMetrics(BaseModel):
    """Aggregated metrics for a region"""
    region_id: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Performance metrics
    total_requests: int = Field(0, ge=0)
    successful_requests: int = Field(0, ge=0)
    failed_requests: int = Field(0, ge=0)
    avg_response_time_ms: float = Field(0.0, ge=0)
    
    # Resource utilization
    resource_metrics: ResourceMetrics = Field(default_factory=ResourceMetrics)
    
    # Cost metrics
    hourly_cost: float = Field(0.0, ge=0)
    daily_cost: float = Field(0.0, ge=0)
    
    # Data transfer
    data_in_gb: float = Field(0.0, ge=0)
    data_out_gb: float = Field(0.0, ge=0)
    cross_region_transfer_gb: float = Field(0.0, ge=0)


class GlobalMetrics(BaseModel):
    """Global metrics across all regions"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Total metrics
    total_regions: int = Field(0, ge=0)
    healthy_regions: int = Field(0, ge=0)
    total_edge_locations: int = Field(0, ge=0)
    
    # Performance
    global_requests_per_second: float = Field(0.0, ge=0)
    global_error_rate: float = Field(0.0, ge=0, le=100)
    global_avg_latency_ms: float = Field(0.0, ge=0)
    
    # Regional breakdown
    region_metrics: List[RegionMetrics] = Field(default_factory=list)
    
    # Cost
    total_hourly_cost: float = Field(0.0, ge=0)
    total_monthly_cost: float = Field(0.0, ge=0)
    
    # Data transfer
    total_data_transfer_gb: float = Field(0.0, ge=0)
    cross_region_transfer_gb: float = Field(0.0, ge=0)