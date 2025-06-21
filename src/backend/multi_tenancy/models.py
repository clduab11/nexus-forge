"""
Multi-tenancy data models and schemas
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field, validator


class TenantTier(str, Enum):
    """Tenant subscription tiers"""

    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class TenantStatus(str, Enum):
    """Tenant status"""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    CANCELLED = "cancelled"


class IsolationLevel(str, Enum):
    """Tenant isolation levels"""

    SHARED = "shared"  # Shared infrastructure with logical separation
    DEDICATED = "dedicated"  # Dedicated compute with shared storage
    ISOLATED = "isolated"  # Complete infrastructure isolation


class ResourceQuota(BaseModel):
    """Resource quota configuration"""

    cpu_cores: Optional[float] = Field(None, description="CPU cores limit")
    memory_gb: Optional[float] = Field(None, description="Memory limit in GB")
    storage_gb: Optional[float] = Field(None, description="Storage limit in GB")
    bandwidth_gbps: Optional[float] = Field(None, description="Bandwidth limit in Gbps")

    # API limits
    api_requests_per_minute: Optional[int] = Field(
        None, description="API requests per minute"
    )
    api_requests_per_day: Optional[int] = Field(
        None, description="API requests per day"
    )

    # Agent limits
    max_agents: Optional[int] = Field(None, description="Maximum number of agents")
    max_workflows: Optional[int] = Field(
        None, description="Maximum number of workflows"
    )
    max_executions_per_day: Optional[int] = Field(
        None, description="Max workflow executions per day"
    )

    # Data limits
    max_users: Optional[int] = Field(None, description="Maximum number of users")
    max_projects: Optional[int] = Field(None, description="Maximum number of projects")

    @validator("*", pre=True)
    def validate_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError("Resource limits must be positive")
        return v


class ResourceUsage(BaseModel):
    """Current resource usage"""

    cpu_usage: float = Field(0.0, description="Current CPU usage")
    memory_usage_gb: float = Field(0.0, description="Current memory usage in GB")
    storage_usage_gb: float = Field(0.0, description="Current storage usage in GB")
    bandwidth_usage_gbps: float = Field(0.0, description="Current bandwidth usage")

    # Current counts
    current_agents: int = Field(0, description="Current number of agents")
    current_workflows: int = Field(0, description="Current number of workflows")
    current_users: int = Field(0, description="Current number of users")
    current_projects: int = Field(0, description="Current number of projects")

    # Daily usage (reset daily)
    api_requests_today: int = Field(0, description="API requests today")
    executions_today: int = Field(0, description="Workflow executions today")

    # Usage history
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class TenantConfiguration(BaseModel):
    """Tenant configuration settings"""

    # Branding
    company_name: str = Field(..., description="Company name")
    logo_url: Optional[str] = Field(None, description="Company logo URL")
    primary_color: Optional[str] = Field(None, description="Primary brand color")
    secondary_color: Optional[str] = Field(None, description="Secondary brand color")

    # Domain settings
    custom_domain: Optional[str] = Field(None, description="Custom domain")
    subdomain: Optional[str] = Field(None, description="Subdomain prefix")

    # Feature flags
    features: Dict[str, bool] = Field(
        default_factory=dict, description="Feature enablement flags"
    )

    # Integration settings
    integrations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Third-party integration configurations"
    )

    # Security settings
    sso_enabled: bool = Field(default=False, description="SSO enabled")
    sso_provider: Optional[str] = Field(None, description="SSO provider")
    sso_config: Dict[str, Any] = Field(
        default_factory=dict, description="SSO configuration"
    )

    # Audit settings
    audit_retention_days: int = Field(
        default=90, description="Audit log retention period"
    )
    export_enabled: bool = Field(default=True, description="Data export enabled")

    # Notification settings
    notification_email: Optional[EmailStr] = Field(
        None, description="Notification email"
    )
    webhook_url: Optional[str] = Field(
        None, description="Webhook URL for notifications"
    )

    @validator("custom_domain")
    def validate_domain(cls, v):
        if v and not v.replace(".", "").replace("-", "").isalnum():
            raise ValueError("Invalid domain format")
        return v


class TenantResource(BaseModel):
    """Tenant resource allocation"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = Field(..., description="Tenant ID")
    resource_type: str = Field(
        ..., description="Resource type (compute, storage, etc.)"
    )
    resource_id: str = Field(..., description="Resource identifier")

    # Resource details
    provider: str = Field(..., description="Cloud provider")
    region: str = Field(..., description="Region")
    zone: Optional[str] = Field(None, description="Availability zone")

    # Configuration
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Resource configuration"
    )

    # Status
    status: str = Field(default="active", description="Resource status")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Cost tracking
    hourly_cost: Optional[float] = Field(None, description="Hourly cost in USD")
    monthly_cost: Optional[float] = Field(None, description="Monthly cost in USD")


class Tenant(BaseModel):
    """Main tenant model"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Tenant name")
    slug: str = Field(..., description="URL-safe tenant identifier")

    # Subscription details
    tier: TenantTier = Field(..., description="Subscription tier")
    status: TenantStatus = Field(
        default=TenantStatus.TRIAL, description="Tenant status"
    )
    isolation_level: IsolationLevel = Field(default=IsolationLevel.SHARED)

    # Contact information
    admin_email: EmailStr = Field(..., description="Tenant administrator email")
    admin_name: str = Field(..., description="Tenant administrator name")
    billing_email: Optional[EmailStr] = Field(None, description="Billing contact email")

    # Configuration
    config: TenantConfiguration = Field(
        default_factory=TenantConfiguration, description="Tenant configuration"
    )

    # Resource management
    quota: ResourceQuota = Field(
        default_factory=ResourceQuota, description="Resource quotas"
    )

    usage: ResourceUsage = Field(
        default_factory=ResourceUsage, description="Current resource usage"
    )

    resources: List[TenantResource] = Field(
        default_factory=list, description="Allocated resources"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    trial_ends_at: Optional[datetime] = Field(None, description="Trial expiration date")
    suspended_at: Optional[datetime] = Field(None, description="Suspension date")

    # Billing
    subscription_id: Optional[str] = Field(None, description="External subscription ID")
    last_payment: Optional[datetime] = Field(None, description="Last payment date")
    next_billing: Optional[datetime] = Field(None, description="Next billing date")

    # Metadata
    tags: List[str] = Field(default_factory=list, description="Tenant tags")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @validator("slug")
    def validate_slug(cls, v):
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Slug must be alphanumeric with hyphens/underscores")
        if len(v) < 3 or len(v) > 50:
            raise ValueError("Slug must be 3-50 characters")
        return v.lower()

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class TenantUser(BaseModel):
    """Tenant user association"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = Field(..., description="Tenant ID")
    user_id: str = Field(..., description="User ID")

    # Role and permissions
    role: str = Field(..., description="User role within tenant")
    permissions: List[str] = Field(default_factory=list, description="User permissions")

    # Status
    status: str = Field(default="active", description="User status in tenant")
    invited_by: Optional[str] = Field(None, description="User who sent invitation")
    invitation_accepted: bool = Field(default=False, description="Invitation accepted")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: Optional[datetime] = Field(None, description="Last activity timestamp")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class TenantInvitation(BaseModel):
    """Tenant user invitation"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = Field(..., description="Tenant ID")
    email: EmailStr = Field(..., description="Invitee email")
    role: str = Field(..., description="Proposed role")

    # Invitation details
    invited_by: str = Field(..., description="User ID who sent invitation")
    invitation_token: str = Field(..., description="Invitation token")
    message: Optional[str] = Field(None, description="Invitation message")

    # Status
    status: str = Field(default="pending", description="Invitation status")
    expires_at: datetime = Field(..., description="Invitation expiration")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accepted_at: Optional[datetime] = Field(None, description="Acceptance timestamp")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class TenantAuditLog(BaseModel):
    """Tenant audit log entry"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = Field(..., description="Tenant ID")

    # Event details
    event_type: str = Field(..., description="Event type")
    event_category: str = Field(..., description="Event category")
    description: str = Field(..., description="Event description")

    # Actor information
    user_id: Optional[str] = Field(None, description="User who performed action")
    user_email: Optional[str] = Field(None, description="User email")
    user_ip: Optional[str] = Field(None, description="User IP address")
    user_agent: Optional[str] = Field(None, description="User agent")

    # Resource information
    resource_type: Optional[str] = Field(None, description="Affected resource type")
    resource_id: Optional[str] = Field(None, description="Affected resource ID")
    resource_name: Optional[str] = Field(None, description="Affected resource name")

    # Event data
    old_values: Optional[Dict[str, Any]] = Field(None, description="Previous values")
    new_values: Optional[Dict[str, Any]] = Field(None, description="New values")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    # Timestamp
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class TenantBilling(BaseModel):
    """Tenant billing information"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = Field(..., description="Tenant ID")

    # Billing period
    period_start: datetime = Field(..., description="Billing period start")
    period_end: datetime = Field(..., description="Billing period end")

    # Usage-based charges
    usage_charges: Dict[str, float] = Field(
        default_factory=dict, description="Usage-based charges by resource type"
    )

    # Fixed charges
    subscription_charge: float = Field(0.0, description="Fixed subscription charge")

    # Totals
    subtotal: float = Field(0.0, description="Subtotal before taxes")
    tax_amount: float = Field(0.0, description="Tax amount")
    total_amount: float = Field(0.0, description="Total amount due")

    # Status
    status: str = Field(default="draft", description="Billing status")
    paid_at: Optional[datetime] = Field(None, description="Payment date")

    # External references
    invoice_id: Optional[str] = Field(None, description="External invoice ID")
    payment_id: Optional[str] = Field(None, description="Payment transaction ID")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
