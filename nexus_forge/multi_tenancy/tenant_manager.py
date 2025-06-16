"""
Tenant Manager
Orchestrates tenant lifecycle operations and management
"""

import asyncio
import secrets
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .models import (
    Tenant, TenantUser, TenantInvitation, TenantAuditLog,
    TenantTier, TenantStatus, ResourceQuota, ResourceUsage
)
from .isolation_manager import TenantIsolationManager
from ..core.cache import RedisCache
from ..integrations.supabase.coordination_client import SupabaseCoordinationClient
from ..core.exceptions import ValidationError, NotFoundError, PermissionError

logger = logging.getLogger(__name__)


class TenantManager:
    """Manages tenant lifecycle and operations"""
    
    def __init__(self):
        self.isolation_manager = TenantIsolationManager()
        self.cache = RedisCache()
        self.supabase = SupabaseCoordinationClient()
        
        # Default quotas by tier
        self.default_quotas = {
            TenantTier.BASIC: ResourceQuota(
                cpu_cores=2.0,
                memory_gb=4.0,
                storage_gb=10.0,
                api_requests_per_minute=100,
                api_requests_per_day=10000,
                max_agents=5,
                max_workflows=10,
                max_executions_per_day=100,
                max_users=5,
                max_projects=3
            ),
            TenantTier.PROFESSIONAL: ResourceQuota(
                cpu_cores=8.0,
                memory_gb=16.0,
                storage_gb=100.0,
                api_requests_per_minute=500,
                api_requests_per_day=100000,
                max_agents=25,
                max_workflows=100,
                max_executions_per_day=1000,
                max_users=25,
                max_projects=10
            ),
            TenantTier.ENTERPRISE: ResourceQuota(
                cpu_cores=32.0,
                memory_gb=64.0,
                storage_gb=1000.0,
                api_requests_per_minute=2000,
                api_requests_per_day=1000000,
                max_agents=100,
                max_workflows=500,
                max_executions_per_day=10000,
                max_users=100,
                max_projects=50
            )
        }
    
    async def create_tenant(
        self,
        name: str,
        slug: str,
        admin_email: str,
        admin_name: str,
        tier: TenantTier = TenantTier.TRIAL,
        trial_days: int = 30
    ) -> Tenant:
        """Create a new tenant with isolation"""
        logger.info(f"Creating tenant: {slug} ({tier})")
        
        # Validate slug uniqueness
        await self._validate_slug_uniqueness(slug)
        
        # Create tenant
        tenant = Tenant(
            name=name,
            slug=slug,
            tier=tier,
            status=TenantStatus.TRIAL if tier == TenantTier.BASIC else TenantStatus.ACTIVE,
            admin_email=admin_email,
            admin_name=admin_name,
            quota=self.default_quotas.get(tier, self.default_quotas[TenantTier.BASIC]),
            trial_ends_at=datetime.utcnow() + timedelta(days=trial_days) if tier == TenantTier.BASIC else None
        )
        
        try:
            # Save tenant to database
            await self._save_tenant(tenant)
            
            # Create isolation infrastructure
            resources = await self.isolation_manager.create_tenant_isolation(tenant)
            tenant.resources = resources
            
            # Create admin user association
            await self._create_admin_user(tenant, admin_email)
            
            # Log creation
            await self._log_audit_event(
                tenant.id,
                "tenant_created",
                "tenant",
                f"Tenant {tenant.name} created",
                user_email=admin_email
            )
            
            # Clear tenant cache
            await self._clear_tenant_cache(tenant.slug)
            
            logger.info(f"Successfully created tenant: {slug}")
            return tenant
            
        except Exception as e:
            # Cleanup on failure
            logger.error(f"Failed to create tenant {slug}: {e}")
            await self._cleanup_failed_tenant(tenant)
            raise
    
    async def get_tenant(self, identifier: str, by_slug: bool = True) -> Tenant:
        """Get tenant by slug or ID"""
        # Check cache first
        cache_key = f"tenant:{'slug' if by_slug else 'id'}:{identifier}"
        cached = await self.cache.get(cache_key)
        if cached:
            return Tenant(**cached)
        
        # Query database
        if by_slug:
            result = await self.supabase.client.table("tenants") \
                .select("*") \
                .eq("slug", identifier) \
                .execute()
        else:
            result = await self.supabase.client.table("tenants") \
                .select("*") \
                .eq("id", identifier) \
                .execute()
        
        if not result.data:
            raise NotFoundError(f"Tenant {identifier} not found")
        
        tenant = Tenant(**result.data[0])
        
        # Load resources
        tenant.resources = await self._load_tenant_resources(tenant.id)
        
        # Cache result
        await self.cache.set(cache_key, tenant.dict(), ttl=1800)  # 30 minutes
        
        return tenant
    
    async def update_tenant(self, tenant: Tenant) -> Tenant:
        """Update tenant information"""
        tenant.updated_at = datetime.utcnow()
        
        # Save to database
        await self._update_tenant(tenant)
        
        # Clear cache
        await self._clear_tenant_cache(tenant.slug)
        
        # Log update
        await self._log_audit_event(
            tenant.id,
            "tenant_updated",
            "tenant",
            f"Tenant {tenant.name} updated"
        )
        
        return tenant
    
    async def upgrade_tenant(
        self, tenant_id: str, new_tier: TenantTier, user_id: str
    ) -> Tenant:
        """Upgrade tenant to a new tier"""
        tenant = await self.get_tenant(tenant_id, by_slug=False)
        old_tier = tenant.tier
        
        logger.info(f"Upgrading tenant {tenant.slug} from {old_tier} to {new_tier}")
        
        if new_tier <= old_tier:
            raise ValidationError("Can only upgrade to higher tier")
        
        try:
            # Update tenant tier and quota
            tenant.tier = new_tier
            tenant.quota = self.default_quotas[new_tier]
            tenant.status = TenantStatus.ACTIVE
            tenant.trial_ends_at = None  # Remove trial limitation
            
            # Update isolation infrastructure
            new_resources = await self.isolation_manager.update_tenant_isolation(
                tenant, new_tier
            )
            
            # Save tenant
            await self._update_tenant(tenant)
            
            # Log upgrade
            await self._log_audit_event(
                tenant.id,
                "tenant_upgraded",
                "tenant",
                f"Tenant upgraded from {old_tier.value} to {new_tier.value}",
                user_id=user_id,
                metadata={"old_tier": old_tier.value, "new_tier": new_tier.value}
            )
            
            # Clear cache
            await self._clear_tenant_cache(tenant.slug)
            
            logger.info(f"Successfully upgraded tenant {tenant.slug}")
            return tenant
            
        except Exception as e:
            logger.error(f"Failed to upgrade tenant {tenant.slug}: {e}")
            # Rollback on failure
            tenant.tier = old_tier
            await self._update_tenant(tenant)
            raise
    
    async def suspend_tenant(self, tenant_id: str, reason: str, user_id: str) -> Tenant:
        """Suspend a tenant"""
        tenant = await self.get_tenant(tenant_id, by_slug=False)
        
        if tenant.status == TenantStatus.SUSPENDED:
            raise ValidationError("Tenant is already suspended")
        
        tenant.status = TenantStatus.SUSPENDED
        tenant.suspended_at = datetime.utcnow()
        
        # Save changes
        await self._update_tenant(tenant)
        
        # Log suspension
        await self._log_audit_event(
            tenant.id,
            "tenant_suspended",
            "tenant",
            f"Tenant suspended: {reason}",
            user_id=user_id,
            metadata={"reason": reason}
        )
        
        # Clear cache
        await self._clear_tenant_cache(tenant.slug)
        
        # TODO: Implement resource suspension (scale down, disable access)
        
        return tenant
    
    async def reactivate_tenant(self, tenant_id: str, user_id: str) -> Tenant:
        """Reactivate a suspended tenant"""
        tenant = await self.get_tenant(tenant_id, by_slug=False)
        
        if tenant.status != TenantStatus.SUSPENDED:
            raise ValidationError("Tenant is not suspended")
        
        tenant.status = TenantStatus.ACTIVE
        tenant.suspended_at = None
        
        # Save changes
        await self._update_tenant(tenant)
        
        # Log reactivation
        await self._log_audit_event(
            tenant.id,
            "tenant_reactivated",
            "tenant",
            "Tenant reactivated",
            user_id=user_id
        )
        
        # Clear cache
        await self._clear_tenant_cache(tenant.slug)
        
        # TODO: Implement resource reactivation
        
        return tenant
    
    async def delete_tenant(self, tenant_id: str, user_id: str) -> bool:
        """Delete a tenant and all its resources"""
        tenant = await self.get_tenant(tenant_id, by_slug=False)
        
        logger.info(f"Deleting tenant: {tenant.slug}")
        
        try:
            # Delete isolation infrastructure
            await self.isolation_manager.delete_tenant_isolation(tenant)
            
            # Delete tenant data (cascade will handle related records)
            await self.supabase.client.table("tenants") \
                .delete() \
                .eq("id", tenant_id) \
                .execute()
            
            # Log deletion
            await self._log_audit_event(
                tenant.id,
                "tenant_deleted",
                "tenant",
                f"Tenant {tenant.name} deleted",
                user_id=user_id
            )
            
            # Clear cache
            await self._clear_tenant_cache(tenant.slug)
            
            logger.info(f"Successfully deleted tenant: {tenant.slug}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete tenant {tenant.slug}: {e}")
            return False
    
    # User management
    
    async def invite_user(
        self,
        tenant_id: str,
        email: str,
        role: str,
        invited_by: str,
        message: Optional[str] = None
    ) -> TenantInvitation:
        """Invite a user to join a tenant"""
        tenant = await self.get_tenant(tenant_id, by_slug=False)
        
        # Check if user is already a member
        existing = await self._get_tenant_user(tenant_id, email)
        if existing:
            raise ValidationError("User is already a member of this tenant")
        
        # Check user limits
        current_users = await self._count_tenant_users(tenant_id)
        if tenant.quota.max_users and current_users >= tenant.quota.max_users:
            raise ValidationError("Tenant user limit reached")
        
        # Create invitation
        invitation = TenantInvitation(
            tenant_id=tenant_id,
            email=email,
            role=role,
            invited_by=invited_by,
            invitation_token=secrets.token_urlsafe(32),
            message=message,
            expires_at=datetime.utcnow() + timedelta(days=7)
        )
        
        # Save invitation
        await self._save_invitation(invitation)
        
        # Send invitation email
        await self._send_invitation_email(invitation, tenant)
        
        # Log invitation
        await self._log_audit_event(
            tenant_id,
            "user_invited",
            "user",
            f"User {email} invited with role {role}",
            user_id=invited_by,
            metadata={"email": email, "role": role}
        )
        
        return invitation
    
    async def accept_invitation(self, token: str, user_id: str) -> TenantUser:
        """Accept a tenant invitation"""
        # Get invitation
        invitation = await self._get_invitation_by_token(token)
        
        if not invitation:
            raise NotFoundError("Invalid invitation token")
        
        if invitation.expires_at < datetime.utcnow():
            raise ValidationError("Invitation has expired")
        
        if invitation.status != "pending":
            raise ValidationError("Invitation is no longer valid")
        
        # Create tenant user
        tenant_user = TenantUser(
            tenant_id=invitation.tenant_id,
            user_id=user_id,
            role=invitation.role,
            invitation_accepted=True
        )
        
        # Save tenant user
        await self._save_tenant_user(tenant_user)
        
        # Update invitation
        invitation.status = "accepted"
        invitation.accepted_at = datetime.utcnow()
        await self._update_invitation(invitation)
        
        # Log acceptance
        await self._log_audit_event(
            invitation.tenant_id,
            "invitation_accepted",
            "user",
            f"User accepted invitation",
            user_id=user_id,
            metadata={"invitation_id": invitation.id}
        )
        
        return tenant_user
    
    async def remove_user(
        self, tenant_id: str, user_id: str, removed_by: str
    ) -> bool:
        """Remove a user from a tenant"""
        # Remove tenant user association
        await self.supabase.client.table("tenant_users") \
            .delete() \
            .eq("tenant_id", tenant_id) \
            .eq("user_id", user_id) \
            .execute()
        
        # Log removal
        await self._log_audit_event(
            tenant_id,
            "user_removed",
            "user",
            "User removed from tenant",
            user_id=removed_by,
            metadata={"removed_user_id": user_id}
        )
        
        return True
    
    async def update_user_role(
        self, tenant_id: str, user_id: str, new_role: str, updated_by: str
    ) -> TenantUser:
        """Update a user's role in a tenant"""
        # Update role
        await self.supabase.client.table("tenant_users") \
            .update({"role": new_role}) \
            .eq("tenant_id", tenant_id) \
            .eq("user_id", user_id) \
            .execute()
        
        # Get updated user
        tenant_user = await self._get_tenant_user_by_id(tenant_id, user_id)
        
        # Log role change
        await self._log_audit_event(
            tenant_id,
            "user_role_updated",
            "user",
            f"User role updated to {new_role}",
            user_id=updated_by,
            metadata={"target_user_id": user_id, "new_role": new_role}
        )
        
        return tenant_user
    
    # Resource management
    
    async def update_resource_usage(
        self, tenant_id: str, usage: ResourceUsage
    ) -> None:
        """Update tenant resource usage"""
        usage.last_updated = datetime.utcnow()
        
        # Update in database
        await self.supabase.client.table("tenants") \
            .update({"usage": usage.dict()}) \
            .eq("id", tenant_id) \
            .execute()
        
        # Update cache
        cache_key = f"tenant:usage:{tenant_id}"
        await self.cache.set(cache_key, usage.dict(), ttl=300)  # 5 minutes
    
    async def check_quota_limits(
        self, tenant_id: str, resource_type: str, requested_amount: float = 1
    ) -> Tuple[bool, Optional[str]]:
        """Check if tenant can allocate more resources"""
        tenant = await self.get_tenant(tenant_id, by_slug=False)
        
        if resource_type == "cpu_cores":
            if tenant.quota.cpu_cores:
                available = tenant.quota.cpu_cores - tenant.usage.cpu_usage
                if requested_amount > available:
                    return False, f"CPU limit exceeded. Available: {available:.2f} cores"
        
        elif resource_type == "memory_gb":
            if tenant.quota.memory_gb:
                available = tenant.quota.memory_gb - tenant.usage.memory_usage_gb
                if requested_amount > available:
                    return False, f"Memory limit exceeded. Available: {available:.2f} GB"
        
        elif resource_type == "storage_gb":
            if tenant.quota.storage_gb:
                available = tenant.quota.storage_gb - tenant.usage.storage_usage_gb
                if requested_amount > available:
                    return False, f"Storage limit exceeded. Available: {available:.2f} GB"
        
        elif resource_type == "agents":
            if tenant.quota.max_agents:
                if tenant.usage.current_agents + requested_amount > tenant.quota.max_agents:
                    return False, f"Agent limit exceeded. Limit: {tenant.quota.max_agents}"
        
        elif resource_type == "workflows":
            if tenant.quota.max_workflows:
                if tenant.usage.current_workflows + requested_amount > tenant.quota.max_workflows:
                    return False, f"Workflow limit exceeded. Limit: {tenant.quota.max_workflows}"
        
        elif resource_type == "users":
            if tenant.quota.max_users:
                if tenant.usage.current_users + requested_amount > tenant.quota.max_users:
                    return False, f"User limit exceeded. Limit: {tenant.quota.max_users}"
        
        return True, None
    
    async def get_tenant_usage_report(self, tenant_id: str) -> Dict[str, Any]:
        """Get detailed usage report for tenant"""
        tenant = await self.get_tenant(tenant_id, by_slug=False)
        
        # Calculate usage percentages
        usage_percentages = {}
        
        if tenant.quota.cpu_cores:
            usage_percentages["cpu"] = (tenant.usage.cpu_usage / tenant.quota.cpu_cores) * 100
        
        if tenant.quota.memory_gb:
            usage_percentages["memory"] = (tenant.usage.memory_usage_gb / tenant.quota.memory_gb) * 100
        
        if tenant.quota.storage_gb:
            usage_percentages["storage"] = (tenant.usage.storage_usage_gb / tenant.quota.storage_gb) * 100
        
        # Get cost information
        total_cost = sum(r.monthly_cost or 0 for r in tenant.resources)
        
        return {
            "tenant_id": tenant_id,
            "tenant_name": tenant.name,
            "tier": tenant.tier.value,
            "status": tenant.status.value,
            "quota": tenant.quota.dict(),
            "usage": tenant.usage.dict(),
            "usage_percentages": usage_percentages,
            "estimated_monthly_cost": total_cost,
            "last_updated": tenant.usage.last_updated.isoformat()
        }
    
    # Audit and monitoring
    
    async def get_audit_logs(
        self,
        tenant_id: str,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[TenantAuditLog]:
        """Get tenant audit logs"""
        query = self.supabase.client.table("tenant_audit_logs") \
            .select("*") \
            .eq("tenant_id", tenant_id)
        
        if event_type:
            query = query.eq("event_type", event_type)
        
        if user_id:
            query = query.eq("user_id", user_id)
        
        query = query.order("created_at", desc=True)
        query = query.range(offset, offset + limit - 1)
        
        result = await query.execute()
        
        return [TenantAuditLog(**log) for log in result.data]
    
    # Private helper methods
    
    async def _validate_slug_uniqueness(self, slug: str) -> None:
        """Validate that tenant slug is unique"""
        result = await self.supabase.client.table("tenants") \
            .select("id") \
            .eq("slug", slug) \
            .execute()
        
        if result.data:
            raise ValidationError(f"Tenant slug '{slug}' is already taken")
    
    async def _save_tenant(self, tenant: Tenant) -> None:
        """Save tenant to database"""
        await self.supabase.client.table("tenants").insert(
            tenant.dict()
        ).execute()
    
    async def _update_tenant(self, tenant: Tenant) -> None:
        """Update tenant in database"""
        await self.supabase.client.table("tenants") \
            .update(tenant.dict()) \
            .eq("id", tenant.id) \
            .execute()
    
    async def _load_tenant_resources(self, tenant_id: str) -> List:
        """Load tenant resources from database"""
        result = await self.supabase.client.table("tenant_resources") \
            .select("*") \
            .eq("tenant_id", tenant_id) \
            .execute()
        
        return [TenantResource(**res) for res in result.data] if result.data else []
    
    async def _create_admin_user(self, tenant: Tenant, admin_email: str) -> None:
        """Create admin user association"""
        # This assumes the user already exists in auth.users
        # In practice, you might create the user here if they don't exist
        
        tenant_user = TenantUser(
            tenant_id=tenant.id,
            user_id="admin-user-id",  # This would be the actual user ID
            role="admin",
            invitation_accepted=True
        )
        
        await self._save_tenant_user(tenant_user)
    
    async def _cleanup_failed_tenant(self, tenant: Tenant) -> None:
        """Cleanup resources for failed tenant creation"""
        try:
            # Delete any created isolation resources
            await self.isolation_manager.delete_tenant_isolation(tenant)
            
            # Delete tenant record if it was saved
            await self.supabase.client.table("tenants") \
                .delete() \
                .eq("id", tenant.id) \
                .execute()
                
        except Exception as e:
            logger.error(f"Failed to cleanup failed tenant {tenant.slug}: {e}")
    
    async def _clear_tenant_cache(self, slug: str) -> None:
        """Clear all cache entries for a tenant"""
        keys_to_delete = [
            f"tenant:slug:{slug}",
            f"tenant:usage:{slug}",
            f"tenant:resources:{slug}"
        ]
        
        for key in keys_to_delete:
            await self.cache.delete(key)
    
    async def _log_audit_event(
        self,
        tenant_id: str,
        event_type: str,
        event_category: str,
        description: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log audit event"""
        audit_log = TenantAuditLog(
            tenant_id=tenant_id,
            event_type=event_type,
            event_category=event_category,
            description=description,
            user_id=user_id,
            user_email=user_email,
            resource_type=resource_type,
            resource_id=resource_id,
            metadata=metadata or {}
        )
        
        await self.supabase.client.table("tenant_audit_logs").insert(
            audit_log.dict()
        ).execute()
    
    # User management helpers
    
    async def _get_tenant_user(self, tenant_id: str, email: str) -> Optional[TenantUser]:
        """Get tenant user by email"""
        # This would need to join with users table to match email
        # For now, simplified implementation
        return None
    
    async def _get_tenant_user_by_id(self, tenant_id: str, user_id: str) -> TenantUser:
        """Get tenant user by ID"""
        result = await self.supabase.client.table("tenant_users") \
            .select("*") \
            .eq("tenant_id", tenant_id) \
            .eq("user_id", user_id) \
            .execute()
        
        if not result.data:
            raise NotFoundError("Tenant user not found")
        
        return TenantUser(**result.data[0])
    
    async def _count_tenant_users(self, tenant_id: str) -> int:
        """Count current tenant users"""
        result = await self.supabase.client.table("tenant_users") \
            .select("id", count="exact") \
            .eq("tenant_id", tenant_id) \
            .execute()
        
        return result.count or 0
    
    async def _save_tenant_user(self, tenant_user: TenantUser) -> None:
        """Save tenant user association"""
        await self.supabase.client.table("tenant_users").insert(
            tenant_user.dict()
        ).execute()
    
    async def _save_invitation(self, invitation: TenantInvitation) -> None:
        """Save tenant invitation"""
        await self.supabase.client.table("tenant_invitations").insert(
            invitation.dict()
        ).execute()
    
    async def _update_invitation(self, invitation: TenantInvitation) -> None:
        """Update invitation status"""
        await self.supabase.client.table("tenant_invitations") \
            .update(invitation.dict()) \
            .eq("id", invitation.id) \
            .execute()
    
    async def _get_invitation_by_token(self, token: str) -> Optional[TenantInvitation]:
        """Get invitation by token"""
        result = await self.supabase.client.table("tenant_invitations") \
            .select("*") \
            .eq("invitation_token", token) \
            .execute()
        
        if result.data:
            return TenantInvitation(**result.data[0])
        return None
    
    async def _send_invitation_email(
        self, invitation: TenantInvitation, tenant: Tenant
    ) -> None:
        """Send invitation email"""
        # In production, this would send actual email
        logger.info(f"Sending invitation email to {invitation.email} for tenant {tenant.name}")
        
        # Email would contain invitation link with token
        invitation_url = f"https://app.nexusforge.ai/invite/{invitation.invitation_token}"
        
        # Log the invitation URL for now
        logger.info(f"Invitation URL: {invitation_url}")