"""
Tenant Context Middleware
Provides tenant isolation and context injection for all requests
"""

import asyncio
from typing import Dict, Optional, Any, Callable, Union
from fastapi import Request, HTTPException, status
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging
import re
from urllib.parse import urlparse
import jwt

from .models import Tenant, TenantStatus, TenantTier
from .tenant_manager import TenantManager
from ..core.cache import RedisCache
from ..core.exceptions import ValidationError, NotFoundError, PermissionError

logger = logging.getLogger(__name__)


class TenantContext:
    """Thread-local tenant context storage"""
    
    def __init__(self):
        self._tenant_id: Optional[str] = None
        self._tenant: Optional[Tenant] = None
        self._user_id: Optional[str] = None
        self._request_id: Optional[str] = None
    
    @property
    def tenant_id(self) -> Optional[str]:
        return self._tenant_id
    
    @property
    def tenant(self) -> Optional[Tenant]:
        return self._tenant
    
    @property
    def user_id(self) -> Optional[str]:
        return self._user_id
    
    @property
    def request_id(self) -> Optional[str]:
        return self._request_id
    
    def set_context(
        self, 
        tenant_id: str, 
        tenant: Tenant, 
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        self._tenant_id = tenant_id
        self._tenant = tenant
        self._user_id = user_id
        self._request_id = request_id
    
    def clear(self):
        self._tenant_id = None
        self._tenant = None
        self._user_id = None
        self._request_id = None


# Global tenant context instance
tenant_context = TenantContext()


class TenantContextMiddleware(BaseHTTPMiddleware):
    """Middleware for tenant context extraction and injection"""
    
    def __init__(
        self,
        app: ASGIApp,
        tenant_manager: Optional[TenantManager] = None,
        cache: Optional[RedisCache] = None,
        default_tenant: Optional[str] = None,
        domain_pattern: Optional[str] = None
    ):
        super().__init__(app)
        self.tenant_manager = tenant_manager or TenantManager()
        self.cache = cache or RedisCache()
        self.default_tenant = default_tenant
        self.domain_pattern = domain_pattern or r"^(?P<tenant>[a-zA-Z0-9-]+)\.nexusforge\.ai$"
        
        # Paths that don't require tenant context
        self.public_paths = {
            "/health",
            "/docs",
            "/openapi.json",
            "/favicon.ico",
            "/static/",
            "/auth/signup",
            "/auth/login",
            "/auth/callback",
            "/api/public/"
        }
        
        # Admin-only paths
        self.admin_paths = {
            "/admin/",
            "/api/admin/"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and inject tenant context"""
        
        # Skip tenant resolution for public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)
        
        try:
            # Extract tenant context from request
            tenant_info = await self._extract_tenant_context(request)
            
            if not tenant_info:
                # No tenant context available
                if self._requires_tenant_context(request.url.path):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Tenant context required"
                    )
                return await call_next(request)
            
            tenant_id, tenant, user_id = tenant_info
            
            # Validate tenant status
            await self._validate_tenant_access(tenant, request)
            
            # Check admin access for admin paths
            if self._is_admin_path(request.url.path):
                await self._validate_admin_access(user_id, tenant)
            
            # Set tenant context
            tenant_context.set_context(
                tenant_id=tenant_id,
                tenant=tenant,
                user_id=user_id,
                request_id=request.headers.get("x-request-id")
            )
            
            # Add tenant headers to request
            request.state.tenant_id = tenant_id
            request.state.tenant = tenant
            request.state.user_id = user_id
            
            # Process request with tenant context
            response = await call_next(request)
            
            # Add tenant info to response headers (for debugging)
            response.headers["x-tenant-id"] = tenant_id
            response.headers["x-tenant-slug"] = tenant.slug
            response.headers["x-tenant-tier"] = tenant.tier.value
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Tenant middleware error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
        finally:
            # Clear context after request
            tenant_context.clear()
    
    async def _extract_tenant_context(self, request: Request) -> Optional[tuple]:
        """Extract tenant context from request"""
        
        # Try different methods to identify tenant
        tenant_slug = None
        user_id = None
        
        # Method 1: Subdomain extraction
        tenant_slug = self._extract_from_subdomain(request)
        
        # Method 2: API key header
        if not tenant_slug:
            tenant_slug, user_id = await self._extract_from_api_key(request)
        
        # Method 3: JWT token
        if not tenant_slug:
            tenant_slug, user_id = await self._extract_from_jwt(request)
        
        # Method 4: Path parameter
        if not tenant_slug:
            tenant_slug = self._extract_from_path(request)
        
        # Method 5: Default tenant
        if not tenant_slug and self.default_tenant:
            tenant_slug = self.default_tenant
        
        if not tenant_slug:
            return None
        
        # Get tenant information
        try:
            tenant = await self.tenant_manager.get_tenant(tenant_slug, by_slug=True)
            return tenant.id, tenant, user_id
        except NotFoundError:
            logger.warning(f"Tenant not found: {tenant_slug}")
            return None
    
    def _extract_from_subdomain(self, request: Request) -> Optional[str]:
        """Extract tenant from subdomain"""
        host = request.headers.get("host", "")
        
        # Remove port if present
        host = host.split(":")[0]
        
        # Match subdomain pattern
        match = re.match(self.domain_pattern, host)
        if match:
            return match.group("tenant")
        
        return None
    
    async def _extract_from_api_key(self, request: Request) -> tuple:
        """Extract tenant from API key header"""
        api_key = request.headers.get("x-api-key") or request.headers.get("authorization")
        
        if not api_key:
            return None, None
        
        # Remove "Bearer " prefix if present
        if api_key.startswith("Bearer "):
            api_key = api_key[7:]
        
        # Look up API key in cache first
        cache_key = f"api_key:{api_key}"
        cached_info = await self.cache.get(cache_key)
        
        if cached_info:
            return cached_info.get("tenant_slug"), cached_info.get("user_id")
        
        # Query database for API key
        # This would require an API keys table
        # For now, return None
        return None, None
    
    async def _extract_from_jwt(self, request: Request) -> tuple:
        """Extract tenant from JWT token"""
        auth_header = request.headers.get("authorization")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            return None, None
        
        token = auth_header[7:]
        
        try:
            # In production, you'd verify the JWT signature
            # For now, just decode without verification
            payload = jwt.decode(token, options={"verify_signature": False})
            
            tenant_slug = payload.get("tenant")
            user_id = payload.get("sub") or payload.get("user_id")
            
            return tenant_slug, user_id
            
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None, None
    
    def _extract_from_path(self, request: Request) -> Optional[str]:
        """Extract tenant from URL path parameter"""
        path = request.url.path
        
        # Look for /tenant/{slug}/ pattern
        match = re.match(r"^/tenant/([a-zA-Z0-9-]+)/", path)
        if match:
            return match.group(1)
        
        return None
    
    async def _validate_tenant_access(self, tenant: Tenant, request: Request) -> None:
        """Validate tenant can access the system"""
        
        # Check tenant status
        if tenant.status == TenantStatus.SUSPENDED:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant account is suspended"
            )
        
        if tenant.status == TenantStatus.CANCELLED:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant account is cancelled"
            )
        
        # Check trial expiration
        if tenant.status == TenantStatus.TRIAL and tenant.trial_ends_at:
            from datetime import datetime
            if datetime.utcnow() > tenant.trial_ends_at:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail="Trial period has expired"
                )
        
        # Check rate limits
        await self._check_rate_limits(tenant, request)
    
    async def _check_rate_limits(self, tenant: Tenant, request: Request) -> None:
        """Check API rate limits for tenant"""
        if not tenant.quota.api_requests_per_minute:
            return  # No rate limiting
        
        # Check minute-based rate limit
        minute_key = f"rate_limit:{tenant.id}:minute:{int(asyncio.get_event_loop().time() // 60)}"
        minute_count = await self.cache.get(minute_key) or 0
        
        if minute_count >= tenant.quota.api_requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": "60"}
            )
        
        # Increment counter
        await self.cache.set(minute_key, minute_count + 1, ttl=60)
        
        # Check daily rate limit if configured
        if tenant.quota.api_requests_per_day:
            from datetime import datetime
            today = datetime.utcnow().strftime("%Y-%m-%d")
            day_key = f"rate_limit:{tenant.id}:day:{today}"
            day_count = await self.cache.get(day_key) or 0
            
            if day_count >= tenant.quota.api_requests_per_day:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Daily rate limit exceeded",
                    headers={"Retry-After": "86400"}
                )
            
            await self.cache.set(day_key, day_count + 1, ttl=86400)
    
    async def _validate_admin_access(self, user_id: Optional[str], tenant: Tenant) -> None:
        """Validate admin access for admin paths"""
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        # Check if user is tenant admin
        # This would require querying the tenant_users table
        # For now, basic validation
        if tenant.admin_email:  # Simple check
            # In production, you'd verify the user has admin role
            pass
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
    
    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (doesn't require tenant context)"""
        return any(path.startswith(public_path) for public_path in self.public_paths)
    
    def _is_admin_path(self, path: str) -> bool:
        """Check if path requires admin access"""
        return any(path.startswith(admin_path) for admin_path in self.admin_paths)
    
    def _requires_tenant_context(self, path: str) -> bool:
        """Check if path requires tenant context"""
        # Most API paths require tenant context
        return path.startswith("/api/") and not path.startswith("/api/public/")


class TenantDatabaseMiddleware:
    """Middleware for database connection routing based on tenant"""
    
    def __init__(self, isolation_manager):
        self.isolation_manager = isolation_manager
    
    async def get_database_connection(self, tenant: Tenant):
        """Get appropriate database connection for tenant"""
        
        if tenant.tier == TenantTier.ENTERPRISE:
            # Enterprise tenants may have dedicated databases
            return await self._get_dedicated_connection(tenant)
        
        elif tenant.tier == TenantTier.PROFESSIONAL:
            # Professional tenants use dedicated schemas
            return await self._get_schema_connection(tenant)
        
        else:
            # Basic tenants use shared database with RLS
            return await self._get_shared_connection(tenant)
    
    async def _get_dedicated_connection(self, tenant: Tenant):
        """Get dedicated database connection for enterprise tenant"""
        # Implementation would return tenant-specific database connection
        pass
    
    async def _get_schema_connection(self, tenant: Tenant):
        """Get schema-isolated connection for professional tenant"""
        # Implementation would return connection with tenant schema
        pass
    
    async def _get_shared_connection(self, tenant: Tenant):
        """Get shared connection with RLS for basic tenant"""
        # Implementation would return shared connection with tenant context
        pass


# Utility functions for accessing tenant context

def get_current_tenant() -> Optional[Tenant]:
    """Get current tenant from context"""
    return tenant_context.tenant

def get_current_tenant_id() -> Optional[str]:
    """Get current tenant ID from context"""
    return tenant_context.tenant_id

def get_current_user_id() -> Optional[str]:
    """Get current user ID from context"""
    return tenant_context.user_id

def require_tenant_context() -> Tenant:
    """Require tenant context, raise exception if not available"""
    tenant = get_current_tenant()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context required"
        )
    return tenant

def require_user_context() -> str:
    """Require user context, raise exception if not available"""
    user_id = get_current_user_id()
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User authentication required"
        )
    return user_id


# Decorators for tenant-aware operations

def tenant_aware(func):
    """Decorator to inject tenant context into function"""
    async def wrapper(*args, **kwargs):
        tenant = get_current_tenant()
        if tenant:
            kwargs["tenant"] = tenant
        return await func(*args, **kwargs)
    return wrapper

def require_tenant(func):
    """Decorator that requires tenant context"""
    async def wrapper(*args, **kwargs):
        tenant = require_tenant_context()
        kwargs["tenant"] = tenant
        return await func(*args, **kwargs)
    return wrapper

def admin_required(func):
    """Decorator that requires admin access"""
    async def wrapper(*args, **kwargs):
        tenant = require_tenant_context()
        user_id = require_user_context()
        
        # Check admin permissions
        # This would need proper implementation
        
        kwargs["tenant"] = tenant
        kwargs["user_id"] = user_id
        return await func(*args, **kwargs)
    return wrapper