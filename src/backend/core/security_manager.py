"""
Enterprise Security Manager for Nexus Forge
Comprehensive security, compliance, and access control system
"""

import asyncio
import base64
import hashlib
import re
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import bcrypt
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from nexus_forge.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    SecurityError,
)
from nexus_forge.core.monitoring import get_logger

logger = get_logger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels"""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class Permission(Enum):
    """System permissions"""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    AGENT_CONTROL = "agent_control"
    WORKFLOW_MANAGE = "workflow_manage"
    SYSTEM_CONFIG = "system_config"


@dataclass
class SecurityPolicy:
    """Security policy definition"""

    policy_id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    severity: str  # 'low', 'medium', 'high', 'critical'
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class AuditEvent:
    """Security audit event"""

    event_id: str
    user_id: Optional[str]
    agent_id: Optional[str]
    action: str
    resource_type: str
    resource_id: Optional[str]
    result: str  # 'success', 'failure', 'blocked'
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    ip_address: Optional[str]
    user_agent: Optional[str]
    details: Dict[str, Any]
    timestamp: datetime


@dataclass
class UserRole:
    """User role with permissions"""

    role_id: str
    name: str
    description: str
    permissions: Set[Permission]
    security_level: SecurityLevel
    resource_access: Dict[str, List[str]]  # resource_type -> [resource_ids]


@dataclass
class SecurityContext:
    """Security context for operations"""

    user_id: Optional[str]
    agent_id: Optional[str]
    roles: List[UserRole]
    security_level: SecurityLevel
    permissions: Set[Permission]
    session_id: str
    ip_address: Optional[str]
    expires_at: datetime


class SecurityManager:
    """
    Enterprise-grade security manager for Nexus Forge
    Handles authentication, authorization, encryption, and compliance
    """

    def __init__(
        self,
        secret_key: str,
        encryption_key: Optional[str] = None,
        jwt_algorithm: str = "HS256",
        session_timeout_hours: int = 8,
        max_failed_attempts: int = 5,
        lockout_duration_minutes: int = 30,
    ):
        """Initialize security manager"""
        self.secret_key = secret_key
        self.jwt_algorithm = jwt_algorithm
        self.session_timeout_hours = session_timeout_hours
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration_minutes = lockout_duration_minutes

        # Setup encryption
        if encryption_key:
            self.encryption_key = encryption_key.encode()
        else:
            self.encryption_key = secrets.token_bytes(32)
        self.fernet = Fernet(base64.urlsafe_b64encode(self.encryption_key))

        # Security state
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.audit_events: List[AuditEvent] = []

        # Rate limiting
        self.rate_limits: Dict[str, List[datetime]] = {}

        # Initialize default security policies
        self._setup_default_policies()

        # Predefined roles
        self._setup_default_roles()

    def _setup_default_policies(self):
        """Setup default security policies"""
        policies = [
            SecurityPolicy(
                policy_id="password_complexity",
                name="Password Complexity",
                description="Enforce strong password requirements",
                rules=[
                    {"type": "min_length", "value": 12},
                    {"type": "require_uppercase", "value": True},
                    {"type": "require_lowercase", "value": True},
                    {"type": "require_numbers", "value": True},
                    {"type": "require_symbols", "value": True},
                    {"type": "no_common_passwords", "value": True},
                ],
                severity="high",
            ),
            SecurityPolicy(
                policy_id="session_security",
                name="Session Security",
                description="Session management and security controls",
                rules=[
                    {"type": "max_concurrent_sessions", "value": 3},
                    {"type": "session_timeout_hours", "value": 8},
                    {"type": "require_2fa", "value": True},
                    {"type": "ip_validation", "value": True},
                ],
                severity="high",
            ),
            SecurityPolicy(
                policy_id="api_security",
                name="API Security",
                description="API access and rate limiting",
                rules=[
                    {"type": "rate_limit_per_minute", "value": 100},
                    {"type": "require_authentication", "value": True},
                    {"type": "log_all_requests", "value": True},
                    {"type": "block_suspicious_patterns", "value": True},
                ],
                severity="medium",
            ),
            SecurityPolicy(
                policy_id="data_protection",
                name="Data Protection",
                description="Data encryption and privacy controls",
                rules=[
                    {"type": "encrypt_at_rest", "value": True},
                    {"type": "encrypt_in_transit", "value": True},
                    {"type": "pii_detection", "value": True},
                    {"type": "data_retention_days", "value": 90},
                ],
                severity="critical",
            ),
            SecurityPolicy(
                policy_id="agent_security",
                name="Agent Security",
                description="AI agent access and behavior controls",
                rules=[
                    {"type": "agent_authentication", "value": True},
                    {"type": "agent_capability_limits", "value": True},
                    {"type": "prompt_injection_protection", "value": True},
                    {"type": "output_filtering", "value": True},
                ],
                severity="high",
            ),
        ]

        for policy in policies:
            policy.created_at = datetime.now(timezone.utc)
            self.security_policies[policy.policy_id] = policy

    def _setup_default_roles(self):
        """Setup default user roles"""
        self.default_roles = {
            "admin": UserRole(
                role_id="admin",
                name="Administrator",
                description="Full system access",
                permissions={
                    Permission.READ,
                    Permission.WRITE,
                    Permission.DELETE,
                    Permission.EXECUTE,
                    Permission.ADMIN,
                    Permission.AGENT_CONTROL,
                    Permission.WORKFLOW_MANAGE,
                    Permission.SYSTEM_CONFIG,
                },
                security_level=SecurityLevel.TOP_SECRET,
                resource_access={"*": ["*"]},
            ),
            "agent_operator": UserRole(
                role_id="agent_operator",
                name="Agent Operator",
                description="Agent management and workflow control",
                permissions={
                    Permission.READ,
                    Permission.WRITE,
                    Permission.EXECUTE,
                    Permission.AGENT_CONTROL,
                    Permission.WORKFLOW_MANAGE,
                },
                security_level=SecurityLevel.SECRET,
                resource_access={"agents": ["*"], "workflows": ["*"], "tasks": ["*"]},
            ),
            "developer": UserRole(
                role_id="developer",
                name="Developer",
                description="Development and testing access",
                permissions={Permission.READ, Permission.WRITE, Permission.EXECUTE},
                security_level=SecurityLevel.CONFIDENTIAL,
                resource_access={
                    "agents": ["read"],
                    "workflows": ["read", "create"],
                    "tasks": ["read"],
                },
            ),
            "viewer": UserRole(
                role_id="viewer",
                name="Viewer",
                description="Read-only access",
                permissions={Permission.READ},
                security_level=SecurityLevel.INTERNAL,
                resource_access={
                    "agents": ["read"],
                    "workflows": ["read"],
                    "tasks": ["read"],
                },
            ),
        }

    # Authentication
    async def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[SecurityContext]:
        """Authenticate user and create security context"""
        try:
            # Check for account lockout
            if await self._is_account_locked(username):
                await self._log_audit_event(
                    None,
                    None,
                    "login_attempt",
                    "user",
                    username,
                    "blocked",
                    "high",
                    ip_address,
                    user_agent,
                    {"reason": "account_locked"},
                )
                raise AuthenticationError(
                    "Account is locked due to too many failed attempts"
                )

            # Validate credentials (mock implementation)
            user_data = await self._validate_credentials(username, password)
            if not user_data:
                await self._record_failed_attempt(username)
                await self._log_audit_event(
                    None,
                    None,
                    "login_attempt",
                    "user",
                    username,
                    "failure",
                    "medium",
                    ip_address,
                    user_agent,
                    {"reason": "invalid_credentials"},
                )
                raise AuthenticationError("Invalid credentials")

            # Create security context
            context = await self._create_security_context(
                user_data, ip_address, user_agent
            )

            # Clear failed attempts
            self.failed_attempts.pop(username, None)

            # Log successful authentication
            await self._log_audit_event(
                context.user_id,
                None,
                "login_success",
                "user",
                context.user_id,
                "success",
                "low",
                ip_address,
                user_agent,
                {},
            )

            return context

        except Exception as e:
            logger.error(f"Authentication failed for {username}: {e}")
            raise

    async def authenticate_agent(
        self, agent_id: str, api_key: str, ip_address: Optional[str] = None
    ) -> Optional[SecurityContext]:
        """Authenticate AI agent"""
        try:
            # Validate agent credentials
            agent_data = await self._validate_agent_credentials(agent_id, api_key)
            if not agent_data:
                await self._log_audit_event(
                    None,
                    agent_id,
                    "agent_auth_attempt",
                    "agent",
                    agent_id,
                    "failure",
                    "high",
                    ip_address,
                    None,
                    {"reason": "invalid_api_key"},
                )
                raise AuthenticationError("Invalid agent credentials")

            # Create agent security context
            context = SecurityContext(
                user_id=None,
                agent_id=agent_id,
                roles=[self.default_roles["agent_operator"]],  # Default agent role
                security_level=SecurityLevel.SECRET,
                permissions={
                    Permission.READ,
                    Permission.WRITE,
                    Permission.EXECUTE,
                    Permission.AGENT_CONTROL,
                },
                session_id=secrets.token_urlsafe(32),
                ip_address=ip_address,
                expires_at=datetime.now(timezone.utc)
                + timedelta(hours=24),  # Longer for agents
            )

            self.active_sessions[context.session_id] = context

            await self._log_audit_event(
                None,
                agent_id,
                "agent_auth_success",
                "agent",
                agent_id,
                "success",
                "low",
                ip_address,
                None,
                {},
            )

            return context

        except Exception as e:
            logger.error(f"Agent authentication failed for {agent_id}: {e}")
            raise

    # Authorization
    async def authorize_action(
        self,
        context: SecurityContext,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
    ) -> bool:
        """Authorize action against security policies"""
        try:
            # Check session validity
            if not await self._is_session_valid(context):
                raise AuthorizationError("Session expired or invalid")

            # Check required permission
            required_permission = self._get_required_permission(action)
            if required_permission not in context.permissions:
                await self._log_audit_event(
                    context.user_id,
                    context.agent_id,
                    action,
                    resource_type,
                    resource_id,
                    "blocked",
                    "medium",
                    context.ip_address,
                    None,
                    {
                        "reason": "insufficient_permissions",
                        "required": required_permission.value,
                    },
                )
                return False

            # Check resource access
            if not await self._check_resource_access(
                context, resource_type, resource_id
            ):
                await self._log_audit_event(
                    context.user_id,
                    context.agent_id,
                    action,
                    resource_type,
                    resource_id,
                    "blocked",
                    "medium",
                    context.ip_address,
                    None,
                    {"reason": "resource_access_denied"},
                )
                return False

            # Check security level
            required_level = self._get_required_security_level(resource_type, action)
            if context.security_level.value < required_level.value:
                await self._log_audit_event(
                    context.user_id,
                    context.agent_id,
                    action,
                    resource_type,
                    resource_id,
                    "blocked",
                    "high",
                    context.ip_address,
                    None,
                    {"reason": "insufficient_security_level"},
                )
                return False

            # Apply rate limiting
            if not await self._check_rate_limit(context, action):
                await self._log_audit_event(
                    context.user_id,
                    context.agent_id,
                    action,
                    resource_type,
                    resource_id,
                    "blocked",
                    "low",
                    context.ip_address,
                    None,
                    {"reason": "rate_limit_exceeded"},
                )
                return False

            # Log authorized action
            await self._log_audit_event(
                context.user_id,
                context.agent_id,
                action,
                resource_type,
                resource_id,
                "success",
                "low",
                context.ip_address,
                None,
                {},
            )

            return True

        except Exception as e:
            logger.error(f"Authorization failed: {e}")
            return False

    # Encryption & Data Protection
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            raise SecurityError("Encryption failed")

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            raise SecurityError("Decryption failed")

    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

    # Input Validation & Sanitization
    async def validate_input(
        self, input_data: str, input_type: str = "general"
    ) -> bool:
        """Validate and sanitize input data"""
        try:
            # Check for prompt injection attempts
            if await self._detect_prompt_injection(input_data):
                await self._log_audit_event(
                    None,
                    None,
                    "prompt_injection_detected",
                    "input",
                    None,
                    "blocked",
                    "critical",
                    None,
                    None,
                    {"input_type": input_type, "detected_patterns": ["injection"]},
                )
                return False

            # Check for malicious patterns
            if await self._detect_malicious_patterns(input_data):
                await self._log_audit_event(
                    None,
                    None,
                    "malicious_input_detected",
                    "input",
                    None,
                    "blocked",
                    "high",
                    None,
                    None,
                    {"input_type": input_type},
                )
                return False

            # Check input length and format
            if not await self._validate_input_format(input_data, input_type):
                return False

            return True

        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False

    async def sanitize_output(self, output_data: str, context: SecurityContext) -> str:
        """Sanitize output data before returning"""
        try:
            # Remove potential PII
            sanitized = await self._remove_pii(output_data)

            # Apply output filtering based on security level
            sanitized = await self._apply_output_filtering(sanitized, context)

            # Remove potential security information
            sanitized = await self._remove_security_info(sanitized)

            return sanitized

        except Exception as e:
            logger.error(f"Output sanitization failed: {e}")
            return output_data  # Return original if sanitization fails

    # Session Management
    async def create_jwt_token(self, context: SecurityContext) -> str:
        """Create JWT token for session"""
        payload = {
            "user_id": context.user_id,
            "agent_id": context.agent_id,
            "session_id": context.session_id,
            "roles": [role.role_id for role in context.roles],
            "security_level": context.security_level.value,
            "iat": datetime.now(timezone.utc).timestamp(),
            "exp": context.expires_at.timestamp(),
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.jwt_algorithm)

    async def validate_jwt_token(self, token: str) -> Optional[SecurityContext]:
        """Validate JWT token and return security context"""
        try:
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.jwt_algorithm]
            )
            session_id = payload.get("session_id")

            if session_id in self.active_sessions:
                context = self.active_sessions[session_id]
                if await self._is_session_valid(context):
                    return context

            return None

        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None

    async def invalidate_session(self, session_id: str):
        """Invalidate active session"""
        if session_id in self.active_sessions:
            context = self.active_sessions.pop(session_id)
            await self._log_audit_event(
                context.user_id,
                context.agent_id,
                "session_invalidated",
                "session",
                session_id,
                "success",
                "low",
                context.ip_address,
                None,
                {},
            )

    # Compliance & Auditing
    async def get_audit_trail(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        action: Optional[str] = None,
    ) -> List[AuditEvent]:
        """Get filtered audit trail"""
        filtered_events = []

        for event in self.audit_events:
            if user_id and event.user_id != user_id:
                continue
            if agent_id and event.agent_id != agent_id:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if action and event.action != action:
                continue

            filtered_events.append(event)

        return filtered_events

    async def export_compliance_report(
        self, report_type: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Export compliance report"""
        audit_events = await self.get_audit_trail(
            start_time=start_date, end_time=end_date
        )

        report = {
            "report_type": report_type,
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "summary": {
                "total_events": len(audit_events),
                "successful_actions": len(
                    [e for e in audit_events if e.result == "success"]
                ),
                "failed_actions": len(
                    [e for e in audit_events if e.result == "failure"]
                ),
                "blocked_actions": len(
                    [e for e in audit_events if e.result == "blocked"]
                ),
                "high_risk_events": len(
                    [e for e in audit_events if e.risk_level in ["high", "critical"]]
                ),
            },
            "events": [asdict(event) for event in audit_events],
            "security_policies": [
                asdict(policy) for policy in self.security_policies.values()
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        return report

    # Helper Methods
    async def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if username not in self.failed_attempts:
            return False

        recent_attempts = [
            attempt
            for attempt in self.failed_attempts[username]
            if attempt
            > datetime.now(timezone.utc)
            - timedelta(minutes=self.lockout_duration_minutes)
        ]

        return len(recent_attempts) >= self.max_failed_attempts

    async def _record_failed_attempt(self, username: str):
        """Record failed login attempt"""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []

        self.failed_attempts[username].append(datetime.now(timezone.utc))

        # Clean old attempts
        cutoff = datetime.now(timezone.utc) - timedelta(
            minutes=self.lockout_duration_minutes
        )
        self.failed_attempts[username] = [
            attempt for attempt in self.failed_attempts[username] if attempt > cutoff
        ]

    async def _validate_credentials(
        self, username: str, password: str
    ) -> Optional[Dict[str, Any]]:
        """Validate user credentials (mock implementation)"""
        # Mock user database
        mock_users = {
            "admin": {
                "user_id": "user-admin-001",
                "username": "admin",
                "password_hash": self.hash_password("admin123!@#"),
                "roles": ["admin"],
                "security_level": "top_secret",
            },
            "operator": {
                "user_id": "user-operator-001",
                "username": "operator",
                "password_hash": self.hash_password("operator123!@#"),
                "roles": ["agent_operator"],
                "security_level": "secret",
            },
        }

        user_data = mock_users.get(username)
        if user_data and self.verify_password(password, user_data["password_hash"]):
            return user_data

        return None

    async def _validate_agent_credentials(
        self, agent_id: str, api_key: str
    ) -> Optional[Dict[str, Any]]:
        """Validate agent credentials (mock implementation)"""
        # Mock agent database
        mock_agents = {
            "starri-orchestrator": {
                "api_key": "starri-key-123",
                "security_level": "secret",
            },
            "gemini-analysis": {
                "api_key": "gemini-key-456",
                "security_level": "secret",
            },
            "jules-coding": {"api_key": "jules-key-789", "security_level": "secret"},
        }

        agent_data = mock_agents.get(agent_id)
        if agent_data and agent_data["api_key"] == api_key:
            return agent_data

        return None

    async def _create_security_context(
        self,
        user_data: Dict[str, Any],
        ip_address: Optional[str],
        user_agent: Optional[str],
    ) -> SecurityContext:
        """Create security context for authenticated user"""
        user_roles = [self.default_roles[role] for role in user_data["roles"]]

        # Aggregate permissions from all roles
        all_permissions = set()
        for role in user_roles:
            all_permissions.update(role.permissions)

        # Get highest security level
        max_security_level = max(role.security_level for role in user_roles)

        context = SecurityContext(
            user_id=user_data["user_id"],
            agent_id=None,
            roles=user_roles,
            security_level=max_security_level,
            permissions=all_permissions,
            session_id=secrets.token_urlsafe(32),
            ip_address=ip_address,
            expires_at=datetime.now(timezone.utc)
            + timedelta(hours=self.session_timeout_hours),
        )

        self.active_sessions[context.session_id] = context
        return context

    async def _is_session_valid(self, context: SecurityContext) -> bool:
        """Check if session is still valid"""
        return (
            context.session_id in self.active_sessions
            and datetime.now(timezone.utc) < context.expires_at
        )

    def _get_required_permission(self, action: str) -> Permission:
        """Get required permission for action"""
        action_permissions = {
            "read": Permission.READ,
            "write": Permission.WRITE,
            "delete": Permission.DELETE,
            "execute": Permission.EXECUTE,
            "admin": Permission.ADMIN,
            "agent_control": Permission.AGENT_CONTROL,
            "workflow_manage": Permission.WORKFLOW_MANAGE,
            "system_config": Permission.SYSTEM_CONFIG,
        }

        return action_permissions.get(action, Permission.READ)

    def _get_required_security_level(
        self, resource_type: str, action: str
    ) -> SecurityLevel:
        """Get required security level for resource access"""
        # Define security requirements
        security_requirements = {
            ("agents", "read"): SecurityLevel.INTERNAL,
            ("agents", "write"): SecurityLevel.SECRET,
            ("agents", "delete"): SecurityLevel.SECRET,
            ("workflows", "read"): SecurityLevel.INTERNAL,
            ("workflows", "write"): SecurityLevel.CONFIDENTIAL,
            ("system", "config"): SecurityLevel.TOP_SECRET,
        }

        return security_requirements.get(
            (resource_type, action), SecurityLevel.INTERNAL
        )

    async def _check_resource_access(
        self, context: SecurityContext, resource_type: str, resource_id: Optional[str]
    ) -> bool:
        """Check if user has access to specific resource"""
        for role in context.roles:
            if resource_type in role.resource_access:
                allowed_resources = role.resource_access[resource_type]
                if "*" in allowed_resources or (
                    resource_id and resource_id in allowed_resources
                ):
                    return True

        return False

    async def _check_rate_limit(self, context: SecurityContext, action: str) -> bool:
        """Check rate limiting"""
        key = f"{context.user_id or context.agent_id}:{action}"
        current_time = datetime.now(timezone.utc)

        if key not in self.rate_limits:
            self.rate_limits[key] = []

        # Clean old requests
        cutoff = current_time - timedelta(minutes=1)
        self.rate_limits[key] = [
            req_time for req_time in self.rate_limits[key] if req_time > cutoff
        ]

        # Check limit (100 requests per minute default)
        if len(self.rate_limits[key]) >= 100:
            return False

        self.rate_limits[key].append(current_time)
        return True

    async def _detect_prompt_injection(self, input_data: str) -> bool:
        """Detect potential prompt injection attempts"""
        injection_patterns = [
            r"ignore.{0,20}previous.{0,20}instructions",
            r"forget.{0,20}everything",
            r"new.{0,20}instruction",
            r"system.{0,20}prompt",
            r"\\n\\n.{0,50}(assistant|user|system):",
            r"<\|im_start\|>",
            r"<\|im_end\|>",
        ]

        for pattern in injection_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                return True

        return False

    async def _detect_malicious_patterns(self, input_data: str) -> bool:
        """Detect malicious patterns in input"""
        malicious_patterns = [
            r"<script.*?>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"eval\s*\(",
            r"exec\s*\(",
            r"\b(union|select|insert|update|delete|drop)\b.*\b(from|into|table)\b",
        ]

        for pattern in malicious_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                return True

        return False

    async def _validate_input_format(self, input_data: str, input_type: str) -> bool:
        """Validate input format and length"""
        max_lengths = {
            "general": 10000,
            "prompt": 50000,
            "code": 100000,
            "email": 254,
            "username": 50,
        }

        max_length = max_lengths.get(input_type, 10000)
        return len(input_data) <= max_length

    async def _remove_pii(self, text: str) -> str:
        """Remove potential PII from text"""
        # Email addresses
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text
        )

        # Phone numbers
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)

        # SSN patterns
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)

        # Credit card patterns
        text = re.sub(
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CREDIT_CARD]", text
        )

        return text

    async def _apply_output_filtering(self, text: str, context: SecurityContext) -> str:
        """Apply output filtering based on security level"""
        if context.security_level == SecurityLevel.PUBLIC:
            # More aggressive filtering for public access
            text = re.sub(
                r"\b(internal|confidential|secret)\b",
                "[FILTERED]",
                text,
                flags=re.IGNORECASE,
            )

        return text

    async def _remove_security_info(self, text: str) -> str:
        """Remove security-sensitive information"""
        # API keys and tokens
        text = re.sub(r"\b[A-Za-z0-9]{32,}\b", "[TOKEN]", text)

        # Password patterns
        text = re.sub(
            r"password[:\s]*[^\s]+", "password: [HIDDEN]", text, flags=re.IGNORECASE
        )

        return text

    async def _log_audit_event(
        self,
        user_id: Optional[str],
        agent_id: Optional[str],
        action: str,
        resource_type: str,
        resource_id: Optional[str],
        result: str,
        risk_level: str,
        ip_address: Optional[str],
        user_agent: Optional[str],
        details: Dict[str, Any],
    ):
        """Log security audit event"""
        event = AuditEvent(
            event_id=secrets.token_urlsafe(16),
            user_id=user_id,
            agent_id=agent_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            result=result,
            risk_level=risk_level,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            timestamp=datetime.now(timezone.utc),
        )

        self.audit_events.append(event)

        # Keep only recent events in memory
        if len(self.audit_events) > 10000:
            self.audit_events = self.audit_events[-5000:]

        # Log high-risk events
        if risk_level in ["high", "critical"]:
            logger.warning(
                f"High-risk security event: {action} by {user_id or agent_id} - {result}"
            )

    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        active_sessions_count = len(self.active_sessions)
        recent_audit_events = len(
            [
                e
                for e in self.audit_events
                if e.timestamp > datetime.now(timezone.utc) - timedelta(hours=1)
            ]
        )

        return {
            "security_policies": len(self.security_policies),
            "active_sessions": active_sessions_count,
            "recent_audit_events": recent_audit_events,
            "failed_attempts": sum(
                len(attempts) for attempts in self.failed_attempts.values()
            ),
            "encryption_enabled": True,
            "rate_limiting_enabled": True,
            "audit_logging_enabled": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
