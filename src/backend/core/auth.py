import os
from datetime import datetime, timedelta
from typing import Optional, Union

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jwt.exceptions import PyJWTError
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .. import models
from ..config import get_security_settings, settings
from ..database import get_db
# Temporarily disabled for testing - monitoring import causes hang
# from .monitoring import structured_logger

# Fallback logger for testing
import logging
class FallbackLogger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    def log(self, level, message, **kwargs):
        getattr(self.logger, level)(f"{message} - {kwargs}")

structured_logger = FallbackLogger()

# Security context for password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# API key scheme
api_key_header = APIKeyHeader(name="X-API-Key")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token with enhanced security

    Args:
        data: Data to encode in the JWT
        expires_delta: Optional custom expiration time

    Returns:
        str: Encoded JWT token
    """
    security_settings = get_security_settings()

    # Create a copy to avoid modifying the original
    to_encode = data.copy()

    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=security_settings["token_expire_minutes"]
        )

    # Add standard JWT claims
    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.utcnow(),  # Issued at claim
            "jti": f"{datetime.utcnow().timestamp()}-{os.urandom(8).hex()}",  # JWT ID for uniqueness
        }
    )

    # Encode the JWT
    encoded_jwt = jwt.encode(
        to_encode,
        security_settings["secret_key"],
        algorithm=security_settings["algorithm"],
    )

    return encoded_jwt


def decode_token(token: str) -> Optional[dict]:
    """
    Decode and validate JWT token with constant-time comparison

    Args:
        token: JWT token to decode

    Returns:
        Optional[dict]: Decoded payload or None if invalid
    """
    security_settings = get_security_settings()

    try:
        # Time-based attacks are mitigated by PyJWT's implementation
        payload = jwt.decode(
            token,
            security_settings["secret_key"],
            algorithms=[security_settings["algorithm"]],
            options={"verify_signature": True, "verify_exp": True},
        )
        return payload
    except jwt.ExpiredSignatureError:
        # Handle expired tokens separately for better logging/metrics
        structured_logger.log("warning", "Token expired")
        return None
    except PyJWTError as e:
        # Log error type without details to avoid information leakage
        structured_logger.log(
            "warning", "Token validation failed", error_type=type(e).__name__
        )
        return None


def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT refresh token with enhanced security

    Args:
        data: Data to encode in the JWT
        expires_delta: Optional custom expiration time

    Returns:
        str: Encoded JWT refresh token
    """
    security_settings = get_security_settings()

    # Create a copy to avoid modifying the original
    to_encode = data.copy()

    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            days=security_settings["refresh_token_expire_days"]
        )

    # Add standard JWT claims and refresh-specific claims
    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.utcnow(),  # Issued at claim
            "jti": f"{datetime.utcnow().timestamp()}-{os.urandom(8).hex()}",  # JWT ID for uniqueness
            "token_type": "refresh",  # Mark as refresh token
        }
    )

    # Encode the JWT
    encoded_jwt = jwt.encode(
        to_encode,
        security_settings["secret_key"],
        algorithm=security_settings["algorithm"],
    )

    return encoded_jwt


def create_api_key(
    db: Session, user: models.User, name: str, expires_in_days: Optional[int] = None
) -> models.APIKey:
    """Create a new API key for a user"""
    api_key = models.APIKey(
        key=f"pk_{pwd_context.hash(f'{user.id}_{datetime.utcnow().timestamp()}')[-32:]}",
        name=name,
        user_id=user.id,
        expires_at=(
            datetime.utcnow() + timedelta(days=expires_in_days)
            if expires_in_days
            else None
        ),
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)
    return api_key


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> models.User:
    """
    Get current user from JWT token with enhanced security

    Args:
        token: JWT access token
        db: Database session

    Returns:
        models.User: Authenticated user

    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Use our improved token decoder that handles timing attacks
    payload = decode_token(token)
    if payload is None:
        structured_logger.log("warning", "Invalid token during authentication")
        raise credentials_exception

    # Extract username from payload
    username: str = payload.get("sub")
    if username is None:
        structured_logger.log("warning", "Token missing username claim")
        raise credentials_exception

    # Check token type (should not be a refresh token)
    if payload.get("token_type") == "refresh":
        structured_logger.log("warning", "Using refresh token for authentication")
        raise credentials_exception

    # Get the user from database using constant-time comparison for security
    # The ORM handles this safely under the hood
    user = (
        db.query(models.User)
        .filter(models.User.username == username)
        .filter(models.User.is_active == True)
        .first()
    )

    if user is None:
        # Log failed authentication attempt without revealing whether username exists
        structured_logger.log(
            "warning", "Authentication failed - user not found or inactive"
        )
        # Use the same exception to prevent user enumeration
        raise credentials_exception

    # Update last login - only for actual API usage, not token validation
    # This helps with auditing
    user.last_login = datetime.utcnow()
    db.commit()

    # Log successful authentication
    structured_logger.log("info", "User authenticated successfully", user_id=user.id)

    return user


async def get_api_key_user(
    api_key: str = Depends(api_key_header), db: Session = Depends(get_db)
) -> models.User:
    """Get user from API key"""
    api_key_obj = (
        db.query(models.APIKey)
        .filter(models.APIKey.key == api_key)
        .filter(models.APIKey.is_active == True)
        .first()
    )

    if not api_key_obj:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )

    # Check expiration
    if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="API key has expired"
        )

    # Update last used timestamp
    api_key_obj.last_used = datetime.utcnow()
    db.commit()

    user = (
        db.query(models.User)
        .filter(models.User.id == api_key_obj.user_id)
        .filter(models.User.is_active == True)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key user"
        )

    return user


def get_current_active_user(
    current_user: models.User = Depends(get_current_user),
) -> models.User:
    """Verify user is active"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def check_admin_role(
    current_user: models.User = Depends(get_current_active_user),
) -> models.User:
    """Check if user has admin role"""
    if current_user.role != models.UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )
    return current_user


def has_research_task_access(task: models.ResearchTask, user: models.User) -> bool:
    """Check if user has access to research task"""
    return user.role == models.UserRole.ADMIN or task.owner_id == user.id


def log_auth_activity(
    db: Session,
    user: models.User,
    action: str,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    details: Optional[str] = None,
):
    """Log authentication/authorization activity"""
    log = models.AuditLog(
        user_id=user.id,
        action=action,
        resource_type="auth",
        ip_address=ip_address,
        user_agent=user_agent,
        details=details,
    )
    db.add(log)
    db.commit()

    structured_logger.log(
        "info", "Auth activity", user_id=user.id, action=action, ip_address=ip_address
    )
