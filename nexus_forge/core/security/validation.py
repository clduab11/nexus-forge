"""
Security validation and sanitization for Parallax Pal

This module provides input validation, sanitization, and standardized
error responses to prevent security vulnerabilities.
"""

from pydantic import BaseModel, validator, Field
import re
import bleach
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ResearchQueryValidator(BaseModel):
    """Validates and sanitizes research queries"""
    
    query: str = Field(..., min_length=3, max_length=1000, description="The research query")
    mode: Optional[str] = Field(
        "comprehensive", 
        pattern="^(quick|comprehensive|continuous)$",
        description="Research mode"
    )
    focus_areas: Optional[List[str]] = Field(
        default_factory=list, 
        max_items=10,
        description="Specific areas to focus on"
    )
    language: Optional[str] = Field(
        "en", 
        pattern="^[a-z]{2}$",
        description="Two-letter language code"
    )
    
    @validator('query')
    def sanitize_query(cls, v):
        """Remove potential security threats from query"""
        
        # Remove any HTML/JavaScript
        v = bleach.clean(v, tags=[], strip=True)
        
        # Check for SQL injection patterns
        sql_patterns = [
            r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b\s+\b(all|from|into|where|table)\b)",
            r"(--|#|\/\*|\*\/)",  # SQL comments
            r"(\bor\b\s*\d+\s*=\s*\d+)",  # OR 1=1 pattern
            r"(\band\b\s*\d+\s*=\s*\d+)",  # AND 1=1 pattern
            r"(;.*(?:drop|delete|update|insert))",  # Command chaining
            r"(xp_cmdshell|sp_executesql)",  # SQL Server specific
            r"(script\s*:)",  # Script protocol
            r"(javascript\s*:)",  # JavaScript protocol
            r"(onerror\s*=)",  # Event handlers
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                logger.warning(f"Potential injection attempt detected: {pattern}")
                raise ValueError("Invalid query format detected")
        
        # Check for command injection patterns
        cmd_patterns = [
            r"([;&|`$])",  # Command separators
            r"(\$\(.*\))",  # Command substitution
            r"(`.*`)",  # Backtick execution
            r"(>\s*/dev/null)",  # Redirect patterns
            r"(rm\s+-rf)",  # Dangerous commands
        ]
        
        for pattern in cmd_patterns:
            if re.search(pattern, v):
                logger.warning(f"Potential command injection detected: {pattern}")
                raise ValueError("Invalid query format detected")
        
        # Length check after cleaning
        if len(v) < 3:
            raise ValueError("Query too short after sanitization")
        
        return v
    
    @validator('focus_areas')
    def validate_focus_areas(cls, v):
        """Sanitize focus areas"""
        if v:
            sanitized = []
            for area in v:
                # Clean each area
                cleaned = bleach.clean(area, tags=[], strip=True)
                if len(cleaned) > 100:
                    raise ValueError("Focus area too long")
                if cleaned:  # Only add non-empty areas
                    sanitized.append(cleaned)
            return sanitized
        return v
    
    class Config:
        # Prevent extra fields
        extra = 'forbid'


class WebSocketMessageValidator(BaseModel):
    """Validates WebSocket messages"""
    
    type: str = Field(..., pattern="^[a-z_]+$", max_length=50)
    session_id: Optional[str] = Field(None, pattern="^[a-f0-9-]{36}$")
    data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('type')
    def validate_message_type(cls, v):
        """Ensure message type is allowed"""
        allowed_types = {
            'research_query',
            'follow_up_question',
            'cancel_research',
            'get_status',
            'export_results',
            'share_research',
            'ping'
        }
        
        if v not in allowed_types:
            raise ValueError(f"Unknown message type: {v}")
        
        return v
    
    @validator('data')
    def validate_data(cls, v):
        """Validate data payload based on requirements"""
        if v and len(str(v)) > 10000:  # 10KB limit
            raise ValueError("Message data too large")
        return v


class ErrorResponse:
    """Standardized error responses without information leakage"""
    
    # Generic error messages that don't reveal system details
    ERRORS = {
        "auth_failed": "Authentication failed. Please check your credentials.",
        "auth_expired": "Your session has expired. Please sign in again.",
        "rate_limited": "Too many requests. Please try again later.",
        "invalid_input": "Invalid input provided. Please check your request.",
        "server_error": "An error occurred. Please try again.",
        "not_found": "The requested resource was not found.",
        "forbidden": "You don't have permission to access this resource.",
        "quota_exceeded": "You've exceeded your usage quota. Please upgrade your plan.",
        "maintenance": "The service is temporarily unavailable for maintenance.",
        "websocket_error": "Connection error. Please refresh and try again.",
        "research_failed": "Unable to complete research. Please try again.",
        "export_failed": "Export failed. Please try a different format.",
        "invalid_session": "Invalid or expired session. Please reconnect."
    }
    
    @classmethod
    def get(cls, error_type: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get standardized error response
        
        Args:
            error_type: Type of error
            request_id: Optional request ID for tracking
            
        Returns:
            Standardized error response dict
        """
        return {
            "error": cls.ERRORS.get(error_type, cls.ERRORS["server_error"]),
            "code": error_type,
            "request_id": request_id
        }
    
    @classmethod
    def get_http_status(cls, error_type: str) -> int:
        """Get appropriate HTTP status code for error type"""
        
        status_map = {
            "auth_failed": 401,
            "auth_expired": 401,
            "forbidden": 403,
            "not_found": 404,
            "rate_limited": 429,
            "invalid_input": 400,
            "quota_exceeded": 402,
            "maintenance": 503,
            "server_error": 500,
            "websocket_error": 500,
            "research_failed": 500,
            "export_failed": 500,
            "invalid_session": 401
        }
        
        return status_map.get(error_type, 500)


class SanitizationUtils:
    """Utility functions for data sanitization"""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent directory traversal"""
        
        # Remove any path components
        filename = filename.replace('/', '').replace('\\', '').replace('..', '')
        
        # Allow only alphanumeric, dash, underscore, and dot
        filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:250] + '.' + ext if ext else name[:255]
        
        return filename or 'unnamed'
    
    @staticmethod
    def sanitize_url(url: str) -> Optional[str]:
        """Sanitize and validate URL"""
        
        # Basic URL pattern
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
        
        if url_pattern.match(url):
            # Additional checks for malicious URLs
            blacklisted_domains = [
                'javascript:',
                'data:',
                'vbscript:',
                'file:',
                'about:'
            ]
            
            for domain in blacklisted_domains:
                if url.lower().startswith(domain):
                    return None
            
            return url
        
        return None
    
    @staticmethod
    def sanitize_json_output(data: Any) -> Any:
        """Sanitize JSON data for safe output"""
        
        if isinstance(data, dict):
            return {k: SanitizationUtils.sanitize_json_output(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [SanitizationUtils.sanitize_json_output(item) for item in data]
        elif isinstance(data, str):
            # Remove any potential XSS
            return bleach.clean(data, tags=[], strip=True)
        else:
            return data


# Validation decorators for common patterns
def validate_user_id(user_id: str) -> bool:
    """Validate user ID format"""
    return bool(re.match(r'^[a-zA-Z0-9_-]{1,128}$', user_id))


def validate_session_id(session_id: str) -> bool:
    """Validate session ID format (UUID)"""
    return bool(re.match(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', session_id))


def validate_api_key(api_key: str) -> bool:
    """Validate API key format"""
    return bool(re.match(r'^pk_[a-zA-Z0-9]{32}$', api_key))