"""
Nexus Forge Custom Exceptions

Centralized exception handling for the application.
"""

from typing import Any, Dict, Optional


class NexusForgeError(Exception):
    """Base exception for Nexus Forge application."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "NEXUS_FORGE_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(NexusForgeError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=details
        )


class BuildError(NexusForgeError):
    """Raised when app building process fails."""
    
    def __init__(self, message: str, phase: str, details: Optional[Dict[str, Any]] = None):
        build_details = {"phase": phase}
        if details:
            build_details.update(details)
            
        super().__init__(
            message=message,
            error_code="BUILD_ERROR",
            status_code=422,
            details=build_details
        )


class AIModelError(NexusForgeError):
    """Raised when AI model interaction fails."""
    
    def __init__(self, message: str, model: str, details: Optional[Dict[str, Any]] = None):
        model_details = {"model": model}
        if details:
            model_details.update(details)
            
        super().__init__(
            message=message,
            error_code="AI_MODEL_ERROR",
            status_code=503,
            details=model_details
        )


class DeploymentError(NexusForgeError):
    """Raised when deployment fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DEPLOYMENT_ERROR",
            status_code=502,
            details=details
        )


class ServiceUnavailableError(NexusForgeError):
    """Raised when a service is temporarily unavailable."""
    
    def __init__(self, message: str, service: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        service_details = {"service": service} if service else {}
        if details:
            service_details.update(details)
            
        super().__init__(
            message=message,
            error_code="SERVICE_UNAVAILABLE",
            status_code=503,
            details=service_details
        )


class RateLimitError(NexusForgeError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        limit_details = {"retry_after": retry_after} if retry_after else {}
        if details:
            limit_details.update(details)
            
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details=limit_details
        )


class AuthenticationError(NexusForgeError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str, service: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        auth_details = {"service": service} if service else {}
        if details:
            auth_details.update(details)
            
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
            details=auth_details
        )


class QuotaExceededError(NexusForgeError):
    """Raised when service quotas are exceeded."""
    
    def __init__(self, message: str, quota_type: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        quota_details = {"quota_type": quota_type} if quota_type else {}
        if details:
            quota_details.update(details)
            
        super().__init__(
            message=message,
            error_code="QUOTA_EXCEEDED",
            status_code=429,
            details=quota_details
        )


class ModelNotFoundError(NexusForgeError):
    """Raised when requested AI model is not found or available."""
    
    def __init__(self, message: str, model: str, details: Optional[Dict[str, Any]] = None):
        model_details = {"model": model}
        if details:
            model_details.update(details)
            
        super().__init__(
            message=message,
            error_code="MODEL_NOT_FOUND",
            status_code=404,
            details=model_details
        )


class ModelTimeoutError(NexusForgeError):
    """Raised when AI model operations timeout."""
    
    def __init__(self, message: str, model: str, timeout: Optional[float] = None, details: Optional[Dict[str, Any]] = None):
        timeout_details = {"model": model, "timeout": timeout}
        if details:
            timeout_details.update(details)
            
        super().__init__(
            message=message,
            error_code="MODEL_TIMEOUT",
            status_code=408,
            details=timeout_details
        )


class InvalidAPIKeyError(AuthenticationError):
    """Raised when API key is invalid or expired."""
    
    def __init__(self, message: str, service: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            service=service,
            details=details
        )
        self.error_code = "INVALID_API_KEY"


class ConfigurationError(NexusForgeError):
    """Raised when service configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        config_details = {"config_key": config_key} if config_key else {}
        if details:
            config_details.update(details)
            
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=400,
            details=config_details
        )


class PayloadTooLargeError(NexusForgeError):
    """Raised when request payload exceeds size limits."""
    
    def __init__(self, message: str, size: Optional[int] = None, max_size: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        size_details = {"size": size, "max_size": max_size}
        if details:
            size_details.update(details)
            
        super().__init__(
            message=message,
            error_code="PAYLOAD_TOO_LARGE",
            status_code=413,
            details=size_details
        )


class OrchestrationError(NexusForgeError):
    """Raised when orchestration process fails."""
    
    def __init__(self, message: str, agent: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        orchestration_details = {"agent": agent} if agent else {}
        if details:
            orchestration_details.update(details)
            
        super().__init__(
            message=message,
            error_code="ORCHESTRATION_ERROR",
            status_code=500,
            details=orchestration_details
        )


class CoordinationError(NexusForgeError):
    """Raised when agent coordination fails."""
    
    def __init__(self, message: str, agents: Optional[list] = None, details: Optional[Dict[str, Any]] = None):
        coordination_details = {"agents": agents} if agents else {}
        if details:
            coordination_details.update(details)
            
        super().__init__(
            message=message,
            error_code="COORDINATION_ERROR",
            status_code=500,
            details=coordination_details
        )


class AgentError(NexusForgeError):
    """Raised when an AI agent fails to execute."""
    
    def __init__(self, message: str, agent_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        agent_details = {"agent_name": agent_name} if agent_name else {}
        if details:
            agent_details.update(details)
            
        super().__init__(
            message=message,
            error_code="AGENT_ERROR",
            status_code=500,
            details=agent_details
        )


class TaskDecompositionError(NexusForgeError):
    """Raised when task decomposition fails."""
    
    def __init__(self, message: str, task: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        task_details = {"task": task} if task else {}
        if details:
            task_details.update(details)
            
        super().__init__(
            message=message,
            error_code="TASK_DECOMPOSITION_ERROR",
            status_code=422,
            details=task_details
        )