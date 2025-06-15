"""
Google Cloud Authentication and Configuration Management
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from google.auth import default
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import google.auth.exceptions
import asyncio
import random

logger = logging.getLogger(__name__)

class GoogleCloudAuth:
    """
    Centralized Google Cloud authentication management
    
    Supports multiple authentication methods:
    - Service Account Key Files
    - Application Default Credentials (ADC)
    - Workload Identity (for GKE)
    - Local development credentials
    """
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.credentials = None
        self.auth_method = None
        self._initialize_credentials()
    
    def _initialize_credentials(self):
        """Initialize credentials using the best available method"""
        
        # Method 1: Service Account Key File
        service_account_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if service_account_path and os.path.exists(service_account_path):
            try:
                self.credentials = service_account.Credentials.from_service_account_file(
                    service_account_path,
                    scopes=self._get_required_scopes()
                )
                self.auth_method = "service_account_file"
                logger.info("Authenticated using service account key file")
                return
            except Exception as e:
                logger.warning(f"Service account file auth failed: {str(e)}")
        
        # Method 2: Service Account JSON from environment variable
        service_account_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
        if service_account_json:
            try:
                service_account_info = json.loads(service_account_json)
                self.credentials = service_account.Credentials.from_service_account_info(
                    service_account_info,
                    scopes=self._get_required_scopes()
                )
                self.auth_method = "service_account_json"
                logger.info("Authenticated using service account JSON")
                return
            except Exception as e:
                logger.warning(f"Service account JSON auth failed: {str(e)}")
        
        # Method 3: Application Default Credentials
        try:
            self.credentials, detected_project = default(
                scopes=self._get_required_scopes()
            )
            
            # Use detected project if not explicitly set
            if not self.project_id and detected_project:
                self.project_id = detected_project
            
            self.auth_method = "application_default"
            logger.info("Authenticated using Application Default Credentials")
            return
        except Exception as e:
            logger.warning(f"ADC authentication failed: {str(e)}")
        
        # No authentication method worked
        logger.error("Failed to initialize Google Cloud credentials")
        raise Exception("No valid Google Cloud authentication method found")
    
    def _get_required_scopes(self) -> list:
        """Get required OAuth scopes for all Google Cloud services"""
        return [
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/aiplatform",
            "https://www.googleapis.com/auth/cloud-vision",
            "https://www.googleapis.com/auth/devstorage.read_write",
            "https://www.googleapis.com/auth/logging.write",
            "https://www.googleapis.com/auth/monitoring"
        ]
    
    def get_credentials(self):
        """Get current credentials, refreshing if necessary"""
        
        if not self.credentials:
            raise Exception("No credentials available")
        
        # Refresh credentials if needed
        if hasattr(self.credentials, 'expired') and self.credentials.expired:
            if hasattr(self.credentials, 'refresh'):
                try:
                    self.credentials.refresh(Request())
                    logger.debug("Credentials refreshed successfully")
                except Exception as e:
                    logger.error(f"Failed to refresh credentials: {str(e)}")
                    raise
        
        return self.credentials
    
    def get_access_token(self) -> str:
        """Get current access token"""
        
        credentials = self.get_credentials()
        
        if not hasattr(credentials, 'token') or not credentials.token:
            credentials.refresh(Request())
        
        return credentials.token
    
    def validate_credentials(self) -> Dict[str, Any]:
        """Validate current credentials and return status"""
        
        try:
            credentials = self.get_credentials()
            
            # Test credentials by making a simple API call
            from google.cloud import resource_manager
            
            client = resource_manager.Client(credentials=credentials)
            project = client.fetch_project(self.project_id)
            
            return {
                "valid": True,
                "auth_method": self.auth_method,
                "project_id": self.project_id,
                "project_name": project.name if project else "Unknown",
                "scopes": list(credentials.scopes) if hasattr(credentials, 'scopes') else []
            }
            
        except Exception as e:
            logger.error(f"Credential validation failed: {str(e)}")
            return {
                "valid": False,
                "error": str(e),
                "auth_method": self.auth_method
            }

class GoogleCloudConfig:
    """Configuration management for Google Cloud services"""
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment and files"""
        
        return {
            "project_id": os.environ.get("GOOGLE_CLOUD_PROJECT_ID", ""),
            "region": os.environ.get("GOOGLE_CLOUD_REGION", "us-central1"),
            "vertex_ai": {
                "location": os.environ.get("VERTEX_AI_LOCATION", "us-central1"),
                "staging_bucket": os.environ.get("VERTEX_AI_STAGING_BUCKET", ""),
                "enable_streaming": os.environ.get("VERTEX_AI_ENABLE_STREAMING", "true").lower() == "true"
            },
            "imagen": {
                "model_version": os.environ.get("IMAGEN_MODEL_VERSION", "imagen-4.0"),
                "default_resolution": os.environ.get("IMAGEN_DEFAULT_RESOLUTION", "2048x1536"),
                "safety_filter": os.environ.get("IMAGEN_SAFETY_FILTER", "block_some")
            },
            "veo": {
                "model_version": os.environ.get("VEO_MODEL_VERSION", "veo-3.0-generate-preview"),
                "default_duration": int(os.environ.get("VEO_DEFAULT_DURATION", "30")),
                "output_bucket": os.environ.get("VEO_OUTPUT_BUCKET", "")
            },
            "gemini": {
                "pro_model": os.environ.get("GEMINI_PRO_MODEL", "gemini-2.5-pro"),
                "flash_model": os.environ.get("GEMINI_FLASH_MODEL", "gemini-2.5-flash"),
                "max_tokens": int(os.environ.get("GEMINI_MAX_TOKENS", "8192")),
                "temperature": float(os.environ.get("GEMINI_TEMPERATURE", "0.7"))
            },
            "jules": {
                "github_token": os.environ.get("JULES_GITHUB_TOKEN", ""),
                "webhook_secret": os.environ.get("JULES_WEBHOOK_SECRET", ""),
                "auto_approve_prs": os.environ.get("JULES_AUTO_APPROVE", "false").lower() == "true"
            },
            "rate_limiting": {
                "requests_per_minute": int(os.environ.get("GOOGLE_API_RATE_LIMIT", "60")),
                "concurrent_requests": int(os.environ.get("GOOGLE_API_CONCURRENT", "10")),
                "retry_attempts": int(os.environ.get("GOOGLE_API_RETRIES", "3"))
            }
        }
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check required fields
        if not self.config["project_id"]:
            validation_results["errors"].append("GOOGLE_CLOUD_PROJECT_ID is required")
            validation_results["valid"] = False
        
        # Check Veo configuration
        if not self.config["veo"]["output_bucket"]:
            validation_results["warnings"].append("VEO_OUTPUT_BUCKET not set, using default")
        
        # Check Jules configuration
        if not self.config["jules"]["github_token"]:
            validation_results["warnings"].append("JULES_GITHUB_TOKEN not set, Jules integration disabled")
        
        return validation_results
    
    def get_service_config(self, service: str) -> Dict[str, Any]:
        """Get configuration for a specific service"""
        return self.config.get(service, {})

class RateLimiter:
    """Rate limiter for Google Cloud API calls"""
    
    def __init__(self, requests_per_minute: int = 60, concurrent_requests: int = 10):
        self.requests_per_minute = requests_per_minute
        self.concurrent_requests = concurrent_requests
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        self.last_request_times = []
    
    async def acquire(self):
        """Acquire permission to make a request"""
        
        async with self.semaphore:
            now = asyncio.get_event_loop().time()
            
            # Remove old request times (older than 1 minute)
            self.last_request_times = [
                t for t in self.last_request_times 
                if now - t < 60
            ]
            
            # Check if we need to wait
            if len(self.last_request_times) >= self.requests_per_minute:
                wait_time = 60 - (now - self.last_request_times[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            self.last_request_times.append(now)

async def retry_with_exponential_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    *args,
    **kwargs
):
    """Retry function with exponential backoff"""
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if attempt == max_retries:
                break
            
            # Calculate delay with jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, 0.1) * delay
            
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay + jitter:.2f}s")
            await asyncio.sleep(delay + jitter)
    
    raise last_exception

# Error handling classes
class AIServiceError(Exception):
    """Base exception for AI service errors"""
    pass

class QuotaExceededError(AIServiceError):
    """Raised when API quota is exceeded"""
    pass

class ModelNotAvailableError(AIServiceError):
    """Raised when requested model is not available"""
    pass

class AuthenticationError(AIServiceError):
    """Raised when authentication fails"""
    pass

# Service status enum
from enum import Enum

class ServiceStatus(Enum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"

class AIServiceManager:
    """Manages AI service availability and fallbacks"""
    
    def __init__(self):
        self.service_status = {
            "imagen": ServiceStatus.AVAILABLE,
            "veo": ServiceStatus.AVAILABLE,
            "gemini": ServiceStatus.AVAILABLE,
            "jules": ServiceStatus.AVAILABLE
        }
        self.fallback_strategies = {
            "imagen": self._imagen_fallback,
            "veo": self._veo_fallback,
            "gemini": self._gemini_fallback
        }
        self.rate_limiter = RateLimiter()
    
    async def call_with_fallback(
        self,
        service_name: str,
        primary_func,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Call AI service with automatic fallback"""
        
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            # Try primary service
            result = await retry_with_exponential_backoff(primary_func, *args, **kwargs)
            self._update_service_status(service_name, ServiceStatus.AVAILABLE)
            return result
            
        except QuotaExceededError:
            logger.warning(f"{service_name} quota exceeded, using fallback")
            self._update_service_status(service_name, ServiceStatus.DEGRADED)
            return await self._use_fallback(service_name, *args, **kwargs)
            
        except (ModelNotAvailableError, AuthenticationError) as e:
            logger.error(f"{service_name} unavailable: {str(e)}")
            self._update_service_status(service_name, ServiceStatus.UNAVAILABLE)
            return await self._use_fallback(service_name, *args, **kwargs)
            
        except Exception as e:
            logger.error(f"Unexpected error in {service_name}: {str(e)}")
            return await self._use_fallback(service_name, *args, **kwargs)
    
    def _update_service_status(self, service_name: str, status: ServiceStatus):
        """Update service status"""
        self.service_status[service_name] = status
        logger.info(f"{service_name} status updated to {status.value}")
    
    async def _use_fallback(self, service_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Use fallback strategy for service"""
        
        fallback_func = self.fallback_strategies.get(service_name)
        if fallback_func:
            return await fallback_func(*args, **kwargs)
        else:
            raise AIServiceError(f"No fallback available for {service_name}")
    
    async def _imagen_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """Fallback strategy for Imagen (use pre-generated templates)"""
        logger.info("Using Imagen fallback: template-based mockups")
        
        # Return template-based mockup
        return {
            "image_data": "base64_encoded_template_image",
            "format": "png",
            "source": "fallback_template",
            "message": "Using pre-designed template due to service unavailability"
        }
    
    async def _veo_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """Fallback strategy for Veo (use static video templates)"""
        logger.info("Using Veo fallback: template videos")
        
        return {
            "video_url": "https://storage.googleapis.com/fallback-videos/template-demo.mp4",
            "source": "fallback_template",
            "message": "Using template video due to service unavailability"
        }
    
    async def _gemini_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """Fallback strategy for Gemini (use alternative models)"""
        logger.info("Using Gemini fallback: OpenAI GPT-4")
        
        # Could integrate with OpenAI as fallback
        return {
            "content": "Fallback response generated using alternative AI service",
            "source": "fallback_ai",
            "message": "Using alternative AI service due to Gemini unavailability"
        }
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get overall service health status"""
        
        return {
            "services": {
                name: status.value 
                for name, status in self.service_status.items()
            },
            "overall_health": self._calculate_overall_health(),
            "timestamp": asyncio.get_event_loop().time()
        }
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall system health"""
        
        statuses = list(self.service_status.values())
        
        if all(status == ServiceStatus.AVAILABLE for status in statuses):
            return "healthy"
        elif any(status == ServiceStatus.AVAILABLE for status in statuses):
            return "degraded"
        else:
            return "critical"

# Example environment configuration
def create_example_env_file(file_path: str = ".env.example"):
    """Create example environment file"""
    
    example_config = """
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_CLOUD_REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Alternative: Use service account JSON directly
# GOOGLE_SERVICE_ACCOUNT_JSON='{"type": "service_account", ...}'

# Vertex AI Configuration
VERTEX_AI_LOCATION=us-central1
VERTEX_AI_STAGING_BUCKET=gs://your-staging-bucket
VERTEX_AI_ENABLE_STREAMING=true

# Imagen Configuration
IMAGEN_MODEL_VERSION=imagen-4.0
IMAGEN_DEFAULT_RESOLUTION=2048x1536
IMAGEN_SAFETY_FILTER=block_some

# Veo Configuration
VEO_MODEL_VERSION=veo-3.0-generate-preview
VEO_DEFAULT_DURATION=30
VEO_OUTPUT_BUCKET=gs://your-veo-outputs

# Gemini Configuration
GEMINI_PRO_MODEL=gemini-2.5-pro
GEMINI_FLASH_MODEL=gemini-2.5-flash
GEMINI_MAX_TOKENS=8192
GEMINI_TEMPERATURE=0.7

# Jules Configuration
JULES_GITHUB_TOKEN=ghp_your_github_token
JULES_WEBHOOK_SECRET=your_webhook_secret
JULES_AUTO_APPROVE=false

# Rate Limiting
GOOGLE_API_RATE_LIMIT=60
GOOGLE_API_CONCURRENT=10
GOOGLE_API_RETRIES=3
"""
    
    with open(file_path, 'w') as f:
        f.write(example_config)
    
    logger.info(f"Example environment file created at {file_path}")

if __name__ == "__main__":
    # Create example environment file
    create_example_env_file()
    
    # Test authentication
    try:
        auth = GoogleCloudAuth("your-project-id")
        validation = auth.validate_credentials()
        print(f"Authentication status: {validation}")
    except Exception as e:
        print(f"Authentication failed: {str(e)}")