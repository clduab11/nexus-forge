from typing import Any, Dict, Union

try:
    from pydantic import EmailStr, HttpUrl, PostgresDsn, validator
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings, PostgresDsn, HttpUrl, EmailStr, validator

import os
from functools import lru_cache


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Parallax Pal"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # Database
    DATABASE_URL: Union[PostgresDsn, str]
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10

    # Redis Cache
    REDIS_URL: str
    REDIS_PASSWORD: str = None
    REDIS_DB: int = 0

    # Frontend
    FRONTEND_URL: HttpUrl
    CORS_ORIGINS: list[str] = ["*"]

    # OAuth Providers
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    GOOGLE_REDIRECT_URI: HttpUrl

    GITHUB_CLIENT_ID: str
    GITHUB_CLIENT_SECRET: str
    GITHUB_REDIRECT_URI: HttpUrl

    FACEBOOK_CLIENT_ID: str
    FACEBOOK_CLIENT_SECRET: str
    FACEBOOK_REDIRECT_URI: HttpUrl

    INSTAGRAM_CLIENT_ID: str
    INSTAGRAM_CLIENT_SECRET: str
    INSTAGRAM_REDIRECT_URI: HttpUrl

    # Email
    SMTP_HOST: str
    SMTP_PORT: int
    SMTP_USER: str
    SMTP_PASSWORD: str
    SMTP_FROM_EMAIL: EmailStr
    SMTP_FROM_NAME: str = "Parallax Pal"

    # Stripe
    STRIPE_SECRET_KEY: str
    STRIPE_PUBLISHABLE_KEY: str
    STRIPE_WEBHOOK_SECRET: str

    # GPU Settings
    GPU_MEMORY_THRESHOLD: float = 0.9  # 90% memory usage threshold
    DEFAULT_MODEL: str = "llama2"
    OLLAMA_API_URL: str = "http://localhost:11434"

    # Google Cloud AI Settings
    GOOGLE_CLOUD_PROJECT_ID: str = None
    GOOGLE_CLOUD_REGION: str = "us-central1"
    VERTEX_AI_LOCATION: str = "us-central1"

    # Jules Integration Settings
    JULES_GITHUB_TOKEN: str = None
    JULES_WEBHOOK_SECRET: str = None
    JULES_AUTO_APPROVE_PRS: bool = False
    JULES_DEFAULT_ORG: str = "nexus-forge"
    JULES_MAX_RETRIES: int = 3
    JULES_TIMEOUT_MINUTES: int = 30

    # Gemini Settings
    GEMINI_PRO_MODEL: str = "gemini-2.5-pro"
    GEMINI_FLASH_MODEL: str = "gemini-2.5-flash"
    GEMINI_MAX_TOKENS: int = 8192
    GEMINI_TEMPERATURE: float = 0.7

    # Imagen Settings
    IMAGEN_MODEL_VERSION: str = "imagen-4.0"
    IMAGEN_DEFAULT_RESOLUTION: str = "2048x1536"
    IMAGEN_SAFETY_FILTER: str = "block_some"

    # Veo Settings
    VEO_MODEL_VERSION: str = "veo-3.0-generate-preview"
    VEO_DEFAULT_DURATION: int = 30
    VEO_OUTPUT_BUCKET: str = None

    # Monitoring
    SENTRY_DSN: str = None
    LOG_LEVEL: str = "INFO"
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    # Security
    ALLOWED_HOSTS: list[str] = ["*"]
    SSL_KEYFILE: str = None
    SSL_CERTFILE: str = None
    SECURE_HEADERS: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow" if os.getenv("TESTING") == "true" else "forbid"

    @validator("DATABASE_URL", pre=True)
    def validate_database_url(cls, v: str) -> str:
        """Ensure DATABASE_URL is properly formatted"""
        if isinstance(v, str):
            # Allow sqlite URLs in testing mode
            if os.getenv("TESTING") == "true" and ("sqlite" in v or ":memory:" in v):
                return v
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            user=v.get("user"),
            password=v.get("password"),
            host=v.get("host"),
            port=v.get("port"),
            path=f"/{v.get('database')}",
        )

    @validator("REDIS_URL", pre=True)
    def validate_redis_url(cls, v: str) -> str:
        """Ensure REDIS_URL is properly formatted"""
        if isinstance(v, str):
            return v
        return f"redis://{v.get('host')}:{v.get('port')}/{v.get('db', 0)}"

    @validator("CORS_ORIGINS", pre=True)
    def validate_cors_origins(cls, v: Any) -> list[str]:
        """Convert CORS_ORIGINS to list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("ALLOWED_HOSTS", pre=True)
    def validate_allowed_hosts(cls, v: Any) -> list[str]:
        """Convert ALLOWED_HOSTS to list"""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def get_db_url() -> str:
    """Get database URL with proper formatting"""
    settings = get_settings()
    return str(settings.DATABASE_URL)


def get_security_settings() -> Dict[str, Any]:
    """Get security-related settings"""
    return {
        "secret_key": settings.SECRET_KEY,
        "algorithm": settings.JWT_ALGORITHM,
        "access_token_expire_minutes": settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        "refresh_token_expire_days": settings.REFRESH_TOKEN_EXPIRE_DAYS,
        "rate_limit_per_minute": settings.RATE_LIMIT_PER_MINUTE,
    }


# Environment-specific settings
ENVIRONMENT_SETTINGS: Dict[str, Dict[str, Any]] = {
    "development": {
        "DEBUG": True,
        "LOG_LEVEL": "DEBUG",
        "ENABLE_METRICS": False,
    },
    "staging": {
        "DEBUG": False,
        "LOG_LEVEL": "INFO",
        "CORS_ORIGINS": ["https://staging.parallaxpal.com"],
        "ALLOWED_HOSTS": ["staging.parallaxpal.com"],
    },
    "production": {
        "DEBUG": False,
        "LOG_LEVEL": "WARNING",
        "SECURE_HEADERS": True,
        "CORS_ORIGINS": ["https://parallaxpal.com"],
        "ALLOWED_HOSTS": ["parallaxpal.com"],
    },
    "testing": {
        "DEBUG": True,
        "LOG_LEVEL": "DEBUG",
        "ENABLE_METRICS": False,
        "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
        "REDIS_URL": "redis://localhost:6379/1",
        "FRONTEND_URL": "http://localhost:3000",
        "SECRET_KEY": "test-secret-key-for-testing-only",
        "GOOGLE_CLIENT_ID": "test-google-client-id",
        "GOOGLE_CLIENT_SECRET": "test-google-client-secret",
        "GOOGLE_REDIRECT_URI": "http://localhost:3000/auth/google/callback",
        "GITHUB_CLIENT_ID": "test-github-client-id",
        "GITHUB_CLIENT_SECRET": "test-github-client-secret",
        "GITHUB_REDIRECT_URI": "http://localhost:3000/auth/github/callback",
        "FACEBOOK_CLIENT_ID": "test-facebook-client-id",
        "FACEBOOK_CLIENT_SECRET": "test-facebook-client-secret",
        "FACEBOOK_REDIRECT_URI": "http://localhost:3000/auth/facebook/callback",
        "INSTAGRAM_CLIENT_ID": "test-instagram-client-id",
        "INSTAGRAM_CLIENT_SECRET": "test-instagram-client-secret",
        "INSTAGRAM_REDIRECT_URI": "http://localhost:3000/auth/instagram/callback",
        "SMTP_HOST": "localhost",
        "SMTP_PORT": 587,
        "SMTP_USER": "test@example.com",
        "SMTP_PASSWORD": "test-password",
        "SMTP_FROM_EMAIL": "test@example.com",
        "STRIPE_SECRET_KEY": "sk_test_123456789",
        "STRIPE_PUBLISHABLE_KEY": "pk_test_123456789",
        "STRIPE_WEBHOOK_SECRET": "whsec_test_123456789",
    },
}

# Load environment-specific settings
settings = get_settings()
env_settings = ENVIRONMENT_SETTINGS.get(settings.ENVIRONMENT, {})
for key, value in env_settings.items():
    setattr(settings, key, value)

# Google AI configurations
GOOGLE_AI_SETTINGS = {
    "project_id": settings.GOOGLE_CLOUD_PROJECT_ID,
    "region": settings.GOOGLE_CLOUD_REGION,
    "jules": {
        "github_token": settings.JULES_GITHUB_TOKEN,
        "webhook_secret": settings.JULES_WEBHOOK_SECRET,
        "auto_approve_prs": settings.JULES_AUTO_APPROVE_PRS,
        "default_org": settings.JULES_DEFAULT_ORG,
        "max_retries": settings.JULES_MAX_RETRIES,
        "timeout_minutes": settings.JULES_TIMEOUT_MINUTES,
    },
    "gemini": {
        "pro_model": settings.GEMINI_PRO_MODEL,
        "flash_model": settings.GEMINI_FLASH_MODEL,
        "max_tokens": settings.GEMINI_MAX_TOKENS,
        "temperature": settings.GEMINI_TEMPERATURE,
    },
    "imagen": {
        "model_version": settings.IMAGEN_MODEL_VERSION,
        "default_resolution": settings.IMAGEN_DEFAULT_RESOLUTION,
        "safety_filter": settings.IMAGEN_SAFETY_FILTER,
    },
    "veo": {
        "model_version": settings.VEO_MODEL_VERSION,
        "default_duration": settings.VEO_DEFAULT_DURATION,
        "output_bucket": settings.VEO_OUTPUT_BUCKET,
    },
}

# OAuth configurations
OAUTH_SETTINGS = {
    "google": {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "redirect_uri": str(settings.GOOGLE_REDIRECT_URI),
        "scope": ["openid", "email", "profile"],
    },
    "github": {
        "client_id": settings.GITHUB_CLIENT_ID,
        "client_secret": settings.GITHUB_CLIENT_SECRET,
        "redirect_uri": str(settings.GITHUB_REDIRECT_URI),
        "scope": ["read:user", "user:email"],
    },
    "facebook": {
        "client_id": settings.FACEBOOK_CLIENT_ID,
        "client_secret": settings.FACEBOOK_CLIENT_SECRET,
        "redirect_uri": str(settings.FACEBOOK_REDIRECT_URI),
        "scope": ["email", "public_profile"],
    },
    "instagram": {
        "client_id": settings.INSTAGRAM_CLIENT_ID,
        "client_secret": settings.INSTAGRAM_CLIENT_SECRET,
        "redirect_uri": str(settings.INSTAGRAM_REDIRECT_URI),
        "scope": ["basic"],
    },
}

# Example .env file template
ENV_TEMPLATE = """
# Application
APP_NAME=Parallax Pal
DEBUG=false
ENVIRONMENT=development
SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/parallaxpal

# Redis
REDIS_URL=redis://localhost:6379/0

# Frontend
FRONTEND_URL=http://localhost:3000

# OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:3000/auth/google/callback

GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret
GITHUB_REDIRECT_URI=http://localhost:3000/auth/github/callback

FACEBOOK_CLIENT_ID=your-facebook-client-id
FACEBOOK_CLIENT_SECRET=your-facebook-client-secret
FACEBOOK_REDIRECT_URI=http://localhost:3000/auth/facebook/callback

INSTAGRAM_CLIENT_ID=your-instagram-client-id
INSTAGRAM_CLIENT_SECRET=your-instagram-client-secret
INSTAGRAM_REDIRECT_URI=http://localhost:3000/auth/instagram/callback

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-specific-password
SMTP_FROM_EMAIL=noreply@parallaxpal.com

# Stripe
STRIPE_SECRET_KEY=your-stripe-secret-key
STRIPE_PUBLISHABLE_KEY=your-stripe-publishable-key
STRIPE_WEBHOOK_SECRET=your-stripe-webhook-secret

# Google Cloud AI
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_CLOUD_REGION=us-central1
VERTEX_AI_LOCATION=us-central1

# Jules Integration
JULES_GITHUB_TOKEN=your-github-token
JULES_WEBHOOK_SECRET=your-webhook-secret
JULES_AUTO_APPROVE_PRS=false
JULES_DEFAULT_ORG=nexus-forge

# Gemini Configuration
GEMINI_PRO_MODEL=gemini-2.5-pro
GEMINI_FLASH_MODEL=gemini-2.5-flash
GEMINI_MAX_TOKENS=8192
GEMINI_TEMPERATURE=0.7

# Imagen Configuration
IMAGEN_MODEL_VERSION=imagen-4.0
IMAGEN_DEFAULT_RESOLUTION=2048x1536
IMAGEN_SAFETY_FILTER=block_some

# Veo Configuration
VEO_MODEL_VERSION=veo-3.0-generate-preview
VEO_DEFAULT_DURATION=30
VEO_OUTPUT_BUCKET=gs://your-veo-outputs

# Monitoring
SENTRY_DSN=your-sentry-dsn
"""


def generate_env_file():
    """Generate .env file template"""
    with open(".env.example", "w") as f:
        f.write(ENV_TEMPLATE.strip())


if __name__ == "__main__":
    # Generate .env.example file when run directly
    generate_env_file()
