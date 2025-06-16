"""
Test-specific configuration that overrides production settings
"""
import os
from typing import Optional
from nexus_forge.config import Settings

class TestSettings(Settings):
    """Test-specific settings that provide defaults for all required fields"""
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///:memory:"
    
    # Redis
    REDIS_PASSWORD: str = "test-password"
    
    # OAuth
    GOOGLE_CLIENT_ID: str = "test-google-client-id"
    GOOGLE_CLIENT_SECRET: str = "test-google-client-secret"
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/auth/google/callback"
    
    GITHUB_CLIENT_ID: str = "test-github-client-id"
    GITHUB_CLIENT_SECRET: str = "test-github-client-secret"
    GITHUB_REDIRECT_URI: str = "http://localhost:8000/auth/github/callback"
    
    FACEBOOK_CLIENT_ID: str = "test-facebook-client-id"
    FACEBOOK_CLIENT_SECRET: str = "test-facebook-client-secret"
    FACEBOOK_REDIRECT_URI: str = "http://localhost:8000/auth/facebook/callback"
    
    INSTAGRAM_CLIENT_ID: str = "test-instagram-client-id"
    INSTAGRAM_CLIENT_SECRET: str = "test-instagram-client-secret"
    INSTAGRAM_REDIRECT_URI: str = "http://localhost:8000/auth/instagram/callback"
    
    # Email
    SMTP_HOST: str = "smtp.test.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = "test@test.com"
    SMTP_PASSWORD: str = "test-password"
    SMTP_FROM_EMAIL: str = "noreply@test.com"
    
    # Stripe
    STRIPE_SECRET_KEY: str = "sk_test_test_key"
    STRIPE_PUBLISHABLE_KEY: str = "pk_test_test_key"
    STRIPE_WEBHOOK_SECRET: str = "whsec_test_secret"
    
    # Optional fields
    SENTRY_DSN: Optional[str] = None
    SSL_KEYFILE: Optional[str] = None
    SSL_CERTFILE: Optional[str] = None
    
    class Config:
        env_file = None  # Don't load from .env file during testing

def get_test_settings():
    """Get test settings instance"""
    return TestSettings()