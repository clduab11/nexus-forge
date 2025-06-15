from enum import Enum
from typing import Optional
from pydantic import BaseModel, HttpUrl

class OAuthProvider(str, Enum):
    GOOGLE = "google"
    GITHUB = "github"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"

class OAuthUserData(BaseModel):
    """Standardized user data from OAuth providers"""
    provider: OAuthProvider
    provider_user_id: str
    email: str
    username: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[HttpUrl] = None
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[int] = None
    raw_data: dict  # Store original provider data

class OAuthConfig(BaseModel):
    """Base OAuth provider configuration"""
    client_id: str
    client_secret: str
    authorize_url: HttpUrl
    token_url: HttpUrl
    userinfo_url: HttpUrl
    scopes: list[str]
    redirect_uri: HttpUrl

class GoogleOAuthConfig(OAuthConfig):
    """Google OAuth specific configuration"""
    authorize_url: HttpUrl = "https://accounts.google.com/o/oauth2/v2/auth"
    token_url: HttpUrl = "https://oauth2.googleapis.com/token"
    userinfo_url: HttpUrl = "https://www.googleapis.com/oauth2/v3/userinfo"
    scopes: list[str] = [
        "openid",
        "email",
        "profile"
    ]

class GitHubOAuthConfig(OAuthConfig):
    """GitHub OAuth specific configuration"""
    authorize_url: HttpUrl = "https://github.com/login/oauth/authorize"
    token_url: HttpUrl = "https://github.com/login/oauth/access_token"
    userinfo_url: HttpUrl = "https://api.github.com/user"
    scopes: list[str] = [
        "read:user",
        "user:email"
    ]

class FacebookOAuthConfig(OAuthConfig):
    """Facebook OAuth specific configuration"""
    authorize_url: HttpUrl = "https://facebook.com/v18.0/dialog/oauth"
    token_url: HttpUrl = "https://graph.facebook.com/v18.0/oauth/access_token"
    userinfo_url: HttpUrl = "https://graph.facebook.com/v18.0/me"
    scopes: list[str] = [
        "email",
        "public_profile"
    ]

class InstagramOAuthConfig(OAuthConfig):
    """Instagram OAuth specific configuration"""
    authorize_url: HttpUrl = "https://api.instagram.com/oauth/authorize"
    token_url: HttpUrl = "https://api.instagram.com/oauth/access_token"
    userinfo_url: HttpUrl = "https://graph.instagram.com/me"
    scopes: list[str] = [
        "basic"
    ]

class OAuthState(BaseModel):
    """OAuth state parameter for security"""
    provider: OAuthProvider
    redirect_uri: str
    nonce: str  # Random string to prevent CSRF

class OAuthError(BaseModel):
    """OAuth error response"""
    error: str
    error_description: Optional[str] = None
    error_uri: Optional[HttpUrl] = None

class OAuthTokenResponse(BaseModel):
    """Standardized OAuth token response"""
    access_token: str
    token_type: str
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    id_token: Optional[str] = None  # For OpenID Connect

class OAuthCallbackParams(BaseModel):
    """Parameters received in OAuth callback"""
    code: str
    state: str
    error: Optional[str] = None
    error_description: Optional[str] = None