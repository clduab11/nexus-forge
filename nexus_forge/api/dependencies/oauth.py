from typing import Optional, Dict, Any
import httpx
import jwt
from fastapi import HTTPException, status
import secrets
import base64
import json
from datetime import datetime, timedelta

from ..schemas.oauth import (
    OAuthProvider,
    OAuthUserData,
    OAuthConfig,
    GoogleOAuthConfig,
    GitHubOAuthConfig,
    FacebookOAuthConfig,
    InstagramOAuthConfig,
    OAuthError,
    OAuthTokenResponse
)
from ...config import settings

class OAuthClient:
    """Base OAuth client implementation"""
    def __init__(self, config: OAuthConfig):
        self.config = config
        self.http_client = httpx.AsyncClient()

    async def get_authorization_url(self, state: str) -> str:
        """Generate authorization URL for OAuth flow"""
        params = {
            'client_id': self.config.client_id,
            'redirect_uri': str(self.config.redirect_uri),
            'scope': ' '.join(self.config.scopes),
            'response_type': 'code',
            'state': state
        }
        return f"{self.config.authorize_url}?{httpx.QueryParams(params)}"

    async def exchange_code(self, code: str) -> OAuthTokenResponse:
        """Exchange authorization code for tokens"""
        data = {
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret,
            'code': code,
            'redirect_uri': str(self.config.redirect_uri),
            'grant_type': 'authorization_code'
        }
        
        async with self.http_client as client:
            response = await client.post(
                str(self.config.token_url),
                data=data,
                headers={'Accept': 'application/json'}
            )
            
            if response.status_code != 200:
                error_data = response.json()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Token exchange failed: {error_data.get('error_description', error_data.get('error'))}"
                )
            
            return OAuthTokenResponse(**response.json())

    async def get_user_data(self, access_token: str) -> Dict[str, Any]:
        """Get user data from provider's API"""
        async with self.http_client as client:
            response = await client.get(
                str(self.config.userinfo_url),
                headers={'Authorization': f'Bearer {access_token}'}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get user data"
                )
            
            return response.json()

class GoogleOAuth(OAuthClient):
    """Google OAuth implementation"""
    def __init__(self):
        super().__init__(GoogleOAuthConfig(
            client_id=settings.GOOGLE_CLIENT_ID,
            client_secret=settings.GOOGLE_CLIENT_SECRET,
            redirect_uri=settings.GOOGLE_REDIRECT_URI
        ))

    async def get_user_data(self, code: str) -> OAuthUserData:
        tokens = await self.exchange_code(code)
        
        # Decode ID token to get user info
        try:
            id_token = tokens.id_token
            if not id_token:
                raise ValueError("No ID token received")
            
            # Google tokens are JWTs that can be decoded without verification
            # for user info (verification happens during token exchange)
            user_info = jwt.decode(id_token, options={"verify_signature": False})
            
            return OAuthUserData(
                provider=OAuthProvider.GOOGLE,
                provider_user_id=user_info['sub'],
                email=user_info['email'],
                username=user_info.get('name'),
                full_name=user_info.get('name'),
                avatar_url=user_info.get('picture'),
                access_token=tokens.access_token,
                refresh_token=tokens.refresh_token,
                expires_at=int(datetime.now().timestamp()) + tokens.expires_in,
                raw_data=user_info
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process Google user data: {str(e)}"
            )

class GitHubOAuth(OAuthClient):
    """GitHub OAuth implementation"""
    def __init__(self):
        super().__init__(GitHubOAuthConfig(
            client_id=settings.GITHUB_CLIENT_ID,
            client_secret=settings.GITHUB_CLIENT_SECRET,
            redirect_uri=settings.GITHUB_REDIRECT_URI
        ))

    async def get_user_data(self, code: str) -> OAuthUserData:
        tokens = await self.exchange_code(code)
        
        # Get user profile
        async with self.http_client as client:
            profile_response = await client.get(
                str(self.config.userinfo_url),
                headers={'Authorization': f'token {tokens.access_token}'}
            )
            
            if profile_response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get GitHub user profile"
                )
            
            profile = profile_response.json()
            
            # Get user email (might be private)
            email_response = await client.get(
                'https://api.github.com/user/emails',
                headers={'Authorization': f'token {tokens.access_token}'}
            )
            
            if email_response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get GitHub user email"
                )
            
            # Get primary email
            emails = email_response.json()
            primary_email = next(
                (email['email'] for email in emails if email['primary']),
                emails[0]['email'] if emails else None
            )
            
            return OAuthUserData(
                provider=OAuthProvider.GITHUB,
                provider_user_id=str(profile['id']),
                email=primary_email,
                username=profile['login'],
                full_name=profile.get('name'),
                avatar_url=profile.get('avatar_url'),
                access_token=tokens.access_token,
                raw_data=profile
            )

class FacebookOAuth(OAuthClient):
    """Facebook OAuth implementation"""
    def __init__(self):
        super().__init__(FacebookOAuthConfig(
            client_id=settings.FACEBOOK_CLIENT_ID,
            client_secret=settings.FACEBOOK_CLIENT_SECRET,
            redirect_uri=settings.FACEBOOK_REDIRECT_URI
        ))

    async def get_user_data(self, code: str) -> OAuthUserData:
        tokens = await self.exchange_code(code)
        
        # Get user profile with email
        async with self.http_client as client:
            response = await client.get(
                f"{self.config.userinfo_url}",
                params={
                    'fields': 'id,email,name,picture',
                    'access_token': tokens.access_token
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get Facebook user data"
                )
            
            profile = response.json()
            
            return OAuthUserData(
                provider=OAuthProvider.FACEBOOK,
                provider_user_id=profile['id'],
                email=profile.get('email'),
                username=None,  # Facebook doesn't provide username
                full_name=profile.get('name'),
                avatar_url=profile.get('picture', {}).get('data', {}).get('url'),
                access_token=tokens.access_token,
                expires_at=int(datetime.now().timestamp()) + tokens.expires_in,
                raw_data=profile
            )

class InstagramOAuth(OAuthClient):
    """Instagram OAuth implementation"""
    def __init__(self):
        super().__init__(InstagramOAuthConfig(
            client_id=settings.INSTAGRAM_CLIENT_ID,
            client_secret=settings.INSTAGRAM_CLIENT_SECRET,
            redirect_uri=settings.INSTAGRAM_REDIRECT_URI
        ))

    async def get_user_data(self, code: str) -> OAuthUserData:
        tokens = await self.exchange_code(code)
        
        # Get user profile
        async with self.http_client as client:
            response = await client.get(
                f"{self.config.userinfo_url}",
                params={
                    'fields': 'id,username',
                    'access_token': tokens.access_token
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get Instagram user data"
                )
            
            profile = response.json()
            
            return OAuthUserData(
                provider=OAuthProvider.INSTAGRAM,
                provider_user_id=profile['id'],
                email=None,  # Instagram Basic Display API doesn't provide email
                username=profile.get('username'),
                access_token=tokens.access_token,
                raw_data=profile
            )

# Initialize OAuth clients
google_oauth = GoogleOAuth()
github_oauth = GitHubOAuth()
facebook_oauth = FacebookOAuth()
instagram_oauth = InstagramOAuth()

def generate_oauth_state() -> str:
    """Generate secure state parameter for OAuth flow"""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8')