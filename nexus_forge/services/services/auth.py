from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
import jwt
from jwt.exceptions import PyJWTError

from ..models import User, RefreshToken
from ..schemas.auth import UserCreate, Token
from ..schemas.oauth import OAuthProvider, OAuthUserData
from ..dependencies.oauth import (
    google_oauth,
    github_oauth,
    facebook_oauth,
    instagram_oauth
)
from ..config import settings
from ..utils.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token
)

class AuthService:
    async def create_user(self, db: Session, user_data: UserCreate) -> User:
        """Create a new user"""
        # Check if user already exists
        if db.query(User).filter(User.email == user_data.email).first():
            raise ValueError("Email already registered")
        
        # Create user
        hashed_password = get_password_hash(user_data.password)
        user = User(
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed_password,
            is_active=False  # Requires email verification
        )
        
        try:
            db.add(user)
            db.commit()
            db.refresh(user)
            return user
        except IntegrityError:
            db.rollback()
            raise ValueError("Username already taken")

    async def verify_email(self, db: Session, token: str) -> None:
        """Verify user's email address"""
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            email = payload.get("sub")
            if not email:
                raise ValueError("Invalid verification token")
        except PyJWTError:
            raise ValueError("Invalid verification token")

        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise ValueError("User not found")
        if user.is_active:
            raise ValueError("Email already verified")

        user.is_active = True
        db.commit()

    async def create_tokens(self, db: Session, user: User) -> Token:
        """Create access and refresh tokens"""
        access_token = create_access_token(data={"sub": user.email})
        refresh_token = create_refresh_token(data={"sub": user.email})

        # Store refresh token
        db_refresh_token = RefreshToken(
            token=refresh_token,
            user_id=user.id,
            expires_at=datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        )
        db.add(db_refresh_token)
        db.commit()

        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer"
        )

    async def refresh_token(self, db: Session, refresh_token: str) -> Token:
        """Get new access token using refresh token"""
        try:
            payload = jwt.decode(
                refresh_token,
                settings.SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            email = payload.get("sub")
            if not email:
                raise ValueError("Invalid refresh token")
        except PyJWTError:
            raise ValueError("Invalid refresh token")

        # Verify refresh token exists and is valid
        db_token = db.query(RefreshToken).filter(
            RefreshToken.token == refresh_token,
            RefreshToken.expires_at > datetime.utcnow()
        ).first()
        if not db_token:
            raise ValueError("Invalid or expired refresh token")

        user = db.query(User).filter(User.email == email).first()
        if not user or not user.is_active:
            raise ValueError("User not found or inactive")

        # Create new tokens
        return await self.create_tokens(db, user)

    async def handle_oauth_login(
        self,
        db: Session,
        provider: OAuthProvider,
        code: str
    ) -> Token:
        """Handle OAuth login flow"""
        oauth_handlers = {
            OAuthProvider.GOOGLE: google_oauth,
            OAuthProvider.GITHUB: github_oauth,
            OAuthProvider.FACEBOOK: facebook_oauth,
            OAuthProvider.INSTAGRAM: instagram_oauth
        }

        if provider not in oauth_handlers:
            raise ValueError(f"Unsupported OAuth provider: {provider}")

        # Get user data from OAuth provider
        oauth_handler = oauth_handlers[provider]
        user_data = await oauth_handler.get_user_data(code)

        # Find or create user
        user = db.query(User).filter(User.email == user_data.email).first()
        if not user:
            user = User(
                email=user_data.email,
                username=user_data.username or user_data.email.split('@')[0],
                is_active=True,  # OAuth users are pre-verified
                oauth_provider=provider,
                oauth_user_id=user_data.provider_user_id
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        # Create tokens
        return await self.create_tokens(db, user)

    async def request_password_reset(self, db: Session, email: str) -> None:
        """Request password reset"""
        user = db.query(User).filter(User.email == email).first()
        if not user:
            # Don't reveal if user exists
            return

        # Create reset token
        reset_token = create_access_token(
            data={"sub": user.email, "type": "reset"},
            expires_delta=timedelta(hours=1)
        )

        # Send reset email
        from ..services.email import EmailService
        await EmailService.send_password_reset_email(user.email, reset_token)

    async def reset_password(
        self,
        db: Session,
        token: str,
        new_password: str
    ) -> None:
        """Reset password using reset token"""
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            email = payload.get("sub")
            token_type = payload.get("type")
            if not email or token_type != "reset":
                raise ValueError("Invalid reset token")
        except PyJWTError:
            raise ValueError("Invalid reset token")

        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise ValueError("User not found")

        # Update password
        user.hashed_password = get_password_hash(new_password)
        db.commit()

        # Invalidate all refresh tokens
        db.query(RefreshToken).filter(
            RefreshToken.user_id == user.id,
            RefreshToken.expires_at > datetime.utcnow()
        ).update({"expires_at": datetime.utcnow()})
        db.commit()

    async def change_password(
        self,
        db: Session,
        user: User,
        current_password: str,
        new_password: str
    ) -> None:
        """Change user's password"""
        if not verify_password(current_password, user.hashed_password):
            raise ValueError("Incorrect current password")

        user.hashed_password = get_password_hash(new_password)
        db.commit()

        # Invalidate all refresh tokens
        db.query(RefreshToken).filter(
            RefreshToken.user_id == user.id,
            RefreshToken.expires_at > datetime.utcnow()
        ).update({"expires_at": datetime.utcnow()})
        db.commit()