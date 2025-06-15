from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from ..dependencies.auth import (
    get_current_user,
    create_access_token,
    create_refresh_token,
    verify_refresh_token,
    get_password_hash,
    authenticate_user
)
from ..dependencies.database import get_db
from ..schemas.auth import (
    Token,
    TokenData,
    UserCreate,
    UserResponse,
    RefreshTokenRequest
)
from ..schemas.oauth import OAuthProvider
from ...services.services.auth import AuthService
from ...services.services.email import EmailService
from ...models import User, RefreshToken
from ...config import settings

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", response_model=Dict[str, str])
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends()
):
    """Register a new user"""
    try:
        user = await auth_service.create_user(db, user_data)
        verification_token = create_access_token(
            data={"sub": user.email},
            expires_delta=timedelta(hours=24)
        )
        await EmailService.send_verification_email(user.email, verification_token)
        return {"message": "Registration successful. Please check your email to verify your account."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends()
):
    """Login with username and password"""
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please verify your email first"
        )
    
    return await auth_service.create_tokens(db, user)

@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token_data: RefreshTokenRequest,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends()
):
    """Get new access token using refresh token"""
    return await auth_service.refresh_token(db, refresh_token_data.refresh_token)

@router.post("/verify-email/{token}", response_model=Dict[str, str])
async def verify_email(
    token: str,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends()
):
    """Verify user's email address"""
    try:
        await auth_service.verify_email(db, token)
        return {"message": "Email verified successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/oauth/{provider}", response_model=Token)
async def oauth_login(
    provider: OAuthProvider,
    code: str,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends()
):
    """Handle OAuth login for different providers"""
    try:
        return await auth_service.handle_oauth_login(db, provider, code)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/logout", response_model=Dict[str, str])
async def logout(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Logout user and invalidate refresh token"""
    db.query(RefreshToken).filter(
        RefreshToken.user_id == current_user.id,
        RefreshToken.expires_at > datetime.utcnow()
    ).update({"expires_at": datetime.utcnow()})
    db.commit()
    return {"message": "Successfully logged out"}

@router.post("/request-password-reset", response_model=Dict[str, str])
async def request_password_reset(
    email: EmailStr,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends()
):
    """Request password reset email"""
    try:
        await auth_service.request_password_reset(db, email)
        return {"message": "If an account exists with this email, a password reset link has been sent"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/reset-password/{token}", response_model=Dict[str, str])
async def reset_password(
    token: str,
    new_password: str,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends()
):
    """Reset password using reset token"""
    try:
        await auth_service.reset_password(db, token, new_password)
        return {"message": "Password reset successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/change-password", response_model=Dict[str, str])
async def change_password(
    current_password: str,
    new_password: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends()
):
    """Change user's password"""
    try:
        await auth_service.change_password(
            db,
            current_user,
            current_password,
            new_password
        )
        return {"message": "Password changed successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))