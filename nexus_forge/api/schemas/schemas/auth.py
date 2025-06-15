from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, EmailStr, constr, validator
from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    VIEWER = "viewer"

class UserBase(BaseModel):
    email: EmailStr
    username: constr(min_length=3, max_length=50)

class UserCreate(UserBase):
    password: constr(min_length=8)
    
    @validator("password")
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one number")
        return v

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[constr(min_length=3, max_length=50)] = None
    is_active: Optional[bool] = None
    role: Optional[UserRole] = None

class UserResponse(UserBase):
    id: int
    is_active: bool
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime]
    is_mfa_enabled: bool
    has_subscription: bool

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    scopes: List[str] = []

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class PasswordReset(BaseModel):
    token: str
    new_password: constr(min_length=8)
    
    @validator("new_password")
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one number")
        return v

class PasswordChange(BaseModel):
    current_password: str
    new_password: constr(min_length=8)
    
    @validator("new_password")
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one number")
        return v

class MFASetup(BaseModel):
    secret: str
    qr_code: str

class MFAVerify(BaseModel):
    code: constr(min_length=6, max_length=6)
    remember_device: bool = False

class APIKeyCreate(BaseModel):
    name: constr(min_length=1, max_length=50)
    expires_in_days: Optional[int] = None

class APIKeyResponse(BaseModel):
    id: int
    name: str
    key: str
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    is_active: bool

    class Config:
        orm_mode = True