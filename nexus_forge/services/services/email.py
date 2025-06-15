from typing import List, Optional
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import EmailStr, BaseModel
from pathlib import Path
import aiofiles
import jinja2
from datetime import datetime

from ..config import settings

# Email templates directory
TEMPLATE_DIR = Path(__file__).parent.parent / "templates" / "email"

# Configure FastMail
conf = ConnectionConfig(
    MAIL_USERNAME=settings.SMTP_USER,
    MAIL_PASSWORD=settings.SMTP_PASSWORD,
    MAIL_FROM=settings.SMTP_FROM_EMAIL,
    MAIL_PORT=settings.SMTP_PORT,
    MAIL_SERVER=settings.SMTP_HOST,
    MAIL_FROM_NAME=settings.SMTP_FROM_NAME,
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True,
    TEMPLATE_FOLDER=str(TEMPLATE_DIR)
)

# Initialize FastMail instance
fastmail = FastMail(conf)

# Initialize Jinja2 environment
template_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=True
)

class EmailTemplate(BaseModel):
    """Base model for email templates"""
    subject: str
    template_name: str
    context: dict

class EmailService:
    @staticmethod
    async def _render_template(template_name: str, context: dict) -> str:
        """Render email template with context"""
        template = template_env.get_template(template_name)
        return template.render(**context)

    @staticmethod
    async def _send_email(
        email: EmailStr,
        subject: str,
        body: str,
        template_name: Optional[str] = None,
        context: Optional[dict] = None
    ) -> None:
        """Send email using FastMail"""
        if template_name and context:
            # Use template if provided
            html_content = await EmailService._render_template(template_name, context)
        else:
            # Use plain body if no template
            html_content = body

        message = MessageSchema(
            subject=subject,
            recipients=[email],
            body=html_content,
            subtype="html"
        )

        await fastmail.send_message(message)

    @staticmethod
    async def send_verification_email(email: EmailStr, token: str) -> None:
        """Send email verification link"""
        verification_url = f"{settings.FRONTEND_URL}/verify-email?token={token}"
        
        context = {
            "verification_url": verification_url,
            "app_name": settings.APP_NAME,
            "support_email": settings.SMTP_FROM_EMAIL,
            "year": datetime.now().year
        }
        
        await EmailService._send_email(
            email=email,
            subject=f"Verify your {settings.APP_NAME} account",
            template_name="verification.html",
            context=context,
            body=""  # Body is provided by template
        )

    @staticmethod
    async def send_password_reset_email(email: EmailStr, token: str) -> None:
        """Send password reset link"""
        reset_url = f"{settings.FRONTEND_URL}/reset-password?token={token}"
        
        context = {
            "reset_url": reset_url,
            "app_name": settings.APP_NAME,
            "support_email": settings.SMTP_FROM_EMAIL,
            "year": datetime.now().year
        }
        
        await EmailService._send_email(
            email=email,
            subject=f"Reset your {settings.APP_NAME} password",
            template_name="password_reset.html",
            context=context,
            body=""  # Body is provided by template
        )

    @staticmethod
    async def send_welcome_email(email: EmailStr, username: str) -> None:
        """Send welcome email after account verification"""
        context = {
            "username": username,
            "app_name": settings.APP_NAME,
            "login_url": f"{settings.FRONTEND_URL}/login",
            "support_email": settings.SMTP_FROM_EMAIL,
            "year": datetime.now().year
        }
        
        await EmailService._send_email(
            email=email,
            subject=f"Welcome to {settings.APP_NAME}!",
            template_name="welcome.html",
            context=context,
            body=""  # Body is provided by template
        )

    @staticmethod
    async def send_subscription_confirmation(
        email: EmailStr,
        plan_name: str,
        amount: float,
        next_billing_date: datetime
    ) -> None:
        """Send subscription confirmation email"""
        context = {
            "plan_name": plan_name,
            "amount": amount,
            "next_billing_date": next_billing_date.strftime("%B %d, %Y"),
            "app_name": settings.APP_NAME,
            "account_url": f"{settings.FRONTEND_URL}/account",
            "support_email": settings.SMTP_FROM_EMAIL,
            "year": datetime.now().year
        }
        
        await EmailService._send_email(
            email=email,
            subject=f"Your {settings.APP_NAME} subscription is active",
            template_name="subscription_confirmation.html",
            context=context,
            body=""  # Body is provided by template
        )

    @staticmethod
    async def send_subscription_canceled(
        email: EmailStr,
        plan_name: str,
        end_date: datetime
    ) -> None:
        """Send subscription cancellation confirmation"""
        context = {
            "plan_name": plan_name,
            "end_date": end_date.strftime("%B %d, %Y"),
            "app_name": settings.APP_NAME,
            "resubscribe_url": f"{settings.FRONTEND_URL}/account/subscription",
            "support_email": settings.SMTP_FROM_EMAIL,
            "year": datetime.now().year
        }
        
        await EmailService._send_email(
            email=email,
            subject=f"Your {settings.APP_NAME} subscription has been canceled",
            template_name="subscription_canceled.html",
            context=context,
            body=""  # Body is provided by template
        )
        
    @staticmethod
    async def send_subscription_reactivated(
        email: EmailStr,
        plan_name: str,
        next_billing_date: datetime
    ) -> None:
        """Send subscription reactivation confirmation"""
        context = {
            "plan_name": plan_name,
            "next_billing_date": next_billing_date.strftime("%B %d, %Y"),
            "app_name": settings.APP_NAME,
            "account_url": f"{settings.FRONTEND_URL}/account",
            "support_email": settings.SMTP_FROM_EMAIL,
            "year": datetime.now().year
        }
        
        await EmailService._send_email(
            email=email,
            subject=f"Your {settings.APP_NAME} subscription has been reactivated",
            template_name="subscription_reactivated.html",
            context=context,
            body=""  # Body is provided by template
        )

    @staticmethod
    async def send_payment_failed(
        email: EmailStr,
        amount: float,
        retry_date: datetime
    ) -> None:
        """Send payment failure notification"""
        context = {
            "amount": amount,
            "retry_date": retry_date.strftime("%B %d, %Y"),
            "app_name": settings.APP_NAME,
            "billing_url": f"{settings.FRONTEND_URL}/billing",
            "support_email": settings.SMTP_FROM_EMAIL,
            "year": datetime.now().year
        }
        
        await EmailService._send_email(
            email=email,
            subject=f"Payment failed for your {settings.APP_NAME} subscription",
            template_name="payment_failed.html",
            context=context,
            body=""  # Body is provided by template
        )

    @staticmethod
    async def send_security_alert(
        email: EmailStr,
        alert_type: str,
        ip_address: str,
        location: str,
        device: str
    ) -> None:
        """Send security alert email"""
        context = {
            "alert_type": alert_type,
            "ip_address": ip_address,
            "location": location,
            "device": device,
            "timestamp": datetime.now().strftime("%B %d, %Y at %H:%M:%S UTC"),
            "app_name": settings.APP_NAME,
            "security_url": f"{settings.FRONTEND_URL}/account/security",
            "support_email": settings.SMTP_FROM_EMAIL,
            "year": datetime.now().year
        }
        
        await EmailService._send_email(
            email=email,
            subject=f"Security Alert - {settings.APP_NAME}",
            template_name="security_alert.html",
            context=context,
            body=""  # Body is provided by template
        )