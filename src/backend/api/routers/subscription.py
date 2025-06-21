from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..dependencies.auth import get_current_user
from ..dependencies.database import get_db
from ..models import SubscriptionStatus, User

router = APIRouter(prefix="/subscription", tags=["subscription"])


@router.get("/features", response_model=Dict[str, bool])
async def get_subscription_features(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    """Get user's subscription features"""
    # Get user's active subscription
    active_sub = next(
        (
            sub
            for sub in current_user.subscriptions
            if sub.status == SubscriptionStatus.ACTIVE
            and sub.current_period_end > datetime.utcnow()
        ),
        None,
    )

    if not active_sub:
        return {"gpu_acceleration": False, "ollama_access": False, "is_pro": False}

    return {
        "gpu_acceleration": active_sub.plan.price >= 79.99,  # Pro plan or higher
        "ollama_access": active_sub.plan.allows_ollama,
        "is_pro": active_sub.plan.price >= 79.99,
    }
