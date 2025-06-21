from enum import Enum as PyEnum

from sqlalchemy import (
    ARRAY,
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class ResearchStatus(str, PyEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class UserRole(str, PyEnum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    VIEWER = "viewer"


class SubscriptionStatus(str, PyEnum):
    ACTIVE = "active"
    CANCELED = "canceled"
    PAST_DUE = "past_due"
    UNPAID = "unpaid"
    TRIALING = "trialing"


class PaymentStatus(str, PyEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


class QueryPurchaseType(str, PyEnum):
    SINGLE = "single"
    PACK_5 = "pack_5"
    PACK_10 = "pack_10"


class AIModelType(str, PyEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(Enum(UserRole), default=UserRole.RESEARCHER)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    mfa_secret = Column(String, nullable=True)
    is_mfa_enabled = Column(Boolean, default=False)
    stripe_customer_id = Column(String, nullable=True)

    # Relationships
    research_tasks = relationship("ResearchTask", back_populates="owner")
    api_keys = relationship("APIKey", back_populates="user")
    refresh_tokens = relationship("RefreshToken", back_populates="user")
    subscriptions = relationship("Subscription", back_populates="user")
    payment_methods = relationship("PaymentMethod", back_populates="user")
    transactions = relationship("Transaction", back_populates="user")


class SubscriptionPlan(Base):
    __tablename__ = "subscription_plans"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    description = Column(Text)
    price = Column(Float)  # Price in USD
    interval = Column(String)  # 'month' or 'year'
    stripe_price_id = Column(String, unique=True)
    features = Column(JSON)  # Store features as JSON
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    allows_ollama = Column(
        Boolean, default=False
    )  # Whether plan includes Ollama access

    # Relationships
    subscriptions = relationship("Subscription", back_populates="plan")


class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    plan_id = Column(Integer, ForeignKey("subscription_plans.id"))
    stripe_subscription_id = Column(String, unique=True)
    status = Column(Enum(SubscriptionStatus))
    current_period_start = Column(DateTime(timezone=True))
    current_period_end = Column(DateTime(timezone=True))
    cancel_at_period_end = Column(Boolean, default=False)
    canceled_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="subscriptions")
    plan = relationship("SubscriptionPlan", back_populates="subscriptions")


class PaymentMethod(Base):
    __tablename__ = "payment_methods"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    stripe_payment_method_id = Column(String, unique=True)
    type = Column(String)  # 'card', 'bank_account', etc.
    last4 = Column(String)  # Last 4 digits of card/account
    exp_month = Column(Integer, nullable=True)  # For cards
    exp_year = Column(Integer, nullable=True)  # For cards
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="payment_methods")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    amount = Column(Float)
    currency = Column(String, default="USD")
    stripe_payment_intent_id = Column(String, unique=True)
    status = Column(Enum(PaymentStatus))
    description = Column(Text, nullable=True)
    meta_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    purchase_type = Column(String, nullable=True)  # 'subscription', 'query_pack', etc.
    query_quantity = Column(Integer, nullable=True)  # For query pack purchases

    # Relationships
    user = relationship("User", back_populates="transactions")
    query_purchase = relationship(
        "QueryPurchase", back_populates="transaction", uselist=False
    )


class QueryPurchase(Base):
    __tablename__ = "query_purchases"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    purchase_type = Column(Enum(QueryPurchaseType))
    quantity = Column(Integer)  # Number of queries purchased
    price_per_query = Column(Float)  # Price per query in USD
    total_amount = Column(Float)  # Total amount paid
    transaction_id = Column(Integer, ForeignKey("transactions.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)  # Optional expiration

    # Relationships
    user = relationship("User", backref="query_purchases")
    transaction = relationship("Transaction", back_populates="query_purchase")


class QueryCredit(Base):
    __tablename__ = "query_credits"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    remaining_queries = Column(Integer, default=0)
    last_updated = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user = relationship("User", backref="query_credits")
    usage_history = relationship("QueryUsage", back_populates="credit")


class QueryUsage(Base):
    __tablename__ = "query_usage"

    id = Column(Integer, primary_key=True, index=True)
    credit_id = Column(Integer, ForeignKey("query_credits.id"))
    task_id = Column(Integer, ForeignKey("research_tasks.id"))
    used_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    credit = relationship("QueryCredit", back_populates="usage_history")
    task = relationship("ResearchTask", backref="query_usage")


class ResearchTask(Base):
    __tablename__ = "research_tasks"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text)
    status = Column(Enum(ResearchStatus), default=ResearchStatus.PENDING)
    result = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    continuous_mode = Column(Boolean, default=False)
    max_iterations = Column(Integer, default=1)
    current_iteration = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"))

    # New fields for multi-model support
    web_results = Column(JSON, nullable=True)  # Store web results with snippets
    ai_analyses = Column(JSON, nullable=True)  # Store analyses from different models
    models_used = Column(ARRAY(String), nullable=True)  # Track which models were used
    used_ollama = Column(Boolean, default=False)  # Track Ollama usage

    # Relationships
    owner = relationship("User", back_populates="research_tasks")
    analytics = relationship("ResearchAnalytics", back_populates="task", uselist=False)
    sources = relationship("ResearchSource", back_populates="task")
    model_results = relationship("ModelResult", back_populates="task")


class ResearchAnalytics(Base):
    __tablename__ = "research_analytics"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("research_tasks.id"), unique=True)
    processing_time = Column(Integer)  # in milliseconds
    token_count = Column(Integer)
    source_count = Column(Integer)
    model_count = Column(Integer)  # Number of AI models used
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Performance metrics
    avg_source_processing_time = Column(Float, nullable=True)
    avg_tokens_per_source = Column(Float, nullable=True)
    cache_hit_rate = Column(Float, nullable=True)
    avg_model_confidence = Column(
        Float, nullable=True
    )  # Average confidence across models

    # Relationships
    task = relationship("ResearchTask", back_populates="analytics")


class ResearchSource(Base):
    __tablename__ = "research_sources"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("research_tasks.id"))
    url = Column(String)
    title = Column(String, nullable=True)
    snippet = Column(Text, nullable=True)  # Store content preview
    source_type = Column(String)  # 'web', 'database', etc.
    relevance_score = Column(Float)
    processed_at = Column(DateTime(timezone=True), server_default=func.now())
    processing_time = Column(Integer)  # in milliseconds

    # Relationships
    task = relationship("ResearchTask", back_populates="sources")
    model_analyses = relationship("ModelResult", back_populates="source")


class ModelResult(Base):
    __tablename__ = "model_results"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("research_tasks.id"))
    source_id = Column(Integer, ForeignKey("research_sources.id"))
    model_type = Column(Enum(AIModelType))
    analysis = Column(Text)
    confidence = Column(Float)
    processing_time = Column(Integer)  # in milliseconds
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    task = relationship("ResearchTask", back_populates="model_results")
    source = relationship("ResearchSource", back_populates="model_analyses")


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    name = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    last_used = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    replaced_by = Column(Integer, ForeignKey("refresh_tokens.id"), nullable=True)

    # Relationships
    user = relationship("User")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String)
    resource_type = Column(String)
    resource_id = Column(Integer)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    ip_address = Column(String)
    user_agent = Column(String)
    details = Column(Text, nullable=True)


# Create indexes
from sqlalchemy import Index

Index("ix_research_tasks_status", ResearchTask.status)
Index(
    "ix_research_tasks_owner_created",
    ResearchTask.owner_id,
    ResearchTask.created_at.desc(),
)
Index("ix_api_keys_active", APIKey.is_active)
Index("ix_audit_logs_user_timestamp", AuditLog.user_id, AuditLog.timestamp.desc())
Index("ix_subscriptions_user_status", Subscription.user_id, Subscription.status)
Index(
    "ix_transactions_user_created", Transaction.user_id, Transaction.created_at.desc()
)
Index(
    "ix_payment_methods_user_default", PaymentMethod.user_id, PaymentMethod.is_default
)
Index("ix_model_results_task_model", ModelResult.task_id, ModelResult.model_type)
Index(
    "ix_research_sources_task_relevance",
    ResearchSource.task_id,
    ResearchSource.relevance_score.desc(),
)
Index(
    "ix_query_purchases_user_created",
    QueryPurchase.user_id,
    QueryPurchase.created_at.desc(),
)
Index("ix_query_credits_user", QueryCredit.user_id)
Index("ix_query_usage_credit", QueryUsage.credit_id, QueryUsage.used_at.desc())
