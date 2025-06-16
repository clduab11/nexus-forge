import logging
import os
from contextlib import contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from .config import get_db_url, settings

logger = logging.getLogger(__name__)

# Create engines for both sync and async operations
db_url = get_db_url()
is_testing = os.getenv("TESTING") == "true"
is_sqlite = "sqlite" in db_url

if is_sqlite:
    # SQLite configuration (mainly for testing)
    engine = create_engine(
        db_url,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=settings.DEBUG,
    )

    async_engine = create_async_engine(
        db_url,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=settings.DEBUG,
    )
else:
    # PostgreSQL configuration (production)
    engine = create_engine(
        db_url,
        poolclass=QueuePool,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=settings.DEBUG,
    )

    async_engine = create_async_engine(
        db_url.replace("postgresql://", "postgresql+asyncpg://"),
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=settings.DEBUG,
    )

# Create session factories
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

AsyncSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=async_engine, class_=AsyncSession
)

# Base class for SQLAlchemy models
Base = declarative_base()


@contextmanager
def get_db() -> Generator:
    """
    Synchronous database session context manager.
    Usage:
        with get_db() as db:
            db.query(Model).all()
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Asynchronous database session context manager.
    Usage:
        async with get_async_db() as db:
            result = await db.execute(select(Model))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            await session.rollback()
            raise
        finally:
            await session.close()


def init_db() -> None:
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise


async def check_db_connection() -> bool:
    """Check database connectivity"""
    try:
        async with get_async_db() as db:
            await db.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {str(e)}")
        return False


class DatabaseHealthCheck:
    @staticmethod
    async def check_health() -> dict:
        """
        Perform comprehensive database health check.
        Returns dict with status and metrics.
        """
        try:
            async with get_async_db() as db:
                # Check basic connectivity
                await db.execute("SELECT 1")

                # Get connection pool stats
                pool_status = {
                    "pool_size": engine.pool.size(),
                    "checkedin": engine.pool.checkedin(),
                    "checkedout": engine.pool.checkedout(),
                    "overflow": engine.pool.overflow(),
                }

                return {
                    "status": "healthy",
                    "pool_metrics": pool_status,
                    "message": "Database connection successful",
                }
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Database connection failed",
            }
