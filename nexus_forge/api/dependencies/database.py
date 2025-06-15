"""Database dependency for FastAPI dependency injection."""
from sqlalchemy.orm import Session
from nexus_forge.database import SessionLocal

def get_db() -> Session:
    """Get database session for dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()