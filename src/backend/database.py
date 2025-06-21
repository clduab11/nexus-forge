import logging
import os
from contextlib import contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool
import threading
import time
from collections import defaultdict

from .config import get_db_url, settings
from .core.cache import RedisCache

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
    # PostgreSQL configuration (production) - HIGHLY OPTIMIZED
    # Dynamically sized pool based on system resources
    import psutil
    cpu_count = psutil.cpu_count()
    
    # Formula: base_pool + (cpu_cores * multiplier)
    pool_size = max(20, settings.DATABASE_POOL_SIZE * 2 + cpu_count * 2)
    max_overflow = max(40, settings.DATABASE_MAX_OVERFLOW * 3 + cpu_count * 3)
    
    engine = create_engine(
        db_url,
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        pool_recycle=1800,  # Recycle connections every 30 min
        pool_timeout=30,  # Timeout for getting connection from pool
        echo=settings.DEBUG,
        query_cache_size=2400,  # Double query plan caching
        connect_args={
            "connect_timeout": 10,
            "application_name": "nexus_forge_hyper_optimized",
            "options": "-c statement_timeout=30000 -c jit=on -c max_parallel_workers_per_gather=4",
            "server_settings": {
                "shared_preload_libraries": "pg_stat_statements",
                "pg_stat_statements.track": "all",
            }
        }
    )

    async_engine = create_async_engine(
        db_url.replace("postgresql://", "postgresql+asyncpg://"),
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        pool_recycle=1800,
        echo=settings.DEBUG,
        query_cache_size=2400,
        connect_args={
            "timeout": 10,
            "command_timeout": 30,
            "server_settings": {
                "application_name": "nexus_forge_async_hyper_optimized",
                "jit": "on",  # Enable JIT for complex queries
                "max_parallel_workers_per_gather": "4",
                "effective_cache_size": "4GB",
                "work_mem": "256MB"
            },
            "prepared_statement_cache_size": 0,  # Disable to prevent cache bloat
            "prepared_statement_name_func": lambda: None,
        }
    )

# Create session factories
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

AsyncSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=async_engine, class_=AsyncSession
)

# Base class for SQLAlchemy models
Base = declarative_base()

# Initialize multi-level cache for query results
query_cache = RedisCache()

# Add prepared statement cache
prepared_statements_cache = {}
ps_cache_lock = threading.Lock()

# Query execution statistics with thread safety
query_stats = {
    "total_queries": 0,
    "cache_hits": 0,
    "slow_queries": 0,
    "query_times": [],
    "connection_wait_times": [],
    "query_patterns": defaultdict(lambda: {"count": 0, "total_time": 0}),
}
stats_lock = threading.Lock()

# Connection pool monitoring
pool_monitor = {
    "max_overflow_reached": 0,
    "connection_timeouts": 0,
    "active_connections": 0,
    "connection_creation_times": [],
}


@contextmanager
def get_db() -> Generator:
    """
    Synchronous database session context manager with performance monitoring.
    Usage:
        with get_db() as db:
            db.query(Model).all()
    """
    start_time = time.time()
    wait_start = time.time()
    
    db = SessionLocal()
    
    # Record connection acquisition time
    wait_time = time.time() - wait_start
    with stats_lock:
        query_stats["connection_wait_times"].append(wait_time)
        if len(query_stats["connection_wait_times"]) > 1000:
            query_stats["connection_wait_times"] = query_stats["connection_wait_times"][-500:]
    
    # Enable query logging for slow queries
    @event.listens_for(db, "after_cursor_execute")
    def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        execution_time = time.time() - start_time
        if execution_time > 1.0:  # Log slow queries (> 1 second)
            logger.warning(f"Slow query detected ({execution_time:.2f}s): {statement[:100]}...")
            with stats_lock:
                query_stats["slow_queries"] += 1
    
    try:
        # Set session-level optimizations
        db.execute("SET SESSION random_page_cost = 1.1")
        db.execute("SET SESSION effective_cache_size = '4GB'")
        
        yield db
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()
        
        # Update connection pool stats
        with stats_lock:
            pool_monitor["active_connections"] = engine.pool.checkedout()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Asynchronous database session context manager with advanced optimization.
    Usage:
        async with get_async_db() as db:
            result = await db.execute(select(Model))
    """
    start_time = time.time()
    wait_start = time.time()
    
    async with AsyncSessionLocal() as session:
        # Record connection acquisition time
        wait_time = time.time() - wait_start
        with stats_lock:
            query_stats["connection_wait_times"].append(wait_time)
        
        # Enable advanced query optimizations
        session.sync_session.execute_options(
            synchronize_session="fetch",
            stream_results=True,  # Stream large result sets
            yield_per=100,  # Batch fetching
        )
        
        # Set session-level optimizations
        await session.execute("SET SESSION work_mem = '256MB'")
        await session.execute("SET SESSION maintenance_work_mem = '512MB'")
        
        try:
            yield session
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            await session.rollback()
            raise
        finally:
            await session.close()
            
            # Update stats
            execution_time = time.time() - start_time
            with stats_lock:
                query_stats["total_queries"] += 1
                query_stats["query_times"].append(execution_time)
                if len(query_stats["query_times"]) > 1000:
                    query_stats["query_times"] = query_stats["query_times"][-500:]


def init_db() -> None:
    """Initialize database tables with optimizations"""
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)
        
        # Create additional indexes for performance
        with engine.connect() as conn:
            # Composite indexes for common query patterns
            conn.execute("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_research_tasks_owner_status_created 
                ON research_tasks(owner_id, status, created_at DESC);
                
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email_active 
                ON users(email, is_active) WHERE is_active = true;
                
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_user_active 
                ON api_keys(user_id, is_active) WHERE is_active = true;
                
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_user_status_created 
                ON transactions(user_id, status, created_at DESC);
                
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_results_task_confidence 
                ON model_results(task_id, confidence DESC);
                
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_research_sources_task_relevance 
                ON research_sources(task_id, relevance_score DESC);
            """)
            
            # Create partial indexes for common filters
            conn.execute("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_research_tasks_pending 
                ON research_tasks(created_at DESC) WHERE status = 'pending';
                
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_subscriptions_active 
                ON subscriptions(user_id, current_period_end) WHERE status = 'active';
                
                -- Additional optimized indexes
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_stripe_customer 
                ON users(stripe_customer_id) WHERE stripe_customer_id IS NOT NULL;
                
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_created_date 
                ON transactions(date_trunc('day', created_at), user_id);
                
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_results_confidence_high 
                ON model_results(task_id, created_at DESC) WHERE confidence > 0.8;
                
                -- BRIN indexes for time-series data
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_research_tasks_created 
                ON research_tasks USING brin(created_at);
                
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_brin_transactions_created 
                ON transactions USING brin(created_at);
            """)
            
            # Add advanced database performance optimizations
            conn.execute("""
                -- Memory and parallel processing optimizations
                ALTER DATABASE nexus_forge SET work_mem = '512MB';
                ALTER DATABASE nexus_forge SET maintenance_work_mem = '1GB';
                ALTER DATABASE nexus_forge SET shared_buffers = '2GB';
                ALTER DATABASE nexus_forge SET effective_cache_size = '6GB';
                
                -- Parallel query optimizations
                ALTER DATABASE nexus_forge SET max_parallel_workers_per_gather = 8;
                ALTER DATABASE nexus_forge SET max_parallel_workers = 16;
                ALTER DATABASE nexus_forge SET max_parallel_maintenance_workers = 4;
                
                -- SSD and I/O optimizations
                ALTER DATABASE nexus_forge SET random_page_cost = 1.0;
                ALTER DATABASE nexus_forge SET effective_io_concurrency = 200;
                ALTER DATABASE nexus_forge SET wal_compression = on;
                
                -- Query planner optimizations
                ALTER DATABASE nexus_forge SET jit = on;
                ALTER DATABASE nexus_forge SET jit_above_cost = 100000;
                ALTER DATABASE nexus_forge SET jit_inline_above_cost = 500000;
                ALTER DATABASE nexus_forge SET jit_optimize_above_cost = 500000;
                
                -- Statistics and monitoring
                ALTER DATABASE nexus_forge SET track_io_timing = on;
                ALTER DATABASE nexus_forge SET track_functions = 'all';
                ALTER DATABASE nexus_forge SET log_min_duration_statement = 1000;
                
                -- Connection and lock optimizations
                ALTER DATABASE nexus_forge SET lock_timeout = '5s';
                ALTER DATABASE nexus_forge SET idle_in_transaction_session_timeout = '30s';
            """)
            
            # Create materialized views for common aggregations
            conn.execute("""
                -- User activity summary
                CREATE MATERIALIZED VIEW IF NOT EXISTS mv_user_activity_summary AS
                SELECT 
                    u.id as user_id,
                    u.email,
                    COUNT(DISTINCT rt.id) as total_tasks,
                    COUNT(DISTINCT rt.id) FILTER (WHERE rt.status = 'completed') as completed_tasks,
                    COUNT(DISTINCT rt.id) FILTER (WHERE rt.created_at > NOW() - INTERVAL '7 days') as recent_tasks,
                    AVG(ra.processing_time) as avg_processing_time,
                    MAX(rt.created_at) as last_activity
                FROM users u
                LEFT JOIN research_tasks rt ON u.id = rt.owner_id
                LEFT JOIN research_analytics ra ON rt.id = ra.task_id
                GROUP BY u.id, u.email;
                
                CREATE UNIQUE INDEX ON mv_user_activity_summary(user_id);
                
                -- Model performance summary
                CREATE MATERIALIZED VIEW IF NOT EXISTS mv_model_performance AS
                SELECT 
                    model_type,
                    COUNT(*) as usage_count,
                    AVG(confidence) as avg_confidence,
                    AVG(processing_time) as avg_processing_time,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY processing_time) as median_processing_time,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY processing_time) as p95_processing_time
                FROM model_results
                WHERE created_at > NOW() - INTERVAL '30 days'
                GROUP BY model_type;
                
                CREATE UNIQUE INDEX ON mv_model_performance(model_type);
            """)
            
        # Schedule automatic refresh of materialized views
        conn.execute("""
            -- Create refresh function
            CREATE OR REPLACE FUNCTION refresh_materialized_views() RETURNS void AS $$
            BEGIN
                REFRESH MATERIALIZED VIEW CONCURRENTLY mv_user_activity_summary;
                REFRESH MATERIALIZED VIEW CONCURRENTLY mv_model_performance;
            END;
            $$ LANGUAGE plpgsql;
        """)
        
        logger.info("Database tables and advanced optimizations created successfully")
        logger.info(f"Connection pool size: {pool_size}, Max overflow: {max_overflow}")
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
        Perform comprehensive database health check with advanced metrics.
        Returns dict with status and detailed metrics.
        """
        try:
            async with get_async_db() as db:
                # Check basic connectivity
                start_time = time.time()
                await db.execute("SELECT 1")
                ping_time = (time.time() - start_time) * 1000

                # Get connection pool stats
                pool_status = {
                    "pool_size": engine.pool.size(),
                    "checkedin": engine.pool.checkedin(),
                    "checkedout": engine.pool.checkedout(),
                    "overflow": engine.pool.overflow(),
                    "total": engine.pool.checkedin() + engine.pool.checkedout(),
                }
                
                # Get query performance stats
                with stats_lock:
                    avg_query_time = sum(query_stats["query_times"]) / len(query_stats["query_times"]) if query_stats["query_times"] else 0
                    avg_wait_time = sum(query_stats["connection_wait_times"]) / len(query_stats["connection_wait_times"]) if query_stats["connection_wait_times"] else 0
                    
                    performance_stats = {
                        "total_queries": query_stats["total_queries"],
                        "slow_queries": query_stats["slow_queries"],
                        "avg_query_time_ms": avg_query_time * 1000,
                        "avg_connection_wait_ms": avg_wait_time * 1000,
                        "cache_hit_rate": (query_stats["cache_hits"] / query_stats["total_queries"] * 100) if query_stats["total_queries"] > 0 else 0,
                    }
                
                # Check database size and table statistics
                db_stats = await db.execute("""
                    SELECT 
                        pg_database_size(current_database()) as db_size,
                        (SELECT count(*) FROM pg_stat_user_tables) as table_count,
                        (SELECT sum(n_tup_ins + n_tup_upd + n_tup_del) FROM pg_stat_user_tables) as total_modifications
                """)
                db_info = db_stats.first()

                return {
                    "status": "healthy",
                    "ping_ms": ping_time,
                    "pool_metrics": pool_status,
                    "performance_metrics": performance_stats,
                    "database_info": {
                        "size_mb": db_info.db_size / (1024 * 1024) if db_info else 0,
                        "table_count": db_info.table_count if db_info else 0,
                        "total_modifications": db_info.total_modifications if db_info else 0,
                    },
                    "pool_monitor": pool_monitor,
                    "message": "Database connection healthy with optimized performance",
                }
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Database connection failed",
            }
