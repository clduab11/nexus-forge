"""
Database & Memory Optimization Implementation Code
Ready-to-use code snippets for integrating optimizations
"""

# ============================================
# 1. OPTIMIZED DATABASE QUERIES
# ============================================

from backend.core.db_optimization import query_builder, get_optimized_user, bulk_fetch_users
from backend.core.integrated_cache_optimizer import cached_query, invalidate_cache_on_update
from backend.core.memory_optimizer import memory_managed

# Example 1: Fetching user with all relationships (cached)
@cached_query(data_type="user_data", ttl=300)
async def get_user_with_full_data(db_session, user_id: int):
    """Get user with optimized eager loading and caching"""
    return await query_builder.get_user_with_relationships(db_session, user_id)


# Example 2: Bulk operations with memory management
@memory_managed(pool_name="query_results")
async def bulk_update_task_statuses(db_session, updates: list):
    """Bulk update with memory pooling"""
    return await query_builder.bulk_update_task_statuses(db_session, updates)


# Example 3: Parallel analytics queries
async def get_dashboard_analytics(db_session, user_ids: list):
    """Fetch analytics for multiple users in parallel"""
    # Uses parallel execution and caching
    analytics = await query_builder.parallel_fetch_user_analytics(db_session, user_ids)
    
    # Also queries materialized view for instant results
    mv_query = """
    SELECT * FROM mv_user_activity_summary 
    WHERE user_id = ANY(:user_ids)
    """
    mv_results = await db_session.execute(mv_query, {"user_ids": user_ids})
    
    return {
        "detailed_analytics": analytics,
        "summary_stats": mv_results.fetchall()
    }


# ============================================
# 2. CACHE MANAGEMENT
# ============================================

# Example 4: Research task with automatic cache invalidation
class ResearchTaskService:
    @cached_query(data_type="research_task", ttl=600)
    async def get_task(self, db_session, task_id: int):
        """Get research task with all related data"""
        return await query_builder.get_research_task_with_all_data(db_session, task_id)
    
    @invalidate_cache_on_update("research_tasks")
    async def update_task(self, db_session, task_id: int, updates: dict):
        """Update task and automatically invalidate related caches"""
        task = await db_session.get(ResearchTask, task_id)
        for key, value in updates.items():
            setattr(task, key, value)
        await db_session.commit()
        return task


# Example 5: Manual cache management for complex scenarios
from backend.core.integrated_cache_optimizer import integrated_cache

async def complex_aggregation_query(db_session, filters: dict):
    """Complex query with manual cache control"""
    cache_key = f"complex_agg:{hash(frozenset(filters.items()))}"
    
    # Try cache first
    result = await integrated_cache.get_cached_query(
        cache_key, 
        data_type="analytics",
        fallback=lambda: _execute_complex_query(db_session, filters)
    )
    
    return result


# ============================================
# 3. MEMORY OPTIMIZATION
# ============================================

# Example 6: AI response processing with memory management
@memory_managed(pool_name="ai_responses")
async def process_ai_response(response_data: dict):
    """Process large AI response with automatic memory pooling"""
    # Large object automatically tracked and pooled
    processed_data = {
        "summary": extract_summary(response_data),
        "entities": extract_entities(response_data),
        "sentiment": analyze_sentiment(response_data),
        "embeddings": generate_embeddings(response_data)
    }
    
    # Register for tracking if very large
    if sys.getsizeof(processed_data) > 10 * 1024 * 1024:  # 10MB
        memory_optimizer.register_large_object("ai_response", processed_data)
    
    return processed_data


# Example 7: Batch processing with memory monitoring
async def batch_process_research_sources(sources: list):
    """Process sources in batches with memory optimization"""
    from backend.core.memory_optimizer import memory_optimizer
    
    batch_size = 100
    results = []
    
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i + batch_size]
        
        # Check memory before processing
        memory_stats = await memory_optimizer.collect_memory_metrics()
        if memory_stats.process_memory_mb > 1024:  # 1GB threshold
            await memory_optimizer.optimize_memory()
        
        # Process batch
        batch_results = await process_source_batch(batch)
        results.extend(batch_results)
        
        # Release batch from memory
        del batch
        gc.collect()
    
    return results


# ============================================
# 4. MONITORING & HEALTH CHECKS
# ============================================

# Example 8: Comprehensive health check endpoint
async def get_system_health():
    """Get complete system health including DB, cache, and memory"""
    from backend.database import DatabaseHealthCheck
    from backend.core.integrated_cache_optimizer import integrated_cache
    from backend.core.memory_optimizer import memory_optimizer
    
    # Parallel health checks
    db_health, cache_health, memory_health = await asyncio.gather(
        DatabaseHealthCheck.check_health(),
        integrated_cache.get_cache_report(),
        asyncio.to_thread(memory_optimizer.get_memory_report)
    )
    
    return {
        "database": db_health,
        "cache": cache_health,
        "memory": memory_health,
        "overall_status": _calculate_overall_status(db_health, cache_health, memory_health)
    }


# Example 9: Performance monitoring decorator
def monitor_performance(operation_name: str):
    """Monitor and log performance of operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                result = await func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                memory_delta = psutil.Process().memory_info().rss / 1024 / 1024 - initial_memory
                
                # Log to query stats
                with stats_lock:
                    query_stats["query_patterns"][operation_name]["count"] += 1
                    query_stats["query_patterns"][operation_name]["total_time"] += execution_time
                
                # Alert if slow or memory-intensive
                if execution_time > 1.0:
                    logger.warning(f"{operation_name} slow: {execution_time:.2f}s")
                if memory_delta > 50:
                    logger.warning(f"{operation_name} memory spike: {memory_delta:.2f}MB")
                
                return result
                
            except Exception as e:
                logger.error(f"{operation_name} failed: {e}")
                raise
        
        return wrapper
    return decorator


# ============================================
# 5. USAGE IN API ENDPOINTS
# ============================================

from fastapi import Depends, APIRouter
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()

@router.get("/users/{user_id}")
@monitor_performance("get_user_endpoint")
async def get_user_endpoint(
    user_id: int,
    db: AsyncSession = Depends(get_async_db)
):
    """Optimized user endpoint with all enhancements"""
    # Automatically uses:
    # - Connection pooling with monitoring
    # - Multi-level caching
    # - Eager loading
    # - Memory management
    user = await get_user_with_full_data(db, user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user


@router.post("/research-tasks/bulk-update")
@monitor_performance("bulk_update_endpoint")
async def bulk_update_endpoint(
    updates: List[TaskUpdate],
    db: AsyncSession = Depends(get_async_db)
):
    """Bulk update with optimization"""
    # Convert to format expected by optimizer
    update_list = [
        {"task_id": u.task_id, "status": u.status}
        for u in updates
    ]
    
    # Execute with all optimizations
    updated_count = await bulk_update_task_statuses(db, update_list)
    
    return {
        "updated": updated_count,
        "message": f"Successfully updated {updated_count} tasks"
    }


# ============================================
# 6. STARTUP CONFIGURATION
# ============================================

async def configure_optimizations():
    """Configure all optimizations on startup"""
    from backend.database import init_db
    from backend.core.memory_optimizer import memory_optimizer
    from backend.core.integrated_cache_optimizer import integrated_cache
    
    # Initialize database with optimizations
    init_db()
    
    # Start memory monitoring
    await memory_optimizer.start_monitoring()
    
    # Warm up caches with common data
    warm_data = {
        "popular_python_patterns": load_code_patterns(),
        "common_queries": load_common_queries(),
        "ui_templates": load_ui_templates()
    }
    integrated_cache.redis_cache.warm_cache(warm_data)
    
    # Schedule periodic optimization
    asyncio.create_task(periodic_optimization_task())
    
    logger.info("All optimizations configured and running")


async def periodic_optimization_task():
    """Run periodic optimization tasks"""
    while True:
        await asyncio.sleep(3600)  # Every hour
        
        try:
            # Refresh materialized views
            async with get_async_db() as db:
                await db.execute("SELECT refresh_materialized_views()")
            
            # Optimize caches
            await integrated_cache.optimize_cache_performance()
            
            # Check and optimize memory if needed
            memory_report = memory_optimizer.get_memory_report()
            if memory_report["current_metrics"]["process_memory_mb"] > 1536:  # 1.5GB
                await memory_optimizer.optimize_memory()
            
        except Exception as e:
            logger.error(f"Periodic optimization failed: {e}")


# ============================================
# 7. TESTING & BENCHMARKING
# ============================================

async def benchmark_optimizations():
    """Benchmark the optimization improvements"""
    results = {
        "query_performance": {},
        "cache_performance": {},
        "memory_efficiency": {}
    }
    
    # Test 1: Query performance
    async with get_async_db() as db:
        # Without optimization (direct query)
        start = time.time()
        user = await db.get(User, 1)
        results["query_performance"]["without_optimization"] = time.time() - start
        
        # With optimization
        start = time.time()
        user = await get_user_with_full_data(db, 1)
        results["query_performance"]["with_optimization"] = time.time() - start
    
    # Test 2: Cache performance
    for i in range(100):
        await integrated_cache.get_cached_query(f"test_key_{i % 10}", "test")
    
    cache_stats = integrated_cache.cache_performance
    results["cache_performance"] = {
        "hit_rate": cache_stats["query_cache_hits"] / 
                   (cache_stats["query_cache_hits"] + cache_stats["query_cache_misses"]),
        "avg_retrieval_ms": cache_stats["avg_retrieval_time"]
    }
    
    # Test 3: Memory efficiency
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Simulate heavy load
    tasks = []
    for i in range(50):
        tasks.append(process_ai_response({"data": "x" * 100000}))
    
    await asyncio.gather(*tasks)
    
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    results["memory_efficiency"] = {
        "memory_increase_mb": final_memory - initial_memory,
        "pooling_stats": memory_optimizer.memory_pools["ai_responses"].get_stats()
    }
    
    return results


if __name__ == "__main__":
    # Run benchmarks
    asyncio.run(benchmark_optimizations())