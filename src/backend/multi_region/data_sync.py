"""
Data Synchronizer
Handles cross-region data replication and synchronization
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from ..core.cache import RedisCache
from ..core.exceptions import NotFoundError, ValidationError
from ..integrations.supabase.coordination_client import SupabaseCoordinationClient
from .models import DataSyncConfig
from .region_manager import RegionManager

logger = logging.getLogger(__name__)


class SyncStatus(str, Enum):
    """Data sync status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class ConflictResolution(str, Enum):
    """Conflict resolution strategies"""

    TIMESTAMP = "timestamp"  # Use latest timestamp
    SOURCE_WINS = "source_wins"  # Source region always wins
    TARGET_WINS = "target_wins"  # Target region always wins
    MANUAL = "manual"  # Require manual resolution
    MERGE = "merge"  # Attempt to merge changes


@dataclass
class SyncOperation:
    """Individual sync operation"""

    id: str
    sync_config_id: str
    source_region_id: str
    target_region_id: str
    table_name: str
    operation_type: str  # insert, update, delete
    record_id: str
    data: Dict[str, Any]
    timestamp: datetime
    status: SyncStatus
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class ConflictRecord:
    """Record with sync conflict"""

    id: str
    table_name: str
    record_id: str
    source_data: Dict[str, Any]
    target_data: Dict[str, Any]
    source_timestamp: datetime
    target_timestamp: datetime
    conflict_type: str
    resolution_strategy: ConflictResolution
    resolved: bool = False
    resolved_data: Optional[Dict[str, Any]] = None


class DataSynchronizer:
    """Manages data synchronization across regions"""

    def __init__(self, region_manager: Optional[RegionManager] = None):
        self.region_manager = region_manager or RegionManager()
        self.cache = RedisCache()
        self.supabase = SupabaseCoordinationClient()

        # Sync configuration
        self.max_batch_size = 1000
        self.max_retry_attempts = 3
        self.sync_timeout_seconds = 300  # 5 minutes
        self.conflict_retention_days = 30

        # Background tasks
        self._sync_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        self._coordinator_task = None

    async def start(self):
        """Start data synchronization service"""
        if self._running:
            return

        self._running = True
        self._coordinator_task = asyncio.create_task(self._sync_coordinator())

        logger.info("Data synchronizer started")

    async def stop(self):
        """Stop data synchronization service"""
        self._running = False

        # Cancel coordinator task
        if self._coordinator_task:
            self._coordinator_task.cancel()

        # Cancel all sync tasks
        for task in list(self._sync_tasks.values()):
            task.cancel()

        self._sync_tasks.clear()

        logger.info("Data synchronizer stopped")

    # Sync Configuration Management

    async def create_sync_config(self, config: DataSyncConfig) -> DataSyncConfig:
        """Create new sync configuration"""
        logger.info(f"Creating sync config: {config.name}")

        # Validate configuration
        await self._validate_sync_config(config)

        # Save to database
        await self.supabase.client.table("data_sync_configs").insert(
            config.dict()
        ).execute()

        # Start sync task if enabled
        await self._start_sync_task(config)

        logger.info(f"Created sync config: {config.id}")
        return config

    async def update_sync_config(self, config: DataSyncConfig) -> DataSyncConfig:
        """Update sync configuration"""
        # Validate configuration
        await self._validate_sync_config(config)

        # Stop existing sync task
        await self._stop_sync_task(config.id)

        # Update in database
        await self.supabase.client.table("data_sync_configs").update(config.dict()).eq(
            "id", config.id
        ).execute()

        # Restart sync task
        await self._start_sync_task(config)

        logger.info(f"Updated sync config: {config.id}")
        return config

    async def delete_sync_config(self, config_id: str) -> bool:
        """Delete sync configuration"""
        # Stop sync task
        await self._stop_sync_task(config_id)

        # Delete from database
        await self.supabase.client.table("data_sync_configs").delete().eq(
            "id", config_id
        ).execute()

        logger.info(f"Deleted sync config: {config_id}")
        return True

    async def list_sync_configs(self) -> List[DataSyncConfig]:
        """List all sync configurations"""
        result = (
            await self.supabase.client.table("data_sync_configs").select("*").execute()
        )

        return [DataSyncConfig(**config) for config in result.data]

    # Data Synchronization

    async def trigger_manual_sync(
        self, config_id: str, table_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Trigger manual synchronization"""
        config = await self._get_sync_config(config_id)

        if not config:
            raise NotFoundError(f"Sync config {config_id} not found")

        logger.info(f"Triggering manual sync for config: {config.name}")

        sync_result = {
            "config_id": config_id,
            "started_at": datetime.utcnow().isoformat(),
            "table_filter": table_name,
            "status": SyncStatus.RUNNING,
            "operations_processed": 0,
            "errors": [],
        }

        try:
            operations = await self._collect_sync_operations(config, table_name)
            sync_result["total_operations"] = len(operations)

            if operations:
                results = await self._process_sync_operations(config, operations)
                sync_result.update(results)

            sync_result["status"] = SyncStatus.COMPLETED
            sync_result["completed_at"] = datetime.utcnow().isoformat()

        except Exception as e:
            logger.error(f"Manual sync failed for config {config_id}: {e}")
            sync_result["status"] = SyncStatus.FAILED
            sync_result["error"] = str(e)
            sync_result["failed_at"] = datetime.utcnow().isoformat()

        # Log sync result
        await self._log_sync_result(sync_result)

        return sync_result

    async def get_sync_status(self, config_id: str) -> Dict[str, Any]:
        """Get current sync status for configuration"""
        config = await self._get_sync_config(config_id)

        if not config:
            raise NotFoundError(f"Sync config {config_id} not found")

        # Get recent sync operations
        recent_ops = await self._get_recent_sync_operations(config_id, hours=24)

        # Calculate metrics
        total_ops = len(recent_ops)
        successful_ops = len(
            [op for op in recent_ops if op.status == SyncStatus.COMPLETED]
        )
        failed_ops = len([op for op in recent_ops if op.status == SyncStatus.FAILED])

        # Get current lag
        lag_seconds = await self._calculate_sync_lag(config)

        # Check if sync task is running
        task_running = (
            config_id in self._sync_tasks and not self._sync_tasks[config_id].done()
        )

        return {
            "config_id": config_id,
            "config_name": config.name,
            "replication_mode": config.replication_mode.value,
            "source_region": config.source_region_id,
            "target_regions": config.target_region_ids,
            "task_running": task_running,
            "last_sync": config.last_sync.isoformat() if config.last_sync else None,
            "sync_lag_seconds": lag_seconds,
            "recent_operations": {
                "total": total_ops,
                "successful": successful_ops,
                "failed": failed_ops,
                "success_rate": (
                    (successful_ops / total_ops * 100) if total_ops > 0 else 0
                ),
            },
            "health_status": (
                "healthy" if failed_ops == 0 and lag_seconds < 600 else "warning"
            ),
        }

    # Conflict Resolution

    async def get_conflicts(
        self, resolved: Optional[bool] = None, table_name: Optional[str] = None
    ) -> List[ConflictRecord]:
        """Get sync conflicts"""
        query = self.supabase.client.table("sync_conflicts").select("*")

        if resolved is not None:
            query = query.eq("resolved", resolved)

        if table_name:
            query = query.eq("table_name", table_name)

        result = await query.order("created_at", desc=True).execute()

        conflicts = []
        for conflict_data in result.data:
            conflicts.append(
                ConflictRecord(
                    id=conflict_data["id"],
                    table_name=conflict_data["table_name"],
                    record_id=conflict_data["record_id"],
                    source_data=conflict_data["source_data"],
                    target_data=conflict_data["target_data"],
                    source_timestamp=datetime.fromisoformat(
                        conflict_data["source_timestamp"]
                    ),
                    target_timestamp=datetime.fromisoformat(
                        conflict_data["target_timestamp"]
                    ),
                    conflict_type=conflict_data["conflict_type"],
                    resolution_strategy=ConflictResolution(
                        conflict_data["resolution_strategy"]
                    ),
                    resolved=conflict_data["resolved"],
                    resolved_data=conflict_data.get("resolved_data"),
                )
            )

        return conflicts

    async def resolve_conflict(
        self,
        conflict_id: str,
        resolution_data: Dict[str, Any],
        strategy: ConflictResolution = ConflictResolution.MANUAL,
    ) -> bool:
        """Resolve a sync conflict"""
        logger.info(f"Resolving conflict: {conflict_id}")

        # Get conflict record
        result = (
            await self.supabase.client.table("sync_conflicts")
            .select("*")
            .eq("id", conflict_id)
            .execute()
        )

        if not result.data:
            raise NotFoundError(f"Conflict {conflict_id} not found")

        conflict_data = result.data[0]

        try:
            # Apply resolution
            if strategy == ConflictResolution.MANUAL:
                resolved_data = resolution_data
            elif strategy == ConflictResolution.TIMESTAMP:
                # Use data with latest timestamp
                source_ts = datetime.fromisoformat(conflict_data["source_timestamp"])
                target_ts = datetime.fromisoformat(conflict_data["target_timestamp"])
                resolved_data = (
                    conflict_data["source_data"]
                    if source_ts > target_ts
                    else conflict_data["target_data"]
                )
            elif strategy == ConflictResolution.SOURCE_WINS:
                resolved_data = conflict_data["source_data"]
            elif strategy == ConflictResolution.TARGET_WINS:
                resolved_data = conflict_data["target_data"]
            else:
                resolved_data = self._merge_conflict_data(
                    conflict_data["source_data"], conflict_data["target_data"]
                )

            # Update target regions with resolved data
            await self._apply_conflict_resolution(conflict_data, resolved_data)

            # Mark conflict as resolved
            await self.supabase.client.table("sync_conflicts").update(
                {
                    "resolved": True,
                    "resolved_data": resolved_data,
                    "resolved_at": datetime.utcnow().isoformat(),
                    "resolution_strategy": strategy.value,
                }
            ).eq("id", conflict_id).execute()

            logger.info(f"Resolved conflict: {conflict_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to resolve conflict {conflict_id}: {e}")
            return False

    # Monitoring and Analytics

    async def get_sync_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get synchronization metrics"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        # Get sync operations in time window
        result = (
            await self.supabase.client.table("sync_operations")
            .select("*")
            .gte("timestamp", start_time.isoformat())
            .lte("timestamp", end_time.isoformat())
            .execute()
        )

        operations = result.data

        # Calculate metrics
        metrics = {
            "time_window_hours": hours,
            "total_operations": len(operations),
            "operations_by_status": {},
            "operations_by_table": {},
            "operations_by_region": {},
            "avg_sync_latency_ms": 0,
            "error_rate_percent": 0,
            "data_transferred_mb": 0,
        }

        # Group by status
        for status in SyncStatus:
            metrics["operations_by_status"][status.value] = len(
                [op for op in operations if op.get("status") == status.value]
            )

        # Group by table
        for op in operations:
            table = op.get("table_name", "unknown")
            metrics["operations_by_table"][table] = (
                metrics["operations_by_table"].get(table, 0) + 1
            )

        # Group by region
        for op in operations:
            region = op.get("target_region_id", "unknown")
            metrics["operations_by_region"][region] = (
                metrics["operations_by_region"].get(region, 0) + 1
            )

        # Calculate error rate
        failed_ops = metrics["operations_by_status"].get("failed", 0)
        if metrics["total_operations"] > 0:
            metrics["error_rate_percent"] = (
                failed_ops / metrics["total_operations"]
            ) * 100

        # Get conflict metrics
        conflict_result = (
            await self.supabase.client.table("sync_conflicts")
            .select("*")
            .gte("created_at", start_time.isoformat())
            .execute()
        )

        conflicts = conflict_result.data
        metrics["conflicts"] = {
            "total": len(conflicts),
            "resolved": len([c for c in conflicts if c.get("resolved", False)]),
            "pending": len([c for c in conflicts if not c.get("resolved", False)]),
        }

        return metrics

    # Private helper methods

    async def _validate_sync_config(self, config: DataSyncConfig) -> None:
        """Validate sync configuration"""
        # Check source region exists
        try:
            await self.region_manager.get_region(config.source_region_id)
        except NotFoundError:
            raise ValidationError(f"Source region {config.source_region_id} not found")

        # Check target regions exist
        for target_id in config.target_region_ids:
            try:
                await self.region_manager.get_region(target_id)
            except NotFoundError:
                raise ValidationError(f"Target region {target_id} not found")

        # Validate sync interval
        if config.sync_interval_seconds < 60:
            raise ValidationError("Sync interval must be at least 60 seconds")

        # Check for circular dependencies
        if config.source_region_id in config.target_region_ids:
            raise ValidationError("Source region cannot be in target regions list")

    async def _get_sync_config(self, config_id: str) -> Optional[DataSyncConfig]:
        """Get sync configuration by ID"""
        result = (
            await self.supabase.client.table("data_sync_configs")
            .select("*")
            .eq("id", config_id)
            .execute()
        )

        if not result.data:
            return None

        return DataSyncConfig(**result.data[0])

    async def _sync_coordinator(self):
        """Background task coordinator for sync operations"""
        while self._running:
            try:
                # Get all active sync configurations
                configs = await self.list_sync_configs()

                for config in configs:
                    # Check if sync task is running
                    if (
                        config.id not in self._sync_tasks
                        or self._sync_tasks[config.id].done()
                    ):
                        # Start sync task
                        await self._start_sync_task(config)

                # Clean up completed tasks
                completed_tasks = [
                    config_id
                    for config_id, task in self._sync_tasks.items()
                    if task.done()
                ]

                for config_id in completed_tasks:
                    del self._sync_tasks[config_id]

                # Wait before next coordination cycle
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Sync coordinator error: {e}")
                await asyncio.sleep(10)  # Short delay on error

    async def _start_sync_task(self, config: DataSyncConfig):
        """Start sync task for configuration"""
        if config.id in self._sync_tasks and not self._sync_tasks[config.id].done():
            return  # Task already running

        task = asyncio.create_task(self._sync_task_loop(config))
        self._sync_tasks[config.id] = task

        logger.info(f"Started sync task for config: {config.name}")

    async def _stop_sync_task(self, config_id: str):
        """Stop sync task for configuration"""
        if config_id in self._sync_tasks:
            self._sync_tasks[config_id].cancel()
            del self._sync_tasks[config_id]
            logger.info(f"Stopped sync task for config: {config_id}")

    async def _sync_task_loop(self, config: DataSyncConfig):
        """Main sync task loop"""
        while self._running:
            try:
                # Collect pending sync operations
                operations = await self._collect_sync_operations(config)

                if operations:
                    # Process sync operations
                    await self._process_sync_operations(config, operations)

                    # Update last sync time
                    config.last_sync = datetime.utcnow()
                    await self.supabase.client.table("data_sync_configs").update(
                        {"last_sync": config.last_sync.isoformat()}
                    ).eq("id", config.id).execute()

                # Wait for next sync interval
                await asyncio.sleep(config.sync_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync task error for config {config.id}: {e}")
                await asyncio.sleep(30)  # Wait before retry

    async def _collect_sync_operations(
        self, config: DataSyncConfig, table_filter: Optional[str] = None
    ) -> List[SyncOperation]:
        """Collect pending sync operations"""
        operations = []

        # Implementation would collect change logs from source region
        # For now, return empty list

        return operations

    async def _process_sync_operations(
        self, config: DataSyncConfig, operations: List[SyncOperation]
    ) -> Dict[str, Any]:
        """Process sync operations"""
        result = {
            "operations_processed": 0,
            "operations_succeeded": 0,
            "operations_failed": 0,
            "conflicts_detected": 0,
            "errors": [],
        }

        # Process operations in batches
        for i in range(0, len(operations), config.batch_size):
            batch = operations[i : i + config.batch_size]

            for operation in batch:
                try:
                    success = await self._apply_sync_operation(operation)

                    if success:
                        result["operations_succeeded"] += 1
                    else:
                        result["operations_failed"] += 1

                except Exception as e:
                    result["operations_failed"] += 1
                    result["errors"].append(str(e))

                result["operations_processed"] += 1

        return result

    async def _apply_sync_operation(self, operation: SyncOperation) -> bool:
        """Apply single sync operation"""
        # Implementation would apply operation to target region
        # For now, return True
        return True

    async def _calculate_sync_lag(self, config: DataSyncConfig) -> int:
        """Calculate sync lag in seconds"""
        if not config.last_sync:
            return 0

        lag = datetime.utcnow() - config.last_sync
        return int(lag.total_seconds())

    async def _get_recent_sync_operations(
        self, config_id: str, hours: int = 24
    ) -> List[SyncOperation]:
        """Get recent sync operations"""
        # Implementation would query sync operations
        # For now, return empty list
        return []

    def _merge_conflict_data(
        self, source_data: Dict[str, Any], target_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge conflicting data records"""
        # Simple merge strategy - can be made more sophisticated
        merged = target_data.copy()

        for key, value in source_data.items():
            if key not in merged or merged[key] != value:
                # Prefer non-null values
                if value is not None and merged.get(key) is None:
                    merged[key] = value

        return merged

    async def _apply_conflict_resolution(
        self, conflict_data: Dict[str, Any], resolved_data: Dict[str, Any]
    ) -> None:
        """Apply conflict resolution to target regions"""
        # Implementation would update target regions with resolved data
        pass

    async def _log_sync_result(self, sync_result: Dict[str, Any]) -> None:
        """Log sync result for monitoring"""
        try:
            await self.supabase.client.table("sync_results").insert(
                sync_result
            ).execute()
        except Exception as e:
            logger.error(f"Failed to log sync result: {e}")
