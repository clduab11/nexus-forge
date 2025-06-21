"""
Advanced Performance Monitoring for Nexus Forge Multi-Agent System
Real-time metrics, observability, and performance optimization
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import aioredis
import GPUtil
import psutil
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from nexus_forge.core.monitoring import get_logger
from nexus_forge.integrations.supabase.coordination_client import (
    SupabaseCoordinationClient,
)

logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""

    metric_type: str
    metric_name: str
    value: float
    unit: str
    labels: Dict[str, str]
    timestamp: datetime
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    workflow_id: Optional[str] = None


@dataclass
class SystemResources:
    """System resource utilization"""

    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    gpu_utilization: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0


@dataclass
class AgentPerformance:
    """Agent-specific performance metrics"""

    agent_id: str
    agent_type: str
    status: str
    tasks_completed: int
    tasks_failed: int
    avg_execution_time: float
    last_execution_time: float
    memory_usage_mb: float
    cpu_utilization: float
    error_rate: float
    throughput_per_hour: float


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for Nexus Forge
    Tracks system resources, agent performance, and workflow metrics
    """

    def __init__(
        self,
        coordination_client: SupabaseCoordinationClient,
        redis_client: aioredis.Redis,
        metrics_port: int = 9090,
        collection_interval: float = 5.0,
        retention_hours: int = 24,
    ):
        """Initialize performance monitor"""
        self.coordination_client = coordination_client
        self.redis_client = redis_client
        self.metrics_port = metrics_port
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours

        # Monitoring state
        self.monitoring_active = False
        self.collection_task: Optional[asyncio.Task] = None

        # Metrics storage
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.agent_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.workflow_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Performance targets (SLAs)
        self.performance_targets = {
            "text_generation_ms": 2000,  # <2s for text
            "image_generation_ms": 5000,  # <5s for images
            "video_generation_ms": 30000,  # <30s for videos
            "api_response_ms": 200,  # <200ms for API responses
            "workflow_completion_minutes": 5,  # <5min for workflows
            "system_cpu_percent": 80,  # <80% CPU usage
            "system_memory_percent": 85,  # <85% memory usage
            "error_rate_percent": 1,  # <1% error rate
        }

        # Prometheus metrics
        self._setup_prometheus_metrics()

        # Alert thresholds
        self.alert_handlers: List[Callable] = []

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for monitoring"""
        # Counters
        self.task_counter = Counter(
            "nexus_forge_tasks_total",
            "Total number of tasks processed",
            ["agent_type", "agent_id", "status"],
        )

        self.api_requests = Counter(
            "nexus_forge_api_requests_total",
            "Total API requests",
            ["method", "endpoint", "status_code"],
        )

        # Histograms
        self.task_duration = Histogram(
            "nexus_forge_task_duration_seconds",
            "Task execution duration",
            ["agent_type", "task_type"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0],
        )

        self.api_latency = Histogram(
            "nexus_forge_api_latency_seconds",
            "API request latency",
            ["method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        )

        # Gauges
        self.active_agents = Gauge(
            "nexus_forge_active_agents", "Number of active agents", ["agent_type"]
        )

        self.system_cpu = Gauge("nexus_forge_system_cpu_percent", "System CPU usage")
        self.system_memory = Gauge(
            "nexus_forge_system_memory_percent", "System memory usage"
        )
        self.system_gpu = Gauge("nexus_forge_system_gpu_percent", "System GPU usage")

        self.queue_size = Gauge(
            "nexus_forge_queue_size", "Task queue size", ["status", "priority"]
        )

        self.workflow_status = Gauge(
            "nexus_forge_workflows", "Workflow counts by status", ["status"]
        )

    async def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return

        try:
            # Start Prometheus metrics server
            start_http_server(self.metrics_port)
            logger.info(
                f"Prometheus metrics server started on port {self.metrics_port}"
            )

            # Start collection task
            self.monitoring_active = True
            self.collection_task = asyncio.create_task(self._collection_loop())

            logger.info("Performance monitoring started")

        except Exception as e:
            logger.error(f"Failed to start performance monitoring: {e}")
            raise

    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False

        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass

        logger.info("Performance monitoring stopped")

    async def _collection_loop(self):
        """Main metrics collection loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                await self._collect_system_metrics()

                # Collect agent metrics
                await self._collect_agent_metrics()

                # Collect workflow metrics
                await self._collect_workflow_metrics()

                # Collect queue metrics
                await self._collect_queue_metrics()

                # Check performance targets and alerts
                await self._check_performance_targets()

                # Cleanup old metrics
                await self._cleanup_old_metrics()

                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.collection_interval)

    async def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            network = psutil.net_io_counters()

            # GPU metrics (if available)
            gpu_utilization = None
            gpu_memory_percent = None

            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_utilization = gpu.load * 100
                    gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
            except:
                pass  # GPU metrics not available

            resources = SystemResources(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_usage_percent=disk.percent,
                gpu_utilization=gpu_utilization,
                gpu_memory_percent=gpu_memory_percent,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
            )

            # Update Prometheus metrics
            self.system_cpu.set(cpu_percent)
            self.system_memory.set(memory.percent)
            if gpu_utilization is not None:
                self.system_gpu.set(gpu_utilization)

            # Store in Supabase
            await self.coordination_client.record_metric(
                metric_type="system_resources",
                metric_value=cpu_percent,
                metric_unit="percent",
                additional_data=asdict(resources),
            )

            # Cache in Redis
            await self.redis_client.setex(
                "system:resources:latest",
                60,  # 1 minute TTL
                asdict(resources).__str__(),
            )

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

    async def _collect_agent_metrics(self):
        """Collect agent performance metrics"""
        try:
            # Query agent states from Supabase
            agents_data = await self._get_active_agents()

            for agent_data in agents_data:
                agent_id = agent_data["agent_id"]
                agent_type = agent_data["type"]

                # Calculate performance metrics
                performance = await self._calculate_agent_performance(agent_id)

                if performance:
                    # Update Prometheus metrics
                    self.active_agents.labels(agent_type=agent_type).set(1)

                    # Store metrics
                    await self.coordination_client.record_metric(
                        metric_type="agent_performance",
                        metric_value=performance.avg_execution_time,
                        metric_unit="ms",
                        agent_id=agent_id,
                        additional_data=asdict(performance),
                    )

                    self.agent_metrics[agent_id] = asdict(performance)

        except Exception as e:
            logger.error(f"Failed to collect agent metrics: {e}")

    async def _collect_workflow_metrics(self):
        """Collect workflow performance metrics"""
        try:
            # Query recent workflows
            workflows_data = await self._get_recent_workflows()

            status_counts = defaultdict(int)

            for workflow_data in workflows_data:
                workflow_id = workflow_data["workflow_id"]
                status = workflow_data["status"]

                status_counts[status] += 1

                # Calculate workflow metrics
                if status in ["completed", "failed"]:
                    duration = await self._calculate_workflow_duration(workflow_id)
                    if duration:
                        await self.coordination_client.record_metric(
                            metric_type="workflow_duration",
                            metric_value=duration,
                            metric_unit="seconds",
                            workflow_id=workflow_id,
                        )

            # Update Prometheus metrics
            for status, count in status_counts.items():
                self.workflow_status.labels(status=status).set(count)

        except Exception as e:
            logger.error(f"Failed to collect workflow metrics: {e}")

    async def _collect_queue_metrics(self):
        """Collect task queue metrics"""
        try:
            # Query task queue status
            queue_data = await self._get_queue_status()

            for status, count in queue_data.items():
                self.queue_size.labels(status=status, priority="normal").set(count)

        except Exception as e:
            logger.error(f"Failed to collect queue metrics: {e}")

    async def _check_performance_targets(self):
        """Check if performance targets are being met"""
        try:
            alerts = []

            # Check system resources
            system_metrics = await self.redis_client.get("system:resources:latest")
            if system_metrics:
                # Parse and check against targets
                # Implementation would check CPU, memory, etc. against targets
                pass

            # Check agent performance
            for agent_id, metrics in self.agent_metrics.items():
                if (
                    metrics.get("error_rate", 0)
                    > self.performance_targets["error_rate_percent"]
                ):
                    alerts.append(
                        {
                            "type": "error_rate_high",
                            "agent_id": agent_id,
                            "value": metrics["error_rate"],
                            "target": self.performance_targets["error_rate_percent"],
                        }
                    )

                if metrics.get("avg_execution_time", 0) > self.performance_targets.get(
                    f"{metrics.get('agent_type')}_generation_ms", 10000
                ):
                    alerts.append(
                        {
                            "type": "execution_time_high",
                            "agent_id": agent_id,
                            "value": metrics["avg_execution_time"],
                            "target": self.performance_targets.get(
                                f"{metrics.get('agent_type')}_generation_ms", 10000
                            ),
                        }
                    )

            # Trigger alerts
            for alert in alerts:
                await self._trigger_alert(alert)

        except Exception as e:
            logger.error(f"Failed to check performance targets: {e}")

    async def _cleanup_old_metrics(self):
        """Cleanup old metrics data"""
        try:
            cutoff_time = datetime.now(timezone.utc).timestamp() - (
                self.retention_hours * 3600
            )

            # Remove old entries from buffer
            while (
                self.metrics_buffer
                and self.metrics_buffer[0].timestamp.timestamp() < cutoff_time
            ):
                self.metrics_buffer.popleft()

        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")

    async def _get_active_agents(self) -> List[Dict[str, Any]]:
        """Get list of active agents"""
        try:
            # This would query the Supabase agents table
            # For now, return mock data
            return [
                {
                    "agent_id": "starri-orchestrator",
                    "type": "orchestrator",
                    "status": "online",
                },
                {"agent_id": "gemini-analysis", "type": "llm", "status": "online"},
                {"agent_id": "jules-coding", "type": "code", "status": "online"},
                {"agent_id": "imagen-generator", "type": "image", "status": "online"},
                {"agent_id": "veo-generator", "type": "video", "status": "online"},
            ]
        except Exception as e:
            logger.error(f"Failed to get active agents: {e}")
            return []

    async def _calculate_agent_performance(
        self, agent_id: str
    ) -> Optional[AgentPerformance]:
        """Calculate performance metrics for specific agent"""
        try:
            # This would analyze tasks, execution times, etc.
            # For now, return mock performance data
            return AgentPerformance(
                agent_id=agent_id,
                agent_type="llm",
                status="online",
                tasks_completed=150,
                tasks_failed=2,
                avg_execution_time=1500.0,  # ms
                last_execution_time=1200.0,  # ms
                memory_usage_mb=512.0,
                cpu_utilization=25.0,
                error_rate=1.3,  # percent
                throughput_per_hour=120.0,
            )
        except Exception as e:
            logger.error(f"Failed to calculate agent performance for {agent_id}: {e}")
            return None

    async def _get_recent_workflows(self) -> List[Dict[str, Any]]:
        """Get recent workflow data"""
        # Mock implementation
        return [
            {"workflow_id": "wf-1", "status": "completed"},
            {"workflow_id": "wf-2", "status": "running"},
            {"workflow_id": "wf-3", "status": "failed"},
        ]

    async def _calculate_workflow_duration(self, workflow_id: str) -> Optional[float]:
        """Calculate workflow execution duration"""
        # Mock implementation
        return 120.5  # seconds

    async def _get_queue_status(self) -> Dict[str, int]:
        """Get task queue status"""
        # Mock implementation
        return {"pending": 5, "running": 3, "completed": 25, "failed": 1}

    async def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger performance alert"""
        try:
            logger.warning(f"Performance alert: {alert}")

            # Call registered alert handlers
            for handler in self.alert_handlers:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")

        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")

    # Public API methods
    async def record_task_execution(
        self,
        agent_id: str,
        task_id: str,
        execution_time_ms: float,
        status: str,
        task_type: str = "unknown",
    ):
        """Record task execution metrics"""
        try:
            # Update Prometheus metrics
            agent_data = await self._get_agent_info(agent_id)
            agent_type = agent_data.get("type", "unknown") if agent_data else "unknown"

            self.task_counter.labels(
                agent_type=agent_type, agent_id=agent_id, status=status
            ).inc()

            self.task_duration.labels(
                agent_type=agent_type, task_type=task_type
            ).observe(
                execution_time_ms / 1000.0
            )  # Convert to seconds

            # Store in coordination system
            await self.coordination_client.record_metric(
                metric_type="task_execution",
                metric_value=execution_time_ms,
                metric_unit="ms",
                agent_id=agent_id,
                task_id=task_id,
                additional_data={"status": status, "task_type": task_type},
            )

        except Exception as e:
            logger.error(f"Failed to record task execution: {e}")

    async def record_api_request(
        self, method: str, endpoint: str, status_code: int, duration_ms: float
    ):
        """Record API request metrics"""
        try:
            # Update Prometheus metrics
            self.api_requests.labels(
                method=method, endpoint=endpoint, status_code=str(status_code)
            ).inc()

            self.api_latency.labels(method=method, endpoint=endpoint).observe(
                duration_ms / 1000.0
            )  # Convert to seconds

        except Exception as e:
            logger.error(f"Failed to record API request: {e}")

    def register_alert_handler(self, handler: Callable):
        """Register alert handler"""
        self.alert_handlers.append(handler)
        logger.info("Registered performance alert handler")

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            system_metrics = await self.redis_client.get("system:resources:latest")

            return {
                "system_resources": system_metrics,
                "active_agents": len(self.agent_metrics),
                "agent_metrics": dict(self.agent_metrics),
                "workflow_metrics": dict(self.workflow_metrics),
                "performance_targets": self.performance_targets,
                "monitoring_active": self.monitoring_active,
                "collection_interval": self.collection_interval,
                "metrics_port": self.metrics_port,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}

    async def _get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information"""
        # This would query agent data from Supabase
        # Mock implementation for now
        agent_types = {
            "starri-orchestrator": "orchestrator",
            "gemini-analysis": "llm",
            "jules-coding": "code",
            "imagen-generator": "image",
            "veo-generator": "video",
        }

        return {"type": agent_types.get(agent_id, "unknown")}
