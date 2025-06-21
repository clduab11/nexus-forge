"""
Google Cloud Monitoring integration for Parallax Pal

Provides comprehensive monitoring, tracing, and custom metrics
for the multi-agent research platform.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional

from google.api_core import exceptions
from google.cloud import logging as cloud_logging
from google.cloud import monitoring_v3, trace_v1
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Gauge, Histogram, Info

logger = logging.getLogger(__name__)


class CloudMonitoringService:
    """Google Cloud Monitoring service for metrics and monitoring"""

    def __init__(self, project_id: str = None):
        """
        Initialize Cloud Monitoring service

        Args:
            project_id: Google Cloud project ID
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("Google Cloud project ID required")

        # Initialize clients
        self.metrics_client = monitoring_v3.MetricServiceClient()
        self.trace_client = trace_v1.TraceServiceClient()

        # Project path
        self.project_path = f"projects/{self.project_id}"

        # Metric descriptors cache
        self._metric_descriptors = {}

        # Initialize OpenTelemetry tracing
        self._setup_tracing()

        # Initialize Prometheus metrics
        self._setup_prometheus_metrics()

        logger.info(f"Cloud Monitoring initialized for project: {self.project_id}")

    def _setup_tracing(self):
        """Setup OpenTelemetry tracing with Cloud Trace export"""

        # Create and configure tracer provider
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)

        # Configure Cloud Trace exporter
        cloud_trace_exporter = CloudTraceSpanExporter(project_id=self.project_id)

        # Add span processor
        tracer_provider.add_span_processor(BatchSpanProcessor(cloud_trace_exporter))

        # Instrument libraries
        FastAPIInstrumentor.instrument()
        RequestsInstrumentor.instrument()
        SQLAlchemyInstrumentor.instrument()

        # Get tracer
        self.tracer = trace.get_tracer(__name__)

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for local collection"""

        # Request metrics
        self.request_count = Counter(
            "parallaxpal_requests_total",
            "Total number of requests",
            ["method", "endpoint", "status"],
        )

        self.request_duration = Histogram(
            "parallaxpal_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
        )

        # Agent metrics
        self.agent_task_count = Counter(
            "parallaxpal_agent_tasks_total",
            "Total number of agent tasks",
            ["agent", "status"],
        )

        self.agent_response_time = Histogram(
            "parallaxpal_agent_response_seconds",
            "Agent response time in seconds",
            ["agent"],
        )

        # Research metrics
        self.research_queries = Counter(
            "parallaxpal_research_queries_total",
            "Total number of research queries",
            ["mode", "user_tier"],
        )

        self.research_duration = Histogram(
            "parallaxpal_research_duration_seconds",
            "Research completion time in seconds",
            ["mode"],
        )

        # System metrics
        self.active_users = Gauge("parallaxpal_active_users", "Number of active users")

        self.websocket_connections = Gauge(
            "parallaxpal_websocket_connections",
            "Number of active WebSocket connections",
        )

        self.cache_hit_rate = Gauge(
            "parallaxpal_cache_hit_rate", "Cache hit rate percentage"
        )

        # Info metrics
        self.system_info = Info("parallaxpal_system", "System information")
        self.system_info.info(
            {"version": "2.0.0", "environment": os.getenv("ENVIRONMENT", "development")}
        )

    async def create_metric_descriptor(
        self,
        metric_type: str,
        metric_kind: monitoring_v3.MetricDescriptor.MetricKind,
        value_type: monitoring_v3.MetricDescriptor.ValueType,
        description: str,
        unit: str = "1",
        labels: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Create a custom metric descriptor

        Args:
            metric_type: Metric type name (e.g., "agent_response_time")
            metric_kind: GAUGE, DELTA, or CUMULATIVE
            value_type: INT64, DOUBLE, etc.
            description: Human-readable description
            unit: Unit of measurement
            labels: Optional label descriptors
        """

        descriptor = monitoring_v3.MetricDescriptor(
            type=f"custom.googleapis.com/parallaxpal/{metric_type}",
            metric_kind=metric_kind,
            value_type=value_type,
            description=description,
            unit=unit,
            display_name=metric_type.replace("_", " ").title(),
        )

        # Add labels if provided
        if labels:
            for label in labels:
                descriptor.labels.append(
                    monitoring_v3.LabelDescriptor(
                        key=label["key"],
                        value_type=label.get("value_type", "STRING"),
                        description=label.get("description", ""),
                    )
                )

        try:
            created = self.metrics_client.create_metric_descriptor(
                name=self.project_path, metric_descriptor=descriptor
            )
            self._metric_descriptors[metric_type] = created
            logger.info(f"Created metric descriptor: {metric_type}")
            return created
        except exceptions.AlreadyExists:
            logger.info(f"Metric descriptor already exists: {metric_type}")
            return None

    async def write_time_series(
        self,
        metric_type: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        resource_type: str = "global",
        resource_labels: Optional[Dict[str, str]] = None,
    ):
        """
        Write a time series data point

        Args:
            metric_type: Metric type name
            value: Metric value
            labels: Metric labels
            resource_type: Resource type (e.g., "global", "gce_instance")
            resource_labels: Resource labels
        """

        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/parallaxpal/{metric_type}"

        # Add metric labels
        if labels:
            for key, val in labels.items():
                series.metric.labels[key] = str(val)

        # Set resource
        series.resource.type = resource_type
        if resource_labels:
            for key, val in resource_labels.items():
                series.resource.labels[key] = str(val)

        # Create point
        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10**9)

        interval = monitoring_v3.TimeInterval(
            end_time={"seconds": seconds, "nanos": nanos}
        )

        point = monitoring_v3.Point(
            interval=interval, value={"double_value": float(value)}
        )

        series.points = [point]

        # Write time series
        try:
            self.metrics_client.create_time_series(
                name=self.project_path, time_series=[series]
            )
        except Exception as e:
            logger.error(f"Error writing time series: {e}")

    def trace_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Create a trace span decorator

        Args:
            name: Span name
            attributes: Span attributes
        """

        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(name) as span:
                    if attributes:
                        span.set_attributes(attributes)

                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(name) as span:
                    if attributes:
                        span.set_attributes(attributes)

                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    @asynccontextmanager
    async def trace_context(
        self, name: str, attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Async context manager for tracing

        Args:
            name: Span name
            attributes: Span attributes
        """
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                span.set_attributes(attributes)

            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    async def record_agent_metrics(
        self, agent_name: str, response_time: float, success: bool, tokens_used: int = 0
    ):
        """
        Record agent performance metrics

        Args:
            agent_name: Name of the agent
            response_time: Response time in seconds
            success: Whether the task succeeded
            tokens_used: Number of tokens used
        """
        # Update Prometheus metrics
        status = "success" if success else "failure"
        self.agent_task_count.labels(agent=agent_name, status=status).inc()
        self.agent_response_time.labels(agent=agent_name).observe(response_time)

        # Write to Cloud Monitoring
        await self.write_time_series(
            "agent_response_time",
            response_time,
            labels={"agent": agent_name, "status": status},
        )

        if tokens_used > 0:
            await self.write_time_series(
                "agent_token_usage", tokens_used, labels={"agent": agent_name}
            )

    async def record_research_metrics(
        self,
        query: str,
        mode: str,
        user_tier: str,
        duration: float,
        sources_found: int = 0,
    ):
        """
        Record research query metrics

        Args:
            query: Research query
            mode: Research mode
            user_tier: User subscription tier
            duration: Query duration in seconds
            sources_found: Number of sources found
        """
        # Update Prometheus metrics
        self.research_queries.labels(mode=mode, user_tier=user_tier).inc()
        self.research_duration.labels(mode=mode).observe(duration)

        # Write to Cloud Monitoring
        await self.write_time_series(
            "research_duration", duration, labels={"mode": mode, "tier": user_tier}
        )

        await self.write_time_series(
            "sources_found", sources_found, labels={"mode": mode}
        )

    async def update_system_metrics(
        self, active_users: int, websocket_connections: int, cache_hit_rate: float
    ):
        """
        Update system-wide metrics

        Args:
            active_users: Number of active users
            websocket_connections: Number of WebSocket connections
            cache_hit_rate: Cache hit rate (0-1)
        """
        # Update Prometheus gauges
        self.active_users.set(active_users)
        self.websocket_connections.set(websocket_connections)
        self.cache_hit_rate.set(cache_hit_rate * 100)

        # Write to Cloud Monitoring
        await self.write_time_series("active_users", active_users)
        await self.write_time_series("websocket_connections", websocket_connections)
        await self.write_time_series("cache_hit_rate", cache_hit_rate * 100)

    async def create_alert_policy(
        self,
        display_name: str,
        condition_filter: str,
        threshold_value: float,
        notification_channel_ids: List[str],
        documentation: str = "",
    ):
        """
        Create an alert policy

        Args:
            display_name: Alert display name
            condition_filter: Monitoring filter
            threshold_value: Threshold for alerting
            notification_channel_ids: Notification channel IDs
            documentation: Alert documentation
        """
        alert_client = monitoring_v3.AlertPolicyServiceClient()

        # Create condition
        condition = monitoring_v3.AlertPolicy.Condition(
            display_name=f"{display_name} condition",
            condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                filter=condition_filter,
                comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                threshold_value=threshold_value,
                duration={"seconds": 60},
                aggregations=[
                    monitoring_v3.Aggregation(
                        alignment_period={"seconds": 60},
                        per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN,
                    )
                ],
            ),
        )

        # Create policy
        policy = monitoring_v3.AlertPolicy(
            display_name=display_name,
            conditions=[condition],
            notification_channels=notification_channel_ids,
            documentation=monitoring_v3.AlertPolicy.Documentation(
                content=documentation, mime_type="text/markdown"
            ),
            combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.AND,
        )

        try:
            created = alert_client.create_alert_policy(
                name=self.project_path, alert_policy=policy
            )
            logger.info(f"Created alert policy: {display_name}")
            return created
        except Exception as e:
            logger.error(f"Error creating alert policy: {e}")
            return None

    async def get_metrics_summary(
        self, metric_type: str, hours: int = 24, aggregation: str = "mean"
    ) -> Dict[str, Any]:
        """
        Get metrics summary for a time period

        Args:
            metric_type: Metric type to query
            hours: Number of hours to look back
            aggregation: Aggregation type (mean, sum, max, min)

        Returns:
            Metrics summary
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        interval = monitoring_v3.TimeInterval(start_time=start_time, end_time=end_time)

        aggregation_map = {
            "mean": monitoring_v3.Aggregation.Aligner.ALIGN_MEAN,
            "sum": monitoring_v3.Aggregation.Aligner.ALIGN_SUM,
            "max": monitoring_v3.Aggregation.Aligner.ALIGN_MAX,
            "min": monitoring_v3.Aggregation.Aligner.ALIGN_MIN,
        }

        results = self.metrics_client.list_time_series(
            request={
                "name": self.project_path,
                "filter": f'metric.type="custom.googleapis.com/parallaxpal/{metric_type}"',
                "interval": interval,
                "aggregation": monitoring_v3.Aggregation(
                    alignment_period={"seconds": 300},  # 5 minutes
                    per_series_aligner=aggregation_map.get(
                        aggregation, monitoring_v3.Aggregation.Aligner.ALIGN_MEAN
                    ),
                ),
            }
        )

        summary = {
            "metric_type": metric_type,
            "period_hours": hours,
            "aggregation": aggregation,
            "time_series": [],
        }

        for result in results:
            series_data = {
                "labels": dict(result.metric.labels),
                "points": [
                    {
                        "time": point.interval.end_time,
                        "value": point.value.double_value or point.value.int64_value,
                    }
                    for point in result.points
                ],
            }
            summary["time_series"].append(series_data)

        return summary


class StructuredLogger:
    """Structured logging with Cloud Logging integration"""

    def __init__(self, name: str = "parallaxpal"):
        """
        Initialize structured logger

        Args:
            name: Logger name
        """
        self.name = name
        self.local_logger = logging.getLogger(name)

        # Initialize Cloud Logging if in production
        if os.getenv("ENVIRONMENT") == "production":
            self.cloud_client = cloud_logging.Client()
            self.cloud_logger = self.cloud_client.logger(name)
        else:
            self.cloud_logger = None

    def log(self, severity: str, message: str, **kwargs):
        """
        Log structured message

        Args:
            severity: Log severity (debug, info, warning, error, critical)
            message: Log message
            **kwargs: Additional structured data
        """
        # Add timestamp
        kwargs["timestamp"] = datetime.now().isoformat()

        # Local logging
        log_method = getattr(
            self.local_logger, severity.lower(), self.local_logger.info
        )
        log_method(message, extra={"structured_data": kwargs})

        # Cloud logging
        if self.cloud_logger:
            severity_map = {
                "debug": "DEBUG",
                "info": "INFO",
                "warning": "WARNING",
                "error": "ERROR",
                "critical": "CRITICAL",
            }

            self.cloud_logger.log_struct(
                {"message": message, **kwargs},
                severity=severity_map.get(severity.lower(), "INFO"),
            )


# Global instances
monitoring_service: Optional[CloudMonitoringService] = None
structured_logger = StructuredLogger()


def initialize_monitoring(project_id: str = None):
    """
    Initialize global monitoring service

    Args:
        project_id: Google Cloud project ID
    """
    global monitoring_service
    monitoring_service = CloudMonitoringService(project_id)
    return monitoring_service


def get_monitoring_service() -> CloudMonitoringService:
    """Get the global monitoring service instance"""
    if not monitoring_service:
        raise RuntimeError("Monitoring service not initialized")
    return monitoring_service
