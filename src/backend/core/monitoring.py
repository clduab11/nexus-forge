import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from collections import defaultdict, deque
from contextvars import ContextVar
from datetime import datetime, timedelta
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

import psutil
import redis
from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

# Optional OpenTelemetry imports with fallback
try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None
# Configure logging - ensure logs directory exists
import os
import threading
from dataclasses import asdict, dataclass

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Context variables for distributed tracing
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
user_id: ContextVar[str] = ContextVar("user_id", default="")
request_id: ContextVar[str] = ContextVar("request_id", default="")

# Prometheus metrics - Core HTTP
REQUEST_COUNT = Counter(
    "nexus_forge_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status", "user_tier"],
)
REQUEST_LATENCY = Histogram(
    "nexus_forge_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
)
RESEARCH_TASKS = Counter(
    "nexus_forge_research_tasks_total", "Total research tasks", ["status", "complexity"]
)
ERROR_COUNT = Counter(
    "nexus_forge_errors_total", "Total errors", ["type", "service", "severity"]
)

# AI-specific metrics
AI_SERVICE_ERRORS = Counter(
    "nexus_forge_ai_errors_total",
    "AI service errors",
    ["service", "error_type", "model"],
)
AI_REQUESTS = Counter(
    "nexus_forge_ai_requests_total",
    "AI service requests",
    ["service", "model", "operation"],
)
AI_MODEL_LATENCY = Histogram(
    "nexus_forge_ai_request_duration_seconds",
    "AI model response latency",
    ["service", "model", "operation"],
)
TOKEN_USAGE = Counter(
    "nexus_forge_tokens_used_total",
    "Total tokens consumed",
    ["service", "model", "type"],
)
TOKEN_QUOTA_USAGE = Gauge(
    "nexus_forge_token_quota_usage_percent", "Token quota usage percentage", ["service"]
)

# System metrics
RETRY_COUNT = Counter(
    "nexus_forge_retry_attempts_total", "Retry attempts", ["service", "function"]
)
CIRCUIT_BREAKER_STATE = Gauge(
    "nexus_forge_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half_open, 2=open)",
    ["service"],
)
ERROR_RATE = Gauge(
    "nexus_forge_error_rate_percentage",
    "Error rate percentage",
    ["service", "time_window"],
)
SERVICE_AVAILABILITY = Gauge(
    "nexus_forge_service_availability_percentage",
    "Service availability percentage",
    ["service"],
)
QUOTA_USAGE = Gauge(
    "nexus_forge_quota_usage_percentage",
    "API quota usage percentage",
    ["service", "quota_type"],
)

# Business metrics
ACTIVE_USERS = Gauge("nexus_forge_active_users", "Number of active users")
CONCURRENT_REQUESTS = Gauge(
    "nexus_forge_concurrent_requests", "Number of concurrent requests"
)
HEALTH_SCORE = Gauge("nexus_forge_health_score", "Overall system health score (0-100)")
DEPLOYMENT_INFO = Info("nexus_forge_deployment_info", "Deployment information")

# Redis metrics
REDIS_COMMANDS = Counter(
    "nexus_forge_redis_commands_total", "Redis commands executed", ["command", "status"]
)
REDIS_CONNECTION_POOL = Gauge(
    "nexus_forge_redis_connection_pool_size", "Redis connection pool size"
)
REDIS_CONNECTION_FAILURES = Counter(
    "nexus_forge_redis_connection_failures_total", "Redis connection failures"
)
REDIS_LATENCY = Histogram(
    "nexus_forge_redis_latency_seconds", "Redis operation latency", ["operation"]
)

# Health checks
HEALTH_CHECK_STATUS = Gauge(
    "nexus_forge_health_check_status",
    "Health check status (1=healthy, 0=unhealthy)",
    ["check_name", "service"],
)
HEALTH_CHECK_LATENCY = Histogram(
    "nexus_forge_health_check_duration_seconds", "Health check latency", ["check_name"]
)

# Database metrics
DB_CONNECTION_POOL_ACTIVE = Gauge(
    "nexus_forge_db_connection_pool_active", "Active database connections"
)
DB_CONNECTION_POOL_SIZE = Gauge(
    "nexus_forge_db_connection_pool_size", "Total database connection pool size"
)
DB_QUERY_DURATION = Histogram(
    "nexus_forge_db_query_duration_seconds", "Database query duration", ["query_type"]
)

# Cache-specific metrics
CACHE_HIT_RATE = Gauge(
    "nexus_forge_cache_hit_rate_percentage",
    "Cache hit rate percentage",
    ["service", "cache_type"],
)
CACHE_OPERATIONS = Counter(
    "nexus_forge_cache_operations_total",
    "Total cache operations",
    ["operation", "service", "status"],
)
CACHE_LATENCY = Histogram(
    "nexus_forge_cache_operation_latency_seconds",
    "Cache operation latency",
    ["operation", "service"],
)
CACHE_SIZE = Gauge("nexus_forge_cache_size_bytes", "Cache size in bytes", ["service"])
CACHE_EVICTIONS = Counter(
    "nexus_forge_cache_evictions_total", "Total cache evictions", ["service", "reason"]
)
CACHE_TTL_EFFECTIVENESS = Histogram(
    "nexus_forge_cache_ttl_effectiveness_ratio",
    "Ratio of actual vs intended TTL",
    ["service", "cache_type"],
)


@dataclass
class RequestContext:
    """Request context for distributed tracing"""

    correlation_id: str
    user_id: str
    request_id: str
    start_time: float
    endpoint: str
    method: str
    user_tier: str = "free"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def duration(self) -> float:
        return time.time() - self.start_time


class DistributedTracing:
    """Distributed tracing with OpenTelemetry"""

    def __init__(self):
        self.tracer_provider = None
        self.tracer = None
        self._setup_tracing()

    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available, tracing disabled")
            return

        try:
            # Configure tracer provider
            trace.set_tracer_provider(TracerProvider())
            self.tracer_provider = trace.get_tracer_provider()

            # Configure Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
                agent_port=int(os.getenv("JAEGER_AGENT_PORT", 6831)),
            )

            # Add span processor
            span_processor = BatchSpanProcessor(jaeger_exporter)
            self.tracer_provider.add_span_processor(span_processor)

            # Get tracer
            self.tracer = trace.get_tracer(__name__)

            logger.info("Distributed tracing initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize distributed tracing: {e}")

    def start_span(self, operation_name: str, **attributes):
        """Start a new trace span"""
        if not self.tracer or not OPENTELEMETRY_AVAILABLE:
            return None

        span = self.tracer.start_span(operation_name)

        # Add correlation ID and other context
        span.set_attribute("correlation_id", correlation_id.get())
        span.set_attribute("user_id", user_id.get())
        span.set_attribute("request_id", request_id.get())

        # Add custom attributes
        for key, value in attributes.items():
            span.set_attribute(key, str(value))

        return span


class BusinessMetrics:
    """Track business-specific metrics"""

    def __init__(self):
        self.active_users_set = set()
        self.concurrent_requests = 0
        self.lock = Lock()
        self._start_metrics_updater()

    def track_user_activity(self, user_id: str):
        """Track active user"""
        with self.lock:
            self.active_users_set.add(user_id)
            ACTIVE_USERS.set(len(self.active_users_set))

    def track_request_start(self):
        """Track concurrent request start"""
        with self.lock:
            self.concurrent_requests += 1
            CONCURRENT_REQUESTS.set(self.concurrent_requests)

    def track_request_end(self):
        """Track concurrent request end"""
        with self.lock:
            self.concurrent_requests = max(0, self.concurrent_requests - 1)
            CONCURRENT_REQUESTS.set(self.concurrent_requests)

    def calculate_health_score(self) -> float:
        """Calculate overall system health score"""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent

            # Weight factors
            cpu_weight = 0.3
            memory_weight = 0.3
            error_weight = 0.4

            # Calculate component scores (0-100)
            cpu_score = max(0, 100 - cpu_percent)
            memory_score = max(0, 100 - memory_percent)

            # Get average error rate across services
            avg_error_rate = self._get_average_error_rate()
            error_score = max(0, 100 - (avg_error_rate * 10))  # Scale error rate

            # Calculate weighted health score
            health_score = (
                cpu_score * cpu_weight
                + memory_score * memory_weight
                + error_score * error_weight
            )

            HEALTH_SCORE.set(health_score)
            return health_score

        except Exception as e:
            logger.error(f"Failed to calculate health score: {e}")
            HEALTH_SCORE.set(0)
            return 0

    def _get_average_error_rate(self) -> float:
        """Get average error rate across all services"""
        try:
            # This would typically query Prometheus for current error rates
            # For now, return a placeholder value
            return 2.5  # 2.5% error rate
        except Exception:
            return 0

    def _start_metrics_updater(self):
        """Start background thread to update metrics"""

        def update_metrics():
            while True:
                try:
                    self.calculate_health_score()

                    # Clear active users older than 5 minutes
                    # This is a simplified approach - in production, you'd track timestamps
                    if len(self.active_users_set) > 1000:  # Prevent memory bloat
                        with self.lock:
                            # Keep only recent users (simplified)
                            recent_users = list(self.active_users_set)[-500:]
                            self.active_users_set = set(recent_users)
                            ACTIVE_USERS.set(len(self.active_users_set))

                    time.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    logger.error(f"Error in metrics updater: {e}")
                    time.sleep(60)

        thread = threading.Thread(target=update_metrics, daemon=True)
        thread.start()
        logger.info("Business metrics updater started")


class ErrorMetrics:
    """Track error metrics and calculate rates."""

    def __init__(self, window_size: int = 300):  # 5-minute window
        self.window_size = window_size
        self.error_timestamps = defaultdict(lambda: deque())
        self.success_timestamps = defaultdict(lambda: deque())
        self.lock = Lock()

        # Alert thresholds
        self.error_rate_threshold = 10.0  # 10% error rate
        self.availability_threshold = 95.0  # 95% availability
        self.consecutive_errors_threshold = 5

    def record_error(self, service: str, error_type: str):
        """Record an error occurrence."""
        with self.lock:
            now = datetime.utcnow()
            self.error_timestamps[service].append(now)
            self._cleanup_old_timestamps(service)

            # Update Prometheus metrics
            ERROR_COUNT.labels(type=error_type, service=service).inc()

            # Check for alerts
            self._check_error_rate_alert(service)
            self._check_availability_alert(service)

    def record_success(self, service: str):
        """Record a successful operation."""
        with self.lock:
            now = datetime.utcnow()
            self.success_timestamps[service].append(now)
            self._cleanup_old_timestamps(service)

    def _cleanup_old_timestamps(self, service: str):
        """Remove timestamps outside the window."""
        cutoff = datetime.utcnow() - timedelta(seconds=self.window_size)

        # Clean error timestamps
        while (
            self.error_timestamps[service]
            and self.error_timestamps[service][0] < cutoff
        ):
            self.error_timestamps[service].popleft()

        # Clean success timestamps
        while (
            self.success_timestamps[service]
            and self.success_timestamps[service][0] < cutoff
        ):
            self.success_timestamps[service].popleft()

    def get_error_rate(self, service: str) -> float:
        """Calculate error rate percentage for a service."""
        with self.lock:
            self._cleanup_old_timestamps(service)

            error_count = len(self.error_timestamps[service])
            success_count = len(self.success_timestamps[service])
            total_count = error_count + success_count

            if total_count == 0:
                return 0.0

            rate = (error_count / total_count) * 100
            ERROR_RATE.labels(service=service, time_window=f"{self.window_size}s").set(
                rate
            )
            return rate

    def get_availability(self, service: str) -> float:
        """Calculate availability percentage for a service."""
        error_rate = self.get_error_rate(service)
        availability = 100.0 - error_rate
        SERVICE_AVAILABILITY.labels(service=service).set(availability)
        return availability

    def _check_error_rate_alert(self, service: str):
        """Check if error rate exceeds threshold."""
        error_rate = self.get_error_rate(service)
        if error_rate > self.error_rate_threshold:
            logger.warning(
                f"High error rate alert: {service} has {error_rate:.2f}% error rate "
                f"(threshold: {self.error_rate_threshold}%)"
            )

    def _check_availability_alert(self, service: str):
        """Check if availability drops below threshold."""
        availability = self.get_availability(service)
        if availability < self.availability_threshold:
            logger.critical(
                f"Low availability alert: {service} has {availability:.2f}% availability "
                f"(threshold: {self.availability_threshold}%)"
            )


class APILogger:
    def __init__(self, start_server: bool = True):
        if start_server and not os.getenv("TESTING"):
            self.start_prometheus()
        self.error_metrics = ErrorMetrics()

    def start_prometheus(self):
        """Start Prometheus metrics server"""
        try:
            prometheus_port = int(os.getenv("PROMETHEUS_PORT", 9090))
            start_http_server(prometheus_port)
            logger.info(f"Prometheus metrics server started on port {prometheus_port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {str(e)}")

    def log_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ):
        """Log HTTP request details"""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_code).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

    def log_research_task(self, status: str):
        """Log research task status"""
        RESEARCH_TASKS.labels(status=status).inc()

    def log_error(
        self, error_type: str, error: Exception, service: Optional[str] = None
    ):
        """Log error details with enhanced metrics"""
        service = service or "unknown"

        # Record error metrics
        self.error_metrics.record_error(service, error_type)

        logger.error(f"Error type: {error_type}")
        logger.error(f"Service: {service}")
        logger.error(f"Error message: {str(error)}")
        logger.error(f"Stacktrace: {''.join(traceback.format_tb(error.__traceback__))}")

    def log_ai_service_error(
        self,
        service: str,
        error_type: str,
        model: str,
        error: Exception,
        operation: Optional[str] = None,
    ):
        """Log AI service specific errors"""
        AI_SERVICE_ERRORS.labels(
            service=service, error_type=error_type, model=model
        ).inc()

        self.log_error(error_type, error, service)

        logger.error(
            f"AI Service Error - Service: {service}, Model: {model}, "
            f"Operation: {operation}, Error: {str(error)}"
        )

    def log_retry_attempt(self, service: str, function: str):
        """Log retry attempts"""
        RETRY_COUNT.labels(service=service, function=function).inc()
        logger.warning(f"Retry attempt for {service}.{function}")

    def log_circuit_breaker_state(self, service: str, state: str):
        """Log circuit breaker state changes"""
        state_mapping = {"closed": 0, "half_open": 1, "open": 2}
        CIRCUIT_BREAKER_STATE.labels(service=service).set(state_mapping.get(state, 0))
        logger.warning(f"Circuit breaker for {service} is now {state}")

    def log_ai_model_latency(
        self, service: str, model: str, operation: str, latency: float
    ):
        """Log AI model operation latency"""
        AI_MODEL_LATENCY.labels(
            service=service, model=model, operation=operation
        ).observe(latency)

    def log_quota_usage(self, service: str, quota_type: str, usage_percentage: float):
        """Log API quota usage"""
        QUOTA_USAGE.labels(service=service, quota_type=quota_type).set(usage_percentage)

        if usage_percentage > 80:
            logger.warning(
                f"High quota usage: {service} {quota_type} at {usage_percentage:.1f}%"
            )
        elif usage_percentage > 95:
            logger.critical(
                f"Critical quota usage: {service} {quota_type} at {usage_percentage:.1f}%"
            )

    def log_cache_hit(self, service: str, cache_type: str, hit_rate: float):
        """Log cache hit metrics"""
        CACHE_HIT_RATE.labels(service=service, cache_type=cache_type).set(hit_rate)
        CACHE_OPERATIONS.labels(
            operation="hit", service=service, status="success"
        ).inc()

    def log_cache_miss(self, service: str, cache_type: str):
        """Log cache miss metrics"""
        CACHE_OPERATIONS.labels(
            operation="miss", service=service, status="success"
        ).inc()

    def log_cache_operation(
        self, operation: str, service: str, latency: float, success: bool = True
    ):
        """Log cache operation metrics"""
        status = "success" if success else "error"
        CACHE_OPERATIONS.labels(
            operation=operation, service=service, status=status
        ).inc()
        CACHE_LATENCY.labels(operation=operation, service=service).observe(latency)

    def log_cache_eviction(self, service: str, reason: str):
        """Log cache eviction events"""
        CACHE_EVICTIONS.labels(service=service, reason=reason).inc()
        logger.info(f"Cache eviction in {service}: {reason}")

    def log_cache_size(self, service: str, size_bytes: int):
        """Log current cache size"""
        CACHE_SIZE.labels(service=service).set(size_bytes)

    def log_cache_ttl_effectiveness(
        self, service: str, cache_type: str, effectiveness_ratio: float
    ):
        """Log how effectively cache TTL is being used"""
        CACHE_TTL_EFFECTIVENESS.labels(service=service, cache_type=cache_type).observe(
            effectiveness_ratio
        )

    def get_service_health(self, service: str) -> Dict[str, Any]:
        """Get comprehensive health metrics for a service"""
        return {
            "error_rate": self.error_metrics.get_error_rate(service),
            "availability": self.error_metrics.get_availability(service),
            "cache_performance": self._get_cache_performance(service),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _get_cache_performance(self, service: str) -> Dict[str, Any]:
        """Get cache performance metrics for a service"""
        try:
            # This would typically query the cache metrics
            # For now, return placeholder metrics
            return {
                "hit_rate": 85.5,  # Placeholder
                "avg_latency_ms": 2.3,
                "total_operations": 1250,
                "evictions_24h": 12,
                "size_mb": 45.7,
            }
        except Exception as e:
            logger.error(f"Failed to get cache performance for {service}: {str(e)}")
            return {"error": "Unable to retrieve cache metrics"}


# Monitoring decorator for API endpoints
def monitor_endpoint(endpoint_name: str):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = 500
            try:
                response = await func(*args, **kwargs)
                status_code = (
                    response.status_code if hasattr(response, "status_code") else 200
                )
                return response
            except Exception as e:
                api_logger.log_error("endpoint_error", e)
                raise
            finally:
                duration = time.time() - start_time
                api_logger.log_request(
                    method=kwargs.get("method", "UNKNOWN"),
                    endpoint=endpoint_name,
                    status_code=status_code,
                    duration=duration,
                )

        return wrapper

    return decorator


class RequestLogMiddleware:
    """Middleware for logging all requests"""

    async def __call__(self, request, call_next):
        start_time = time.time()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            duration = time.time() - start_time
            status_code = response.status_code if response else 500
            api_logger.log_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=status_code,
                duration=duration,
            )


class StructuredLogger:
    """Structured logging with JSON format and error tracking"""

    def __init__(self, service_name: str):
        self.service_name = service_name

    def log(self, level: str, message: str, **kwargs):
        """Log a message with structured data"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "level": level,
            "message": message,
            **kwargs,
        }
        log_message = json.dumps(log_data)

        # Record metrics based on log level
        if level == "error":
            api_logger.error_metrics.record_error(
                self.service_name, kwargs.get("error_type", "unknown_error")
            )
            logger.error(log_message)
        elif level == "warning":
            logger.warning(log_message)
        else:
            api_logger.error_metrics.record_success(self.service_name)
            logger.info(log_message)

    def log_ai_operation(
        self,
        operation: str,
        model: str,
        success: bool,
        latency: Optional[float] = None,
        error: Optional[Exception] = None,
        **kwargs,
    ):
        """Log AI operation with comprehensive metrics"""
        log_data = {
            "operation": operation,
            "model": model,
            "success": success,
            "latency": latency,
            **kwargs,
        }

        if success:
            if latency:
                api_logger.log_ai_model_latency(
                    self.service_name, model, operation, latency
                )
            # Log cache performance if available
            if "cache_metadata" in log_data:
                cache_meta = log_data["cache_metadata"]
                if cache_meta.get("hit"):
                    api_logger.log_cache_hit(self.service_name, "ai_response", 100.0)
                else:
                    api_logger.log_cache_miss(self.service_name, "ai_response")

            self.log("info", f"AI operation successful: {operation}", **log_data)
        else:
            error_type = type(error).__name__ if error else "unknown_error"
            api_logger.log_ai_service_error(
                self.service_name,
                error_type,
                model,
                error or Exception("Unknown error"),
                operation,
            )
            log_data["error"] = str(error) if error else "Unknown error"
            self.log(
                "error",
                f"AI operation failed: {operation}",
                error_type=error_type,
                **log_data,
            )


def setup_monitoring(app):
    """Setup monitoring for the FastAPI application"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Initialize monitoring components
    global api_logger
    api_logger = APILogger(start_server=True)

    # Add middleware
    app.middleware("http")(RequestLogMiddleware())

    # Create structured logger
    return StructuredLogger("parallax-pal-api")


def get_health_dashboard() -> Dict[str, Any]:
    """Get comprehensive health dashboard data"""
    services = ["gemini", "jules", "imagen", "veo", "nexus_forge"]

    dashboard = {
        "timestamp": datetime.utcnow().isoformat(),
        "services": {},
        "overall_health": "healthy",
        "cache_summary": _get_cache_summary(),
    }

    total_availability = 0
    healthy_services = 0

    for service in services:
        health = api_logger.get_service_health(service)
        dashboard["services"][service] = health

        if health["availability"] >= 95:
            healthy_services += 1

        total_availability += health["availability"]

    # Calculate overall health
    avg_availability = total_availability / len(services)
    if avg_availability < 90:
        dashboard["overall_health"] = "critical"
    elif avg_availability < 95:
        dashboard["overall_health"] = "degraded"
    elif healthy_services < len(services):
        dashboard["overall_health"] = "warning"

    return dashboard


def _get_cache_summary() -> Dict[str, Any]:
    """Get overall cache performance summary"""
    try:
        # Import here to avoid circular imports
        from nexus_forge.core.caching_decorators import get_cache_instance

        cache = get_cache_instance()
        stats = cache.get_cache_stats()

        return {
            "overall_hit_rate": stats.get("hit_rate", 0),
            "total_operations": stats.get("total_hits", 0)
            + stats.get("total_misses", 0),
            "memory_usage": stats.get("redis_memory_used", "Unknown"),
            "health_status": "healthy" if stats.get("hit_rate", 0) > 70 else "degraded",
            "recommendations": _generate_cache_recommendations_internal(stats),
        }
    except Exception as e:
        logger.error(f"Failed to get cache summary: {str(e)}")
        return {"health_status": "unknown", "error": str(e)}


def _generate_cache_recommendations_internal(stats: Dict[str, Any]) -> list:
    """Generate cache optimization recommendations"""
    recommendations = []

    hit_rate = stats.get("hit_rate", 0)
    if hit_rate < 50:
        recommendations.append("Low cache hit rate - consider increasing TTL values")
    elif hit_rate > 90:
        recommendations.append(
            "Excellent cache performance - consider expanding to more endpoints"
        )

    evictions = stats.get("evictions", 0)
    if evictions > 100:
        recommendations.append(
            "High eviction rate - consider increasing cache size or optimizing data"
        )

    memory_usage = stats.get("redis_memory_used", "")
    if "MB" in memory_usage and int(memory_usage.replace("MB", "").strip()) > 500:
        recommendations.append("High memory usage - monitor for potential memory leaks")

    return recommendations


def create_ai_operation_monitor(service_name: str):
    """Create a monitoring decorator for AI operations"""

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error = None

            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error = e
                raise
            finally:
                latency = time.time() - start_time
                structured_logger.log_ai_operation(
                    operation=func.__name__,
                    model=kwargs.get("model", "unknown"),
                    success=success,
                    latency=latency,
                    error=error,
                )

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error = None

            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error = e
                raise
            finally:
                latency = time.time() - start_time
                structured_logger.log_ai_operation(
                    operation=func.__name__,
                    model=kwargs.get("model", "unknown"),
                    success=success,
                    latency=latency,
                    error=error,
                )

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Initialize API logger (but don't start server automatically during imports)
api_logger = APILogger(start_server=False)

# Initialize structured logger
structured_logger = StructuredLogger("nexus-forge-api")


def get_logger(name: str = "nexus-forge") -> StructuredLogger:
    """Get a structured logger instance for the given service name."""
    return StructuredLogger(name)


def create_correlation_context(
    correlation_id_val: str, user_id_val: str, request_id_val: str
):
    """Create context manager for request correlation"""

    class CorrelationContext:
        def __enter__(self):
            correlation_id.set(correlation_id_val)
            user_id.set(user_id_val)
            request_id.set(request_id_val)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            correlation_id.set("")
            user_id.set("")
            request_id.set("")

    return CorrelationContext()


def get_request_context() -> Dict[str, str]:
    """Get current request context"""
    return {
        "correlation_id": correlation_id.get(),
        "user_id": user_id.get(),
        "request_id": request_id.get(),
    }


# Cache monitoring functions
def monitor_cache_performance():
    """Monitor and report cache performance metrics"""
    try:
        from nexus_forge.core.caching_decorators import get_cache_instance

        cache = get_cache_instance()
        health = cache.health_check()

        if health["status"] == "healthy":
            stats = health["cache_stats"]
            api_logger.log_cache_operation(
                "health_check", "redis_cache", health["latency_ms"] / 1000, True
            )

            # Log current metrics
            api_logger.log_cache_hit("redis_cache", "general", stats.get("hit_rate", 0))

            logger.info(
                f"Cache health check passed - Hit rate: {stats.get('hit_rate', 0)}%, "
                f"Latency: {health['latency_ms']}ms"
            )
        else:
            logger.error(
                f"Cache health check failed: {health.get('error', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"Failed to monitor cache performance: {str(e)}")


def setup_cache_monitoring():
    """Setup periodic cache monitoring"""
    import threading
    import time

    def periodic_monitor():
        while True:
            try:
                monitor_cache_performance()
                time.sleep(300)  # Monitor every 5 minutes
            except Exception as e:
                logger.error(f"Error in cache monitoring thread: {str(e)}")
                time.sleep(60)  # Retry after 1 minute on error

    # Start monitoring thread
    monitor_thread = threading.Thread(target=periodic_monitor, daemon=True)
    monitor_thread.start()
    logger.info("Cache monitoring thread started")
