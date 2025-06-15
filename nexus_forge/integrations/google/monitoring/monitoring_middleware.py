"""Monitoring middleware for FastAPI to instrument all requests."""

import time
import traceback
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging
from .cloud_monitoring import CloudMonitoringService
from .metrics_config import APP_METRICS

logger = logging.getLogger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware to monitor all HTTP requests."""
    
    def __init__(self, app: ASGIApp, monitoring_service: CloudMonitoringService):
        super().__init__(app)
        self.monitoring = monitoring_service
        
    async def dispatch(self, request: Request, call_next):
        # Start timing
        start_time = time.time()
        
        # Extract request metadata
        method = request.method
        path = request.url.path
        user_tier = request.headers.get("X-User-Tier", "free")
        
        # Create trace span
        with self.monitoring.create_trace_span(
            f"http_{method.lower()}_{path}",
            attributes={
                "http.method": method,
                "http.url": str(request.url),
                "http.path": path,
                "user.tier": user_tier
            }
        ) as span:
            try:
                # Process request
                response = await call_next(request)
                
                # Record metrics
                duration = time.time() - start_time
                status_code = response.status_code
                
                # Update request metrics
                self.monitoring.increment_counter(
                    "http_requests_total",
                    labels={
                        "method": method,
                        "path": path,
                        "status": str(status_code),
                        "user_tier": user_tier
                    }
                )
                
                self.monitoring.observe_histogram(
                    "http_request_duration_seconds",
                    duration,
                    labels={
                        "method": method,
                        "path": path
                    }
                )
                
                # Add response headers
                response.headers["X-Request-Duration"] = f"{duration:.3f}"
                response.headers["X-Trace-ID"] = span.span_id if hasattr(span, 'span_id') else ""
                
                # Log slow requests
                if duration > 5.0:
                    logger.warning(
                        f"Slow request: {method} {path} took {duration:.2f}s",
                        extra={
                            "duration": duration,
                            "method": method,
                            "path": path,
                            "status_code": status_code
                        }
                    )
                
                return response
                
            except Exception as e:
                # Record error metrics
                self.monitoring.increment_counter(
                    "http_errors_total",
                    labels={
                        "method": method,
                        "path": path,
                        "error_type": type(e).__name__
                    }
                )
                
                # Log error with trace
                logger.error(
                    f"Request error: {method} {path}",
                    extra={
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "method": method,
                        "path": path
                    }
                )
                
                # Re-raise the exception
                raise


class MonitoringRoute(APIRoute):
    """Custom route class to add monitoring to specific endpoints."""
    
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        monitoring_service = CloudMonitoringService()
        
        async def monitored_route_handler(request: Request) -> Response:
            # Extract endpoint info
            endpoint = self.endpoint.__name__
            
            # Create trace span for this specific endpoint
            with monitoring_service.create_trace_span(
                f"endpoint_{endpoint}",
                attributes={
                    "endpoint.name": endpoint,
                    "endpoint.path": self.path,
                    "endpoint.methods": ",".join(self.methods or [])
                }
            ):
                return await original_route_handler(request)
        
        return monitored_route_handler


class WebSocketMonitoringMiddleware:
    """Middleware to monitor WebSocket connections."""
    
    def __init__(self, app: ASGIApp, monitoring_service: CloudMonitoringService):
        self.app = app
        self.monitoring = monitoring_service
        self.active_connections = {}
        
    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            connection_id = f"{scope['client'][0]}:{scope['client'][1]}"
            
            # Monitor WebSocket lifecycle
            async def monitored_receive():
                message = await receive()
                
                if message["type"] == "websocket.connect":
                    self.active_connections[connection_id] = time.time()
                    self.monitoring.set_gauge(
                        "active_websocket_connections",
                        len(self.active_connections),
                        labels={"user_tier": scope.get("user_tier", "free")}
                    )
                    
                elif message["type"] == "websocket.disconnect":
                    if connection_id in self.active_connections:
                        duration = time.time() - self.active_connections[connection_id]
                        del self.active_connections[connection_id]
                        
                        self.monitoring.observe_histogram(
                            "websocket_connection_duration_seconds",
                            duration,
                            labels={"close_code": str(message.get("code", 1000))}
                        )
                        
                        self.monitoring.set_gauge(
                            "active_websocket_connections",
                            len(self.active_connections),
                            labels={"user_tier": scope.get("user_tier", "free")}
                        )
                
                return message
            
            async def monitored_send(message):
                if message["type"] == "websocket.send":
                    # Monitor message size
                    if "bytes" in message:
                        size = len(message["bytes"])
                    elif "text" in message:
                        size = len(message["text"].encode())
                    else:
                        size = 0
                    
                    self.monitoring.observe_histogram(
                        "websocket_message_size_bytes",
                        size,
                        labels={"direction": "outbound"}
                    )
                
                await send(message)
            
            await self.app(scope, monitored_receive, monitored_send)
        else:
            await self.app(scope, receive, send)


def setup_request_monitoring(app, monitoring_service: CloudMonitoringService):
    """Setup request monitoring for FastAPI app."""
    # Add HTTP middleware
    app.add_middleware(MonitoringMiddleware, monitoring_service=monitoring_service)
    
    # Add custom exception handler with monitoring
    @app.exception_handler(Exception)
    async def monitored_exception_handler(request: Request, exc: Exception):
        # Record exception metrics
        monitoring_service.increment_counter(
            "unhandled_exceptions_total",
            labels={
                "exception_type": type(exc).__name__,
                "path": request.url.path
            }
        )
        
        # Log with structured data
        logger.error(
            "Unhandled exception",
            extra={
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "path": request.url.path,
                "method": request.method
            }
        )
        
        # Return error response
        return Response(
            content="Internal server error",
            status_code=500
        )
    
    # Add startup/shutdown monitoring
    @app.on_event("startup")
    async def startup_monitoring():
        monitoring_service.increment_counter("app_starts_total")
        logger.info("Application started", extra={"event": "startup"})
    
    @app.on_event("shutdown")
    async def shutdown_monitoring():
        monitoring_service.increment_counter("app_stops_total")
        logger.info("Application shutdown", extra={"event": "shutdown"})


class AgentMonitoringMixin:
    """Mixin for agents to add monitoring capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitoring = CloudMonitoringService()
        self._agent_name = self.__class__.__name__
    
    @property
    def monitored_invoke(self):
        """Decorator to monitor agent invocations."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                
                # Create trace span
                with self.monitoring.create_trace_span(
                    f"agent_{self._agent_name}_invoke",
                    attributes={
                        "agent.name": self._agent_name,
                        "agent.method": func.__name__
                    }
                ) as span:
                    try:
                        # Invoke agent
                        result = await func(*args, **kwargs)
                        
                        # Record success metrics
                        duration = time.time() - start_time
                        self.monitoring.increment_counter(
                            "agent_invocations_total",
                            labels={
                                "agent_name": self._agent_name,
                                "status": "success"
                            }
                        )
                        
                        self.monitoring.observe_histogram(
                            "agent_response_time_seconds",
                            duration,
                            labels={"agent_name": self._agent_name}
                        )
                        
                        # Record token usage if available
                        if hasattr(result, "token_usage"):
                            self.monitoring.increment_counter(
                                "agent_token_usage_total",
                                result.token_usage.get("input_tokens", 0),
                                labels={
                                    "agent": self._agent_name,
                                    "token_type": "input"
                                }
                            )
                            self.monitoring.increment_counter(
                                "agent_token_usage_total",
                                result.token_usage.get("output_tokens", 0),
                                labels={
                                    "agent": self._agent_name,
                                    "token_type": "output"
                                }
                            )
                        
                        return result
                        
                    except Exception as e:
                        # Record error metrics
                        self.monitoring.increment_counter(
                            "agent_invocations_total",
                            labels={
                                "agent_name": self._agent_name,
                                "status": "failed"
                            }
                        )
                        
                        self.monitoring.increment_counter(
                            "agent_errors_total",
                            labels={
                                "agent_name": self._agent_name,
                                "error_type": type(e).__name__
                            }
                        )
                        
                        logger.error(
                            f"Agent {self._agent_name} error",
                            extra={
                                "agent": self._agent_name,
                                "error": str(e),
                                "traceback": traceback.format_exc()
                            }
                        )
                        
                        raise
            
            return wrapper
        return decorator
    
    def record_tool_usage(self, tool_name: str, status: str = "success"):
        """Record tool usage by this agent."""
        self.monitoring.increment_counter(
            "agent_tool_usage_total",
            labels={
                "agent_name": self._agent_name,
                "tool_name": tool_name,
                "status": status
            }
        )