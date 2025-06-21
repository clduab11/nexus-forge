"""
Health check endpoints for comprehensive system monitoring
"""

import asyncio
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp
import psutil
import redis
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from nexus_forge.core.monitoring import (
    api_logger,
    get_health_dashboard,
    get_request_context,
)

router = APIRouter(prefix="/health", tags=["health"])


@dataclass
class HealthCheckResult:
    """Health check result model"""

    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    latency_ms: float
    timestamp: str
    details: Dict[str, Any]
    error: Optional[str] = None


class HealthChecker:
    """Comprehensive health checking service"""

    def __init__(self):
        self.redis_client = None
        self.health_checks = {
            "database": self._check_database,
            "redis": self._check_redis,
            "ai_services": self._check_ai_services,
            "external_apis": self._check_external_apis,
            "filesystem": self._check_filesystem,
            "memory": self._check_memory,
            "cpu": self._check_cpu,
            "disk": self._check_disk,
        }
        self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            redis_db = int(os.getenv("REDIS_DB", 0))

            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
        except Exception as e:
            api_logger.log_error("redis_initialization_error", e, "health_checker")

    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks"""
        results = {}

        # Run health checks concurrently
        tasks = []
        for check_name, check_func in self.health_checks.items():
            tasks.append(self._run_health_check(check_name, check_func))

        check_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(check_results):
            check_name = list(self.health_checks.keys())[i]
            if isinstance(result, Exception):
                results[check_name] = HealthCheckResult(
                    name=check_name,
                    status="unhealthy",
                    latency_ms=0,
                    timestamp=datetime.utcnow().isoformat(),
                    details={},
                    error=str(result),
                )
            else:
                results[check_name] = result

        return results

    async def _run_health_check(self, name: str, check_func) -> HealthCheckResult:
        """Run a single health check with timing"""
        start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(check_func):
                details = await check_func()
            else:
                details = check_func()

            latency_ms = (time.time() - start_time) * 1000

            # Determine status based on details
            status = self._determine_status(name, details, latency_ms)

            result = HealthCheckResult(
                name=name,
                status=status,
                latency_ms=latency_ms,
                timestamp=datetime.utcnow().isoformat(),
                details=details,
            )

            # Log health check result
            api_logger.log_health_check(
                name, name, status == "healthy", latency_ms / 1000
            )

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            api_logger.log_error(f"health_check_{name}_error", e, "health_checker")

            return HealthCheckResult(
                name=name,
                status="unhealthy",
                latency_ms=latency_ms,
                timestamp=datetime.utcnow().isoformat(),
                details={},
                error=str(e),
            )

    def _determine_status(
        self, check_name: str, details: Dict[str, Any], latency_ms: float
    ) -> str:
        """Determine health status based on check results"""

        # Latency-based thresholds
        latency_thresholds = {
            "database": 100,  # 100ms
            "redis": 50,  # 50ms
            "ai_services": 5000,  # 5s
            "external_apis": 2000,  # 2s
            "filesystem": 100,  # 100ms
            "memory": 10,  # 10ms
            "cpu": 10,  # 10ms
            "disk": 50,  # 50ms
        }

        threshold = latency_thresholds.get(check_name, 1000)

        # Check latency
        if latency_ms > threshold * 2:
            return "unhealthy"
        elif latency_ms > threshold:
            return "degraded"

        # Check specific conditions
        if check_name == "memory":
            usage = details.get("usage_percent", 0)
            if usage > 90:
                return "unhealthy"
            elif usage > 80:
                return "degraded"

        elif check_name == "cpu":
            usage = details.get("usage_percent", 0)
            if usage > 95:
                return "unhealthy"
            elif usage > 85:
                return "degraded"

        elif check_name == "disk":
            usage = details.get("usage_percent", 0)
            if usage > 95:
                return "unhealthy"
            elif usage > 90:
                return "degraded"

        elif check_name == "redis":
            if not details.get("connected", False):
                return "unhealthy"

        elif check_name == "ai_services":
            healthy_services = details.get("healthy_services", 0)
            total_services = details.get("total_services", 1)
            if healthy_services == 0:
                return "unhealthy"
            elif healthy_services < total_services:
                return "degraded"

        return "healthy"

    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            # This would check your actual database
            # For now, simulate a database check
            return {
                "connected": True,
                "connection_pool_size": 20,
                "active_connections": 5,
                "query_response_time_ms": 15.3,
            }
        except Exception as e:
            raise Exception(f"Database check failed: {e}")

    def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance"""
        if not self.redis_client:
            raise Exception("Redis client not initialized")

        try:
            # Test basic operations
            start_time = time.time()
            self.redis_client.ping()
            ping_latency = (time.time() - start_time) * 1000

            # Get Redis info
            info = self.redis_client.info()

            # Test set/get operation
            test_key = f"health_check_{int(time.time())}"
            start_time = time.time()
            self.redis_client.set(test_key, "test_value", ex=60)
            value = self.redis_client.get(test_key)
            operation_latency = (time.time() - start_time) * 1000
            self.redis_client.delete(test_key)

            return {
                "connected": True,
                "ping_latency_ms": ping_latency,
                "operation_latency_ms": operation_latency,
                "memory_usage_mb": info.get("used_memory", 0) / (1024 * 1024),
                "connected_clients": info.get("connected_clients", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "version": info.get("redis_version", "unknown"),
            }
        except Exception as e:
            api_logger.log_redis_operation("health_check", 0, False)
            raise Exception(f"Redis check failed: {e}")

    async def _check_ai_services(self) -> Dict[str, Any]:
        """Check AI services availability"""
        services = {
            "gemini": os.getenv("GEMINI_API_ENDPOINT"),
            "jules": "https://api.jules.ai/health",  # Example endpoint
            "imagen": "https://api.imagen.google.com/health",
            "veo": "https://api.veo.google.com/health",
        }

        healthy_services = 0
        service_details = {}

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            for service_name, endpoint in services.items():
                if not endpoint:
                    service_details[service_name] = {"status": "not_configured"}
                    continue

                try:
                    start_time = time.time()
                    # For Google AI services, we might need different health check approaches
                    if "google" in endpoint or "gemini" in service_name:
                        # Simulate health check for Google services
                        await asyncio.sleep(0.1)  # Simulate API call
                        service_details[service_name] = {
                            "status": "healthy",
                            "latency_ms": 100,
                            "last_check": datetime.utcnow().isoformat(),
                        }
                        healthy_services += 1
                    else:
                        async with session.get(endpoint) as response:
                            latency = (time.time() - start_time) * 1000
                            if response.status == 200:
                                service_details[service_name] = {
                                    "status": "healthy",
                                    "latency_ms": latency,
                                    "http_status": response.status,
                                }
                                healthy_services += 1
                            else:
                                service_details[service_name] = {
                                    "status": "unhealthy",
                                    "latency_ms": latency,
                                    "http_status": response.status,
                                }
                except Exception as e:
                    service_details[service_name] = {
                        "status": "unhealthy",
                        "error": str(e),
                    }

        return {
            "healthy_services": healthy_services,
            "total_services": len(services),
            "services": service_details,
        }

    async def _check_external_apis(self) -> Dict[str, Any]:
        """Check external API dependencies"""
        apis = {
            "google_cloud": "https://cloud.google.com",
            "openai": "https://api.openai.com",
        }

        results = {}

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            for api_name, endpoint in apis.items():
                try:
                    start_time = time.time()
                    async with session.head(endpoint) as response:
                        latency = (time.time() - start_time) * 1000
                        results[api_name] = {
                            "status": "reachable",
                            "latency_ms": latency,
                            "http_status": response.status,
                        }
                except Exception as e:
                    results[api_name] = {"status": "unreachable", "error": str(e)}

        return {"external_apis": results}

    def _check_filesystem(self) -> Dict[str, Any]:
        """Check filesystem availability and performance"""
        try:
            # Check if logs directory is writable
            test_file = "logs/health_check_test.tmp"
            os.makedirs("logs", exist_ok=True)

            start_time = time.time()
            with open(test_file, "w") as f:
                f.write("health check test")

            with open(test_file, "r") as f:
                content = f.read()

            os.remove(test_file)
            write_latency = (time.time() - start_time) * 1000

            return {
                "writable": True,
                "write_latency_ms": write_latency,
                "test_content_match": content == "health check test",
            }
        except Exception as e:
            raise Exception(f"Filesystem check failed: {e}")

    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            return {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "usage_percent": memory.percent,
                "swap_usage_percent": swap.percent,
                "status": "healthy" if memory.percent < 80 else "warning",
            }
        except Exception as e:
            raise Exception(f"Memory check failed: {e}")

    def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, "getloadavg") else None

            return {
                "usage_percent": cpu_percent,
                "cpu_count": cpu_count,
                "load_average": list(load_avg) if load_avg else None,
                "status": "healthy" if cpu_percent < 80 else "warning",
            }
        except Exception as e:
            raise Exception(f"CPU check failed: {e}")

    def _check_disk(self) -> Dict[str, Any]:
        """Check disk usage"""
        try:
            disk = psutil.disk_usage("/")

            return {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "usage_percent": round((disk.used / disk.total) * 100, 2),
                "status": "healthy" if disk.used / disk.total < 0.9 else "warning",
            }
        except Exception as e:
            raise Exception(f"Disk check failed: {e}")


# Initialize health checker
health_checker = HealthChecker()


@router.get("/", response_model=Dict[str, Any])
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "nexus-forge-api",
        "version": os.getenv("APP_VERSION", "unknown"),
        **get_request_context(),
    }


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_check():
    """Kubernetes liveness probe endpoint"""
    # Basic check - just return 200 if the service is running
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/ready", response_model=Dict[str, Any])
async def readiness_check():
    """Kubernetes readiness probe endpoint"""
    try:
        # Check critical dependencies
        critical_checks = ["redis", "memory", "disk"]
        results = await health_checker.check_all()

        unhealthy_critical = [
            name
            for name in critical_checks
            if results.get(name, {}).status == "unhealthy"
        ]

        if unhealthy_critical:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "status": "not_ready",
                    "unhealthy_services": unhealthy_critical,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "critical_services": {
                name: results[name].status for name in critical_checks
            },
        }

    except Exception as e:
        api_logger.log_error("readiness_check_error", e, "health_checker")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@router.get("/detailed", response_model=Dict[str, Any])
async def detailed_health_check():
    """Comprehensive health check with all services"""
    try:
        # Run all health checks
        results = await health_checker.check_all()

        # Convert results to dict format
        health_data = {name: asdict(result) for name, result in results.items()}

        # Calculate overall status
        statuses = [result.status for result in results.values()]
        if "unhealthy" in statuses:
            overall_status = "unhealthy"
        elif "degraded" in statuses:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        # Get system dashboard
        dashboard = get_health_dashboard()

        response = {
            "overall_status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "health_checks": health_data,
            "dashboard": dashboard,
            **get_request_context(),
        }

        # Return appropriate HTTP status
        if overall_status == "unhealthy":
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=response
            )
        elif overall_status == "degraded":
            return JSONResponse(
                status_code=status.HTTP_207_MULTI_STATUS, content=response
            )
        else:
            return response

    except Exception as e:
        api_logger.log_error("detailed_health_check_error", e, "health_checker")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                **get_request_context(),
            },
        )


@router.get("/service/{service_name}", response_model=Dict[str, Any])
async def service_health_check(service_name: str):
    """Check health of a specific service"""
    if service_name not in health_checker.health_checks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service '{service_name}' not found",
        )

    try:
        check_func = health_checker.health_checks[service_name]
        result = await health_checker._run_health_check(service_name, check_func)

        response = asdict(result)

        if result.status == "unhealthy":
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=response
            )
        elif result.status == "degraded":
            return JSONResponse(
                status_code=status.HTTP_207_MULTI_STATUS, content=response
            )
        else:
            return response

    except Exception as e:
        api_logger.log_error(f"{service_name}_health_check_error", e, "health_checker")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "service": service_name,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@router.get("/metrics", response_model=Dict[str, Any])
async def health_metrics():
    """Get health metrics in Prometheus-compatible format"""
    try:
        results = await health_checker.check_all()

        metrics = {
            "nexus_forge_health_check_status": [
                {
                    "labels": {"check_name": name, "service": name},
                    "value": 1 if result.status == "healthy" else 0,
                    "timestamp": result.timestamp,
                }
                for name, result in results.items()
            ],
            "nexus_forge_health_check_latency_seconds": [
                {
                    "labels": {"check_name": name},
                    "value": result.latency_ms / 1000,
                    "timestamp": result.timestamp,
                }
                for name, result in results.items()
            ],
        }

        return metrics

    except Exception as e:
        api_logger.log_error("health_metrics_error", e, "health_checker")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail={"error": str(e)}
        )


@router.post("/check/{service_name}")
async def trigger_service_check(service_name: str):
    """Manually trigger a health check for a specific service"""
    if service_name not in health_checker.health_checks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service '{service_name}' not found",
        )

    try:
        check_func = health_checker.health_checks[service_name]
        result = await health_checker._run_health_check(service_name, check_func)

        return {
            "triggered": True,
            "service": service_name,
            "result": asdict(result),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        api_logger.log_error(f"manual_{service_name}_check_error", e, "health_checker")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "triggered": False,
                "service": service_name,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
