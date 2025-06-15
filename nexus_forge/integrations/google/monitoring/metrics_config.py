"""Custom metrics definitions and configurations for Parallax Pal monitoring."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class MetricType(Enum):
    """Types of metrics we track."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Definition of a custom metric."""
    name: str
    type: MetricType
    description: str
    unit: str
    labels: List[str]
    buckets: Optional[List[float]] = None  # For histograms


# Application Metrics
APP_METRICS = {
    "research_queries_total": MetricDefinition(
        name="parallax_pal_research_queries_total",
        type=MetricType.COUNTER,
        description="Total number of research queries processed",
        unit="1",
        labels=["agent", "status", "user_tier"]
    ),
    "research_query_duration": MetricDefinition(
        name="parallax_pal_research_query_duration_seconds",
        type=MetricType.HISTOGRAM,
        description="Duration of research query processing",
        unit="s",
        labels=["agent", "complexity"],
        buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
    ),
    "active_websocket_connections": MetricDefinition(
        name="parallax_pal_active_websocket_connections",
        type=MetricType.GAUGE,
        description="Number of active WebSocket connections",
        unit="1",
        labels=["user_tier"]
    ),
    "knowledge_graph_nodes": MetricDefinition(
        name="parallax_pal_knowledge_graph_nodes_total",
        type=MetricType.COUNTER,
        description="Total number of nodes in knowledge graphs",
        unit="1",
        labels=["query_id", "node_type"]
    ),
    "agent_token_usage": MetricDefinition(
        name="parallax_pal_agent_token_usage_total",
        type=MetricType.COUNTER,
        description="Total tokens used by agents",
        unit="1",
        labels=["agent", "token_type"]  # token_type: input/output
    ),
    "export_operations": MetricDefinition(
        name="parallax_pal_export_operations_total",
        type=MetricType.COUNTER,
        description="Total number of export operations",
        unit="1",
        labels=["format", "status", "user_tier"]
    ),
    "voice_interactions": MetricDefinition(
        name="parallax_pal_voice_interactions_total",
        type=MetricType.COUNTER,
        description="Total number of voice interactions",
        unit="1",
        labels=["type", "status"]  # type: speech_to_text/text_to_speech
    ),
    "collaborative_sessions": MetricDefinition(
        name="parallax_pal_collaborative_sessions_active",
        type=MetricType.GAUGE,
        description="Number of active collaborative research sessions",
        unit="1",
        labels=["session_size"]
    ),
    "cache_operations": MetricDefinition(
        name="parallax_pal_cache_operations_total",
        type=MetricType.COUNTER,
        description="Total cache operations",
        unit="1",
        labels=["operation", "status"]  # operation: get/set/delete, status: hit/miss
    ),
    "rate_limit_hits": MetricDefinition(
        name="parallax_pal_rate_limit_hits_total",
        type=MetricType.COUNTER,
        description="Number of rate limit hits",
        unit="1",
        labels=["endpoint", "user_tier"]
    )
}

# System Metrics
SYSTEM_METRICS = {
    "cpu_usage": MetricDefinition(
        name="parallax_pal_cpu_usage_percent",
        type=MetricType.GAUGE,
        description="CPU usage percentage",
        unit="%",
        labels=["service"]
    ),
    "memory_usage": MetricDefinition(
        name="parallax_pal_memory_usage_bytes",
        type=MetricType.GAUGE,
        description="Memory usage in bytes",
        unit="By",
        labels=["service", "type"]  # type: rss/vms
    ),
    "gpu_usage": MetricDefinition(
        name="parallax_pal_gpu_usage_percent",
        type=MetricType.GAUGE,
        description="GPU usage percentage",
        unit="%",
        labels=["gpu_id", "metric_type"]  # metric_type: compute/memory
    ),
    "redis_connection_pool": MetricDefinition(
        name="parallax_pal_redis_connections",
        type=MetricType.GAUGE,
        description="Redis connection pool status",
        unit="1",
        labels=["status"]  # status: active/idle
    ),
    "database_connections": MetricDefinition(
        name="parallax_pal_database_connections",
        type=MetricType.GAUGE,
        description="Database connection pool status",
        unit="1",
        labels=["status", "database"]
    )
}

# Agent-specific Metrics
AGENT_METRICS = {
    "agent_invocations": MetricDefinition(
        name="parallax_pal_agent_invocations_total",
        type=MetricType.COUNTER,
        description="Total agent invocations",
        unit="1",
        labels=["agent_name", "status"]
    ),
    "agent_response_time": MetricDefinition(
        name="parallax_pal_agent_response_time_seconds",
        type=MetricType.HISTOGRAM,
        description="Agent response time",
        unit="s",
        labels=["agent_name"],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    ),
    "agent_error_rate": MetricDefinition(
        name="parallax_pal_agent_errors_total",
        type=MetricType.COUNTER,
        description="Total agent errors",
        unit="1",
        labels=["agent_name", "error_type"]
    ),
    "agent_tool_usage": MetricDefinition(
        name="parallax_pal_agent_tool_usage_total",
        type=MetricType.COUNTER,
        description="Agent tool usage statistics",
        unit="1",
        labels=["agent_name", "tool_name", "status"]
    )
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    "name": "Parallax Pal Monitoring Dashboard",
    "refresh_interval": 30,  # seconds
    "sections": [
        {
            "title": "System Overview",
            "panels": [
                {
                    "title": "Active Users",
                    "type": "stat",
                    "metric": "parallax_pal_active_websocket_connections",
                    "aggregation": "sum"
                },
                {
                    "title": "Research Queries (24h)",
                    "type": "stat",
                    "metric": "parallax_pal_research_queries_total",
                    "aggregation": "sum",
                    "time_range": "24h"
                },
                {
                    "title": "Average Query Duration",
                    "type": "gauge",
                    "metric": "parallax_pal_research_query_duration_seconds",
                    "aggregation": "avg"
                },
                {
                    "title": "Error Rate",
                    "type": "stat",
                    "metric": "parallax_pal_agent_errors_total",
                    "aggregation": "rate",
                    "time_range": "5m"
                }
            ]
        },
        {
            "title": "Agent Performance",
            "panels": [
                {
                    "title": "Agent Response Times",
                    "type": "graph",
                    "metrics": ["parallax_pal_agent_response_time_seconds"],
                    "group_by": ["agent_name"],
                    "visualization": "line"
                },
                {
                    "title": "Agent Invocations",
                    "type": "graph",
                    "metrics": ["parallax_pal_agent_invocations_total"],
                    "group_by": ["agent_name"],
                    "visualization": "stacked_area"
                },
                {
                    "title": "Token Usage by Agent",
                    "type": "graph",
                    "metrics": ["parallax_pal_agent_token_usage_total"],
                    "group_by": ["agent", "token_type"],
                    "visualization": "stacked_bar"
                }
            ]
        },
        {
            "title": "System Resources",
            "panels": [
                {
                    "title": "CPU Usage",
                    "type": "graph",
                    "metrics": ["parallax_pal_cpu_usage_percent"],
                    "group_by": ["service"],
                    "visualization": "line"
                },
                {
                    "title": "Memory Usage",
                    "type": "graph",
                    "metrics": ["parallax_pal_memory_usage_bytes"],
                    "group_by": ["service"],
                    "visualization": "line"
                },
                {
                    "title": "GPU Usage",
                    "type": "graph",
                    "metrics": ["parallax_pal_gpu_usage_percent"],
                    "group_by": ["gpu_id", "metric_type"],
                    "visualization": "line"
                }
            ]
        },
        {
            "title": "Feature Usage",
            "panels": [
                {
                    "title": "Export Operations",
                    "type": "pie",
                    "metric": "parallax_pal_export_operations_total",
                    "group_by": ["format"]
                },
                {
                    "title": "Voice Interactions",
                    "type": "graph",
                    "metrics": ["parallax_pal_voice_interactions_total"],
                    "group_by": ["type"],
                    "visualization": "bar"
                },
                {
                    "title": "Collaborative Sessions",
                    "type": "stat",
                    "metric": "parallax_pal_collaborative_sessions_active",
                    "aggregation": "current"
                }
            ]
        },
        {
            "title": "Cache & Performance",
            "panels": [
                {
                    "title": "Cache Hit Rate",
                    "type": "gauge",
                    "metric": "parallax_pal_cache_operations_total",
                    "calculation": "hit_rate"
                },
                {
                    "title": "Rate Limit Hits",
                    "type": "graph",
                    "metrics": ["parallax_pal_rate_limit_hits_total"],
                    "group_by": ["endpoint", "user_tier"],
                    "visualization": "heatmap"
                }
            ]
        }
    ]
}

# Alert Thresholds
ALERT_THRESHOLDS = {
    "high_error_rate": {
        "metric": "parallax_pal_agent_errors_total",
        "condition": "rate > 0.05",  # 5% error rate
        "duration": "5m",
        "severity": "critical"
    },
    "slow_queries": {
        "metric": "parallax_pal_research_query_duration_seconds",
        "condition": "p95 > 30",  # 95th percentile > 30 seconds
        "duration": "10m",
        "severity": "warning"
    },
    "high_cpu_usage": {
        "metric": "parallax_pal_cpu_usage_percent",
        "condition": "avg > 80",
        "duration": "15m",
        "severity": "warning"
    },
    "high_memory_usage": {
        "metric": "parallax_pal_memory_usage_bytes",
        "condition": "avg > 0.9 * max",  # 90% of max memory
        "duration": "10m",
        "severity": "critical"
    },
    "rate_limit_abuse": {
        "metric": "parallax_pal_rate_limit_hits_total",
        "condition": "rate > 100",  # More than 100 hits per minute
        "duration": "5m",
        "severity": "warning"
    },
    "low_cache_hit_rate": {
        "metric": "parallax_pal_cache_operations_total",
        "condition": "hit_rate < 0.7",  # Less than 70% hit rate
        "duration": "30m",
        "severity": "info"
    }
}


def get_all_metrics() -> Dict[str, MetricDefinition]:
    """Get all metric definitions."""
    all_metrics = {}
    all_metrics.update(APP_METRICS)
    all_metrics.update(SYSTEM_METRICS)
    all_metrics.update(AGENT_METRICS)
    return all_metrics


def get_prometheus_config() -> str:
    """Generate Prometheus configuration for all metrics."""
    config_lines = ["# Parallax Pal Prometheus Metrics Configuration\n"]
    
    for metric_name, metric_def in get_all_metrics().items():
        config_lines.append(f"# {metric_def.description}")
        config_lines.append(f"# TYPE {metric_def.name} {metric_def.type.value}")
        if metric_def.unit != "1":
            config_lines.append(f"# UNIT {metric_def.name} {metric_def.unit}")
        config_lines.append("")
    
    return "\n".join(config_lines)