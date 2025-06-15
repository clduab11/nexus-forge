"""Alert policy configurations for Parallax Pal monitoring."""

import json
from typing import Dict, List, Any
from datetime import timedelta
from google.cloud import monitoring_v3
from google.api_core import retry
import logging

logger = logging.getLogger(__name__)


class AlertPolicyManager:
    """Manages alert policies for Parallax Pal monitoring."""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.project_name = f"projects/{project_id}"
        self.alert_client = monitoring_v3.AlertPolicyServiceClient()
        self.notification_client = monitoring_v3.NotificationChannelServiceClient()
        
    def create_notification_channels(self) -> Dict[str, str]:
        """Create notification channels for alerts."""
        channels = {}
        
        # Email notification channel
        email_channel = monitoring_v3.NotificationChannel(
            type_="email",
            display_name="Parallax Pal Alerts Email",
            description="Email notifications for Parallax Pal alerts",
            labels={"email_address": "alerts@parallaxpal.com"},
            enabled=True
        )
        
        # Slack notification channel
        slack_channel = monitoring_v3.NotificationChannel(
            type_="slack",
            display_name="Parallax Pal Alerts Slack",
            description="Slack notifications for Parallax Pal alerts",
            labels={"channel_name": "#parallax-pal-alerts"},
            enabled=True
        )
        
        # PagerDuty for critical alerts
        pagerduty_channel = monitoring_v3.NotificationChannel(
            type_="pagerduty",
            display_name="Parallax Pal Critical Alerts",
            description="PagerDuty notifications for critical alerts",
            labels={"service_key": "YOUR_PAGERDUTY_SERVICE_KEY"},
            enabled=True
        )
        
        try:
            email_created = self.notification_client.create_notification_channel(
                name=self.project_name,
                notification_channel=email_channel
            )
            channels["email"] = email_created.name
            
            slack_created = self.notification_client.create_notification_channel(
                name=self.project_name,
                notification_channel=slack_channel
            )
            channels["slack"] = slack_created.name
            
            pagerduty_created = self.notification_client.create_notification_channel(
                name=self.project_name,
                notification_channel=pagerduty_channel
            )
            channels["pagerduty"] = pagerduty_created.name
            
        except Exception as e:
            logger.error(f"Error creating notification channels: {e}")
            
        return channels
    
    def create_alert_policies(self, notification_channels: Dict[str, str]):
        """Create all alert policies."""
        policies = [
            self._create_high_error_rate_policy(notification_channels),
            self._create_slow_query_policy(notification_channels),
            self._create_high_cpu_policy(notification_channels),
            self._create_high_memory_policy(notification_channels),
            self._create_rate_limit_policy(notification_channels),
            self._create_websocket_connection_policy(notification_channels),
            self._create_agent_failure_policy(notification_channels),
            self._create_gpu_usage_policy(notification_channels),
            self._create_token_usage_policy(notification_channels),
            self._create_cache_performance_policy(notification_channels)
        ]
        
        created_policies = []
        for policy in policies:
            try:
                created = self.alert_client.create_alert_policy(
                    name=self.project_name,
                    alert_policy=policy
                )
                created_policies.append(created)
                logger.info(f"Created alert policy: {created.display_name}")
            except Exception as e:
                logger.error(f"Error creating alert policy {policy.display_name}: {e}")
                
        return created_policies
    
    def _create_high_error_rate_policy(self, channels: Dict[str, str]) -> monitoring_v3.AlertPolicy:
        """Create alert policy for high error rates."""
        return monitoring_v3.AlertPolicy(
            display_name="High Error Rate",
            documentation=monitoring_v3.AlertPolicy.Documentation(
                content="Agent error rate is above 5% for 5 minutes. This indicates system instability."
            ),
            conditions=[
                monitoring_v3.AlertPolicy.Condition(
                    display_name="Error rate > 5%",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter='metric.type="custom.googleapis.com/parallax_pal/agent_errors_total" '
                               'resource.type="global"',
                        aggregations=[
                            monitoring_v3.Aggregation(
                                alignment_period={"seconds": 60},
                                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_RATE
                            )
                        ],
                        comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                        threshold_value=0.05,
                        duration={"seconds": 300}
                    )
                )
            ],
            notification_channels=[channels["pagerduty"], channels["slack"]],
            alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
                auto_close={"seconds": 1800}  # Auto-close after 30 minutes
            )
        )
    
    def _create_slow_query_policy(self, channels: Dict[str, str]) -> monitoring_v3.AlertPolicy:
        """Create alert policy for slow queries."""
        return monitoring_v3.AlertPolicy(
            display_name="Slow Research Queries",
            documentation=monitoring_v3.AlertPolicy.Documentation(
                content="95th percentile query duration exceeds 30 seconds. Users may be experiencing delays."
            ),
            conditions=[
                monitoring_v3.AlertPolicy.Condition(
                    display_name="P95 query duration > 30s",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter='metric.type="custom.googleapis.com/parallax_pal/research_query_duration_seconds" '
                               'resource.type="global"',
                        aggregations=[
                            monitoring_v3.Aggregation(
                                alignment_period={"seconds": 300},
                                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_PERCENTILE_95
                            )
                        ],
                        comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                        threshold_value=30,
                        duration={"seconds": 600}
                    )
                )
            ],
            notification_channels=[channels["slack"], channels["email"]],
            alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
                auto_close={"seconds": 3600}
            )
        )
    
    def _create_high_cpu_policy(self, channels: Dict[str, str]) -> monitoring_v3.AlertPolicy:
        """Create alert policy for high CPU usage."""
        return monitoring_v3.AlertPolicy(
            display_name="High CPU Usage",
            documentation=monitoring_v3.AlertPolicy.Documentation(
                content="CPU usage exceeds 80% for 15 minutes. Consider scaling up resources."
            ),
            conditions=[
                monitoring_v3.AlertPolicy.Condition(
                    display_name="CPU usage > 80%",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter='metric.type="custom.googleapis.com/parallax_pal/cpu_usage_percent" '
                               'resource.type="global"',
                        aggregations=[
                            monitoring_v3.Aggregation(
                                alignment_period={"seconds": 60},
                                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN
                            )
                        ],
                        comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                        threshold_value=80,
                        duration={"seconds": 900}
                    )
                )
            ],
            notification_channels=[channels["slack"]],
            alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
                auto_close={"seconds": 3600}
            )
        )
    
    def _create_high_memory_policy(self, channels: Dict[str, str]) -> monitoring_v3.AlertPolicy:
        """Create alert policy for high memory usage."""
        return monitoring_v3.AlertPolicy(
            display_name="High Memory Usage",
            documentation=monitoring_v3.AlertPolicy.Documentation(
                content="Memory usage exceeds 90% of available memory. Risk of OOM errors."
            ),
            conditions=[
                monitoring_v3.AlertPolicy.Condition(
                    display_name="Memory usage > 90%",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter='metric.type="custom.googleapis.com/parallax_pal/memory_usage_bytes" '
                               'resource.type="global"',
                        aggregations=[
                            monitoring_v3.Aggregation(
                                alignment_period={"seconds": 60},
                                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN
                            )
                        ],
                        comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                        threshold_value=0.9,  # This would need to be calculated based on max memory
                        duration={"seconds": 600}
                    )
                )
            ],
            notification_channels=[channels["pagerduty"], channels["slack"]],
            alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
                auto_close={"seconds": 1800}
            )
        )
    
    def _create_rate_limit_policy(self, channels: Dict[str, str]) -> monitoring_v3.AlertPolicy:
        """Create alert policy for rate limit abuse."""
        return monitoring_v3.AlertPolicy(
            display_name="Rate Limit Abuse",
            documentation=monitoring_v3.AlertPolicy.Documentation(
                content="High rate of rate limit hits detected. Possible abuse or misconfigured client."
            ),
            conditions=[
                monitoring_v3.AlertPolicy.Condition(
                    display_name="Rate limit hits > 100/min",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter='metric.type="custom.googleapis.com/parallax_pal/rate_limit_hits_total" '
                               'resource.type="global"',
                        aggregations=[
                            monitoring_v3.Aggregation(
                                alignment_period={"seconds": 60},
                                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_RATE
                            )
                        ],
                        comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                        threshold_value=100,
                        duration={"seconds": 300}
                    )
                )
            ],
            notification_channels=[channels["slack"]],
            alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
                auto_close={"seconds": 1800}
            )
        )
    
    def _create_websocket_connection_policy(self, channels: Dict[str, str]) -> monitoring_v3.AlertPolicy:
        """Create alert policy for WebSocket connection spikes."""
        return monitoring_v3.AlertPolicy(
            display_name="WebSocket Connection Spike",
            documentation=monitoring_v3.AlertPolicy.Documentation(
                content="Sudden spike in WebSocket connections. Possible DDoS or system issue."
            ),
            conditions=[
                monitoring_v3.AlertPolicy.Condition(
                    display_name="WebSocket connections spike",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter='metric.type="custom.googleapis.com/parallax_pal/active_websocket_connections" '
                               'resource.type="global"',
                        aggregations=[
                            monitoring_v3.Aggregation(
                                alignment_period={"seconds": 60},
                                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_DELTA
                            )
                        ],
                        comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                        threshold_value=100,  # More than 100 new connections per minute
                        duration={"seconds": 180}
                    )
                )
            ],
            notification_channels=[channels["slack"], channels["pagerduty"]],
            alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
                auto_close={"seconds": 1800}
            )
        )
    
    def _create_agent_failure_policy(self, channels: Dict[str, str]) -> monitoring_v3.AlertPolicy:
        """Create alert policy for agent failures."""
        return monitoring_v3.AlertPolicy(
            display_name="Agent Failures",
            documentation=monitoring_v3.AlertPolicy.Documentation(
                content="Multiple agent failures detected. Check agent configurations and API limits."
            ),
            conditions=[
                monitoring_v3.AlertPolicy.Condition(
                    display_name="Agent failure rate high",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter='metric.type="custom.googleapis.com/parallax_pal/agent_invocations_total" '
                               'metric.label.status="failed" '
                               'resource.type="global"',
                        aggregations=[
                            monitoring_v3.Aggregation(
                                alignment_period={"seconds": 300},
                                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_COUNT
                            )
                        ],
                        comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                        threshold_value=10,  # More than 10 failures in 5 minutes
                        duration={"seconds": 300}
                    )
                )
            ],
            notification_channels=[channels["slack"], channels["email"]],
            alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
                auto_close={"seconds": 3600}
            )
        )
    
    def _create_gpu_usage_policy(self, channels: Dict[str, str]) -> monitoring_v3.AlertPolicy:
        """Create alert policy for GPU usage."""
        return monitoring_v3.AlertPolicy(
            display_name="High GPU Usage",
            documentation=monitoring_v3.AlertPolicy.Documentation(
                content="GPU usage is consistently high. May need to optimize workloads or add GPU resources."
            ),
            conditions=[
                monitoring_v3.AlertPolicy.Condition(
                    display_name="GPU usage > 90%",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter='metric.type="custom.googleapis.com/parallax_pal/gpu_usage_percent" '
                               'metric.label.metric_type="compute" '
                               'resource.type="global"',
                        aggregations=[
                            monitoring_v3.Aggregation(
                                alignment_period={"seconds": 300},
                                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN
                            )
                        ],
                        comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                        threshold_value=90,
                        duration={"seconds": 1800}  # 30 minutes
                    )
                )
            ],
            notification_channels=[channels["slack"]],
            alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
                auto_close={"seconds": 3600}
            )
        )
    
    def _create_token_usage_policy(self, channels: Dict[str, str]) -> monitoring_v3.AlertPolicy:
        """Create alert policy for excessive token usage."""
        return monitoring_v3.AlertPolicy(
            display_name="Excessive Token Usage",
            documentation=monitoring_v3.AlertPolicy.Documentation(
                content="Token usage is unusually high. Check for inefficient queries or potential abuse."
            ),
            conditions=[
                monitoring_v3.AlertPolicy.Condition(
                    display_name="Token usage spike",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter='metric.type="custom.googleapis.com/parallax_pal/agent_token_usage_total" '
                               'resource.type="global"',
                        aggregations=[
                            monitoring_v3.Aggregation(
                                alignment_period={"seconds": 3600},  # 1 hour
                                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_RATE
                            )
                        ],
                        comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                        threshold_value=1000000,  # 1M tokens per hour
                        duration={"seconds": 3600}
                    )
                )
            ],
            notification_channels=[channels["email"], channels["slack"]],
            alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
                auto_close={"seconds": 7200}
            )
        )
    
    def _create_cache_performance_policy(self, channels: Dict[str, str]) -> monitoring_v3.AlertPolicy:
        """Create alert policy for cache performance."""
        return monitoring_v3.AlertPolicy(
            display_name="Low Cache Hit Rate",
            documentation=monitoring_v3.AlertPolicy.Documentation(
                content="Cache hit rate is below 70%. Consider reviewing cache configuration."
            ),
            conditions=[
                monitoring_v3.AlertPolicy.Condition(
                    display_name="Cache hit rate < 70%",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter='metric.type="custom.googleapis.com/parallax_pal/cache_operations_total" '
                               'resource.type="global"',
                        # This would need custom calculation for hit rate
                        aggregations=[
                            monitoring_v3.Aggregation(
                                alignment_period={"seconds": 1800},  # 30 minutes
                                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN
                            )
                        ],
                        comparison=monitoring_v3.ComparisonType.COMPARISON_LT,
                        threshold_value=0.7,
                        duration={"seconds": 1800}
                    )
                )
            ],
            notification_channels=[channels["email"]],
            alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
                auto_close={"seconds": 7200}
            )
        )


# Alert severity levels and escalation paths
ESCALATION_POLICIES = {
    "critical": {
        "channels": ["pagerduty", "slack", "email"],
        "escalation_delay": 300,  # 5 minutes
        "repeat_interval": 900  # 15 minutes
    },
    "warning": {
        "channels": ["slack", "email"],
        "escalation_delay": 1800,  # 30 minutes
        "repeat_interval": 3600  # 1 hour
    },
    "info": {
        "channels": ["email"],
        "escalation_delay": 3600,  # 1 hour
        "repeat_interval": 86400  # 24 hours
    }
}

# Alert routing rules based on time and day
ALERT_ROUTING_RULES = {
    "business_hours": {
        "days": ["MON", "TUE", "WED", "THU", "FRI"],
        "hours": {"start": 9, "end": 17},
        "timezone": "America/Los_Angeles",
        "channels": ["slack", "email"]
    },
    "after_hours": {
        "channels": ["email"],
        "critical_only": True
    },
    "weekends": {
        "days": ["SAT", "SUN"],
        "channels": ["email"],
        "critical_only": True
    }
}