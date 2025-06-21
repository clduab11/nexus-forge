#!/usr/bin/env python3
"""
Deploy monitoring infrastructure for Parallax Pal.

This script:
1. Creates custom metrics in Google Cloud Monitoring
2. Sets up alert policies
3. Creates monitoring dashboard
4. Configures notification channels
"""

import argparse
import os

import yaml
from google.cloud import monitoring_v3
from google.cloud.monitoring_dashboard import v1 as dashboard_v1

from .alert_policies import AlertPolicyManager
from .cloud_monitoring import CloudMonitoringService
from .metrics_config import get_all_metrics


def create_custom_metrics(project_id: str):
    """Create all custom metrics in Google Cloud Monitoring."""
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    created_metrics = []

    for metric_name, metric_def in get_all_metrics().items():
        try:
            # Create metric descriptor
            descriptor = monitoring_v3.MetricDescriptor(
                type=f"custom.googleapis.com/parallax_pal/{metric_name}",
                display_name=metric_def.name,
                description=metric_def.description,
                metric_kind=(
                    monitoring_v3.MetricDescriptor.MetricKind.GAUGE
                    if metric_def.type.value == "gauge"
                    else monitoring_v3.MetricDescriptor.MetricKind.CUMULATIVE
                ),
                value_type=monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
                unit=metric_def.unit,
                labels=[
                    monitoring_v3.LabelDescriptor(
                        key=label,
                        value_type=monitoring_v3.LabelDescriptor.ValueType.STRING,
                    )
                    for label in metric_def.labels
                ],
            )

            created = client.create_metric_descriptor(
                name=project_name, metric_descriptor=descriptor
            )
            created_metrics.append(created.type)
            print(f"✓ Created metric: {metric_name}")

        except Exception as e:
            print(f"✗ Failed to create metric {metric_name}: {e}")

    return created_metrics


def deploy_dashboard(project_id: str):
    """Deploy the monitoring dashboard."""
    dashboard_client = dashboard_v1.DashboardsServiceClient()
    project_name = f"projects/{project_id}"

    # Load dashboard configuration
    dashboard_config_path = os.path.join(
        os.path.dirname(__file__), "dashboard_config.yaml"
    )

    with open(dashboard_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create dashboard
    dashboard = dashboard_v1.Dashboard(config["dashboard"])

    try:
        created = dashboard_client.create_dashboard(
            parent=project_name, dashboard=dashboard
        )
        print(f"✓ Created dashboard: {created.display_name}")
        print(
            f"  View at: https://console.cloud.google.com/monitoring/dashboards/custom/{created.name.split('/')[-1]}"
        )
        return created
    except Exception as e:
        print(f"✗ Failed to create dashboard: {e}")
        return None


def setup_notification_channels(project_id: str):
    """Set up notification channels."""
    manager = AlertPolicyManager(project_id)

    print("\nSetting up notification channels...")
    channels = manager.create_notification_channels()

    for channel_type, channel_name in channels.items():
        if channel_name:
            print(f"✓ Created {channel_type} notification channel")
        else:
            print(f"✗ Failed to create {channel_type} notification channel")

    return channels


def create_alert_policies(project_id: str, notification_channels: dict):
    """Create all alert policies."""
    manager = AlertPolicyManager(project_id)

    print("\nCreating alert policies...")
    policies = manager.create_alert_policies(notification_channels)

    print(f"✓ Created {len(policies)} alert policies")

    return policies


def verify_monitoring_setup(project_id: str):
    """Verify that monitoring is properly set up."""
    print("\nVerifying monitoring setup...")

    monitoring_service = CloudMonitoringService(project_id)

    # Test metric writing
    try:
        monitoring_service.increment_counter(
            "health_check_performed", labels={"check_type": "setup_verification"}
        )
        print("✓ Successfully wrote test metric")
    except Exception as e:
        print(f"✗ Failed to write test metric: {e}")

    # Test trace creation
    try:
        with monitoring_service.create_trace_span("setup_verification"):
            pass
        print("✓ Successfully created test trace")
    except Exception as e:
        print(f"✗ Failed to create test trace: {e}")

    # Test logging
    try:
        monitoring_service.logger.log("Setup verification completed")
        print("✓ Successfully wrote test log")
    except Exception as e:
        print(f"✗ Failed to write test log: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy monitoring infrastructure for Parallax Pal"
    )
    parser.add_argument("--project-id", required=True, help="Google Cloud Project ID")
    parser.add_argument(
        "--skip-metrics", action="store_true", help="Skip creating custom metrics"
    )
    parser.add_argument(
        "--skip-dashboard", action="store_true", help="Skip creating dashboard"
    )
    parser.add_argument(
        "--skip-alerts", action="store_true", help="Skip creating alert policies"
    )
    parser.add_argument(
        "--skip-channels",
        action="store_true",
        help="Skip creating notification channels",
    )

    args = parser.parse_args()

    print(f"Deploying monitoring for project: {args.project_id}")

    # Create custom metrics
    if not args.skip_metrics:
        print("\nCreating custom metrics...")
        metrics = create_custom_metrics(args.project_id)
        print(f"Created {len(metrics)} custom metrics")

    # Set up notification channels
    notification_channels = {}
    if not args.skip_channels:
        notification_channels = setup_notification_channels(args.project_id)

    # Create alert policies
    if not args.skip_alerts and notification_channels:
        create_alert_policies(args.project_id, notification_channels)

    # Deploy dashboard
    if not args.skip_dashboard:
        print("\nDeploying dashboard...")
        deploy_dashboard(args.project_id)

    # Verify setup
    verify_monitoring_setup(args.project_id)

    print("\n✓ Monitoring deployment completed!")
    print("\nNext steps:")
    print(
        "1. Update notification channel settings (email addresses, Slack webhooks, etc.)"
    )
    print("2. Test alert policies by triggering conditions")
    print("3. Customize dashboard layout as needed")
    print("4. Set up log-based metrics for additional insights")


if __name__ == "__main__":
    main()
