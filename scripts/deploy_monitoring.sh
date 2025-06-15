#!/bin/bash
set -euo pipefail

# Monitoring Stack Deployment Script for Nexus Forge
# Sets up comprehensive monitoring, alerting, and observability

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-nexus-forge}"
NOTIFICATION_EMAIL="${NOTIFICATION_EMAIL:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "\n${BLUE}===== $1 =====${NC}"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed."
        exit 1
    fi
    
    if [ -z "$PROJECT_ID" ]; then
        log_error "GCP_PROJECT_ID environment variable is not set."
        exit 1
    fi
    
    if [ -z "$NOTIFICATION_EMAIL" ]; then
        log_warn "NOTIFICATION_EMAIL not set. Alerts will not be sent via email."
    fi
    
    log_info "Prerequisites check passed."
}

setup_notification_channels() {
    log_section "Setting up Notification Channels"
    
    if [ -n "$NOTIFICATION_EMAIL" ]; then
        # Create email notification channel
        EMAIL_CHANNEL_NAME="nexus-forge-email-alerts"
        
        # Check if channel already exists
        if ! gcloud alpha monitoring channels list \
            --filter="displayName:${EMAIL_CHANNEL_NAME}" \
            --format="value(name)" | head -1; then
            
            log_info "Creating email notification channel..."
            
            cat > /tmp/email_channel.json << EOF
{
  "type": "email",
  "displayName": "${EMAIL_CHANNEL_NAME}",
  "description": "Email notifications for Nexus Forge alerts",
  "labels": {
    "email_address": "${NOTIFICATION_EMAIL}"
  },
  "enabled": true
}
EOF
            
            gcloud alpha monitoring channels create \
                --channel-content-from-file=/tmp/email_channel.json
            
            rm /tmp/email_channel.json
            log_info "Email notification channel created."
        else
            log_info "Email notification channel already exists."
        fi
    fi
    
    # Create Pub/Sub notification channel for integration with other systems
    PUBSUB_CHANNEL_NAME="nexus-forge-pubsub-alerts"
    ALERT_TOPIC="nexus-forge-alert-notifications"
    
    # Create Pub/Sub topic if it doesn't exist
    if ! gcloud pubsub topics describe "$ALERT_TOPIC" &> /dev/null; then
        log_info "Creating alert notification topic: $ALERT_TOPIC"
        gcloud pubsub topics create "$ALERT_TOPIC"
    fi
    
    # Create Pub/Sub notification channel
    if ! gcloud alpha monitoring channels list \
        --filter="displayName:${PUBSUB_CHANNEL_NAME}" \
        --format="value(name)" | head -1; then
        
        log_info "Creating Pub/Sub notification channel..."
        
        cat > /tmp/pubsub_channel.json << EOF
{
  "type": "pubsub",
  "displayName": "${PUBSUB_CHANNEL_NAME}",
  "description": "Pub/Sub notifications for Nexus Forge alerts",
  "labels": {
    "topic": "projects/${PROJECT_ID}/topics/${ALERT_TOPIC}"
  },
  "enabled": true
}
EOF
        
        gcloud alpha monitoring channels create \
            --channel-content-from-file=/tmp/pubsub_channel.json
        
        rm /tmp/pubsub_channel.json
        log_info "Pub/Sub notification channel created."
    else
        log_info "Pub/Sub notification channel already exists."
    fi
}

create_alert_policies() {
    log_section "Creating Alert Policies"
    
    # Get notification channel names
    EMAIL_CHANNEL=""
    if [ -n "$NOTIFICATION_EMAIL" ]; then
        EMAIL_CHANNEL=$(gcloud alpha monitoring channels list \
            --filter="displayName:nexus-forge-email-alerts" \
            --format="value(name)" | head -1)
    fi
    
    PUBSUB_CHANNEL=$(gcloud alpha monitoring channels list \
        --filter="displayName:nexus-forge-pubsub-alerts" \
        --format="value(name)" | head -1)
    
    # Build notification channels array
    CHANNELS="[]"
    if [ -n "$EMAIL_CHANNEL" ] && [ -n "$PUBSUB_CHANNEL" ]; then
        CHANNELS="[\"$EMAIL_CHANNEL\", \"$PUBSUB_CHANNEL\"]"
    elif [ -n "$EMAIL_CHANNEL" ]; then
        CHANNELS="[\"$EMAIL_CHANNEL\"]"
    elif [ -n "$PUBSUB_CHANNEL" ]; then
        CHANNELS="[\"$PUBSUB_CHANNEL\"]"
    fi
    
    # 1. High Error Rate Alert
    log_info "Creating high error rate alert policy..."
    cat > /tmp/error_rate_policy.json << EOF
{
  "displayName": "Nexus Forge - High Error Rate",
  "documentation": {
    "content": "Alert when error rate exceeds 5% over 5 minutes",
    "mimeType": "text/markdown"
  },
  "conditions": [
    {
      "displayName": "Error rate condition",
      "conditionThreshold": {
        "filter": "resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${SERVICE_NAME}\" AND metric.type=\"run.googleapis.com/request_count\"",
        "aggregations": [
          {
            "alignmentPeriod": "300s",
            "perSeriesAligner": "ALIGN_RATE",
            "crossSeriesReducer": "REDUCE_SUM",
            "groupByFields": ["resource.label.service_name"]
          }
        ],
        "comparison": "COMPARISON_GREATER_THAN",
        "thresholdValue": 0.05,
        "duration": "300s",
        "evaluationMissingData": "EVALUATION_MISSING_DATA_INACTIVE"
      }
    }
  ],
  "alertStrategy": {
    "autoClose": "1800s"
  },
  "combiner": "OR",
  "enabled": true,
  "notificationChannels": ${CHANNELS}
}
EOF
    
    gcloud alpha monitoring policies create \
        --policy-from-file=/tmp/error_rate_policy.json || true
    
    # 2. High Memory Usage Alert
    log_info "Creating high memory usage alert policy..."
    cat > /tmp/memory_policy.json << EOF
{
  "displayName": "Nexus Forge - High Memory Usage",
  "documentation": {
    "content": "Alert when memory usage exceeds 80% for 5 minutes",
    "mimeType": "text/markdown"
  },
  "conditions": [
    {
      "displayName": "Memory usage condition",
      "conditionThreshold": {
        "filter": "resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${SERVICE_NAME}\" AND metric.type=\"run.googleapis.com/container/memory/utilizations\"",
        "aggregations": [
          {
            "alignmentPeriod": "300s",
            "perSeriesAligner": "ALIGN_MEAN",
            "crossSeriesReducer": "REDUCE_MEAN"
          }
        ],
        "comparison": "COMPARISON_GREATER_THAN",
        "thresholdValue": 0.8,
        "duration": "300s"
      }
    }
  ],
  "alertStrategy": {
    "autoClose": "1800s"
  },
  "combiner": "OR",
  "enabled": true,
  "notificationChannels": ${CHANNELS}
}
EOF
    
    gcloud alpha monitoring policies create \
        --policy-from-file=/tmp/memory_policy.json || true
    
    # 3. High CPU Usage Alert
    log_info "Creating high CPU usage alert policy..."
    cat > /tmp/cpu_policy.json << EOF
{
  "displayName": "Nexus Forge - High CPU Usage",
  "documentation": {
    "content": "Alert when CPU usage exceeds 80% for 5 minutes",
    "mimeType": "text/markdown"
  },
  "conditions": [
    {
      "displayName": "CPU usage condition",
      "conditionThreshold": {
        "filter": "resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${SERVICE_NAME}\" AND metric.type=\"run.googleapis.com/container/cpu/utilizations\"",
        "aggregations": [
          {
            "alignmentPeriod": "300s",
            "perSeriesAligner": "ALIGN_MEAN",
            "crossSeriesReducer": "REDUCE_MEAN"
          }
        ],
        "comparison": "COMPARISON_GREATER_THAN",
        "thresholdValue": 0.8,
        "duration": "300s"
      }
    }
  ],
  "alertStrategy": {
    "autoClose": "1800s"
  },
  "combiner": "OR",
  "enabled": true,
  "notificationChannels": ${CHANNELS}
}
EOF
    
    gcloud alpha monitoring policies create \
        --policy-from-file=/tmp/cpu_policy.json || true
    
    # 4. High Response Time Alert
    log_info "Creating high response time alert policy..."
    cat > /tmp/latency_policy.json << EOF
{
  "displayName": "Nexus Forge - High Response Time",
  "documentation": {
    "content": "Alert when 95th percentile response time exceeds 2 seconds",
    "mimeType": "text/markdown"
  },
  "conditions": [
    {
      "displayName": "Response time condition",
      "conditionThreshold": {
        "filter": "resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${SERVICE_NAME}\" AND metric.type=\"run.googleapis.com/request_latencies\"",
        "aggregations": [
          {
            "alignmentPeriod": "300s",
            "perSeriesAligner": "ALIGN_DELTA",
            "crossSeriesReducer": "REDUCE_PERCENTILE_95"
          }
        ],
        "comparison": "COMPARISON_GREATER_THAN",
        "thresholdValue": 2000,
        "duration": "300s"
      }
    }
  ],
  "alertStrategy": {
    "autoClose": "1800s"
  },
  "combiner": "OR",
  "enabled": true,
  "notificationChannels": ${CHANNELS}
}
EOF
    
    gcloud alpha monitoring policies create \
        --policy-from-file=/tmp/latency_policy.json || true
    
    # 5. Service Down Alert
    log_info "Creating service down alert policy..."
    cat > /tmp/uptime_policy.json << EOF
{
  "displayName": "Nexus Forge - Service Down",
  "documentation": {
    "content": "Alert when service is down or not responding",
    "mimeType": "text/markdown"
  },
  "conditions": [
    {
      "displayName": "Service uptime condition",
      "conditionThreshold": {
        "filter": "resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${SERVICE_NAME}\" AND metric.type=\"run.googleapis.com/request_count\"",
        "aggregations": [
          {
            "alignmentPeriod": "300s",
            "perSeriesAligner": "ALIGN_RATE",
            "crossSeriesReducer": "REDUCE_SUM"
          }
        ],
        "comparison": "COMPARISON_LESS_THAN",
        "thresholdValue": 0.01,
        "duration": "300s"
      }
    }
  ],
  "alertStrategy": {
    "autoClose": "900s"
  },
  "combiner": "OR",
  "enabled": true,
  "notificationChannels": ${CHANNELS}
}
EOF
    
    gcloud alpha monitoring policies create \
        --policy-from-file=/tmp/uptime_policy.json || true
    
    # Clean up temp files
    rm -f /tmp/*_policy.json
    
    log_info "Alert policies created successfully."
}

create_uptime_checks() {
    log_section "Creating Uptime Checks"
    
    # Get the Cloud Run service URL
    SERVICE_URL=""
    if gcloud run services describe "$SERVICE_NAME" --region="$REGION" &> /dev/null; then
        SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
            --region="$REGION" \
            --format="value(status.url)")
    fi
    
    if [ -n "$SERVICE_URL" ]; then
        log_info "Creating uptime check for: $SERVICE_URL"
        
        cat > /tmp/uptime_check.json << EOF
{
  "displayName": "Nexus Forge Health Check",
  "monitoredResource": {
    "type": "uptime_url",
    "labels": {
      "project_id": "${PROJECT_ID}",
      "host": "$(echo $SERVICE_URL | sed 's|https\?://||' | cut -d'/' -f1)"
    }
  },
  "httpCheck": {
    "path": "/health",
    "port": 443,
    "useSsl": true,
    "validateSsl": true
  },
  "period": "300s",
  "timeout": "10s",
  "contentMatchers": [
    {
      "content": "healthy",
      "matcher": "CONTAINS_STRING"
    }
  ],
  "selectedRegions": [
    "USA",
    "EUROPE",
    "ASIA_PACIFIC"
  ]
}
EOF
        
        gcloud monitoring uptime create \
            --uptime-check-config-from-file=/tmp/uptime_check.json || true
        
        rm /tmp/uptime_check.json
        
        log_info "Uptime check created successfully."
    else
        log_warn "Service not deployed yet. Skipping uptime check creation."
        log_info "Run this script again after deploying the service."
    fi
}

create_custom_dashboards() {
    log_section "Creating Custom Dashboards"
    
    log_info "Creating Nexus Forge monitoring dashboard..."
    
    cat > /tmp/dashboard.json << EOF
{
  "displayName": "Nexus Forge - Production Monitoring",
  "mosaicLayout": {
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Request Rate",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${SERVICE_NAME}\" AND metric.type=\"run.googleapis.com/request_count\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE",
                      "crossSeriesReducer": "REDUCE_SUM"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              }
            ],
            "timeshiftDuration": "0s",
            "yAxis": {
              "label": "Requests/sec",
              "scale": "LINEAR"
            }
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "xPos": 6,
        "widget": {
          "title": "Error Rate",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${SERVICE_NAME}\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.label.response_code_class!=\"2xx\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE",
                      "crossSeriesReducer": "REDUCE_SUM"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              }
            ],
            "timeshiftDuration": "0s",
            "yAxis": {
              "label": "Errors/sec",
              "scale": "LINEAR"
            }
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "yPos": 4,
        "widget": {
          "title": "Response Time (95th percentile)",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${SERVICE_NAME}\" AND metric.type=\"run.googleapis.com/request_latencies\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_DELTA",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              }
            ],
            "timeshiftDuration": "0s",
            "yAxis": {
              "label": "Latency (ms)",
              "scale": "LINEAR"
            }
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "xPos": 6,
        "yPos": 4,
        "widget": {
          "title": "Memory Usage",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${SERVICE_NAME}\" AND metric.type=\"run.googleapis.com/container/memory/utilizations\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MEAN",
                      "crossSeriesReducer": "REDUCE_MEAN"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              }
            ],
            "timeshiftDuration": "0s",
            "yAxis": {
              "label": "Memory %",
              "scale": "LINEAR"
            }
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "yPos": 8,
        "widget": {
          "title": "CPU Usage",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${SERVICE_NAME}\" AND metric.type=\"run.googleapis.com/container/cpu/utilizations\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MEAN",
                      "crossSeriesReducer": "REDUCE_MEAN"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              }
            ],
            "timeshiftDuration": "0s",
            "yAxis": {
              "label": "CPU %",
              "scale": "LINEAR"
            }
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "xPos": 6,
        "yPos": 8,
        "widget": {
          "title": "Active Instances",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\" AND resource.label.service_name=\"${SERVICE_NAME}\" AND metric.type=\"run.googleapis.com/container/instance_count\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MEAN",
                      "crossSeriesReducer": "REDUCE_SUM"
                    }
                  }
                },
                "plotType": "LINE",
                "targetAxis": "Y1"
              }
            ],
            "timeshiftDuration": "0s",
            "yAxis": {
              "label": "Instances",
              "scale": "LINEAR"
            }
          }
        }
      }
    ]
  }
}
EOF
    
    gcloud monitoring dashboards create \
        --config-from-file=/tmp/dashboard.json || true
    
    rm /tmp/dashboard.json
    
    log_info "Dashboard created successfully."
}

setup_log_based_metrics() {
    log_section "Setting up Log-based Metrics"
    
    # Error count metric
    log_info "Creating error count log-based metric..."
    gcloud logging metrics create nexus_forge_error_count \
        --description="Count of error logs from Nexus Forge" \
        --log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="nexus-forge" AND severity>=ERROR' || true
    
    # API call duration metric
    log_info "Creating API call duration log-based metric..."
    gcloud logging metrics create nexus_forge_api_duration \
        --description="API call duration from Nexus Forge logs" \
        --log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="nexus-forge" AND jsonPayload.duration' \
        --value-extractor='EXTRACT(jsonPayload.duration)' || true
    
    # User activity metric
    log_info "Creating user activity log-based metric..."
    gcloud logging metrics create nexus_forge_user_activity \
        --description="User activity events from Nexus Forge" \
        --log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="nexus-forge" AND jsonPayload.event_type="user_action"' || true
    
    log_info "Log-based metrics created successfully."
}

create_scheduled_jobs() {
    log_section "Setting up Scheduled Monitoring Jobs"
    
    # Health check job
    log_info "Creating health check scheduled job..."
    
    SERVICE_URL=""
    if gcloud run services describe "$SERVICE_NAME" --region="$REGION" &> /dev/null; then
        SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
            --region="$REGION" \
            --format="value(status.url)")
        
        gcloud scheduler jobs create http nexus-forge-health-check \
            --schedule="*/5 * * * *" \
            --uri="${SERVICE_URL}/health" \
            --http-method=GET \
            --time-zone="UTC" \
            --description="Regular health check for Nexus Forge service" || true
    else
        log_warn "Service not deployed. Skipping health check job creation."
    fi
    
    # Metrics collection job
    log_info "Creating metrics collection job..."
    gcloud scheduler jobs create pubsub nexus-forge-metrics-collection \
        --schedule="0 */6 * * *" \
        --topic="nexus-forge-metrics" \
        --message-body='{"action": "collect_metrics", "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}' \
        --time-zone="UTC" \
        --description="Collect and aggregate custom metrics" || true
    
    log_info "Scheduled jobs created successfully."
}

print_monitoring_summary() {
    log_section "Monitoring Setup Summary"
    
    log_info "Monitoring stack deployed successfully!"
    log_info ""
    log_info "Created Components:"
    log_info "- Notification channels (email, Pub/Sub)"
    log_info "- Alert policies (error rate, memory, CPU, latency, uptime)"
    log_info "- Uptime checks"
    log_info "- Custom dashboard"
    log_info "- Log-based metrics"
    log_info "- Scheduled monitoring jobs"
    log_info ""
    log_info "Access your monitoring:"
    log_info "- Dashboards: https://console.cloud.google.com/monitoring/dashboards"
    log_info "- Alerts: https://console.cloud.google.com/monitoring/alerting"
    log_info "- Uptime checks: https://console.cloud.google.com/monitoring/uptime"
    log_info "- Logs: https://console.cloud.google.com/logs"
    log_info ""
    
    if [ -n "$NOTIFICATION_EMAIL" ]; then
        log_info "Email alerts will be sent to: $NOTIFICATION_EMAIL"
    else
        log_warn "No email configured for alerts. Set NOTIFICATION_EMAIL environment variable."
    fi
}

main() {
    log_info "Starting monitoring stack deployment for Nexus Forge..."
    
    check_prerequisites
    gcloud config set project "$PROJECT_ID"
    
    setup_notification_channels
    create_alert_policies
    create_uptime_checks
    create_custom_dashboards
    setup_log_based_metrics
    create_scheduled_jobs
    print_monitoring_summary
    
    log_info "Monitoring deployment completed successfully!"
}

# Run main function
main "$@"