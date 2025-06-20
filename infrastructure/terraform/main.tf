# Nexus Forge Enterprise Infrastructure - Terraform Configuration
# Multi-region, highly available deployment on Google Cloud Platform

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  backend "gcs" {
    bucket = "nexus-forge-terraform-state"
    prefix = "terraform/state"
  }
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "primary_region" {
  description = "Primary GCP region"
  type        = string
  default     = "us-central1"
}

variable "secondary_region" {
  description = "Secondary GCP region for DR"
  type        = string
  default     = "us-east1"
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "nexusforge.ai"
}

# Provider Configuration
provider "google" {
  project = var.project_id
  region  = var.primary_region
}

provider "google-beta" {
  project = var.project_id
  region  = var.primary_region
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "cloudrun.googleapis.com",
    "cloudsql.googleapis.com",
    "redis.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudkms.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "aiplatform.googleapis.com",
    "iap.googleapis.com",
    "cloudarmor.googleapis.com",
  ])
  
  service = each.key
  
  disable_on_destroy = false
}

# KMS for encryption
resource "google_kms_key_ring" "nexus_forge" {
  name     = "nexus-forge-keyring"
  location = "global"
}

resource "google_kms_crypto_key" "data_encryption" {
  name     = "data-encryption-key"
  key_ring = google_kms_key_ring.nexus_forge.id
  
  rotation_period = "7776000s" # 90 days
  
  lifecycle {
    prevent_destroy = true
  }
}

# VPC Network
module "vpc" {
  source = "./modules/vpc"
  
  project_id   = var.project_id
  network_name = "nexus-forge-vpc"
  
  regions = {
    primary   = var.primary_region
    secondary = var.secondary_region
  }
  
  enable_flow_logs = true
  enable_private_google_access = true
}

# GKE Clusters (Multi-region)
module "gke_primary" {
  source = "./modules/gke"
  
  project_id     = var.project_id
  cluster_name   = "nexus-forge-primary"
  region         = var.primary_region
  network        = module.vpc.network_name
  subnetwork     = module.vpc.subnets["primary"].name
  
  node_pools = {
    default = {
      machine_type = "n2-standard-4"
      min_count    = 3
      max_count    = 10
      disk_size_gb = 100
      disk_type    = "pd-ssd"
      
      labels = {
        workload = "general"
      }
    }
    
    gpu = {
      machine_type = "n1-standard-4"
      min_count    = 0
      max_count    = 5
      disk_size_gb = 100
      
      accelerator_type  = "nvidia-tesla-t4"
      accelerator_count = 1
      
      labels = {
        workload = "ai-processing"
      }
      
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "present"
        effect = "NO_SCHEDULE"
      }]
    }
  }
  
  enable_autopilot      = false
  enable_private_nodes  = true
  enable_shielded_nodes = true
  enable_workload_identity = true
  
  cluster_autoscaling = {
    enabled = true
    resource_limits = {
      cpu = {
        minimum = 10
        maximum = 1000
      }
      memory = {
        minimum = 40
        maximum = 4000
      }
    }
  }
}

module "gke_secondary" {
  source = "./modules/gke"
  
  project_id     = var.project_id
  cluster_name   = "nexus-forge-secondary"
  region         = var.secondary_region
  network        = module.vpc.network_name
  subnetwork     = module.vpc.subnets["secondary"].name
  
  node_pools = {
    default = {
      machine_type = "n2-standard-4"
      min_count    = 2
      max_count    = 8
      disk_size_gb = 100
      disk_type    = "pd-ssd"
      
      labels = {
        workload = "general"
      }
    }
  }
  
  enable_autopilot      = false
  enable_private_nodes  = true
  enable_shielded_nodes = true
  enable_workload_identity = true
}

# Cloud SQL (Multi-region with HA)
module "cloud_sql" {
  source = "./modules/cloud_sql"
  
  project_id        = var.project_id
  instance_name     = "nexus-forge-db"
  database_version  = "POSTGRES_15"
  region           = var.primary_region
  tier             = "db-custom-4-16384"
  
  high_availability = {
    enabled                     = true
    location_preference_zone    = "${var.primary_region}-a"
    secondary_zone              = "${var.primary_region}-b"
    point_in_time_recovery      = true
    transaction_log_retention_days = 7
  }
  
  backup_configuration = {
    enabled                        = true
    start_time                     = "02:00"
    point_in_time_recovery_enabled = true
    transaction_log_retention_days = 7
    retained_backups              = 30
    retention_unit                = "COUNT"
  }
  
  replica_configuration = {
    enabled            = true
    replica_region     = var.secondary_region
    replica_tier       = "db-custom-2-8192"
    failover_enabled   = true
  }
  
  encryption_key_name = google_kms_crypto_key.data_encryption.id
  
  database_flags = {
    "cloudsql.enable_pgaudit" = "on"
    "log_statement"           = "all"
    "log_checkpoints"         = "on"
  }
}

# Redis Cluster (Memorystore)
module "redis" {
  source = "./modules/redis"
  
  project_id    = var.project_id
  instance_name = "nexus-forge-cache"
  region        = var.primary_region
  
  memory_size_gb = 16
  replica_count  = 2
  
  redis_configs = {
    "maxmemory-policy"  = "allkeys-lru"
    "notify-keyspace-events" = "Ex"
  }
  
  enable_auth = true
  transit_encryption_mode = "SERVER_AUTHENTICATION"
}

# Cloud Storage Buckets
module "storage" {
  source = "./modules/storage"
  
  project_id = var.project_id
  
  buckets = {
    assets = {
      name               = "${var.project_id}-nexus-forge-assets"
      location           = "US"
      storage_class      = "STANDARD"
      enable_versioning  = true
      enable_encryption  = true
      encryption_key     = google_kms_crypto_key.data_encryption.id
      
      lifecycle_rules = [{
        action = {
          type = "SetStorageClass"
          storage_class = "NEARLINE"
        }
        condition = {
          age = 30
        }
      }]
    }
    
    backups = {
      name               = "${var.project_id}-nexus-forge-backups"
      location           = "US"
      storage_class      = "NEARLINE"
      enable_versioning  = true
      enable_encryption  = true
      encryption_key     = google_kms_crypto_key.data_encryption.id
      
      lifecycle_rules = [{
        action = {
          type = "Delete"
        }
        condition = {
          age = 90
        }
      }]
    }
  }
}

# Global Load Balancer
module "load_balancer" {
  source = "./modules/load_balancer"
  
  project_id = var.project_id
  name       = "nexus-forge-lb"
  
  backends = {
    primary = {
      group = module.gke_primary.instance_group
      capacity_scaler = 1.0
      max_utilization = 0.8
    }
    secondary = {
      group = module.gke_secondary.instance_group
      capacity_scaler = 0.5
      max_utilization = 0.8
    }
  }
  
  ssl_certificates = [module.ssl.certificate_id]
  
  security_policy = module.cloud_armor.policy_id
  
  enable_cdn = true
  cdn_policy = {
    cache_mode = "CACHE_ALL_STATIC"
    default_ttl = 3600
    max_ttl     = 86400
    
    negative_caching = true
    negative_caching_policy = [{
      code = 404
      ttl  = 300
    }]
  }
  
  custom_headers = {
    "X-Frame-Options"        = "DENY"
    "X-Content-Type-Options" = "nosniff"
    "X-XSS-Protection"       = "1; mode=block"
    "Referrer-Policy"        = "strict-origin-when-cross-origin"
  }
}

# Cloud Armor (WAF)
module "cloud_armor" {
  source = "./modules/cloud_armor"
  
  project_id  = var.project_id
  policy_name = "nexus-forge-security-policy"
  
  rules = {
    rate_limit = {
      priority = 1000
      action   = "rate_based_ban"
      
      rate_limit_options = {
        rate_limit_threshold = {
          count        = 100
          interval_sec = 60
        }
        ban_duration_sec = 600
        conform_action   = "allow"
        exceed_action    = "deny(429)"
      }
    }
    
    owasp_rules = {
      priority = 2000
      action   = "deny(403)"
      
      expression = {
        expression = "evaluatePreconfiguredExpr('xss-stable') || evaluatePreconfiguredExpr('sqli-stable')"
      }
    }
    
    geo_blocking = {
      priority = 3000
      action   = "deny(403)"
      
      expression = {
        expression = "origin.region_code in ['XX', 'YY']"
      }
    }
  }
  
  adaptive_protection = {
    enabled = true
  }
}

# SSL Certificate
module "ssl" {
  source = "./modules/ssl"
  
  project_id  = var.project_id
  domain_name = var.domain_name
  
  dns_authorization = true
}

# Monitoring and Alerting
module "monitoring" {
  source = "./modules/monitoring"
  
  project_id = var.project_id
  
  notification_channels = {
    email = {
      type   = "email"
      labels = {
        email_address = "soc@${var.domain_name}"
      }
    }
    
    pagerduty = {
      type   = "pagerduty"
      labels = {
        service_key = data.google_secret_manager_secret_version.pagerduty_key.secret_data
      }
    }
  }
  
  uptime_checks = {
    api = {
      display_name   = "API Health Check"
      host           = "api.${var.domain_name}"
      path           = "/health"
      port           = 443
      use_ssl        = true
      check_interval = "60s"
    }
  }
  
  alert_policies = {
    high_error_rate = {
      display_name = "High Error Rate"
      conditions = [{
        display_name = "Error rate > 5%"
        condition_threshold = {
          filter          = "metric.type=\"logging.googleapis.com/user/error_rate\""
          duration        = "300s"
          comparison      = "COMPARISON_GT"
          threshold_value = 5
        }
      }]
      notification_channels = ["email", "pagerduty"]
    }
    
    database_connection_pool = {
      display_name = "Database Connection Pool Exhausted"
      conditions = [{
        display_name = "Connection pool > 90%"
        condition_threshold = {
          filter          = "metric.type=\"cloudsql.googleapis.com/database/postgresql/num_backends_by_state\""
          duration        = "180s"
          comparison      = "COMPARISON_GT"
          threshold_value = 90
        }
      }]
      notification_channels = ["email"]
    }
  }
  
  dashboards = {
    main = {
      display_name = "Nexus Forge Main Dashboard"
      grid_layout = {
        widgets = [
          {
            title = "Request Rate"
            xy_chart = {
              data_sets = [{
                time_series_query = {
                  time_series_filter = {
                    filter = "metric.type=\"loadbalancing.googleapis.com/https/request_count\""
                  }
                }
              }]
            }
          },
          {
            title = "Error Rate"
            xy_chart = {
              data_sets = [{
                time_series_query = {
                  time_series_filter = {
                    filter = "metric.type=\"logging.googleapis.com/user/error_rate\""
                  }
                }
              }]
            }
          }
        ]
      }
    }
  }
}

# Service Accounts and IAM
module "iam" {
  source = "./modules/iam"
  
  project_id = var.project_id
  
  service_accounts = {
    nexus_forge_app = {
      account_id   = "nexus-forge-app"
      display_name = "Nexus Forge Application"
      roles = [
        "roles/aiplatform.user",
        "roles/cloudsql.client",
        "roles/storage.objectAdmin",
        "roles/secretmanager.secretAccessor",
        "roles/cloudkms.cryptoKeyEncrypterDecrypter",
      ]
    }
    
    nexus_forge_ci = {
      account_id   = "nexus-forge-ci"
      display_name = "Nexus Forge CI/CD"
      roles = [
        "roles/container.developer",
        "roles/storage.admin",
        "roles/cloudbuild.builds.editor",
      ]
    }
  }
  
  workload_identity_bindings = {
    app = {
      service_account = "nexus-forge-app"
      namespace       = "default"
      ksa_name        = "nexus-forge-ksa"
    }
  }
}

# Secrets Management
module "secrets" {
  source = "./modules/secrets"
  
  project_id = var.project_id
  
  secrets = {
    database_password = {
      secret_id = "nexus-forge-db-password"
      data      = random_password.db_password.result
    }
    
    redis_password = {
      secret_id = "nexus-forge-redis-password"
      data      = random_password.redis_password.result
    }
    
    jwt_secret = {
      secret_id = "nexus-forge-jwt-secret"
      data      = random_password.jwt_secret.result
    }
  }
  
  encryption_key = google_kms_crypto_key.data_encryption.id
}

# Random passwords
resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "random_password" "redis_password" {
  length  = 32
  special = true
}

resource "random_password" "jwt_secret" {
  length  = 64
  special = true
}

# Outputs
output "load_balancer_ip" {
  value       = module.load_balancer.external_ip
  description = "External IP address of the load balancer"
}

output "gke_cluster_endpoints" {
  value = {
    primary   = module.gke_primary.endpoint
    secondary = module.gke_secondary.endpoint
  }
  description = "GKE cluster endpoints"
}

output "database_connection_name" {
  value       = module.cloud_sql.connection_name
  description = "Cloud SQL connection name"
}

output "redis_host" {
  value       = module.redis.host
  description = "Redis instance host"
}

output "storage_buckets" {
  value = module.storage.bucket_urls
  description = "Storage bucket URLs"
}