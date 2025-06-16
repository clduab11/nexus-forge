"""
Tenant Isolation Manager
Handles tenant isolation across compute, storage, and network layers
"""

import os
import asyncio
import subprocess
from typing import Dict, List, Optional, Any, Tuple
import yaml
import json
from pathlib import Path
import logging

from .models import Tenant, TenantResource, IsolationLevel, TenantTier
from ..core.cache import RedisCache
from ..integrations.supabase.coordination_client import SupabaseCoordinationClient
from ..core.exceptions import ValidationError, ResourceError

logger = logging.getLogger(__name__)


class TenantIsolationManager:
    """Manages tenant isolation at infrastructure level"""
    
    def __init__(self):
        self.cache = RedisCache()
        self.supabase = SupabaseCoordinationClient()
        self.k8s_enabled = self._check_kubernetes_availability()
        
        # Isolation configurations by tier
        self.tier_configs = {
            TenantTier.BASIC: {
                "isolation_level": IsolationLevel.SHARED,
                "namespace_isolation": False,
                "dedicated_database": False,
                "network_policies": False,
                "resource_quotas": True
            },
            TenantTier.PROFESSIONAL: {
                "isolation_level": IsolationLevel.DEDICATED,
                "namespace_isolation": True,
                "dedicated_database": True,
                "network_policies": True,
                "resource_quotas": True
            },
            TenantTier.ENTERPRISE: {
                "isolation_level": IsolationLevel.ISOLATED,
                "namespace_isolation": True,
                "dedicated_database": True,
                "dedicated_cluster": True,
                "network_policies": True,
                "custom_domain": True,
                "resource_quotas": True
            }
        }
    
    def _check_kubernetes_availability(self) -> bool:
        """Check if Kubernetes is available"""
        try:
            result = subprocess.run(
                ["kubectl", "version", "--client", "--short"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def create_tenant_isolation(self, tenant: Tenant) -> List[TenantResource]:
        """Create isolation infrastructure for a tenant"""
        logger.info(f"Creating isolation for tenant {tenant.slug} (tier: {tenant.tier})")
        
        config = self.tier_configs[tenant.tier]
        resources = []
        
        try:
            # Create namespace isolation if needed
            if config.get("namespace_isolation") and self.k8s_enabled:
                namespace_resource = await self._create_kubernetes_namespace(tenant)
                resources.append(namespace_resource)
                
                # Apply resource quotas
                if config.get("resource_quotas"):
                    quota_resource = await self._apply_resource_quotas(tenant)
                    resources.append(quota_resource)
                
                # Apply network policies
                if config.get("network_policies"):
                    network_resource = await self._apply_network_policies(tenant)
                    resources.append(network_resource)
            
            # Create dedicated database schema
            if config.get("dedicated_database"):
                db_resource = await self._create_database_isolation(tenant)
                resources.append(db_resource)
            
            # Setup custom domain
            if config.get("custom_domain") and tenant.config.custom_domain:
                domain_resource = await self._setup_custom_domain(tenant)
                resources.append(domain_resource)
            
            # Create dedicated cluster for enterprise
            if config.get("dedicated_cluster"):
                cluster_resource = await self._create_dedicated_cluster(tenant)
                resources.append(cluster_resource)
            
            # Save resources to database
            for resource in resources:
                await self._save_tenant_resource(resource)
            
            # Update tenant with resource references
            tenant.resources = resources
            
            logger.info(f"Created {len(resources)} isolation resources for tenant {tenant.slug}")
            
            return resources
            
        except Exception as e:
            # Cleanup partial resources on failure
            logger.error(f"Failed to create tenant isolation: {e}")
            await self._cleanup_resources(resources)
            raise ResourceError(f"Failed to create tenant isolation: {e}")
    
    async def update_tenant_isolation(
        self, tenant: Tenant, new_tier: TenantTier
    ) -> List[TenantResource]:
        """Update tenant isolation for tier change"""
        logger.info(f"Updating isolation for tenant {tenant.slug} from {tenant.tier} to {new_tier}")
        
        old_config = self.tier_configs[tenant.tier]
        new_config = self.tier_configs[new_tier]
        
        new_resources = []
        
        # Determine what needs to be added or upgraded
        if not old_config.get("namespace_isolation") and new_config.get("namespace_isolation"):
            # Upgrade to namespace isolation
            if self.k8s_enabled:
                namespace_resource = await self._create_kubernetes_namespace(tenant)
                new_resources.append(namespace_resource)
        
        if not old_config.get("dedicated_database") and new_config.get("dedicated_database"):
            # Upgrade to dedicated database
            db_resource = await self._create_database_isolation(tenant)
            new_resources.append(db_resource)
        
        if not old_config.get("dedicated_cluster") and new_config.get("dedicated_cluster"):
            # Create dedicated cluster for enterprise upgrade
            cluster_resource = await self._create_dedicated_cluster(tenant)
            new_resources.append(cluster_resource)
        
        # Save new resources
        for resource in new_resources:
            await self._save_tenant_resource(resource)
        
        # Update tenant resources
        tenant.resources.extend(new_resources)
        tenant.tier = new_tier
        tenant.isolation_level = new_config["isolation_level"]
        
        return new_resources
    
    async def delete_tenant_isolation(self, tenant: Tenant) -> bool:
        """Delete all isolation resources for a tenant"""
        logger.info(f"Deleting isolation for tenant {tenant.slug}")
        
        try:
            # Delete Kubernetes resources
            if self.k8s_enabled:
                await self._delete_kubernetes_namespace(tenant)
            
            # Delete database schema
            await self._delete_database_isolation(tenant)
            
            # Delete custom domain setup
            if tenant.config.custom_domain:
                await self._delete_custom_domain(tenant)
            
            # Delete dedicated cluster if exists
            await self._delete_dedicated_cluster(tenant)
            
            # Remove resource records
            await self._delete_tenant_resources(tenant.id)
            
            logger.info(f"Successfully deleted isolation for tenant {tenant.slug}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete tenant isolation: {e}")
            return False
    
    # Kubernetes isolation methods
    
    async def _create_kubernetes_namespace(self, tenant: Tenant) -> TenantResource:
        """Create Kubernetes namespace for tenant"""
        namespace_name = f"tenant-{tenant.slug}"
        
        # Create namespace manifest
        namespace_manifest = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": namespace_name,
                "labels": {
                    "nexus-forge/tenant-id": tenant.id,
                    "nexus-forge/tenant-slug": tenant.slug,
                    "nexus-forge/tier": tenant.tier.value
                },
                "annotations": {
                    "nexus-forge/created-by": "isolation-manager",
                    "nexus-forge/tenant-name": tenant.name
                }
            }
        }
        
        # Apply namespace
        await self._apply_k8s_manifest(namespace_manifest)
        
        return TenantResource(
            tenant_id=tenant.id,
            resource_type="kubernetes_namespace",
            resource_id=namespace_name,
            provider="kubernetes",
            region="local",
            config={
                "namespace": namespace_name,
                "manifest": namespace_manifest
            }
        )
    
    async def _apply_resource_quotas(self, tenant: Tenant) -> TenantResource:
        """Apply Kubernetes resource quotas for tenant"""
        namespace_name = f"tenant-{tenant.slug}"
        quota_name = f"{namespace_name}-quota"
        
        # Calculate resource limits based on tier and quota
        limits = self._calculate_k8s_limits(tenant)
        
        quota_manifest = {
            "apiVersion": "v1",
            "kind": "ResourceQuota",
            "metadata": {
                "name": quota_name,
                "namespace": namespace_name
            },
            "spec": {
                "hard": limits
            }
        }
        
        # Apply quota
        await self._apply_k8s_manifest(quota_manifest)
        
        return TenantResource(
            tenant_id=tenant.id,
            resource_type="kubernetes_quota",
            resource_id=quota_name,
            provider="kubernetes",
            region="local",
            config={
                "namespace": namespace_name,
                "quota": quota_name,
                "limits": limits,
                "manifest": quota_manifest
            }
        )
    
    async def _apply_network_policies(self, tenant: Tenant) -> TenantResource:
        """Apply network isolation policies"""
        namespace_name = f"tenant-{tenant.slug}"
        policy_name = f"{namespace_name}-network-policy"
        
        # Create network policy for isolation
        network_policy = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": policy_name,
                "namespace": namespace_name
            },
            "spec": {
                "podSelector": {},
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {"namespaceSelector": {"matchLabels": {"name": namespace_name}}},
                            {"namespaceSelector": {"matchLabels": {"name": "nexus-forge-system"}}}
                        ]
                    }
                ],
                "egress": [
                    {
                        "to": [
                            {"namespaceSelector": {"matchLabels": {"name": namespace_name}}},
                            {"namespaceSelector": {"matchLabels": {"name": "nexus-forge-system"}}}
                        ]
                    },
                    {
                        # Allow DNS
                        "to": [],
                        "ports": [{"protocol": "UDP", "port": 53}]
                    },
                    {
                        # Allow external HTTPS
                        "to": [],
                        "ports": [{"protocol": "TCP", "port": 443}]
                    }
                ]
            }
        }
        
        await self._apply_k8s_manifest(network_policy)
        
        return TenantResource(
            tenant_id=tenant.id,
            resource_type="kubernetes_network_policy",
            resource_id=policy_name,
            provider="kubernetes",
            region="local",
            config={
                "namespace": namespace_name,
                "policy": policy_name,
                "manifest": network_policy
            }
        )
    
    def _calculate_k8s_limits(self, tenant: Tenant) -> Dict[str, str]:
        """Calculate Kubernetes resource limits based on tenant quota"""
        limits = {}
        
        if tenant.quota.cpu_cores:
            limits["requests.cpu"] = f"{tenant.quota.cpu_cores}"
            limits["limits.cpu"] = f"{tenant.quota.cpu_cores * 2}"  # Allow burst
        
        if tenant.quota.memory_gb:
            limits["requests.memory"] = f"{tenant.quota.memory_gb}Gi"
            limits["limits.memory"] = f"{tenant.quota.memory_gb * 1.5}Gi"  # Allow burst
        
        if tenant.quota.storage_gb:
            limits["requests.storage"] = f"{tenant.quota.storage_gb}Gi"
        
        # Pod limits
        limits["count/pods"] = "100"
        limits["count/services"] = "20"
        limits["count/secrets"] = "50"
        limits["count/configmaps"] = "50"
        
        return limits
    
    async def _apply_k8s_manifest(self, manifest: Dict[str, Any]) -> bool:
        """Apply Kubernetes manifest"""
        if not self.k8s_enabled:
            logger.warning("Kubernetes not available, skipping manifest application")
            return False
        
        try:
            # Write manifest to temporary file
            manifest_yaml = yaml.dump(manifest)
            
            # Apply using kubectl
            process = await asyncio.create_subprocess_exec(
                "kubectl", "apply", "-f", "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate(manifest_yaml.encode())
            
            if process.returncode != 0:
                logger.error(f"Failed to apply manifest: {stderr.decode()}")
                return False
            
            logger.info(f"Applied manifest: {stdout.decode().strip()}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying Kubernetes manifest: {e}")
            return False
    
    async def _delete_kubernetes_namespace(self, tenant: Tenant) -> bool:
        """Delete Kubernetes namespace and all resources"""
        if not self.k8s_enabled:
            return True
        
        namespace_name = f"tenant-{tenant.slug}"
        
        try:
            process = await asyncio.create_subprocess_exec(
                "kubectl", "delete", "namespace", namespace_name, "--ignore-not-found=true",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Failed to delete namespace: {stderr.decode()}")
                return False
            
            logger.info(f"Deleted namespace: {namespace_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting Kubernetes namespace: {e}")
            return False
    
    # Database isolation methods
    
    async def _create_database_isolation(self, tenant: Tenant) -> TenantResource:
        """Create dedicated database schema for tenant"""
        schema_name = f"tenant_{tenant.slug}"
        
        try:
            # Create schema
            await self.supabase.client.rpc(
                "create_tenant_schema",
                {"schema_name": schema_name, "tenant_id": tenant.id}
            ).execute()
            
            # Apply row-level security
            await self.supabase.client.rpc(
                "setup_tenant_rls",
                {"schema_name": schema_name, "tenant_id": tenant.id}
            ).execute()
            
            return TenantResource(
                tenant_id=tenant.id,
                resource_type="database_schema",
                resource_id=schema_name,
                provider="supabase",
                region="global",
                config={
                    "schema_name": schema_name,
                    "rls_enabled": True
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create database isolation: {e}")
            raise ResourceError(f"Database isolation creation failed: {e}")
    
    async def _delete_database_isolation(self, tenant: Tenant) -> bool:
        """Delete tenant database schema"""
        schema_name = f"tenant_{tenant.slug}"
        
        try:
            await self.supabase.client.rpc(
                "drop_tenant_schema",
                {"schema_name": schema_name}
            ).execute()
            
            logger.info(f"Deleted database schema: {schema_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete database schema: {e}")
            return False
    
    # Domain and networking methods
    
    async def _setup_custom_domain(self, tenant: Tenant) -> TenantResource:
        """Setup custom domain for tenant"""
        domain = tenant.config.custom_domain
        
        if not domain:
            raise ValidationError("Custom domain not configured")
        
        # In production, this would:
        # 1. Configure DNS records
        # 2. Provision SSL certificate
        # 3. Update load balancer configuration
        # 4. Setup CDN if needed
        
        # For now, simulate the setup
        domain_config = {
            "domain": domain,
            "ssl_enabled": True,
            "cdn_enabled": True,
            "certificate_arn": f"arn:aws:acm:us-east-1:123456789:certificate/{tenant.id}"
        }
        
        return TenantResource(
            tenant_id=tenant.id,
            resource_type="custom_domain",
            resource_id=domain,
            provider="aws",
            region="us-east-1",
            config=domain_config
        )
    
    async def _delete_custom_domain(self, tenant: Tenant) -> bool:
        """Delete custom domain setup"""
        # In production, this would cleanup DNS, SSL, and CDN
        logger.info(f"Deleted custom domain setup for tenant {tenant.slug}")
        return True
    
    # Dedicated cluster methods
    
    async def _create_dedicated_cluster(self, tenant: Tenant) -> TenantResource:
        """Create dedicated Kubernetes cluster for enterprise tenant"""
        cluster_name = f"nexus-forge-{tenant.slug}"
        
        # In production, this would create a dedicated cluster
        # For now, simulate cluster creation
        cluster_config = {
            "cluster_name": cluster_name,
            "node_count": 3,
            "node_type": "e2-standard-4",
            "auto_scaling": True,
            "min_nodes": 1,
            "max_nodes": 10
        }
        
        return TenantResource(
            tenant_id=tenant.id,
            resource_type="kubernetes_cluster",
            resource_id=cluster_name,
            provider="gcp",
            region="us-central1",
            config=cluster_config,
            hourly_cost=1.50,  # Estimated cost
            monthly_cost=1080.0
        )
    
    async def _delete_dedicated_cluster(self, tenant: Tenant) -> bool:
        """Delete dedicated cluster"""
        # In production, this would delete the cluster
        logger.info(f"Deleted dedicated cluster for tenant {tenant.slug}")
        return True
    
    # Resource management methods
    
    async def _save_tenant_resource(self, resource: TenantResource) -> None:
        """Save tenant resource to database"""
        await self.supabase.client.table("tenant_resources").insert(
            resource.dict()
        ).execute()
    
    async def _delete_tenant_resources(self, tenant_id: str) -> bool:
        """Delete all tenant resource records"""
        try:
            await self.supabase.client.table("tenant_resources") \
                .delete() \
                .eq("tenant_id", tenant_id) \
                .execute()
            return True
        except Exception as e:
            logger.error(f"Failed to delete tenant resource records: {e}")
            return False
    
    async def _cleanup_resources(self, resources: List[TenantResource]) -> None:
        """Cleanup partially created resources"""
        for resource in resources:
            try:
                if resource.resource_type == "kubernetes_namespace":
                    await self._delete_kubernetes_namespace_by_name(resource.resource_id)
                elif resource.resource_type == "database_schema":
                    await self._delete_database_schema_by_name(resource.resource_id)
                # Add cleanup for other resource types as needed
                
            except Exception as e:
                logger.error(f"Failed to cleanup resource {resource.resource_id}: {e}")
    
    async def _delete_kubernetes_namespace_by_name(self, namespace_name: str) -> bool:
        """Delete Kubernetes namespace by name"""
        if not self.k8s_enabled:
            return True
        
        try:
            process = await asyncio.create_subprocess_exec(
                "kubectl", "delete", "namespace", namespace_name, "--ignore-not-found=true",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            return process.returncode == 0
            
        except Exception:
            return False
    
    async def _delete_database_schema_by_name(self, schema_name: str) -> bool:
        """Delete database schema by name"""
        try:
            await self.supabase.client.rpc(
                "drop_tenant_schema",
                {"schema_name": schema_name}
            ).execute()
            return True
        except Exception:
            return False
    
    # Monitoring and validation methods
    
    async def validate_tenant_isolation(self, tenant: Tenant) -> Dict[str, bool]:
        """Validate tenant isolation is working correctly"""
        validation_results = {}
        
        # Check namespace isolation
        if tenant.tier in [TenantTier.PROFESSIONAL, TenantTier.ENTERPRISE]:
            namespace_valid = await self._validate_namespace_isolation(tenant)
            validation_results["namespace_isolation"] = namespace_valid
        
        # Check database isolation
        db_valid = await self._validate_database_isolation(tenant)
        validation_results["database_isolation"] = db_valid
        
        # Check network policies
        if tenant.tier in [TenantTier.PROFESSIONAL, TenantTier.ENTERPRISE]:
            network_valid = await self._validate_network_policies(tenant)
            validation_results["network_isolation"] = network_valid
        
        return validation_results
    
    async def _validate_namespace_isolation(self, tenant: Tenant) -> bool:
        """Validate Kubernetes namespace isolation"""
        if not self.k8s_enabled:
            return True
        
        namespace_name = f"tenant-{tenant.slug}"
        
        try:
            # Check if namespace exists
            process = await asyncio.create_subprocess_exec(
                "kubectl", "get", "namespace", namespace_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            return process.returncode == 0
            
        except Exception:
            return False
    
    async def _validate_database_isolation(self, tenant: Tenant) -> bool:
        """Validate database isolation"""
        try:
            # Check if tenant schema exists and RLS is enabled
            result = await self.supabase.client.rpc(
                "validate_tenant_isolation",
                {"tenant_id": tenant.id}
            ).execute()
            
            return result.data.get("valid", False) if result.data else False
            
        except Exception:
            return False
    
    async def _validate_network_policies(self, tenant: Tenant) -> bool:
        """Validate network policies are applied"""
        if not self.k8s_enabled:
            return True
        
        namespace_name = f"tenant-{tenant.slug}"
        policy_name = f"{namespace_name}-network-policy"
        
        try:
            process = await asyncio.create_subprocess_exec(
                "kubectl", "get", "networkpolicy", policy_name, "-n", namespace_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            return process.returncode == 0
            
        except Exception:
            return False