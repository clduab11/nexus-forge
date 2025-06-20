"""
Dynamic Agent Loading System with Hot-Reload and MCP Integration
Advanced implementation for runtime agent loading and management
"""

import asyncio
import hashlib
import importlib.util
import inspect
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

import aiofiles
import docker
import yaml
from pydantic import BaseModel, Field, validator
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ...core.exceptions import (
    AgentLoadError,
    CompatibilityError,
    SecurityException,
    ValidationError,
)
from ...core.logging import get_logger
from ...core.metrics import MetricsCollector
from ..agents.base import BaseAgent
from .models import AgentManifest, AgentPackage
from .security_scanner import SecurityScanner

logger = get_logger(__name__)


class AgentLoadStrategy(str, Enum):
    """Agent loading strategies"""
    
    IMMEDIATE = "immediate"  # Load immediately
    LAZY = "lazy"  # Load on first use
    PRELOAD = "preload"  # Preload based on prediction
    ON_DEMAND = "on_demand"  # Load when requested


class AgentIsolationLevel(str, Enum):
    """Agent isolation levels"""
    
    NONE = "none"  # No isolation (fastest)
    PROCESS = "process"  # Process isolation
    CONTAINER = "container"  # Container isolation
    VM = "vm"  # Virtual machine isolation (most secure)


@dataclass
class LoadedAgent:
    """Loaded agent instance with metadata"""
    
    id: str
    name: str
    version: str
    instance: BaseAgent
    manifest: AgentManifest
    load_time: datetime = field(default_factory=datetime.utcnow)
    isolation_level: AgentIsolationLevel = AgentIsolationLevel.PROCESS
    sandbox_id: Optional[str] = None
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    health_status: str = "healthy"
    last_health_check: Optional[datetime] = None


class DynamicAgentLoader:
    """
    Advanced dynamic agent loading system with hot-reload support
    """
    
    def __init__(
        self,
        base_path: str = "/opt/nexus-forge/agents",
        isolation_level: AgentIsolationLevel = AgentIsolationLevel.PROCESS,
        enable_hot_reload: bool = True,
        max_loaded_agents: int = 100,
    ):
        self.base_path = Path(base_path)
        self.isolation_level = isolation_level
        self.enable_hot_reload = enable_hot_reload
        self.max_loaded_agents = max_loaded_agents
        
        # Agent registry
        self.loaded_agents: Dict[str, LoadedAgent] = {}
        self.agent_cache: Dict[str, Any] = {}
        self.loading_queue: asyncio.Queue = asyncio.Queue()
        
        # Components
        self.sandbox_manager = SandboxManager()
        self.validator = AgentValidator()
        self.compatibility_checker = CompatibilityChecker()
        self.performance_monitor = PerformanceMonitor()
        self.hot_reload_manager = HotReloadManager() if enable_hot_reload else None
        
        # Metrics
        self.metrics = MetricsCollector()
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start the dynamic agent loader"""
        # Create base directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Start background tasks
        self._background_tasks.append(
            asyncio.create_task(self._loading_worker())
        )
        self._background_tasks.append(
            asyncio.create_task(self._health_check_worker())
        )
        
        if self.enable_hot_reload:
            self._background_tasks.append(
                asyncio.create_task(self.hot_reload_manager.start_watching(self.base_path))
            )
        
        logger.info("Dynamic agent loader started")
    
    async def stop(self):
        """Stop the dynamic agent loader"""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Unload all agents
        for agent_id in list(self.loaded_agents.keys()):
            await self.unload_agent(agent_id)
        
        logger.info("Dynamic agent loader stopped")
    
    async def load_agent_from_marketplace(
        self,
        agent_id: str,
        version: str = "latest",
        config: Optional[Dict[str, Any]] = None,
        strategy: AgentLoadStrategy = AgentLoadStrategy.IMMEDIATE,
    ) -> LoadedAgent:
        """
        Load agent from marketplace with dynamic loading
        
        Args:
            agent_id: Agent identifier
            version: Agent version
            config: Agent configuration
            strategy: Loading strategy
        
        Returns:
            Loaded agent instance
        """
        start_time = time.time()
        
        try:
            # Check if already loaded
            cache_key = f"{agent_id}:{version}"
            if cache_key in self.loaded_agents:
                logger.info(f"Agent {agent_id} already loaded")
                return self.loaded_agents[cache_key]
            
            # Download agent package
            package = await self._download_agent_package(agent_id, version)
            
            # Validate package
            validation_result = await self.validator.validate_package(package)
            if not validation_result.is_valid:
                raise ValidationError(
                    f"Agent validation failed: {validation_result.errors}"
                )
            
            # Check compatibility
            compatibility = await self.compatibility_checker.check(
                package.manifest,
                self._get_system_info(),
            )
            if not compatibility.is_compatible:
                raise CompatibilityError(
                    f"Agent not compatible: {compatibility.reason}"
                )
            
            # Extract package
            agent_path = await self._extract_package(package)
            
            # Load based on strategy
            if strategy == AgentLoadStrategy.IMMEDIATE:
                loaded_agent = await self._load_agent_immediate(
                    package, agent_path, config
                )
            elif strategy == AgentLoadStrategy.LAZY:
                loaded_agent = await self._prepare_lazy_loading(
                    package, agent_path, config
                )
            elif strategy == AgentLoadStrategy.PRELOAD:
                await self.loading_queue.put((package, agent_path, config))
                loaded_agent = await self._wait_for_loading(cache_key)
            else:  # ON_DEMAND
                loaded_agent = await self._load_on_demand(
                    package, agent_path, config
                )
            
            # Store in registry
            self.loaded_agents[cache_key] = loaded_agent
            
            # Update metrics
            load_time = time.time() - start_time
            await self.metrics.record("agent_load_time", load_time, {
                "agent_id": agent_id,
                "version": version,
                "strategy": strategy.value,
            })
            
            logger.info(
                f"Successfully loaded agent {agent_id}:{version} in {load_time:.2f}s"
            )
            
            return loaded_agent
        
        except Exception as e:
            logger.error(f"Failed to load agent {agent_id}: {e}")
            await self.metrics.increment("agent_load_failures", {
                "agent_id": agent_id,
                "error_type": type(e).__name__,
            })
            raise AgentLoadError(f"Failed to load agent: {e}")
    
    async def unload_agent(self, agent_id: str, force: bool = False) -> bool:
        """
        Unload agent and free resources
        
        Args:
            agent_id: Agent identifier (format: "id:version")
            force: Force unload even if agent is in use
        
        Returns:
            Success status
        """
        if agent_id not in self.loaded_agents:
            logger.warning(f"Agent {agent_id} not loaded")
            return False
        
        loaded_agent = self.loaded_agents[agent_id]
        
        try:
            # Check if agent is in use
            if not force and await self._is_agent_in_use(loaded_agent):
                logger.warning(f"Agent {agent_id} is in use, cannot unload")
                return False
            
            # Stop agent
            if hasattr(loaded_agent.instance, "stop"):
                await loaded_agent.instance.stop()
            
            # Clean up sandbox
            if loaded_agent.sandbox_id:
                await self.sandbox_manager.destroy_sandbox(loaded_agent.sandbox_id)
            
            # Remove from registry
            del self.loaded_agents[agent_id]
            
            # Clear cache
            self._clear_agent_cache(agent_id)
            
            logger.info(f"Successfully unloaded agent {agent_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error unloading agent {agent_id}: {e}")
            return False
    
    async def reload_agent(self, agent_id: str) -> LoadedAgent:
        """Hot-reload agent without losing state"""
        if agent_id not in self.loaded_agents:
            raise ValueError(f"Agent {agent_id} not loaded")
        
        old_agent = self.loaded_agents[agent_id]
        
        try:
            # Save agent state
            state = await self._save_agent_state(old_agent)
            
            # Unload old version
            await self.unload_agent(agent_id, force=True)
            
            # Load new version
            new_agent = await self.load_agent_from_marketplace(
                old_agent.id,
                old_agent.version,
                state.get("config"),
            )
            
            # Restore state
            await self._restore_agent_state(new_agent, state)
            
            logger.info(f"Successfully hot-reloaded agent {agent_id}")
            return new_agent
        
        except Exception as e:
            logger.error(f"Failed to reload agent {agent_id}: {e}")
            # Try to restore old agent
            self.loaded_agents[agent_id] = old_agent
            raise
    
    async def get_loaded_agents(self) -> List[Dict[str, Any]]:
        """Get information about all loaded agents"""
        return [
            {
                "id": agent.id,
                "name": agent.name,
                "version": agent.version,
                "load_time": agent.load_time.isoformat(),
                "isolation_level": agent.isolation_level.value,
                "health_status": agent.health_status,
                "metrics": agent.metrics,
            }
            for agent in self.loaded_agents.values()
        ]
    
    async def _load_agent_immediate(
        self,
        package: AgentPackage,
        agent_path: Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> LoadedAgent:
        """Load agent immediately"""
        # Create sandbox if required
        sandbox_id = None
        if self.isolation_level != AgentIsolationLevel.NONE:
            sandbox = await self.sandbox_manager.create_sandbox(
                agent_id=package.id,
                agent_path=str(agent_path),
                isolation_level=self.isolation_level,
                resource_limits=self._get_resource_limits(package.manifest),
            )
            sandbox_id = sandbox.id
        
        # Load agent module
        agent_class = await self._load_agent_module(
            agent_path, package.manifest.main_class
        )
        
        # Instantiate agent
        if config is None:
            config = {}
        
        agent_instance = agent_class(config)
        
        # Initialize agent
        if hasattr(agent_instance, "initialize"):
            await agent_instance.initialize()
        
        # Create loaded agent entry
        loaded_agent = LoadedAgent(
            id=package.id,
            name=package.manifest.name,
            version=package.manifest.version,
            instance=agent_instance,
            manifest=package.manifest,
            isolation_level=self.isolation_level,
            sandbox_id=sandbox_id,
            resource_limits=self._get_resource_limits(package.manifest),
        )
        
        return loaded_agent
    
    async def _prepare_lazy_loading(
        self,
        package: AgentPackage,
        agent_path: Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> LoadedAgent:
        """Prepare agent for lazy loading"""
        # Create a proxy that loads the real agent on first use
        proxy = LazyAgentProxy(
            loader=self,
            package=package,
            agent_path=agent_path,
            config=config,
        )
        
        return LoadedAgent(
            id=package.id,
            name=package.manifest.name,
            version=package.manifest.version,
            instance=proxy,
            manifest=package.manifest,
            isolation_level=self.isolation_level,
        )
    
    async def _load_on_demand(
        self,
        package: AgentPackage,
        agent_path: Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> LoadedAgent:
        """Load agent on demand with caching"""
        # Similar to immediate but with additional caching logic
        return await self._load_agent_immediate(package, agent_path, config)
    
    async def _download_agent_package(
        self,
        agent_id: str,
        version: str,
    ) -> AgentPackage:
        """Download agent package from marketplace"""
        # Implementation would download from marketplace
        # For now, return mock package
        return AgentPackage(
            id=agent_id,
            manifest=AgentManifest(
                name=agent_id,
                version=version,
                display_name="Test Agent",
                description="Test agent for dynamic loading",
                author="test",
                license="MIT",
                category="utility",
                capabilities=["test"],
                main_class="test_agent.TestAgent",
                nexus_forge_version=">=1.0.0",
            ),
            author_id="test-author",
            author_email="test@example.com",
        )
    
    async def _extract_package(self, package: AgentPackage) -> Path:
        """Extract agent package to filesystem"""
        agent_dir = self.base_path / package.id / package.manifest.version
        agent_dir.mkdir(parents=True, exist_ok=True)
        
        # Implementation would extract actual package
        # For now, create a dummy agent file
        agent_file = agent_dir / "test_agent.py"
        agent_code = '''
from nexus_forge.agents.base import BaseAgent

class TestAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.name = "Test Agent"
    
    async def process(self, input_data):
        return {"result": f"Processed: {input_data}"}
'''
        
        async with aiofiles.open(agent_file, "w") as f:
            await f.write(agent_code)
        
        return agent_dir
    
    async def _load_agent_module(
        self,
        agent_path: Path,
        main_class: str,
    ) -> Type[BaseAgent]:
        """Dynamically load agent module"""
        # Parse module and class names
        module_name, class_name = main_class.rsplit(".", 1)
        module_file = agent_path / f"{module_name.replace('.', '/')}.py"
        
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(
            f"dynamic_agents.{module_name}",
            module_file,
        )
        
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {module_file}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        
        # Get agent class
        agent_class = getattr(module, class_name)
        
        # Validate it's a proper agent class
        if not issubclass(agent_class, BaseAgent):
            raise TypeError(f"{class_name} is not a valid agent class")
        
        return agent_class
    
    def _get_resource_limits(self, manifest: AgentManifest) -> Dict[str, Any]:
        """Get resource limits from manifest"""
        return {
            "memory": manifest.requirements.get("memory", "1Gi"),
            "cpu": manifest.requirements.get("cpu", "1.0"),
            "disk": manifest.requirements.get("disk", "10Gi"),
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for compatibility checking"""
        return {
            "platform": sys.platform,
            "python_version": sys.version,
            "nexus_forge_version": "1.0.0",  # Would get from package
        }
    
    async def _is_agent_in_use(self, agent: LoadedAgent) -> bool:
        """Check if agent is currently in use"""
        # Implementation would check active tasks
        return False
    
    async def _save_agent_state(self, agent: LoadedAgent) -> Dict[str, Any]:
        """Save agent state for hot-reload"""
        state = {
            "config": getattr(agent.instance, "config", {}),
            "metrics": agent.metrics,
        }
        
        # Save custom state if agent supports it
        if hasattr(agent.instance, "save_state"):
            state["custom"] = await agent.instance.save_state()
        
        return state
    
    async def _restore_agent_state(
        self,
        agent: LoadedAgent,
        state: Dict[str, Any],
    ):
        """Restore agent state after hot-reload"""
        # Restore metrics
        agent.metrics = state.get("metrics", {})
        
        # Restore custom state if agent supports it
        if hasattr(agent.instance, "restore_state") and "custom" in state:
            await agent.instance.restore_state(state["custom"])
    
    def _clear_agent_cache(self, agent_id: str):
        """Clear agent from cache"""
        keys_to_remove = [k for k in self.agent_cache if k.startswith(agent_id)]
        for key in keys_to_remove:
            del self.agent_cache[key]
    
    async def _loading_worker(self):
        """Background worker for loading agents"""
        while True:
            try:
                package, agent_path, config = await self.loading_queue.get()
                await self._load_agent_immediate(package, agent_path, config)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in loading worker: {e}")
    
    async def _health_check_worker(self):
        """Background worker for health checks"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                for agent_id, agent in self.loaded_agents.items():
                    try:
                        # Perform health check
                        if hasattr(agent.instance, "health_check"):
                            health_result = await agent.instance.health_check()
                            agent.health_status = (
                                "healthy" if health_result else "unhealthy"
                            )
                        else:
                            agent.health_status = "unknown"
                        
                        agent.last_health_check = datetime.utcnow()
                        
                        # Update metrics
                        await self.performance_monitor.update_agent_metrics(agent)
                    
                    except Exception as e:
                        logger.error(f"Health check failed for {agent_id}: {e}")
                        agent.health_status = "error"
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check worker: {e}")
    
    async def _wait_for_loading(
        self,
        cache_key: str,
        timeout: float = 30.0,
    ) -> LoadedAgent:
        """Wait for agent to be loaded"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if cache_key in self.loaded_agents:
                return self.loaded_agents[cache_key]
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Timeout waiting for agent {cache_key} to load")


class LazyAgentProxy:
    """Proxy for lazy loading agents"""
    
    def __init__(
        self,
        loader: DynamicAgentLoader,
        package: AgentPackage,
        agent_path: Path,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._loader = loader
        self._package = package
        self._agent_path = agent_path
        self._config = config
        self._real_agent: Optional[BaseAgent] = None
    
    async def _ensure_loaded(self):
        """Ensure the real agent is loaded"""
        if self._real_agent is None:
            loaded = await self._loader._load_agent_immediate(
                self._package,
                self._agent_path,
                self._config,
            )
            self._real_agent = loaded.instance
    
    def __getattr__(self, name: str):
        """Proxy attribute access to real agent"""
        async def async_wrapper(*args, **kwargs):
            await self._ensure_loaded()
            method = getattr(self._real_agent, name)
            if asyncio.iscoroutinefunction(method):
                return await method(*args, **kwargs)
            return method(*args, **kwargs)
        
        return async_wrapper


class SandboxManager:
    """Manage sandboxed execution environments"""
    
    def __init__(self):
        self.sandboxes: Dict[str, Any] = {}
        self.docker_client = None
    
    async def create_sandbox(
        self,
        agent_id: str,
        agent_path: str,
        isolation_level: AgentIsolationLevel,
        resource_limits: Dict[str, Any],
    ) -> Any:
        """Create sandbox based on isolation level"""
        sandbox_id = f"{agent_id}-{int(time.time())}"
        
        if isolation_level == AgentIsolationLevel.CONTAINER:
            sandbox = await self._create_container_sandbox(
                sandbox_id, agent_path, resource_limits
            )
        elif isolation_level == AgentIsolationLevel.PROCESS:
            sandbox = await self._create_process_sandbox(
                sandbox_id, agent_path, resource_limits
            )
        else:
            sandbox = {"id": sandbox_id, "type": "none"}
        
        self.sandboxes[sandbox_id] = sandbox
        return sandbox
    
    async def destroy_sandbox(self, sandbox_id: str):
        """Destroy sandbox and clean up resources"""
        if sandbox_id not in self.sandboxes:
            return
        
        sandbox = self.sandboxes[sandbox_id]
        
        if sandbox.get("type") == "container":
            await self._destroy_container_sandbox(sandbox)
        elif sandbox.get("type") == "process":
            await self._destroy_process_sandbox(sandbox)
        
        del self.sandboxes[sandbox_id]
    
    async def _create_container_sandbox(
        self,
        sandbox_id: str,
        agent_path: str,
        resource_limits: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create Docker container sandbox"""
        if self.docker_client is None:
            self.docker_client = docker.from_env()
        
        # Create container with resource limits
        container = self.docker_client.containers.create(
            image="nexusforge/agent-runtime:latest",
            name=f"agent-sandbox-{sandbox_id}",
            volumes={agent_path: {"bind": "/app", "mode": "ro"}},
            mem_limit=resource_limits.get("memory", "1g"),
            cpu_quota=int(float(resource_limits.get("cpu", 1.0)) * 100000),
            detach=True,
            remove=True,
        )
        
        return {
            "id": sandbox_id,
            "type": "container",
            "container": container,
        }
    
    async def _create_process_sandbox(
        self,
        sandbox_id: str,
        agent_path: str,
        resource_limits: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create process-level sandbox"""
        return {
            "id": sandbox_id,
            "type": "process",
            "path": agent_path,
            "limits": resource_limits,
        }
    
    async def _destroy_container_sandbox(self, sandbox: Dict[str, Any]):
        """Destroy container sandbox"""
        container = sandbox.get("container")
        if container:
            try:
                container.stop(timeout=10)
                container.remove()
            except Exception as e:
                logger.error(f"Error destroying container: {e}")
    
    async def _destroy_process_sandbox(self, sandbox: Dict[str, Any]):
        """Destroy process sandbox"""
        # Clean up process resources
        pass


class AgentValidator:
    """Validate agent packages before loading"""
    
    def __init__(self):
        self.security_scanner = SecurityScanner()
    
    async def validate_package(self, package: AgentPackage) -> "ValidationResult":
        """Validate agent package"""
        errors = []
        warnings = []
        
        # Validate manifest
        manifest_errors = self._validate_manifest(package.manifest)
        errors.extend(manifest_errors)
        
        # Security scan
        if package.security_report:
            if not package.security_report.passed:
                errors.append("Security scan failed")
        
        # Performance check
        if package.performance_metrics:
            perf_warnings = self._check_performance(package.performance_metrics)
            warnings.extend(perf_warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    def _validate_manifest(self, manifest: AgentManifest) -> List[str]:
        """Validate agent manifest"""
        errors = []
        
        # Check required fields
        if not manifest.main_class:
            errors.append("main_class is required")
        
        # Validate version format
        if not self._is_valid_version(manifest.version):
            errors.append(f"Invalid version format: {manifest.version}")
        
        return errors
    
    def _is_valid_version(self, version: str) -> bool:
        """Check if version format is valid"""
        import re
        return bool(re.match(r"^\d+\.\d+\.\d+$", version))
    
    def _check_performance(
        self,
        metrics: "PerformanceMetrics",
    ) -> List[str]:
        """Check performance metrics for warnings"""
        warnings = []
        
        if metrics.avg_response_time > 1000:  # 1 second
            warnings.append("High average response time")
        
        if metrics.avg_memory_mb > 1024:  # 1 GB
            warnings.append("High memory usage")
        
        return warnings


@dataclass
class ValidationResult:
    """Validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class CompatibilityChecker:
    """Check agent compatibility with system"""
    
    async def check(
        self,
        manifest: AgentManifest,
        system_info: Dict[str, Any],
    ) -> "CompatibilityResult":
        """Check if agent is compatible with system"""
        issues = []
        
        # Check platform compatibility
        if system_info["platform"] not in manifest.supported_platforms:
            issues.append(f"Platform {system_info['platform']} not supported")
        
        # Check version compatibility
        # Implementation would check version constraints
        
        return CompatibilityResult(
            is_compatible=len(issues) == 0,
            reason="; ".join(issues) if issues else None,
        )


@dataclass
class CompatibilityResult:
    """Compatibility check result"""
    is_compatible: bool
    reason: Optional[str]


class PerformanceMonitor:
    """Monitor agent performance"""
    
    async def update_agent_metrics(self, agent: LoadedAgent):
        """Update agent performance metrics"""
        # Implementation would collect actual metrics
        agent.metrics.update({
            "cpu_usage": 0.5,
            "memory_usage": 256,
            "requests_processed": 100,
            "errors": 0,
        })


class HotReloadManager:
    """Manage hot-reload functionality"""
    
    def __init__(self):
        self.observer = Observer()
        self.reload_callbacks: Dict[str, Callable] = {}
    
    async def start_watching(self, base_path: Path):
        """Start watching for file changes"""
        event_handler = AgentFileEventHandler(self)
        self.observer.schedule(event_handler, str(base_path), recursive=True)
        self.observer.start()
        
        logger.info(f"Started watching {base_path} for changes")
    
    def register_reload_callback(self, pattern: str, callback: Callable):
        """Register callback for reload events"""
        self.reload_callbacks[pattern] = callback
    
    async def trigger_reload(self, file_path: str):
        """Trigger reload for changed file"""
        for pattern, callback in self.reload_callbacks.items():
            if file_path.endswith(pattern):
                await callback(file_path)


class AgentFileEventHandler(FileSystemEventHandler):
    """Handle file system events for hot-reload"""
    
    def __init__(self, reload_manager: HotReloadManager):
        self.reload_manager = reload_manager
    
    def on_modified(self, event):
        """Handle file modification"""
        if not event.is_directory and event.src_path.endswith(".py"):
            asyncio.create_task(
                self.reload_manager.trigger_reload(event.src_path)
            )