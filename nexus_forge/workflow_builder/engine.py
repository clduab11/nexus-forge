"""
Workflow Engine
Main orchestrator for visual workflow creation, compilation, and execution
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ..core.cache import RedisCache
from ..core.exceptions import NotFoundError, ValidationError
from ..integrations.supabase.coordination_client import SupabaseCoordinationClient
from .compiler import CompiledWorkflow, WorkflowCompiler
from .executor import WorkflowExecutor
from .models import (
    ExecutionStatus,
    NodeType,
    Workflow,
    WorkflowExecution,
    WorkflowNode,
    WorkflowTemplate,
)

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """Main workflow engine for Nexus Forge"""

    def __init__(self):
        self.compiler = WorkflowCompiler()
        self.executor = WorkflowExecutor()
        self.cache = RedisCache()
        self.supabase = SupabaseCoordinationClient()

        # Node type registry
        self.node_types: Dict[str, Dict[str, Any]] = {}
        self._register_built_in_node_types()

        # Template registry
        self.templates: Dict[str, WorkflowTemplate] = {}

        # Compiled workflow cache
        self.compiled_cache: Dict[str, CompiledWorkflow] = {}

    def _register_built_in_node_types(self):
        """Register built-in node types"""
        self.node_types.update(
            {
                "trigger": {
                    "name": "Trigger",
                    "category": "triggers",
                    "description": "Workflow trigger node",
                    "icon": "play",
                    "color": "#4CAF50",
                    "inputs": [],
                    "outputs": [{"name": "output", "type": "any"}],
                    "config_schema": {
                        "type": "object",
                        "properties": {
                            "trigger_type": {
                                "type": "string",
                                "enum": ["manual", "webhook", "schedule"],
                                "default": "manual",
                            }
                        },
                    },
                },
                "http_request": {
                    "name": "HTTP Request",
                    "category": "network",
                    "description": "Make HTTP requests to external APIs",
                    "icon": "globe",
                    "color": "#2196F3",
                    "inputs": [{"name": "input", "type": "any"}],
                    "outputs": [
                        {"name": "response", "type": "object"},
                        {"name": "error", "type": "error"},
                    ],
                    "config_schema": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "format": "uri"},
                            "method": {
                                "type": "string",
                                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                                "default": "GET",
                            },
                            "headers": {"type": "object"},
                            "body": {"type": "object"},
                        },
                        "required": ["url"],
                    },
                },
                "condition": {
                    "name": "Condition",
                    "category": "logic",
                    "description": "Conditional branching based on expressions",
                    "icon": "split",
                    "color": "#FF9800",
                    "inputs": [{"name": "input", "type": "any"}],
                    "outputs": [
                        {"name": "true", "type": "any"},
                        {"name": "false", "type": "any"},
                    ],
                    "config_schema": {
                        "type": "object",
                        "properties": {
                            "condition": {
                                "type": "string",
                                "description": "JavaScript-like expression",
                            }
                        },
                        "required": ["condition"],
                    },
                },
                "ai_agent": {
                    "name": "AI Agent",
                    "category": "ai",
                    "description": "Execute AI agent operations",
                    "icon": "brain",
                    "color": "#9C27B0",
                    "inputs": [{"name": "input", "type": "any"}],
                    "outputs": [
                        {"name": "result", "type": "object"},
                        {"name": "error", "type": "error"},
                    ],
                    "config_schema": {
                        "type": "object",
                        "properties": {
                            "agent_type": {
                                "type": "string",
                                "enum": [
                                    "text_generation",
                                    "data_analysis",
                                    "code_generation",
                                ],
                                "default": "text_generation",
                            },
                            "model": {"type": "string"},
                            "prompt": {"type": "string"},
                        },
                    },
                },
                "transformation": {
                    "name": "Data Transform",
                    "category": "data",
                    "description": "Transform data using templates or expressions",
                    "icon": "transform",
                    "color": "#607D8B",
                    "inputs": [{"name": "input", "type": "any"}],
                    "outputs": [{"name": "output", "type": "any"}],
                    "config_schema": {
                        "type": "object",
                        "properties": {
                            "transform_type": {
                                "type": "string",
                                "enum": ["template", "json_path", "passthrough"],
                                "default": "passthrough",
                            },
                            "template": {"type": "string"},
                            "json_path": {"type": "string"},
                        },
                    },
                },
                "email": {
                    "name": "Send Email",
                    "category": "communication",
                    "description": "Send email notifications",
                    "icon": "mail",
                    "color": "#F44336",
                    "inputs": [{"name": "input", "type": "any"}],
                    "outputs": [
                        {"name": "sent", "type": "object"},
                        {"name": "error", "type": "error"},
                    ],
                    "config_schema": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string", "format": "email"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["to", "subject", "body"],
                    },
                },
                "delay": {
                    "name": "Delay",
                    "category": "utility",
                    "description": "Add delay between operations",
                    "icon": "clock",
                    "color": "#795548",
                    "inputs": [{"name": "input", "type": "any"}],
                    "outputs": [{"name": "output", "type": "any"}],
                    "config_schema": {
                        "type": "object",
                        "properties": {
                            "delay_seconds": {
                                "type": "number",
                                "minimum": 0,
                                "default": 1,
                            }
                        },
                    },
                },
            }
        )

    # Workflow Management

    async def create_workflow(
        self,
        name: str,
        description: str,
        author_id: str,
        organization: Optional[str] = None,
    ) -> Workflow:
        """Create a new workflow"""
        workflow = Workflow(
            name=name,
            description=description,
            author_id=author_id,
            organization=organization,
        )

        # Save to database
        await self._save_workflow(workflow)

        return workflow

    async def get_workflow(self, workflow_id: str) -> Workflow:
        """Get workflow by ID"""
        # Check cache first
        cache_key = f"workflow:{workflow_id}"
        cached = await self.cache.get(cache_key)
        if cached:
            return Workflow(**cached)

        # Query database
        result = (
            await self.supabase.client.table("workflows")
            .select("*")
            .eq("id", workflow_id)
            .execute()
        )

        if not result.data:
            raise NotFoundError(f"Workflow {workflow_id} not found")

        workflow = Workflow(**result.data[0])

        # Cache result
        await self.cache.set(cache_key, workflow.dict(), ttl=3600)

        return workflow

    async def update_workflow(self, workflow: Workflow) -> Workflow:
        """Update an existing workflow"""
        workflow.updated_at = datetime.utcnow()

        # Save to database
        await self._update_workflow(workflow)

        # Clear cache
        await self.cache.delete(f"workflow:{workflow.id}")
        await self.cache.delete(f"compiled:{workflow.id}")

        # Clear compiled cache
        self.compiled_cache.pop(workflow.id, None)

        return workflow

    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow"""
        try:
            await self.supabase.client.table("workflows").delete().eq(
                "id", workflow_id
            ).execute()

            # Clear caches
            await self.cache.delete(f"workflow:{workflow_id}")
            await self.cache.delete(f"compiled:{workflow_id}")
            self.compiled_cache.pop(workflow_id, None)

            return True
        except Exception as e:
            logger.error(f"Failed to delete workflow {workflow_id}: {e}")
            return False

    async def list_workflows(
        self,
        author_id: Optional[str] = None,
        organization: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Workflow]:
        """List workflows with filtering"""
        query = self.supabase.client.table("workflows").select("*")

        if author_id:
            query = query.eq("author_id", author_id)

        if organization:
            query = query.eq("organization", organization)

        if category:
            query = query.eq("category", category)

        query = query.order("updated_at", desc=True)
        query = query.range(offset, offset + limit - 1)

        result = await query.execute()

        return [Workflow(**data) for data in result.data]

    # Node Management

    def get_node_types(self) -> Dict[str, Dict[str, Any]]:
        """Get all available node types"""
        return self.node_types.copy()

    def get_node_type(self, node_type: str) -> Dict[str, Any]:
        """Get specific node type definition"""
        if node_type not in self.node_types:
            raise NotFoundError(f"Node type {node_type} not found")

        return self.node_types[node_type].copy()

    def register_node_type(self, type_id: str, definition: Dict[str, Any]) -> None:
        """Register a custom node type"""
        self.node_types[type_id] = definition
        logger.info(f"Registered node type: {type_id}")

    def validate_node(self, node: WorkflowNode) -> List[str]:
        """Validate a workflow node"""
        errors = []

        # Check if node type exists
        if node.type.value not in self.node_types:
            errors.append(f"Unknown node type: {node.type}")
            return errors

        # Validate using compiler
        compiler_errors = self.compiler.validate_node_configuration(node)
        errors.extend(compiler_errors)

        return errors

    # Workflow Compilation

    async def compile_workflow(self, workflow: Workflow) -> CompiledWorkflow:
        """Compile workflow for execution"""
        # Check cache
        if workflow.id in self.compiled_cache:
            return self.compiled_cache[workflow.id]

        # Compile workflow
        compiled = self.compiler.compile(workflow)

        # Cache compiled workflow
        self.compiled_cache[workflow.id] = compiled

        # Cache in Redis
        await self.cache.set(
            f"compiled:{workflow.id}",
            {
                "workflow_id": compiled.workflow_id,
                "entry_points": compiled.entry_points,
                "variables": compiled.variables,
            },
            ttl=3600,
        )

        return compiled

    async def validate_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """Validate workflow and return validation report"""
        errors = []
        warnings = []

        try:
            # Attempt compilation
            compiled = await self.compile_workflow(workflow)

            # Validate individual nodes
            for node in workflow.nodes:
                node_errors = self.validate_node(node)
                if node_errors:
                    errors.extend(
                        [f"Node {node.name}: {error}" for error in node_errors]
                    )

            # Generate execution plan
            execution_plan = self.compiler.generate_execution_plan(compiled)

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "execution_plan": execution_plan,
                "compiled": True,
            }

        except ValidationError as e:
            errors.append(str(e))
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "compiled": False,
            }

    # Workflow Execution

    async def execute_workflow(
        self,
        workflow_id: str,
        trigger_data: Dict[str, Any],
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> WorkflowExecution:
        """Execute a workflow"""
        # Get workflow
        workflow = await self.get_workflow(workflow_id)

        # Compile workflow
        compiled = await self.compile_workflow(workflow)

        # Execute workflow
        execution = await self.executor.execute_workflow(
            compiled_workflow=compiled,
            trigger_data=trigger_data,
            user_id=user_id,
            tenant_id=tenant_id,
        )

        # Update workflow stats
        await self._update_workflow_stats(workflow_id)

        return execution

    async def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID"""
        try:
            result = (
                await self.supabase.client.table("workflow_executions")
                .select("*")
                .eq("id", execution_id)
                .execute()
            )

            if result.data:
                return WorkflowExecution(**result.data[0])
        except Exception as e:
            logger.error(f"Failed to get execution {execution_id}: {e}")

        return None

    async def list_executions(
        self,
        workflow_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[WorkflowExecution]:
        """List workflow executions"""
        query = self.supabase.client.table("workflow_executions").select("*")

        if workflow_id:
            query = query.eq("workflow_id", workflow_id)

        if user_id:
            query = query.eq("user_id", user_id)

        if status:
            query = query.eq("status", status.value)

        query = query.order("started_at", desc=True)
        query = query.range(offset, offset + limit - 1)

        result = await query.execute()

        return [WorkflowExecution(**data) for data in result.data]

    # Execution Control

    async def pause_execution(self, execution_id: str) -> bool:
        """Pause workflow execution"""
        return await self.executor.pause_execution(execution_id)

    async def resume_execution(self, execution_id: str) -> bool:
        """Resume workflow execution"""
        return await self.executor.resume_execution(execution_id)

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel workflow execution"""
        return await self.executor.cancel_execution(execution_id)

    # Template Management

    async def create_template(
        self,
        name: str,
        description: str,
        category: str,
        workflow: Workflow,
        author: str,
        config_schema: Optional[Dict[str, Any]] = None,
    ) -> WorkflowTemplate:
        """Create workflow template"""
        template = WorkflowTemplate(
            name=name,
            description=description,
            category=category,
            workflow=workflow,
            author=author,
            config_schema=config_schema or {},
        )

        # Save to database
        await self._save_template(template)

        return template

    async def get_template(self, template_id: str) -> WorkflowTemplate:
        """Get workflow template"""
        result = (
            await self.supabase.client.table("workflow_templates")
            .select("*")
            .eq("id", template_id)
            .execute()
        )

        if not result.data:
            raise NotFoundError(f"Template {template_id} not found")

        return WorkflowTemplate(**result.data[0])

    async def list_templates(
        self,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        limit: int = 50,
    ) -> List[WorkflowTemplate]:
        """List workflow templates"""
        query = (
            self.supabase.client.table("workflow_templates")
            .select("*")
            .eq("published", True)
        )

        if category:
            query = query.eq("category", category)

        if difficulty:
            query = query.eq("difficulty", difficulty)

        query = query.order("usage_count", desc=True)
        query = query.limit(limit)

        result = await query.execute()

        return [WorkflowTemplate(**data) for data in result.data]

    async def create_from_template(
        self, template_id: str, name: str, author_id: str, config: Dict[str, Any]
    ) -> Workflow:
        """Create workflow from template"""
        template = await self.get_template(template_id)

        # Clone template workflow
        workflow_data = template.workflow.dict()
        workflow_data.update(
            {
                "id": None,  # Generate new ID
                "name": name,
                "author_id": author_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
        )

        workflow = Workflow(**workflow_data)

        # Apply template configuration
        # This would customize the workflow based on the config
        # For now, just save as-is

        await self._save_workflow(workflow)

        # Update template usage
        await self._increment_template_usage(template_id)

        return workflow

    # Statistics and Analytics

    async def get_workflow_stats(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow statistics"""
        # Get execution stats
        exec_query = (
            self.supabase.client.table("workflow_executions")
            .select("status", count="exact")
            .eq("workflow_id", workflow_id)
        )

        exec_result = await exec_query.execute()

        # Count by status
        status_counts = {}
        for status in ExecutionStatus:
            count_query = (
                self.supabase.client.table("workflow_executions")
                .select("id", count="exact")
                .eq("workflow_id", workflow_id)
                .eq("status", status.value)
            )

            count_result = await count_query.execute()
            status_counts[status.value] = count_result.count or 0

        return {
            "total_executions": exec_result.count or 0,
            "status_counts": status_counts,
            "success_rate": self._calculate_success_rate(status_counts),
            "last_execution": await self._get_last_execution_time(workflow_id),
        }

    def _calculate_success_rate(self, status_counts: Dict[str, int]) -> float:
        """Calculate workflow success rate"""
        total = sum(status_counts.values())
        if total == 0:
            return 0.0

        successful = status_counts.get(ExecutionStatus.COMPLETED.value, 0)
        return successful / total

    async def _get_last_execution_time(self, workflow_id: str) -> Optional[str]:
        """Get last execution time for workflow"""
        query = (
            self.supabase.client.table("workflow_executions")
            .select("started_at")
            .eq("workflow_id", workflow_id)
            .order("started_at", desc=True)
            .limit(1)
        )

        result = await query.execute()

        if result.data:
            return result.data[0]["started_at"]
        return None

    # Private helper methods

    async def _save_workflow(self, workflow: Workflow):
        """Save workflow to database"""
        await self.supabase.client.table("workflows").insert(workflow.dict()).execute()

    async def _update_workflow(self, workflow: Workflow):
        """Update workflow in database"""
        await self.supabase.client.table("workflows").update(workflow.dict()).eq(
            "id", workflow.id
        ).execute()

    async def _save_template(self, template: WorkflowTemplate):
        """Save template to database"""
        await self.supabase.client.table("workflow_templates").insert(
            template.dict()
        ).execute()

    async def _update_workflow_stats(self, workflow_id: str):
        """Update workflow execution statistics"""
        await self.supabase.client.rpc(
            "increment_workflow_executions", {"workflow_id": workflow_id}
        ).execute()

    async def _increment_template_usage(self, template_id: str):
        """Increment template usage count"""
        await self.supabase.client.rpc(
            "increment_template_usage", {"template_id": template_id}
        ).execute()
