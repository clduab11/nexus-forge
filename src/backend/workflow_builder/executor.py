"""
Workflow Executor
Executes compiled workflows with real-time monitoring
"""

import asyncio
import logging
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from ..core.cache import RedisCache
from ..core.exceptions import ExecutionError, TimeoutError
from ..integrations.supabase.coordination_client import SupabaseCoordinationClient
from ..websockets.manager import WebSocketManager
from .compiler import CompiledWorkflow
from .models import (
    ExecutionStatus,
    NodeExecutionResult,
    NodeType,
    WorkflowExecution,
    WorkflowNode,
)

logger = logging.getLogger(__name__)


class ExecutionContext:
    """Context for workflow execution"""

    def __init__(
        self,
        execution_id: str,
        workflow: CompiledWorkflow,
        trigger_data: Dict[str, Any],
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ):
        self.execution_id = execution_id
        self.workflow = workflow
        self.trigger_data = trigger_data
        self.user_id = user_id
        self.tenant_id = tenant_id

        # Runtime state
        self.variables = workflow.variables.copy()
        self.variables.update(trigger_data)

        self.node_states: Dict[str, Dict[str, Any]] = {}
        self.execution_results: Dict[str, Any] = {}

        # Execution control
        self.should_stop = False
        self.paused_nodes: set = set()
        self.breakpoints: set = set()

        # Monitoring
        self.start_time = time.time()
        self.node_timings: Dict[str, float] = {}


class NodeExecutor:
    """Executes individual workflow nodes"""

    def __init__(self):
        self.http_session: Optional[aiohttp.ClientSession] = None

    async def execute_node(
        self, node: WorkflowNode, context: ExecutionContext, input_data: Dict[str, Any]
    ) -> NodeExecutionResult:
        """Execute a single workflow node"""
        result = NodeExecutionResult(
            node_id=node.id, status=ExecutionStatus.RUNNING, input_data=input_data
        )

        try:
            # Check if node should be skipped
            if not node.enabled:
                result.status = ExecutionStatus.COMPLETED
                result.output_data = input_data  # Pass through
                return result

            # Execute based on node type
            if node.type == NodeType.TRIGGER:
                output = await self._execute_trigger(node, context, input_data)
            elif node.type == NodeType.ACTION:
                output = await self._execute_action(node, context, input_data)
            elif node.type == NodeType.CONDITION:
                output = await self._execute_condition(node, context, input_data)
            elif node.type == NodeType.TRANSFORMATION:
                output = await self._execute_transformation(node, context, input_data)
            elif node.type == NodeType.AI_AGENT:
                output = await self._execute_ai_agent(node, context, input_data)
            elif node.type == NodeType.HTTP_REQUEST:
                output = await self._execute_http_request(node, context, input_data)
            elif node.type == NodeType.DATABASE:
                output = await self._execute_database(node, context, input_data)
            elif node.type == NodeType.EMAIL:
                output = await self._execute_email(node, context, input_data)
            elif node.type == NodeType.DELAY:
                output = await self._execute_delay(node, context, input_data)
            elif node.type == NodeType.SUB_WORKFLOW:
                output = await self._execute_sub_workflow(node, context, input_data)
            else:
                raise ExecutionError(f"Unsupported node type: {node.type}")

            result.status = ExecutionStatus.COMPLETED
            result.output_data = output

        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            logger.error(f"Node {node.id} execution failed: {e}")

            # Handle retries
            if node.config.retries > result.retry_count:
                logger.info(
                    f"Retrying node {node.id}, attempt {result.retry_count + 1}"
                )
                await asyncio.sleep(node.config.retry_delay)
                result.retry_count += 1
                # Recursively retry
                return await self.execute_node(node, context, input_data)

        finally:
            result.completed_at = datetime.utcnow()
            if result.started_at:
                delta = result.completed_at - result.started_at
                result.duration_ms = int(delta.total_seconds() * 1000)

        return result

    async def _execute_trigger(
        self, node: WorkflowNode, context: ExecutionContext, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute trigger node"""
        # Triggers typically just pass through the trigger data
        return context.trigger_data

    async def _execute_action(
        self, node: WorkflowNode, context: ExecutionContext, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute action node"""
        # Generic action execution based on configuration
        settings = node.config.settings
        action_type = settings.get("action_type", "log")

        if action_type == "log":
            message = settings.get("message", "Action executed")
            logger.info(f"Action node {node.name}: {message}")
            return {"logged": True, "message": message}

        elif action_type == "variable_set":
            var_name = settings.get("variable_name")
            var_value = settings.get("variable_value")
            if var_name:
                context.variables[var_name] = var_value
            return {"variable_set": var_name, "value": var_value}

        else:
            return {"action_type": action_type, "executed": True}

    async def _execute_condition(
        self, node: WorkflowNode, context: ExecutionContext, input_data: Dict[str, Any]
    ) -> bool:
        """Execute condition node"""
        condition = node.config.settings.get("condition", "true")

        # Create evaluation context
        eval_context = {
            "input": input_data,
            "variables": context.variables,
            "data": input_data,
        }

        try:
            # Evaluate condition
            result = eval(condition, {"__builtins__": {}}, eval_context)
            return bool(result)
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False

    async def _execute_transformation(
        self, node: WorkflowNode, context: ExecutionContext, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute transformation node"""
        settings = node.config.settings
        transform_type = settings.get("transform_type", "passthrough")

        if transform_type == "passthrough":
            return input_data

        elif transform_type == "json_path":
            json_path = settings.get("json_path", "$")
            # Simplified JSONPath - just support basic dot notation
            if json_path.startswith("$."):
                keys = json_path[2:].split(".")
                result = input_data
                for key in keys:
                    if isinstance(result, dict) and key in result:
                        result = result[key]
                    else:
                        result = None
                        break
                return {"extracted": result}
            return input_data

        elif transform_type == "template":
            template = settings.get("template", "{}")
            # Simple template substitution
            try:
                result = template.format(**input_data, **context.variables)
                return {"transformed": result}
            except Exception:
                return {"transformed": template}

        else:
            return input_data

    async def _execute_ai_agent(
        self, node: WorkflowNode, context: ExecutionContext, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute AI agent node"""
        settings = node.config.settings
        agent_type = settings.get("agent_type", "generic")

        # Simulate AI agent execution
        await asyncio.sleep(0.5)  # Simulate processing time

        if agent_type == "text_generation":
            prompt = settings.get("prompt", "Generate text")
            return {
                "generated_text": f"AI generated response for: {prompt}",
                "model": settings.get("model", "default"),
                "input": input_data,
            }

        elif agent_type == "data_analysis":
            return {
                "analysis": "Data analysis complete",
                "insights": ["Pattern 1", "Pattern 2"],
                "confidence": 0.85,
            }

        else:
            return {"agent_type": agent_type, "processed": True, "input": input_data}

    async def _execute_http_request(
        self, node: WorkflowNode, context: ExecutionContext, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute HTTP request node"""
        settings = node.config.settings
        url = settings.get("url")
        method = settings.get("method", "GET").upper()
        headers = settings.get("headers", {})

        if not url:
            raise ExecutionError("HTTP node requires URL")

        # Initialize HTTP session if not exists
        if not self.http_session:
            self.http_session = aiohttp.ClientSession()

        try:
            timeout = aiohttp.ClientTimeout(total=node.config.timeout or 30)

            if method == "GET":
                async with self.http_session.get(
                    url, headers=headers, timeout=timeout
                ) as response:
                    content = await response.text()
                    return {
                        "status_code": response.status,
                        "headers": dict(response.headers),
                        "content": content,
                        "url": str(response.url),
                    }

            elif method == "POST":
                data = settings.get("body", input_data)
                async with self.http_session.post(
                    url, json=data, headers=headers, timeout=timeout
                ) as response:
                    content = await response.text()
                    return {
                        "status_code": response.status,
                        "headers": dict(response.headers),
                        "content": content,
                    }

            else:
                raise ExecutionError(f"Unsupported HTTP method: {method}")

        except asyncio.TimeoutError:
            raise TimeoutError(f"HTTP request to {url} timed out")
        except Exception as e:
            raise ExecutionError(f"HTTP request failed: {e}")

    async def _execute_database(
        self, node: WorkflowNode, context: ExecutionContext, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute database node"""
        settings = node.config.settings
        query = settings.get("query")

        if not query:
            raise ExecutionError("Database node requires query")

        # Simulate database execution
        await asyncio.sleep(0.1)

        return {
            "query": query,
            "rows_affected": 1,
            "execution_time_ms": 50,
            "results": [{"id": 1, "status": "success"}],
        }

    async def _execute_email(
        self, node: WorkflowNode, context: ExecutionContext, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute email node"""
        settings = node.config.settings

        to = settings.get("to")
        subject = settings.get("subject", "Workflow Notification")
        body = settings.get("body", "")

        if not to:
            raise ExecutionError("Email node requires 'to' address")

        # Simulate email sending
        await asyncio.sleep(0.2)

        return {
            "sent": True,
            "to": to,
            "subject": subject,
            "message_id": f"msg_{int(time.time())}",
        }

    async def _execute_delay(
        self, node: WorkflowNode, context: ExecutionContext, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute delay node"""
        settings = node.config.settings
        delay_seconds = settings.get("delay_seconds", 1)

        await asyncio.sleep(delay_seconds)

        return {"delayed": True, "seconds": delay_seconds, "input": input_data}

    async def _execute_sub_workflow(
        self, node: WorkflowNode, context: ExecutionContext, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute sub-workflow node"""
        settings = node.config.settings
        sub_workflow_id = settings.get("workflow_id")

        if not sub_workflow_id:
            raise ExecutionError("Sub-workflow node requires workflow_id")

        # This would load and execute the sub-workflow
        # For now, simulate execution
        await asyncio.sleep(1)

        return {
            "sub_workflow_id": sub_workflow_id,
            "completed": True,
            "output": input_data,
        }

    async def cleanup(self):
        """Cleanup resources"""
        if self.http_session:
            await self.http_session.close()


class WorkflowExecutor:
    """Main workflow executor"""

    def __init__(self):
        self.cache = RedisCache()
        self.supabase = SupabaseCoordinationClient()
        self.websocket_manager = WebSocketManager()
        self.node_executor = NodeExecutor()

        # Execution monitoring
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.execution_callbacks: Dict[str, List[Callable]] = {}

    async def execute_workflow(
        self,
        compiled_workflow: CompiledWorkflow,
        trigger_data: Dict[str, Any],
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        execution_id: Optional[str] = None,
    ) -> WorkflowExecution:
        """
        Execute a compiled workflow

        Args:
            compiled_workflow: Compiled workflow to execute
            trigger_data: Data that triggered the workflow
            user_id: User who triggered execution
            tenant_id: Tenant context
            execution_id: Optional custom execution ID

        Returns:
            Workflow execution result
        """
        # Create execution record
        execution = WorkflowExecution(
            id=execution_id or f"exec_{int(time.time())}_{len(self.active_executions)}",
            workflow_id=compiled_workflow.workflow_id,
            workflow_version="1.0.0",
            trigger_type="manual",
            trigger_data=trigger_data,
            user_id=user_id,
            tenant_id=tenant_id,
        )

        # Create execution context
        context = ExecutionContext(
            execution_id=execution.id,
            workflow=compiled_workflow,
            trigger_data=trigger_data,
            user_id=user_id,
            tenant_id=tenant_id,
        )

        # Register active execution
        self.active_executions[execution.id] = context

        try:
            # Save execution to database
            await self._save_execution(execution)

            # Notify start
            await self._notify_execution_start(execution)

            # Execute workflow
            await self._execute_workflow_internal(execution, context, compiled_workflow)

            # Update final status
            execution.completed_at = datetime.utcnow()
            execution.duration_ms = int((time.time() - context.start_time) * 1000)
            execution.nodes_executed = len(context.node_states)

            if execution.status == ExecutionStatus.RUNNING:
                execution.status = ExecutionStatus.COMPLETED

        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.utcnow()
            logger.error(f"Workflow execution {execution.id} failed: {e}")
            logger.error(traceback.format_exc())

        finally:
            # Update execution record
            await self._update_execution(execution)

            # Notify completion
            await self._notify_execution_complete(execution)

            # Cleanup
            self.active_executions.pop(execution.id, None)
            await self.node_executor.cleanup()

        return execution

    async def _execute_workflow_internal(
        self,
        execution: WorkflowExecution,
        context: ExecutionContext,
        compiled_workflow: CompiledWorkflow,
    ):
        """Internal workflow execution logic"""
        # Start from entry points
        current_nodes = compiled_workflow.entry_points.copy()
        completed_nodes = set()

        # Execution loop
        while current_nodes and not context.should_stop:
            next_nodes = []

            # Execute current batch of nodes in parallel
            tasks = []
            for node_id in current_nodes:
                if node_id not in completed_nodes:
                    task = self._execute_node_with_context(
                        node_id, execution, context, compiled_workflow
                    )
                    tasks.append((node_id, task))

            # Wait for all nodes in current batch
            if tasks:
                results = await asyncio.gather(
                    *[task for _, task in tasks], return_exceptions=True
                )

                for (node_id, _), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        # Handle node execution error
                        execution.status = ExecutionStatus.FAILED
                        execution.error = f"Node {node_id} failed: {result}"
                        return

                    completed_nodes.add(node_id)

                    # Determine next nodes based on result
                    node_result = context.execution_results.get(node_id)
                    error = context.node_states.get(node_id, {}).get("error", False)

                    next_batch = compiled_workflow.get_next_nodes(
                        node_id, node_result, error
                    )
                    next_nodes.extend(next_batch)

            # Remove duplicates and already completed nodes
            current_nodes = list(set(next_nodes) - completed_nodes)

            # Update execution status
            execution.current_node = current_nodes[0] if current_nodes else None
            execution.node_states = context.node_states.copy()
            execution.variables = context.variables.copy()

            # Notify progress
            await self._notify_execution_progress(
                execution, completed_nodes, current_nodes
            )

    async def _execute_node_with_context(
        self,
        node_id: str,
        execution: WorkflowExecution,
        context: ExecutionContext,
        compiled_workflow: CompiledWorkflow,
    ):
        """Execute a single node with full context"""
        node = compiled_workflow.node_mapping[node_id]

        # Check for breakpoint
        if node.breakpoint or node_id in context.breakpoints:
            execution.status = ExecutionStatus.PAUSED
            await self._notify_breakpoint(execution, node_id)
            # Wait for resume signal
            while execution.status == ExecutionStatus.PAUSED:
                await asyncio.sleep(0.1)

        # Gather input data from previous nodes
        input_data = await self._gather_node_inputs(node_id, context, compiled_workflow)

        # Execute the node
        start_time = time.time()
        result = await self.node_executor.execute_node(node, context, input_data)
        duration = time.time() - start_time

        # Store results
        context.node_states[node_id] = {
            "status": result.status.value,
            "duration_ms": result.duration_ms,
            "error": result.error is not None,
            "retry_count": result.retry_count,
        }

        if result.output_data is not None:
            context.execution_results[node_id] = result.output_data

        context.node_timings[node_id] = duration

        # Update variables if node produced them
        if result.output_data and isinstance(result.output_data, dict):
            variables_update = result.output_data.get("__variables__", {})
            context.variables.update(variables_update)

        # Notify node completion
        await self._notify_node_complete(execution, node_id, result)

        return result

    async def _gather_node_inputs(
        self,
        node_id: str,
        context: ExecutionContext,
        compiled_workflow: CompiledWorkflow,
    ) -> Dict[str, Any]:
        """Gather input data for a node from connected sources"""
        input_data = {}

        # Find incoming connections
        for source, target, edge_data in compiled_workflow.execution_graph.edges(
            data=True
        ):
            if target == node_id:
                source_result = context.execution_results.get(source, {})
                target_port = edge_data.get("target_port", "input")

                # Apply transformation if specified
                transform = edge_data.get("transform")
                if transform and source_result:
                    # Simple transformation - in production use a proper expression engine
                    try:
                        transformed = eval(transform, {"data": source_result})
                        input_data[target_port] = transformed
                    except Exception:
                        input_data[target_port] = source_result
                else:
                    input_data[target_port] = source_result

        # If no inputs, use trigger data or variables
        if not input_data:
            input_data = context.trigger_data.copy()
            input_data.update(context.variables)

        return input_data

    async def pause_execution(self, execution_id: str) -> bool:
        """Pause workflow execution"""
        if execution_id in self.active_executions:
            context = self.active_executions[execution_id]
            context.should_stop = True

            # Update database
            execution = await self._get_execution(execution_id)
            if execution:
                execution.status = ExecutionStatus.PAUSED
                await self._update_execution(execution)

            return True
        return False

    async def resume_execution(self, execution_id: str) -> bool:
        """Resume paused workflow execution"""
        execution = await self._get_execution(execution_id)
        if execution and execution.status == ExecutionStatus.PAUSED:
            execution.status = ExecutionStatus.RUNNING
            await self._update_execution(execution)

            # Notify resume
            await self._notify_execution_resume(execution)
            return True
        return False

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel workflow execution"""
        if execution_id in self.active_executions:
            context = self.active_executions[execution_id]
            context.should_stop = True

            # Update database
            execution = await self._get_execution(execution_id)
            if execution:
                execution.status = ExecutionStatus.CANCELLED
                execution.completed_at = datetime.utcnow()
                await self._update_execution(execution)

            return True
        return False

    # Database and notification methods

    async def _save_execution(self, execution: WorkflowExecution):
        """Save execution to database"""
        try:
            await self.supabase.client.table("workflow_executions").insert(
                execution.dict()
            ).execute()
        except Exception as e:
            logger.error(f"Failed to save execution: {e}")

    async def _update_execution(self, execution: WorkflowExecution):
        """Update execution in database"""
        try:
            await self.supabase.client.table("workflow_executions").update(
                execution.dict()
            ).eq("id", execution.id).execute()
        except Exception as e:
            logger.error(f"Failed to update execution: {e}")

    async def _get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution from database"""
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
            logger.error(f"Failed to get execution: {e}")

        return None

    async def _notify_execution_start(self, execution: WorkflowExecution):
        """Notify execution start"""
        await self.websocket_manager.broadcast(
            {
                "type": "workflow_execution_start",
                "execution_id": execution.id,
                "workflow_id": execution.workflow_id,
            }
        )

    async def _notify_execution_progress(
        self, execution: WorkflowExecution, completed: set, current: List[str]
    ):
        """Notify execution progress"""
        await self.websocket_manager.broadcast(
            {
                "type": "workflow_execution_progress",
                "execution_id": execution.id,
                "completed_nodes": list(completed),
                "current_nodes": current,
                "status": execution.status.value,
            }
        )

    async def _notify_execution_complete(self, execution: WorkflowExecution):
        """Notify execution completion"""
        await self.websocket_manager.broadcast(
            {
                "type": "workflow_execution_complete",
                "execution_id": execution.id,
                "status": execution.status.value,
                "duration_ms": execution.duration_ms,
                "error": execution.error,
            }
        )

    async def _notify_execution_resume(self, execution: WorkflowExecution):
        """Notify execution resume"""
        await self.websocket_manager.broadcast(
            {"type": "workflow_execution_resume", "execution_id": execution.id}
        )

    async def _notify_node_complete(
        self, execution: WorkflowExecution, node_id: str, result: NodeExecutionResult
    ):
        """Notify node completion"""
        await self.websocket_manager.broadcast(
            {
                "type": "workflow_node_complete",
                "execution_id": execution.id,
                "node_id": node_id,
                "status": result.status.value,
                "duration_ms": result.duration_ms,
                "error": result.error,
            }
        )

    async def _notify_breakpoint(self, execution: WorkflowExecution, node_id: str):
        """Notify breakpoint hit"""
        await self.websocket_manager.broadcast(
            {
                "type": "workflow_breakpoint",
                "execution_id": execution.id,
                "node_id": node_id,
            }
        )
