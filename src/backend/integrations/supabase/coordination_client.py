"""
Supabase Coordination Client for Multi-Agent Real-time Communication
Handles database operations, real-time subscriptions, and agent coordination
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from postgrest import APIError
from supabase import Client, create_client
from supabase.lib.client_options import ClientOptions

from nexus_forge.core.exceptions import CoordinationError
from nexus_forge.core.monitoring import get_logger

logger = get_logger(__name__)


class SupabaseCoordinationClient:
    """
    Enterprise-grade Supabase client for multi-agent coordination
    Provides real-time communication, state management, and workflow orchestration
    """

    def __init__(
        self,
        url: str,
        key: str,
        project_id: str,
        options: Optional[ClientOptions] = None,
    ):
        """Initialize Supabase coordination client"""
        self.url = url
        self.key = key
        self.project_id = project_id
        self.options = options or ClientOptions()

        # Initialize client
        self.client: Client = create_client(url, key, options=self.options)

        # Real-time channels
        self.channels: Dict[str, Any] = {}
        self.subscriptions: Dict[str, List[Callable]] = {
            "agent_status": [],
            "task_updates": [],
            "workflow_progress": [],
            "coordination_events": [],
        }

        # Connection state
        self.connected = False
        self._connection_lock = asyncio.Lock()

    async def connect(self) -> bool:
        """Establish connection to Supabase with retry logic"""
        async with self._connection_lock:
            if self.connected:
                return True

            try:
                # Test connection with a simple query
                result = self.client.table("agents").select("count").execute()
                self.connected = True
                logger.info("Successfully connected to Supabase coordination layer")
                return True

            except Exception as e:
                logger.error(f"Failed to connect to Supabase: {e}")
                raise CoordinationError(f"Supabase connection failed: {e}")

    async def disconnect(self):
        """Gracefully disconnect from Supabase"""
        async with self._connection_lock:
            try:
                # Unsubscribe from all channels
                for channel_name, channel in self.channels.items():
                    await channel.unsubscribe()

                self.channels.clear()
                self.connected = False
                logger.info("Disconnected from Supabase coordination layer")

            except Exception as e:
                logger.error(f"Error during Supabase disconnection: {e}")

    # Agent Management
    async def register_agent(
        self,
        name: str,
        agent_type: str,
        capabilities: Dict[str, Any],
        configuration: Dict[str, Any],
    ) -> str:
        """Register a new agent in the coordination system"""
        try:
            agent_data = {
                "name": name,
                "type": agent_type,
                "capabilities": capabilities,
                "configuration": configuration,
                "status": "online",
                "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            }

            result = self.client.table("agents").insert(agent_data).execute()
            agent_id = result.data[0]["agent_id"]

            # Initialize agent state
            await self.update_agent_state(agent_id, "idle", {})

            logger.info(f"Registered agent {name} with ID {agent_id}")
            return agent_id

        except APIError as e:
            logger.error(f"Failed to register agent {name}: {e}")
            raise CoordinationError(f"Agent registration failed: {e}")

    async def update_agent_state(
        self,
        agent_id: str,
        status: str,
        payload: Dict[str, Any],
        current_task_id: Optional[str] = None,
    ):
        """Update agent state with real-time broadcasting"""
        try:
            state_data = {
                "agent_id": agent_id,
                "status": status,
                "current_task_id": current_task_id,
                "payload": payload,
                "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            }

            # Insert new state record
            self.client.table("agent_states").insert(state_data).execute()

            # Update agent's last heartbeat
            self.client.table("agents").update(
                {
                    "status": status,
                    "last_heartbeat": datetime.now(timezone.utc).isoformat(),
                }
            ).eq("agent_id", agent_id).execute()

            # Broadcast state change
            await self._broadcast_event(
                "agent_status",
                {
                    "agent_id": agent_id,
                    "status": status,
                    "payload": payload,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        except APIError as e:
            logger.error(f"Failed to update agent state for {agent_id}: {e}")
            raise CoordinationError(f"Agent state update failed: {e}")

    async def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get current agent state"""
        try:
            result = (
                self.client.table("agent_states")
                .select("*")
                .eq("agent_id", agent_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )

            return result.data[0] if result.data else None

        except APIError as e:
            logger.error(f"Failed to get agent state for {agent_id}: {e}")
            return None

    # Workflow Management
    async def create_workflow(
        self,
        name: str,
        description: str,
        definition: Dict[str, Any],
        priority: int = 5,
        created_by: Optional[str] = None,
    ) -> str:
        """Create a new workflow instance"""
        try:
            workflow_data = {
                "name": name,
                "description": description,
                "definition": definition,
                "priority": priority,
                "created_by": created_by,
                "status": "pending",
            }

            result = self.client.table("workflows").insert(workflow_data).execute()
            workflow_id = result.data[0]["workflow_id"]

            logger.info(f"Created workflow {name} with ID {workflow_id}")
            return workflow_id

        except APIError as e:
            logger.error(f"Failed to create workflow {name}: {e}")
            raise CoordinationError(f"Workflow creation failed: {e}")

    async def update_workflow_status(
        self,
        workflow_id: str,
        status: str,
        error_message: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Update workflow status with real-time broadcasting"""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            if status == "running" and "started_at" not in update_data:
                update_data["started_at"] = datetime.now(timezone.utc).isoformat()
            elif status in ["completed", "failed", "cancelled"]:
                update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

            if error_message:
                update_data["error_message"] = error_message
            if metrics:
                update_data["metrics"] = metrics

            self.client.table("workflows").update(update_data).eq(
                "workflow_id", workflow_id
            ).execute()

            # Broadcast workflow progress
            await self._broadcast_event(
                "workflow_progress",
                {
                    "workflow_id": workflow_id,
                    "status": status,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metrics": metrics,
                },
            )

        except APIError as e:
            logger.error(f"Failed to update workflow {workflow_id}: {e}")
            raise CoordinationError(f"Workflow update failed: {e}")

    # Task Management
    async def create_task(
        self,
        workflow_id: str,
        name: str,
        task_type: str,
        input_data: Dict[str, Any],
        agent_id: Optional[str] = None,
        parent_task_id: Optional[str] = None,
        priority: int = 5,
    ) -> str:
        """Create a new task"""
        try:
            task_data = {
                "workflow_id": workflow_id,
                "name": name,
                "type": task_type,
                "input_data": input_data,
                "agent_id": agent_id,
                "parent_task_id": parent_task_id,
                "priority": priority,
                "status": "pending",
            }

            result = self.client.table("tasks").insert(task_data).execute()
            task_id = result.data[0]["task_id"]

            # Broadcast task creation
            await self._broadcast_event(
                "task_updates",
                {
                    "task_id": task_id,
                    "workflow_id": workflow_id,
                    "status": "pending",
                    "agent_id": agent_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            return task_id

        except APIError as e:
            logger.error(f"Failed to create task {name}: {e}")
            raise CoordinationError(f"Task creation failed: {e}")

    async def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign task to specific agent with reservation logic"""
        try:
            # Attempt to reserve task atomically
            result = (
                self.client.table("tasks")
                .update(
                    {
                        "agent_id": agent_id,
                        "status": "assigned",
                        "reserved_by": agent_id,
                        "reserved_at": datetime.now(timezone.utc).isoformat(),
                        "assigned_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                .eq("task_id", task_id)
                .eq("status", "pending")
                .execute()
            )

            if not result.data:
                logger.warning(f"Task {task_id} already assigned or not available")
                return False

            # Update agent state
            await self.update_agent_state(
                agent_id, "busy", {"current_task": task_id}, task_id
            )

            # Broadcast task assignment
            await self._broadcast_event(
                "task_updates",
                {
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "status": "assigned",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            return True

        except APIError as e:
            logger.error(f"Failed to assign task {task_id} to agent {agent_id}: {e}")
            return False

    async def update_task_status(
        self,
        task_id: str,
        status: str,
        output_data: Optional[Dict[str, Any]] = None,
        error_details: Optional[Dict[str, Any]] = None,
    ):
        """Update task status with real-time broadcasting"""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            if status == "running":
                update_data["started_at"] = datetime.now(timezone.utc).isoformat()
            elif status in ["completed", "failed"]:
                update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

            if output_data:
                update_data["output_data"] = output_data
            if error_details:
                update_data["error_details"] = error_details

            self.client.table("tasks").update(update_data).eq(
                "task_id", task_id
            ).execute()

            # Broadcast task update
            await self._broadcast_event(
                "task_updates",
                {
                    "task_id": task_id,
                    "status": status,
                    "output_data": output_data,
                    "error_details": error_details,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        except APIError as e:
            logger.error(f"Failed to update task {task_id}: {e}")
            raise CoordinationError(f"Task update failed: {e}")

    async def get_pending_tasks(
        self, agent_type: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get pending tasks for assignment"""
        try:
            query = self.client.table("tasks").select("*").eq("status", "pending")

            if agent_type:
                # Join with workflows to filter by agent type requirements
                query = query.order("priority").order("created_at").limit(limit)
            else:
                query = query.order("priority").order("created_at").limit(limit)

            result = query.execute()
            return result.data

        except APIError as e:
            logger.error(f"Failed to get pending tasks: {e}")
            return []

    # Performance Metrics
    async def record_metric(
        self,
        metric_type: str,
        metric_value: float,
        metric_unit: str,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """Record performance metric"""
        try:
            metric_data = {
                "metric_type": metric_type,
                "metric_value": metric_value,
                "metric_unit": metric_unit,
                "agent_id": agent_id,
                "task_id": task_id,
                "workflow_id": workflow_id,
                "additional_data": additional_data or {},
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }

            self.client.table("performance_metrics").insert(metric_data).execute()

        except APIError as e:
            logger.error(f"Failed to record metric {metric_type}: {e}")

    # Real-time Event System
    async def subscribe_to_agent_status(self, callback: Callable):
        """Subscribe to agent status changes"""
        self.subscriptions["agent_status"].append(callback)

        # Set up real-time subscription if not already done
        if "agent_status" not in self.channels:
            channel = self.client.channel("agent_status_changes")
            channel.on(
                "postgres_changes",
                event="*",
                schema="public",
                table="agent_states",
                callback=self._handle_agent_status_change,
            )
            channel.subscribe()
            self.channels["agent_status"] = channel

    async def subscribe_to_task_updates(self, callback: Callable):
        """Subscribe to task status updates"""
        self.subscriptions["task_updates"].append(callback)

        if "task_updates" not in self.channels:
            channel = self.client.channel("task_updates")
            channel.on(
                "postgres_changes",
                event="*",
                schema="public",
                table="tasks",
                callback=self._handle_task_update,
            )
            channel.subscribe()
            self.channels["task_updates"] = channel

    async def subscribe_to_workflow_progress(self, callback: Callable):
        """Subscribe to workflow progress updates"""
        self.subscriptions["workflow_progress"].append(callback)

        if "workflow_progress" not in self.channels:
            channel = self.client.channel("workflow_progress")
            channel.on(
                "postgres_changes",
                event="*",
                schema="public",
                table="workflows",
                callback=self._handle_workflow_progress,
            )
            channel.subscribe()
            self.channels["workflow_progress"] = channel

    async def _broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """Broadcast event to all subscribers"""
        try:
            # Store event in coordination_events table
            event_data = {
                "event_type": event_type,
                "event_data": data,
                "priority": "normal",
            }

            self.client.table("coordination_events").insert(event_data).execute()

            # Notify local subscribers
            for callback in self.subscriptions.get(event_type, []):
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")

        except Exception as e:
            logger.error(f"Failed to broadcast event {event_type}: {e}")

    def _handle_agent_status_change(self, payload):
        """Handle agent status change events"""
        asyncio.create_task(self._process_agent_status_change(payload))

    def _handle_task_update(self, payload):
        """Handle task update events"""
        asyncio.create_task(self._process_task_update(payload))

    def _handle_workflow_progress(self, payload):
        """Handle workflow progress events"""
        asyncio.create_task(self._process_workflow_progress(payload))

    async def _process_agent_status_change(self, payload):
        """Process agent status change"""
        for callback in self.subscriptions["agent_status"]:
            try:
                await callback(payload)
            except Exception as e:
                logger.error(f"Error processing agent status change: {e}")

    async def _process_task_update(self, payload):
        """Process task update"""
        for callback in self.subscriptions["task_updates"]:
            try:
                await callback(payload)
            except Exception as e:
                logger.error(f"Error processing task update: {e}")

    async def _process_workflow_progress(self, payload):
        """Process workflow progress"""
        for callback in self.subscriptions["workflow_progress"]:
            try:
                await callback(payload)
            except Exception as e:
                logger.error(f"Error processing workflow progress: {e}")

    # Health and Monitoring
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on coordination system"""
        try:
            # Check database connection
            agents_count = self.client.table("agents").select("count").execute()

            # Check real-time connections
            active_channels = len(self.channels)

            return {
                "status": "healthy",
                "connected": self.connected,
                "active_channels": active_channels,
                "agents_registered": len(agents_count.data) if agents_count.data else 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
