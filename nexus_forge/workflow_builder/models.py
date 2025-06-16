"""
Workflow Builder data models and schemas
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class NodeType(str, Enum):
    """Types of workflow nodes"""

    TRIGGER = "trigger"
    ACTION = "action"
    CONDITION = "condition"
    TRANSFORMATION = "transformation"
    AI_AGENT = "ai_agent"
    SUB_WORKFLOW = "sub_workflow"
    WEBHOOK = "webhook"
    HTTP_REQUEST = "http_request"
    DATABASE = "database"
    EMAIL = "email"
    DELAY = "delay"


class ConnectionType(str, Enum):
    """Types of connections between nodes"""

    DEFAULT = "default"
    SUCCESS = "success"
    ERROR = "error"
    CONDITION_TRUE = "condition_true"
    CONDITION_FALSE = "condition_false"


class ExecutionStatus(str, Enum):
    """Workflow execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class NodePosition(BaseModel):
    """Position of a node on the canvas"""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")


class NodePort(BaseModel):
    """Input/output port for a node"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Port name")
    type: str = Field(..., description="Port data type")
    required: bool = Field(default=True, description="Whether port is required")
    description: Optional[str] = Field(None, description="Port description")


class NodeConfig(BaseModel):
    """Configuration for a workflow node"""

    # Basic settings
    timeout: Optional[int] = Field(None, description="Timeout in seconds")
    retries: int = Field(default=0, description="Number of retries on failure")
    retry_delay: int = Field(default=5, description="Delay between retries in seconds")

    # Conditional execution
    condition: Optional[str] = Field(
        None, description="JavaScript condition expression"
    )

    # Error handling
    continue_on_error: bool = Field(
        default=False, description="Continue workflow on error"
    )
    error_path: Optional[str] = Field(None, description="Path to take on error")

    # Node-specific configuration
    settings: Dict[str, Any] = Field(
        default_factory=dict, description="Node-specific settings"
    )


class WorkflowNode(BaseModel):
    """Individual node in a workflow"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: NodeType = Field(..., description="Node type")
    name: str = Field(..., description="Node display name")
    description: Optional[str] = Field(None, description="Node description")

    # Visual properties
    position: NodePosition = Field(..., description="Position on canvas")

    # Port definitions
    inputs: List[NodePort] = Field(default_factory=list, description="Input ports")
    outputs: List[NodePort] = Field(default_factory=list, description="Output ports")

    # Configuration
    config: NodeConfig = Field(
        default_factory=NodeConfig, description="Node configuration"
    )

    # Metadata
    version: str = Field(default="1.0.0", description="Node version")
    category: str = Field(default="general", description="Node category")
    icon: Optional[str] = Field(None, description="Node icon")
    color: Optional[str] = Field(None, description="Node color")

    # Runtime data
    enabled: bool = Field(default=True, description="Whether node is enabled")
    breakpoint: bool = Field(default=False, description="Whether to pause at this node")


class WorkflowConnection(BaseModel):
    """Connection between workflow nodes"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str = Field(..., description="Source node ID")
    source_port: str = Field(..., description="Source port name")
    target_node_id: str = Field(..., description="Target node ID")
    target_port: str = Field(..., description="Target port name")
    type: ConnectionType = Field(
        default=ConnectionType.DEFAULT, description="Connection type"
    )

    # Visual properties
    label: Optional[str] = Field(None, description="Connection label")
    color: Optional[str] = Field(None, description="Connection color")

    # Configuration
    condition: Optional[str] = Field(None, description="Condition for this connection")
    transform: Optional[str] = Field(None, description="Data transformation")


class WorkflowVariable(BaseModel):
    """Global workflow variable"""

    name: str = Field(..., description="Variable name")
    type: str = Field(..., description="Variable type")
    value: Any = Field(None, description="Variable value")
    description: Optional[str] = Field(None, description="Variable description")
    secret: bool = Field(default=False, description="Whether variable is secret")


class WorkflowTrigger(BaseModel):
    """Workflow trigger configuration"""

    type: str = Field(..., description="Trigger type (manual, webhook, schedule, etc.)")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Trigger configuration"
    )
    enabled: bool = Field(default=True, description="Whether trigger is enabled")


class Workflow(BaseModel):
    """Complete workflow definition"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    version: str = Field(default="1.0.0", description="Workflow version")

    # Structure
    nodes: List[WorkflowNode] = Field(
        default_factory=list, description="Workflow nodes"
    )
    connections: List[WorkflowConnection] = Field(
        default_factory=list, description="Node connections"
    )

    # Configuration
    variables: List[WorkflowVariable] = Field(
        default_factory=list, description="Global variables"
    )
    triggers: List[WorkflowTrigger] = Field(
        default_factory=list, description="Workflow triggers"
    )

    # Metadata
    author_id: str = Field(..., description="Workflow author")
    organization: Optional[str] = Field(None, description="Organization")
    tags: List[str] = Field(default_factory=list, description="Workflow tags")
    category: str = Field(default="general", description="Workflow category")

    # Status
    published: bool = Field(default=False, description="Whether workflow is published")
    public: bool = Field(default=False, description="Whether workflow is public")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Statistics
    executions: int = Field(default=0, description="Number of executions")
    last_executed: Optional[datetime] = Field(None, description="Last execution time")

    @validator("nodes")
    def validate_nodes(cls, v):
        """Validate node IDs are unique"""
        node_ids = [node.id for node in v]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Node IDs must be unique")
        return v

    @validator("connections")
    def validate_connections(cls, v, values):
        """Validate connections reference existing nodes"""
        if "nodes" in values:
            node_ids = {node.id for node in values["nodes"]}
            for conn in v:
                if conn.source_node_id not in node_ids:
                    raise ValueError(f"Source node {conn.source_node_id} not found")
                if conn.target_node_id not in node_ids:
                    raise ValueError(f"Target node {conn.target_node_id} not found")
        return v


class WorkflowExecution(BaseModel):
    """Workflow execution instance"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = Field(..., description="Workflow ID")
    workflow_version: str = Field(..., description="Workflow version")

    # Execution details
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    trigger_type: str = Field(..., description="What triggered the execution")
    trigger_data: Dict[str, Any] = Field(
        default_factory=dict, description="Trigger data"
    )

    # Runtime state
    current_node: Optional[str] = Field(None, description="Currently executing node")
    node_states: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="State of each node"
    )
    variables: Dict[str, Any] = Field(
        default_factory=dict, description="Runtime variable values"
    )

    # Results
    output: Optional[Dict[str, Any]] = Field(None, description="Execution output")
    error: Optional[str] = Field(None, description="Error message if failed")

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)

    # Performance metrics
    duration_ms: Optional[int] = Field(
        None, description="Execution duration in milliseconds"
    )
    nodes_executed: int = Field(default=0, description="Number of nodes executed")

    # User context
    user_id: Optional[str] = Field(None, description="User who triggered execution")
    tenant_id: Optional[str] = Field(None, description="Tenant context")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class NodeExecutionResult(BaseModel):
    """Result of executing a single node"""

    node_id: str = Field(..., description="Node that was executed")
    status: ExecutionStatus = Field(..., description="Execution status")

    # Input/output data
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output data")

    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: Optional[int] = Field(None)

    # Error handling
    error: Optional[str] = Field(None, description="Error message")
    retry_count: int = Field(default=0, description="Number of retries attempted")

    # Debugging
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Debug information")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class WorkflowTemplate(BaseModel):
    """Workflow template for common patterns"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    category: str = Field(..., description="Template category")

    # Template definition
    workflow: Workflow = Field(..., description="Template workflow")

    # Configuration schema
    config_schema: Dict[str, Any] = Field(
        default_factory=dict, description="JSON schema for template configuration"
    )

    # Metadata
    author: str = Field(..., description="Template author")
    tags: List[str] = Field(default_factory=list, description="Template tags")
    difficulty: str = Field(default="beginner", description="Difficulty level")

    # Usage stats
    usage_count: int = Field(default=0, description="Number of times used")
    rating: Optional[float] = Field(None, description="Average rating")

    # Publishing
    published: bool = Field(default=False, description="Whether template is published")

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
