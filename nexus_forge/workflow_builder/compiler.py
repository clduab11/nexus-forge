"""
Workflow Compiler
Converts visual workflow definitions into executable format
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import networkx as nx
from collections import defaultdict, deque

from .models import (
    Workflow, WorkflowNode, WorkflowConnection, NodeType,
    ConnectionType, ExecutionStatus
)
from ..core.exceptions import ValidationError


class CompiledWorkflow:
    """Compiled workflow ready for execution"""
    
    def __init__(
        self,
        workflow_id: str,
        execution_graph: nx.DiGraph,
        node_mapping: Dict[str, WorkflowNode],
        entry_points: List[str],
        variables: Dict[str, Any]
    ):
        self.workflow_id = workflow_id
        self.execution_graph = execution_graph
        self.node_mapping = node_mapping
        self.entry_points = entry_points
        self.variables = variables
        
        # Pre-compute execution order
        self.execution_order = self._compute_execution_order()
        
        # Build connection mappings
        self.connections = self._build_connection_mappings()
    
    def _compute_execution_order(self) -> List[str]:
        """Compute topological execution order"""
        try:
            return list(nx.topological_sort(self.execution_graph))
        except nx.NetworkXError:
            # Graph has cycles - return empty list
            return []
    
    def _build_connection_mappings(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build connection mappings for each node"""
        connections = defaultdict(list)
        
        for source, target, data in self.execution_graph.edges(data=True):
            connections[source].append({
                "target": target,
                "connection_type": data.get("connection_type", ConnectionType.DEFAULT),
                "condition": data.get("condition"),
                "transform": data.get("transform"),
                "source_port": data.get("source_port"),
                "target_port": data.get("target_port")
            })
        
        return dict(connections)
    
    def get_next_nodes(
        self, 
        current_node: str, 
        execution_result: Any = None,
        error: bool = False
    ) -> List[str]:
        """Get next nodes to execute based on current node and result"""
        if current_node not in self.connections:
            return []
        
        next_nodes = []
        
        for connection in self.connections[current_node]:
            conn_type = connection["connection_type"]
            condition = connection["condition"]
            
            # Determine if this connection should be taken
            should_take = False
            
            if error and conn_type == ConnectionType.ERROR:
                should_take = True
            elif not error and conn_type in [ConnectionType.DEFAULT, ConnectionType.SUCCESS]:
                should_take = True
            elif conn_type == ConnectionType.CONDITION_TRUE and execution_result:
                should_take = True
            elif conn_type == ConnectionType.CONDITION_FALSE and not execution_result:
                should_take = True
            
            # Evaluate additional condition if present
            if should_take and condition:
                should_take = self._evaluate_condition(condition, execution_result)
            
            if should_take:
                next_nodes.append(connection["target"])
        
        return next_nodes
    
    def _evaluate_condition(self, condition: str, context: Any) -> bool:
        """Evaluate a JavaScript-like condition"""
        # Simplified condition evaluation
        # In production, use a safe JavaScript evaluator
        try:
            # Replace common JavaScript expressions with Python equivalents
            python_condition = condition.replace("&&", " and ").replace("||", " or ")
            
            # Create evaluation context
            eval_context = {
                "result": context,
                "data": context,
                "output": context
            }
            
            return bool(eval(python_condition, {"__builtins__": {}}, eval_context))
        except Exception:
            return False


class WorkflowCompiler:
    """Compiles visual workflows into executable format"""
    
    def compile(self, workflow: Workflow) -> CompiledWorkflow:
        """
        Compile a workflow into executable format
        
        Args:
            workflow: Visual workflow definition
            
        Returns:
            Compiled workflow ready for execution
            
        Raises:
            ValidationError: If workflow is invalid
        """
        # Validate workflow structure
        self._validate_workflow(workflow)
        
        # Build execution graph
        execution_graph = self._build_execution_graph(workflow)
        
        # Validate graph properties
        self._validate_graph(execution_graph, workflow)
        
        # Create node mapping
        node_mapping = {node.id: node for node in workflow.nodes}
        
        # Find entry points
        entry_points = self._find_entry_points(workflow)
        
        # Prepare variables
        variables = {var.name: var.value for var in workflow.variables}
        
        return CompiledWorkflow(
            workflow_id=workflow.id,
            execution_graph=execution_graph,
            node_mapping=node_mapping,
            entry_points=entry_points,
            variables=variables
        )
    
    def _validate_workflow(self, workflow: Workflow) -> None:
        """Validate workflow structure"""
        if not workflow.nodes:
            raise ValidationError("Workflow must have at least one node")
        
        # Check for duplicate node IDs (already validated by Pydantic)
        node_ids = {node.id for node in workflow.nodes}
        
        # Validate connections reference existing nodes
        for connection in workflow.connections:
            if connection.source_node_id not in node_ids:
                raise ValidationError(
                    f"Connection references non-existent source node: {connection.source_node_id}"
                )
            if connection.target_node_id not in node_ids:
                raise ValidationError(
                    f"Connection references non-existent target node: {connection.target_node_id}"
                )
        
        # Validate node ports
        self._validate_node_ports(workflow)
        
        # Check for trigger nodes
        trigger_nodes = [n for n in workflow.nodes if n.type == NodeType.TRIGGER]
        if not trigger_nodes and not workflow.triggers:
            raise ValidationError("Workflow must have at least one trigger")
    
    def _validate_node_ports(self, workflow: Workflow) -> None:
        """Validate node port connections"""
        node_mapping = {node.id: node for node in workflow.nodes}
        
        for connection in workflow.connections:
            source_node = node_mapping[connection.source_node_id]
            target_node = node_mapping[connection.target_node_id]
            
            # Check source port exists
            source_ports = {port.name for port in source_node.outputs}
            if connection.source_port not in source_ports:
                raise ValidationError(
                    f"Source port '{connection.source_port}' not found in node '{source_node.name}'"
                )
            
            # Check target port exists
            target_ports = {port.name for port in target_node.inputs}
            if connection.target_port not in target_ports:
                raise ValidationError(
                    f"Target port '{connection.target_port}' not found in node '{target_node.name}'"
                )
    
    def _build_execution_graph(self, workflow: Workflow) -> nx.DiGraph:
        """Build NetworkX execution graph"""
        graph = nx.DiGraph()
        
        # Add nodes
        for node in workflow.nodes:
            graph.add_node(node.id, node=node)
        
        # Add edges
        for connection in workflow.connections:
            graph.add_edge(
                connection.source_node_id,
                connection.target_node_id,
                connection_id=connection.id,
                connection_type=connection.type,
                condition=connection.condition,
                transform=connection.transform,
                source_port=connection.source_port,
                target_port=connection.target_port
            )
        
        return graph
    
    def _validate_graph(self, graph: nx.DiGraph, workflow: Workflow) -> None:
        """Validate graph properties"""
        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            cycle_names = []
            node_mapping = {node.id: node.name for node in workflow.nodes}
            
            for cycle in cycles[:3]:  # Show first 3 cycles
                cycle_names.append(" -> ".join(node_mapping[node_id] for node_id in cycle))
            
            raise ValidationError(
                f"Workflow contains cycles: {', '.join(cycle_names)}"
            )
        
        # Check for unreachable nodes
        entry_points = self._find_entry_points(workflow)
        if entry_points:
            reachable = set()
            for entry in entry_points:
                reachable.update(nx.descendants(graph, entry))
                reachable.add(entry)
            
            all_nodes = set(graph.nodes())
            unreachable = all_nodes - reachable
            
            if unreachable:
                node_mapping = {node.id: node.name for node in workflow.nodes}
                unreachable_names = [node_mapping[node_id] for node_id in unreachable]
                raise ValidationError(
                    f"Unreachable nodes detected: {', '.join(unreachable_names)}"
                )
    
    def _find_entry_points(self, workflow: Workflow) -> List[str]:
        """Find workflow entry points (nodes with no incoming connections)"""
        incoming = set()
        all_nodes = set()
        
        for node in workflow.nodes:
            all_nodes.add(node.id)
        
        for connection in workflow.connections:
            incoming.add(connection.target_node_id)
        
        # Entry points are nodes with no incoming connections
        entry_points = list(all_nodes - incoming)
        
        # If no natural entry points, look for trigger nodes
        if not entry_points:
            trigger_nodes = [
                node.id for node in workflow.nodes 
                if node.type == NodeType.TRIGGER
            ]
            entry_points = trigger_nodes
        
        return entry_points
    
    def validate_node_configuration(self, node: WorkflowNode) -> List[str]:
        """
        Validate node-specific configuration
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Type-specific validation
        if node.type == NodeType.HTTP_REQUEST:
            errors.extend(self._validate_http_node(node))
        elif node.type == NodeType.DATABASE:
            errors.extend(self._validate_database_node(node))
        elif node.type == NodeType.AI_AGENT:
            errors.extend(self._validate_ai_agent_node(node))
        elif node.type == NodeType.CONDITION:
            errors.extend(self._validate_condition_node(node))
        elif node.type == NodeType.EMAIL:
            errors.extend(self._validate_email_node(node))
        
        # General configuration validation
        if node.config.timeout and node.config.timeout <= 0:
            errors.append("Timeout must be positive")
        
        if node.config.retries < 0:
            errors.append("Retries cannot be negative")
        
        return errors
    
    def _validate_http_node(self, node: WorkflowNode) -> List[str]:
        """Validate HTTP request node"""
        errors = []
        settings = node.config.settings
        
        if "url" not in settings:
            errors.append("HTTP node requires URL")
        elif not settings["url"].startswith(("http://", "https://")):
            errors.append("HTTP URL must start with http:// or https://")
        
        if "method" not in settings:
            errors.append("HTTP node requires method")
        elif settings["method"] not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
            errors.append("Invalid HTTP method")
        
        return errors
    
    def _validate_database_node(self, node: WorkflowNode) -> List[str]:
        """Validate database node"""
        errors = []
        settings = node.config.settings
        
        if "connection" not in settings:
            errors.append("Database node requires connection")
        
        if "query" not in settings:
            errors.append("Database node requires query")
        
        return errors
    
    def _validate_ai_agent_node(self, node: WorkflowNode) -> List[str]:
        """Validate AI agent node"""
        errors = []
        settings = node.config.settings
        
        if "agent_id" not in settings and "agent_type" not in settings:
            errors.append("AI agent node requires agent_id or agent_type")
        
        return errors
    
    def _validate_condition_node(self, node: WorkflowNode) -> List[str]:
        """Validate condition node"""
        errors = []
        settings = node.config.settings
        
        if "condition" not in settings:
            errors.append("Condition node requires condition expression")
        
        # Basic syntax check for condition
        condition = settings.get("condition", "")
        if condition:
            try:
                # Try to parse as Python expression
                compile(condition, "<condition>", "eval")
            except SyntaxError:
                errors.append("Invalid condition syntax")
        
        return errors
    
    def _validate_email_node(self, node: WorkflowNode) -> List[str]:
        """Validate email node"""
        errors = []
        settings = node.config.settings
        
        required_fields = ["to", "subject", "body"]
        for field in required_fields:
            if field not in settings:
                errors.append(f"Email node requires {field}")
        
        # Validate email format
        if "to" in settings:
            email = settings["to"]
            if "@" not in email:
                errors.append("Invalid email address format")
        
        return errors
    
    def generate_execution_plan(self, compiled_workflow: CompiledWorkflow) -> Dict[str, Any]:
        """Generate human-readable execution plan"""
        plan = {
            "workflow_id": compiled_workflow.workflow_id,
            "total_nodes": len(compiled_workflow.node_mapping),
            "entry_points": [],
            "execution_paths": [],
            "variables": list(compiled_workflow.variables.keys()),
            "estimated_duration": self._estimate_duration(compiled_workflow)
        }
        
        # Document entry points
        for entry_id in compiled_workflow.entry_points:
            node = compiled_workflow.node_mapping[entry_id]
            plan["entry_points"].append({
                "id": entry_id,
                "name": node.name,
                "type": node.type.value
            })
        
        # Generate execution paths
        for entry_id in compiled_workflow.entry_points:
            path = self._trace_execution_path(compiled_workflow, entry_id)
            plan["execution_paths"].append({
                "start": entry_id,
                "nodes": path
            })
        
        return plan
    
    def _estimate_duration(self, compiled_workflow: CompiledWorkflow) -> int:
        """Estimate workflow execution duration in seconds"""
        total_duration = 0
        
        for node in compiled_workflow.node_mapping.values():
            # Base duration by node type
            if node.type == NodeType.HTTP_REQUEST:
                duration = 2  # 2 seconds for HTTP calls
            elif node.type == NodeType.DATABASE:
                duration = 1  # 1 second for DB queries
            elif node.type == NodeType.AI_AGENT:
                duration = 10  # 10 seconds for AI processing
            elif node.type == NodeType.EMAIL:
                duration = 3  # 3 seconds for email sending
            else:
                duration = 1  # 1 second for other operations
            
            # Add configured timeout if present
            if node.config.timeout:
                duration = min(duration, node.config.timeout)
            
            total_duration += duration
        
        return total_duration
    
    def _trace_execution_path(
        self, compiled_workflow: CompiledWorkflow, start_node: str
    ) -> List[Dict[str, str]]:
        """Trace execution path from start node"""
        path = []
        visited = set()
        queue = deque([start_node])
        
        while queue and len(path) < 100:  # Prevent infinite loops
            current = queue.popleft()
            
            if current in visited:
                continue
            
            visited.add(current)
            node = compiled_workflow.node_mapping[current]
            
            path.append({
                "id": current,
                "name": node.name,
                "type": node.type.value
            })
            
            # Add next nodes to queue
            next_nodes = compiled_workflow.get_next_nodes(current)
            queue.extend(next_nodes)
        
        return path