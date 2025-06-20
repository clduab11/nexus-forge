/**
 * Workflow Serialization Service
 * Handles saving, loading, and validation of workflow definitions
 */

import {
  WorkflowDefinition,
  WorkflowNode,
  WorkflowConnection,
  WorkflowVariable,
  WorkflowTrigger,
  WorkflowSettings,
  NodeType,
  AgentType,
  ConnectionType,
  DataType,
  ExecutionMode,
  TriggerType
} from '../types/workflow.types';

export class WorkflowSerializer {
  private static instance: WorkflowSerializer;

  static getInstance(): WorkflowSerializer {
    if (!WorkflowSerializer.instance) {
      WorkflowSerializer.instance = new WorkflowSerializer();
    }
    return WorkflowSerializer.instance;
  }

  /**
   * Serialize workflow to JSON string
   */
  serialize(workflow: WorkflowDefinition): string {
    try {
      const serialized = {
        ...workflow,
        metadata: {
          ...workflow.metadata,
          created: workflow.metadata.created.toISOString(),
          modified: workflow.metadata.modified.toISOString()
        }
      };
      return JSON.stringify(serialized, null, 2);
    } catch (error) {
      throw new Error(`Failed to serialize workflow: ${error.message}`);
    }
  }

  /**
   * Deserialize JSON string to workflow definition
   */
  deserialize(json: string): WorkflowDefinition {
    try {
      const parsed = JSON.parse(json);
      return {
        ...parsed,
        metadata: {
          ...parsed.metadata,
          created: new Date(parsed.metadata.created),
          modified: new Date(parsed.metadata.modified)
        }
      };
    } catch (error) {
      throw new Error(`Failed to deserialize workflow: ${error.message}`);
    }
  }

  /**
   * Export workflow to file
   */
  async exportToFile(workflow: WorkflowDefinition): Promise<Blob> {
    const serialized = this.serialize(workflow);
    return new Blob([serialized], { type: 'application/json' });
  }

  /**
   * Import workflow from file
   */
  async importFromFile(file: File): Promise<WorkflowDefinition> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (event) => {
        try {
          const json = event.target?.result as string;
          const workflow = this.deserialize(json);
          resolve(workflow);
        } catch (error) {
          reject(new Error(`Failed to import workflow: ${error.message}`));
        }
      };
      
      reader.onerror = () => {
        reject(new Error('Failed to read workflow file'));
      };
      
      reader.readAsText(file);
    });
  }

  /**
   * Validate workflow definition
   */
  validate(workflow: WorkflowDefinition): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    // Validate metadata
    if (!workflow.id) {
      errors.push({
        field: 'id',
        message: 'Workflow ID is required'
      });
    }

    if (!workflow.metadata.name) {
      errors.push({
        field: 'metadata.name',
        message: 'Workflow name is required'
      });
    }

    // Validate nodes
    if (!workflow.nodes || workflow.nodes.length === 0) {
      errors.push({
        field: 'nodes',
        message: 'Workflow must have at least one node'
      });
    } else {
      workflow.nodes.forEach((node, index) => {
        const nodeErrors = this.validateNode(node, index);
        errors.push(...nodeErrors);
      });
    }

    // Validate connections
    workflow.connections.forEach((connection, index) => {
      const connectionErrors = this.validateConnection(connection, workflow.nodes, index);
      errors.push(...connectionErrors);
    });

    // Validate DAG (no cycles)
    const hasCycle = this.detectCycles(workflow.nodes, workflow.connections);
    if (hasCycle) {
      errors.push({
        field: 'connections',
        message: 'Workflow contains cycles. Only directed acyclic graphs are supported'
      });
    }

    // Validate variables
    workflow.variables.forEach((variable, index) => {
      if (!variable.name) {
        errors.push({
          field: `variables[${index}].name`,
          message: 'Variable name is required'
        });
      }
    });

    // Check for warnings
    const orphanNodes = this.findOrphanNodes(workflow.nodes, workflow.connections);
    if (orphanNodes.length > 0) {
      warnings.push({
        field: 'nodes',
        message: `Found ${orphanNodes.length} orphan node(s) with no connections`,
        nodeIds: orphanNodes
      });
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Compress workflow for storage
   */
  compress(workflow: WorkflowDefinition): CompressedWorkflow {
    // Remove redundant data and optimize structure
    const compressed: CompressedWorkflow = {
      v: workflow.version,
      m: {
        n: workflow.metadata.name,
        d: workflow.metadata.description,
        t: workflow.metadata.tags
      },
      n: workflow.nodes.map(node => ({
        i: node.id,
        t: node.type,
        p: [node.position.x, node.position.y],
        d: {
          l: node.data.label,
          a: node.data.agentType,
          c: node.data.config,
          i: node.data.inputs.map(p => ({ i: p.id, n: p.name, t: p.type, r: p.required })),
          o: node.data.outputs.map(p => ({ i: p.id, n: p.name, t: p.type, m: p.multiple }))
        }
      })),
      c: workflow.connections.map(conn => ({
        i: conn.id,
        s: [conn.source.nodeId, conn.source.portId],
        t: [conn.target.nodeId, conn.target.portId],
        y: conn.type
      })),
      var: workflow.variables.map(v => ({
        n: v.name,
        t: v.type,
        v: v.defaultValue,
        s: v.scope
      }))
    };

    return compressed;
  }

  /**
   * Decompress workflow from storage
   */
  decompress(compressed: CompressedWorkflow): WorkflowDefinition {
    const workflow: WorkflowDefinition = {
      id: this.generateId(),
      version: compressed.v,
      metadata: {
        name: compressed.m.n,
        description: compressed.m.d,
        tags: compressed.m.t,
        author: '',
        created: new Date(),
        modified: new Date(),
        category: null,
        estimatedDuration: 0,
        requiredAgents: []
      },
      nodes: compressed.n.map(node => ({
        id: node.i,
        type: node.t as NodeType,
        position: { x: node.p[0], y: node.p[1] },
        data: {
          label: node.d.l,
          agentType: node.d.a as AgentType,
          config: node.d.c,
          inputs: node.d.i.map(p => ({
            id: p.i,
            name: p.n,
            type: p.t as DataType,
            required: p.r
          })),
          outputs: node.d.o.map(p => ({
            id: p.i,
            name: p.n,
            type: p.t as DataType,
            multiple: p.m
          }))
        },
        state: {
          enabled: true,
          locked: false,
          collapsed: false
        }
      })),
      connections: compressed.c.map(conn => ({
        id: conn.i,
        source: {
          nodeId: conn.s[0],
          portId: conn.s[1]
        },
        target: {
          nodeId: conn.t[0],
          portId: conn.t[1]
        },
        type: conn.y as ConnectionType
      })),
      variables: compressed.var.map(v => ({
        name: v.n,
        type: v.t as DataType,
        defaultValue: v.v,
        scope: v.s
      })),
      triggers: [],
      settings: this.getDefaultSettings()
    };

    return workflow;
  }

  /**
   * Clone workflow with new ID
   */
  clone(workflow: WorkflowDefinition, newName?: string): WorkflowDefinition {
    const cloned = JSON.parse(JSON.stringify(workflow));
    
    // Generate new IDs
    cloned.id = this.generateId();
    cloned.metadata.name = newName || `${workflow.metadata.name} (Copy)`;
    cloned.metadata.created = new Date();
    cloned.metadata.modified = new Date();
    
    // Generate new node IDs but maintain references
    const nodeIdMap = new Map<string, string>();
    cloned.nodes.forEach(node => {
      const newId = this.generateId();
      nodeIdMap.set(node.id, newId);
      node.id = newId;
    });
    
    // Update connection references
    cloned.connections.forEach(conn => {
      conn.id = this.generateId();
      conn.source.nodeId = nodeIdMap.get(conn.source.nodeId) || conn.source.nodeId;
      conn.target.nodeId = nodeIdMap.get(conn.target.nodeId) || conn.target.nodeId;
    });
    
    return cloned;
  }

  /**
   * Merge multiple workflows
   */
  merge(workflows: WorkflowDefinition[], name: string): WorkflowDefinition {
    const merged: WorkflowDefinition = {
      id: this.generateId(),
      version: '1.0.0',
      metadata: {
        name,
        description: `Merged from ${workflows.length} workflows`,
        author: '',
        created: new Date(),
        modified: new Date(),
        tags: [],
        category: null,
        estimatedDuration: 0,
        requiredAgents: []
      },
      nodes: [],
      connections: [],
      variables: [],
      triggers: [],
      settings: this.getDefaultSettings()
    };

    // Merge nodes with offset to prevent overlap
    let xOffset = 0;
    workflows.forEach((workflow, index) => {
      const nodeIdMap = new Map<string, string>();
      
      workflow.nodes.forEach(node => {
        const newNode = { ...node };
        newNode.id = this.generateId();
        newNode.position.x += xOffset;
        nodeIdMap.set(node.id, newNode.id);
        merged.nodes.push(newNode);
      });
      
      workflow.connections.forEach(conn => {
        const newConn = { ...conn };
        newConn.id = this.generateId();
        newConn.source.nodeId = nodeIdMap.get(conn.source.nodeId) || conn.source.nodeId;
        newConn.target.nodeId = nodeIdMap.get(conn.target.nodeId) || conn.target.nodeId;
        merged.connections.push(newConn);
      });
      
      // Calculate offset for next workflow
      const maxX = Math.max(...workflow.nodes.map(n => n.position.x));
      xOffset += maxX + 200;
    });

    return merged;
  }

  /**
   * Convert workflow to executable format
   */
  toExecutable(workflow: WorkflowDefinition): ExecutableWorkflow {
    const executable: ExecutableWorkflow = {
      id: workflow.id,
      tasks: [],
      dependencies: new Map(),
      variables: new Map()
    };

    // Create execution order
    const executionOrder = this.topologicalSort(workflow.nodes, workflow.connections);
    
    // Build task list
    executionOrder.forEach(nodeId => {
      const node = workflow.nodes.find(n => n.id === nodeId);
      if (node && node.type === NodeType.AGENT) {
        executable.tasks.push({
          id: node.id,
          agentType: node.data.agentType,
          config: node.data.config,
          inputs: this.getNodeInputs(node, workflow.connections),
          outputs: node.data.outputs.map(o => o.id)
        });
      }
    });

    // Build dependency map
    workflow.connections.forEach(conn => {
      if (!executable.dependencies.has(conn.target.nodeId)) {
        executable.dependencies.set(conn.target.nodeId, []);
      }
      executable.dependencies.get(conn.target.nodeId).push(conn.source.nodeId);
    });

    // Add variables
    workflow.variables.forEach(variable => {
      executable.variables.set(variable.name, variable.defaultValue);
    });

    return executable;
  }

  // Private helper methods
  private validateNode(node: WorkflowNode, index: number): ValidationError[] {
    const errors: ValidationError[] = [];

    if (!node.id) {
      errors.push({
        field: `nodes[${index}].id`,
        message: 'Node ID is required'
      });
    }

    if (!node.type) {
      errors.push({
        field: `nodes[${index}].type`,
        message: 'Node type is required'
      });
    }

    if (!node.data.label) {
      errors.push({
        field: `nodes[${index}].data.label`,
        message: 'Node label is required'
      });
    }

    if (node.type === NodeType.AGENT && !node.data.agentType) {
      errors.push({
        field: `nodes[${index}].data.agentType`,
        message: 'Agent type is required for agent nodes'
      });
    }

    return errors;
  }

  private validateConnection(
    connection: WorkflowConnection,
    nodes: WorkflowNode[],
    index: number
  ): ValidationError[] {
    const errors: ValidationError[] = [];

    if (!connection.id) {
      errors.push({
        field: `connections[${index}].id`,
        message: 'Connection ID is required'
      });
    }

    const sourceNode = nodes.find(n => n.id === connection.source.nodeId);
    const targetNode = nodes.find(n => n.id === connection.target.nodeId);

    if (!sourceNode) {
      errors.push({
        field: `connections[${index}].source`,
        message: `Source node ${connection.source.nodeId} not found`
      });
    }

    if (!targetNode) {
      errors.push({
        field: `connections[${index}].target`,
        message: `Target node ${connection.target.nodeId} not found`
      });
    }

    if (sourceNode && targetNode) {
      const sourcePort = sourceNode.data.outputs.find(p => p.id === connection.source.portId);
      const targetPort = targetNode.data.inputs.find(p => p.id === connection.target.portId);

      if (!sourcePort) {
        errors.push({
          field: `connections[${index}].source.portId`,
          message: `Source port ${connection.source.portId} not found`
        });
      }

      if (!targetPort) {
        errors.push({
          field: `connections[${index}].target.portId`,
          message: `Target port ${connection.target.portId} not found`
        });
      }

      // Type compatibility check
      if (sourcePort && targetPort && sourcePort.type !== targetPort.type && targetPort.type !== DataType.ANY) {
        errors.push({
          field: `connections[${index}]`,
          message: `Type mismatch: ${sourcePort.type} cannot connect to ${targetPort.type}`
        });
      }
    }

    return errors;
  }

  private detectCycles(nodes: WorkflowNode[], connections: WorkflowConnection[]): boolean {
    const adjacencyList = new Map<string, string[]>();
    
    // Build adjacency list
    nodes.forEach(node => adjacencyList.set(node.id, []));
    connections.forEach(conn => {
      adjacencyList.get(conn.source.nodeId)?.push(conn.target.nodeId);
    });

    const visited = new Set<string>();
    const recursionStack = new Set<string>();

    const hasCycle = (nodeId: string): boolean => {
      visited.add(nodeId);
      recursionStack.add(nodeId);

      const neighbors = adjacencyList.get(nodeId) || [];
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          if (hasCycle(neighbor)) return true;
        } else if (recursionStack.has(neighbor)) {
          return true;
        }
      }

      recursionStack.delete(nodeId);
      return false;
    };

    for (const node of nodes) {
      if (!visited.has(node.id)) {
        if (hasCycle(node.id)) return true;
      }
    }

    return false;
  }

  private findOrphanNodes(nodes: WorkflowNode[], connections: WorkflowConnection[]): string[] {
    const connectedNodes = new Set<string>();
    
    connections.forEach(conn => {
      connectedNodes.add(conn.source.nodeId);
      connectedNodes.add(conn.target.nodeId);
    });

    return nodes
      .filter(node => !connectedNodes.has(node.id))
      .map(node => node.id);
  }

  private topologicalSort(nodes: WorkflowNode[], connections: WorkflowConnection[]): string[] {
    const adjacencyList = new Map<string, string[]>();
    const inDegree = new Map<string, number>();
    
    // Initialize
    nodes.forEach(node => {
      adjacencyList.set(node.id, []);
      inDegree.set(node.id, 0);
    });
    
    // Build graph
    connections.forEach(conn => {
      adjacencyList.get(conn.source.nodeId)?.push(conn.target.nodeId);
      inDegree.set(conn.target.nodeId, (inDegree.get(conn.target.nodeId) || 0) + 1);
    });
    
    // Find nodes with no dependencies
    const queue: string[] = [];
    inDegree.forEach((degree, nodeId) => {
      if (degree === 0) queue.push(nodeId);
    });
    
    const sorted: string[] = [];
    
    while (queue.length > 0) {
      const nodeId = queue.shift()!;
      sorted.push(nodeId);
      
      const neighbors = adjacencyList.get(nodeId) || [];
      neighbors.forEach(neighbor => {
        const degree = (inDegree.get(neighbor) || 0) - 1;
        inDegree.set(neighbor, degree);
        if (degree === 0) queue.push(neighbor);
      });
    }
    
    return sorted;
  }

  private getNodeInputs(node: WorkflowNode, connections: WorkflowConnection[]): Map<string, string> {
    const inputs = new Map<string, string>();
    
    connections
      .filter(conn => conn.target.nodeId === node.id)
      .forEach(conn => {
        inputs.set(conn.target.portId, conn.source.nodeId);
      });
    
    return inputs;
  }

  private generateId(): string {
    return `wf_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private getDefaultSettings(): WorkflowSettings {
    return {
      executionMode: ExecutionMode.SEQUENTIAL,
      errorHandling: {
        onNodeError: 'stop',
        onConnectionError: 'stop',
        notifyOnError: true,
        errorTimeout: 300
      },
      timeout: 3600,
      maxRetries: 3,
      logging: {
        level: 'info',
        destinations: [{ type: 'console', config: {} }],
        includeData: false
      },
      notifications: {
        onStart: false,
        onComplete: true,
        onError: true,
        channels: []
      },
      security: {
        authentication: false,
        authorization: [],
        encryption: false,
        auditLog: true
      }
    };
  }
}

// Type definitions for serialization
interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
}

interface ValidationError {
  field: string;
  message: string;
}

interface ValidationWarning {
  field: string;
  message: string;
  nodeIds?: string[];
}

interface CompressedWorkflow {
  v: string;
  m: {
    n: string;
    d: string;
    t: string[];
  };
  n: Array<{
    i: string;
    t: string;
    p: [number, number];
    d: {
      l: string;
      a?: string;
      c: any;
      i: Array<{ i: string; n: string; t: string; r: boolean }>;
      o: Array<{ i: string; n: string; t: string; m: boolean }>;
    };
  }>;
  c: Array<{
    i: string;
    s: [string, string];
    t: [string, string];
    y: string;
  }>;
  var: Array<{
    n: string;
    t: string;
    v: any;
    s: string;
  }>;
}

interface ExecutableWorkflow {
  id: string;
  tasks: ExecutableTask[];
  dependencies: Map<string, string[]>;
  variables: Map<string, any>;
}

interface ExecutableTask {
  id: string;
  agentType: AgentType;
  config: any;
  inputs: Map<string, string>;
  outputs: string[];
}

// Export singleton instance
export const workflowSerializer = WorkflowSerializer.getInstance();