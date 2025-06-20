/**
 * Workflow Executor Service
 * Handles workflow execution and integration with the orchestrator
 */

import {
  WorkflowDefinition,
  WorkflowNode,
  WorkflowConnection,
  WorkflowExecution,
  NodeExecutionState,
  NodeExecutionStatus,
  ExecutionStatus,
  NodeType,
  ConnectionType,
  AgentType
} from '../types/workflow.types';
import { nexusForgeApi } from './nexusForgeApi';
import { workflowSerializer } from './workflowSerializer';

export interface ExecutionOptions {
  debug?: boolean;
  parallel?: boolean;
  maxConcurrency?: number;
  timeout?: number;
  onNodeStart?: (nodeId: string) => void;
  onNodeComplete?: (nodeId: string, output: any) => void;
  onNodeError?: (nodeId: string, error: Error) => void;
  onProgress?: (progress: number) => void;
  // Performance optimization options
  enableCaching?: boolean;
  cacheTTL?: number;
  batchSize?: number;
  workerPoolSize?: number;
  enableMetrics?: boolean;
  useWebSocket?: boolean;
}

export class WorkflowExecutor {
  private static instance: WorkflowExecutor;
  private activeExecutions: Map<string, WorkflowExecution> = new Map();
  private nodeStates: Map<string, NodeExecutionState> = new Map();
  
  // Performance optimization components
  private resultCache: Map<string, { result: any; timestamp: number }> = new Map();
  private workerPool: Promise<void>[] = [];
  private metricsCollector: Map<string, any> = new Map();
  private connectionPool: Map<string, any> = new Map();
  private websocketConnections: Map<string, WebSocket> = new Map();
  private pendingBatches: Map<string, any[]> = new Map();
  private batchTimers: Map<string, NodeJS.Timeout> = new Map();
  
  static getInstance(): WorkflowExecutor {
    if (!WorkflowExecutor.instance) {
      WorkflowExecutor.instance = new WorkflowExecutor();
    }
    return WorkflowExecutor.instance;
  }
  
  /**
   * Execute a workflow
   */
  async execute(
    workflow: WorkflowDefinition,
    options: ExecutionOptions = {}
  ): Promise<WorkflowExecution> {
    // Validate workflow first
    const validation = workflowSerializer.validate(workflow);
    if (!validation.isValid) {
      throw new Error(`Workflow validation failed: ${validation.errors.join(', ')}`);
    }
    
    // Create execution instance
    const execution: WorkflowExecution = {
      id: `exec_${Date.now()}`,
      workflowId: workflow.id,
      status: ExecutionStatus.INITIALIZING,
      startTime: new Date(),
      nodeStates: new Map(),
      variables: new Map(workflow.variables.map(v => [v.name, v.defaultValue])),
      logs: [],
      metrics: {
        totalDuration: 0,
        nodeDurations: new Map(),
        resourceUsage: {
          cpuUsage: 0,
          memoryUsage: 0,
          networkUsage: 0,
          apiCalls: 0
        },
        dataProcessed: {
          inputSize: 0,
          outputSize: 0,
          recordsProcessed: 0,
          transformations: 0
        }
      }
    };
    
    this.activeExecutions.set(execution.id, execution);
    
    try {
      // Initialize execution
      await this.initializeExecution(workflow, execution);
      
      // Execute based on execution mode
      if (workflow.settings.executionMode === 'parallel' || options.parallel) {
        await this.executeParallel(workflow, execution, options);
      } else {
        await this.executeSequential(workflow, execution, options);
      }
      
      // Complete execution
      execution.status = ExecutionStatus.COMPLETED;
      execution.endTime = new Date();
      execution.metrics.totalDuration = 
        execution.endTime.getTime() - execution.startTime.getTime();
      
    } catch (error) {
      execution.status = ExecutionStatus.FAILED;
      execution.endTime = new Date();
      execution.error = {
        code: 'EXECUTION_ERROR',
        message: error.message,
        stack: error.stack
      };
      throw error;
    } finally {
      this.activeExecutions.delete(execution.id);
    }
    
    return execution;
  }
  
  /**
   * Cancel an active execution
   */
  async cancel(executionId: string): Promise<void> {
    const execution = this.activeExecutions.get(executionId);
    if (!execution) {
      throw new Error(`Execution ${executionId} not found`);
    }
    
    execution.status = ExecutionStatus.CANCELLED;
    execution.endTime = new Date();
    
    // Cancel all running nodes
    execution.nodeStates.forEach((state, nodeId) => {
      if (state.status === NodeExecutionStatus.RUNNING) {
        state.status = NodeExecutionStatus.SKIPPED;
        state.endTime = new Date();
      }
    });
  }
  
  /**
   * Initialize execution
   */
  private async initializeExecution(
    workflow: WorkflowDefinition,
    execution: WorkflowExecution
  ): Promise<void> {
    execution.status = ExecutionStatus.RUNNING;
    
    // Initialize node states
    workflow.nodes.forEach(node => {
      const state: NodeExecutionState = {
        nodeId: node.id,
        status: NodeExecutionStatus.WAITING,
        progress: 0,
        retries: 0
      };
      execution.nodeStates.set(node.id, state);
    });
    
    // Log start
    this.log(execution, 'info', 'Workflow execution started');
  }
  
  /**
   * Execute workflow sequentially
   */
  private async executeSequential(
    workflow: WorkflowDefinition,
    execution: WorkflowExecution,
    options: ExecutionOptions
  ): Promise<void> {
    // Get execution order
    const executionOrder = this.getExecutionOrder(workflow);
    
    // Execute nodes in order
    for (const nodeId of executionOrder) {
      if (execution.status === ExecutionStatus.CANCELLED) break;
      
      const node = workflow.nodes.find(n => n.id === nodeId);
      if (!node) continue;
      
      await this.executeNode(node, workflow, execution, options);
    }
  }
  
  /**
   * Execute workflow in parallel with enhanced concurrency control
   */
  private async executeParallel(
    workflow: WorkflowDefinition,
    execution: WorkflowExecution,
    options: ExecutionOptions
  ): Promise<void> {
    const maxConcurrency = options.maxConcurrency || 5;
    const workerPoolSize = options.workerPoolSize || maxConcurrency * 2;
    const executionOrder = this.getParallelExecutionGroups(workflow);
    
    // Initialize worker pool
    const semaphore = this.createSemaphore(maxConcurrency);
    const taskQueue: Array<() => Promise<void>> = [];
    
    // Prepare all tasks
    for (const group of executionOrder) {
      if (execution.status === ExecutionStatus.CANCELLED) break;
      
      for (const nodeId of group) {
        const node = workflow.nodes.find(n => n.id === nodeId);
        if (node) {
          taskQueue.push(async () => {
            await semaphore.acquire();
            try {
              await this.executeNode(node, workflow, execution, options);
            } finally {
              semaphore.release();
            }
          });
        }
      }
    }
    
    // Execute with worker pool pattern
    const workers = Array(workerPoolSize).fill(null).map(async () => {
      while (taskQueue.length > 0) {
        const task = taskQueue.shift();
        if (task) {
          await task();
        }
      }
    });
    
    await Promise.all(workers);
  }
  
  /**
   * Create a semaphore for concurrency control
   */
  private createSemaphore(limit: number) {
    let current = 0;
    const queue: Array<() => void> = [];
    
    return {
      acquire: async () => {
        if (current < limit) {
          current++;
          return;
        }
        
        await new Promise<void>(resolve => {
          queue.push(resolve);
        });
      },
      release: () => {
        current--;
        const next = queue.shift();
        if (next) {
          current++;
          next();
        }
      }
    };
  }
  
  /**
   * Execute a single node
   */
  private async executeNode(
    node: WorkflowNode,
    workflow: WorkflowDefinition,
    execution: WorkflowExecution,
    options: ExecutionOptions
  ): Promise<void> {
    const nodeState = execution.nodeStates.get(node.id)!;
    
    try {
      // Check if node should be skipped
      if (!node.state.enabled || await this.shouldSkipNode(node, workflow, execution)) {
        nodeState.status = NodeExecutionStatus.SKIPPED;
        return;
      }
      
      // Start node execution
      nodeState.status = NodeExecutionStatus.RUNNING;
      nodeState.startTime = new Date();
      nodeState.progress = 0;
      
      if (options.onNodeStart) {
        options.onNodeStart(node.id);
      }
      
      this.log(execution, 'info', `Executing node: ${node.data.label}`, node.id);
      
      // Gather inputs
      const inputs = await this.gatherNodeInputs(node, workflow, execution);
      nodeState.input = inputs;
      
      // Execute based on node type
      let output: any;
      switch (node.type) {
        case NodeType.AGENT:
          output = await this.executeAgentNode(node, inputs, execution, options);
          break;
        case NodeType.TRANSFORM:
          output = await this.executeTransformNode(node, inputs);
          break;
        case NodeType.CONDITION:
          output = await this.executeConditionNode(node, inputs);
          break;
        case NodeType.PARALLEL:
          output = await this.executeParallelNode(node, inputs);
          break;
        default:
          output = inputs;
      }
      
      // Complete node execution
      nodeState.status = NodeExecutionStatus.COMPLETED;
      nodeState.endTime = new Date();
      nodeState.progress = 100;
      nodeState.output = output;
      
      // Update metrics
      const duration = nodeState.endTime.getTime() - nodeState.startTime.getTime();
      execution.metrics.nodeDurations.set(node.id, duration);
      
      if (options.onNodeComplete) {
        options.onNodeComplete(node.id, output);
      }
      
      this.log(execution, 'info', `Node completed: ${node.data.label}`, node.id);
      
    } catch (error) {
      nodeState.status = NodeExecutionStatus.FAILED;
      nodeState.endTime = new Date();
      nodeState.error = {
        code: 'NODE_EXECUTION_ERROR',
        message: error.message,
        nodeId: node.id,
        input: nodeState.input
      };
      
      if (options.onNodeError) {
        options.onNodeError(node.id, error);
      }
      
      this.log(execution, 'error', `Node failed: ${error.message}`, node.id);
      
      // Handle error based on settings
      if (workflow.settings.errorHandling.onNodeError === 'stop') {
        throw error;
      } else if (workflow.settings.errorHandling.onNodeError === 'retry') {
        if (nodeState.retries < workflow.settings.maxRetries) {
          nodeState.retries++;
          this.log(execution, 'info', `Retrying node (attempt ${nodeState.retries})`, node.id);
          await this.executeNode(node, workflow, execution, options);
        } else {
          throw error;
        }
      }
    }
  }
  
  /**
   * Execute agent node with caching and batching
   */
  private async executeAgentNode(
    node: WorkflowNode,
    inputs: any,
    execution: WorkflowExecution,
    options: ExecutionOptions
  ): Promise<any> {
    const agentType = node.data.agentType!;
    const config = node.data.config;
    
    // Check cache if enabled
    if (options.enableCaching) {
      const cacheKey = this.generateCacheKey(node, inputs);
      const cached = this.getFromCache(cacheKey, options.cacheTTL || 300000); // 5 min default
      if (cached) {
        this.log(execution, 'debug', `Cache hit for node: ${node.data.label}`, node.id);
        return cached;
      }
    }
    
    // Create agent task
    const task = {
      agent_type: agentType.toLowerCase(),
      task_description: this.buildTaskDescription(node, inputs),
      config: {
        ...config,
        inputs,
        workflow_context: {
          workflow_id: execution.workflowId,
          execution_id: execution.id,
          node_id: node.id
        }
      }
    };
    
    // Use WebSocket if available and enabled
    if (options.useWebSocket) {
      return await this.executeAgentViaWebSocket(node, task, execution, options);
    }
    
    // Batch API calls if enabled
    if (options.batchSize && options.batchSize > 1) {
      return await this.batchAgentExecution(node, task, execution, options);
    }
    
    // Submit to orchestrator
    const response = await nexusForgeApi.createProject({
      name: `${node.data.label} Task`,
      description: task.task_description,
      config: task
    });
    
    // Poll for completion with exponential backoff
    const result = await this.pollAgentExecutionOptimized(response.id, node.id, execution, options);
    
    // Cache result if enabled
    if (options.enableCaching && result) {
      const cacheKey = this.generateCacheKey(node, inputs);
      this.addToCache(cacheKey, result);
    }
    
    return result;
  }
  
  /**
   * Execute agent via WebSocket for real-time updates
   */
  private async executeAgentViaWebSocket(
    node: WorkflowNode,
    task: any,
    execution: WorkflowExecution,
    options: ExecutionOptions
  ): Promise<any> {
    return new Promise((resolve, reject) => {
      const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:3000/ws';
      const ws = new WebSocket(wsUrl);
      
      const timeout = setTimeout(() => {
        ws.close();
        reject(new Error('WebSocket timeout'));
      }, options.timeout || 600000);
      
      ws.onopen = () => {
        ws.send(JSON.stringify({
          type: 'execute_agent',
          payload: task
        }));
      };
      
      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        
        if (message.type === 'progress') {
          const nodeState = execution.nodeStates.get(node.id);
          if (nodeState) {
            nodeState.progress = message.progress;
          }
          if (options.onProgress) {
            options.onProgress(message.progress);
          }
        } else if (message.type === 'complete') {
          clearTimeout(timeout);
          ws.close();
          resolve(message.result);
        } else if (message.type === 'error') {
          clearTimeout(timeout);
          ws.close();
          reject(new Error(message.error));
        }
      };
      
      ws.onerror = (error) => {
        clearTimeout(timeout);
        reject(error);
      };
      
      this.websocketConnections.set(node.id, ws);
    });
  }
  
  /**
   * Batch agent executions for efficiency
   */
  private async batchAgentExecution(
    node: WorkflowNode,
    task: any,
    execution: WorkflowExecution,
    options: ExecutionOptions
  ): Promise<any> {
    const batchKey = `${task.agent_type}_batch`;
    
    if (!this.pendingBatches.has(batchKey)) {
      this.pendingBatches.set(batchKey, []);
    }
    
    const batch = this.pendingBatches.get(batchKey)!;
    
    return new Promise((resolve, reject) => {
      batch.push({ task, resolve, reject, node, execution });
      
      // Clear existing timer
      if (this.batchTimers.has(batchKey)) {
        clearTimeout(this.batchTimers.get(batchKey)!);
      }
      
      // Set new timer or execute immediately if batch is full
      if (batch.length >= (options.batchSize || 10)) {
        this.executeBatch(batchKey, options);
      } else {
        const timer = setTimeout(() => {
          this.executeBatch(batchKey, options);
        }, 100); // 100ms debounce
        this.batchTimers.set(batchKey, timer);
      }
    });
  }
  
  /**
   * Execute a batch of agent tasks
   */
  private async executeBatch(batchKey: string, options: ExecutionOptions) {
    const batch = this.pendingBatches.get(batchKey);
    if (!batch || batch.length === 0) return;
    
    this.pendingBatches.set(batchKey, []);
    this.batchTimers.delete(batchKey);
    
    try {
      // Create batch request
      const batchRequest = {
        tasks: batch.map(item => item.task),
        batch_id: `batch_${Date.now()}`
      };
      
      // Submit batch
      const response = await nexusForgeApi.createBatchProject(batchRequest);
      
      // Poll for batch completion
      const results = await this.pollBatchExecution(response.batch_id, options);
      
      // Resolve individual promises
      batch.forEach((item, index) => {
        if (results[index].success) {
          item.resolve(results[index].result);
        } else {
          item.reject(new Error(results[index].error));
        }
      });
    } catch (error) {
      // Reject all promises in batch
      batch.forEach(item => item.reject(error));
    }
  }
  
  /**
   * Build task description from node and inputs
   */
  private buildTaskDescription(node: WorkflowNode, inputs: any): string {
    let description = node.data.description || node.data.label;
    
    // Add input context
    if (inputs && Object.keys(inputs).length > 0) {
      description += '\n\nInputs:\n';
      Object.entries(inputs).forEach(([key, value]) => {
        description += `- ${key}: ${JSON.stringify(value, null, 2)}\n`;
      });
    }
    
    return description;
  }
  
  /**
   * Poll agent execution with exponential backoff and jitter
   */
  private async pollAgentExecutionOptimized(
    projectId: string,
    nodeId: string,
    execution: WorkflowExecution,
    options: ExecutionOptions
  ): Promise<any> {
    const nodeState = execution.nodeStates.get(nodeId)!;
    const timeout = options.timeout || 600000; // 10 minutes default
    const startTime = Date.now();
    
    let pollInterval = 500; // Start with 500ms
    const maxInterval = 5000; // Max 5 seconds
    const backoffMultiplier = 1.5;
    
    // Track consecutive unchanged polls
    let lastProgress = -1;
    let unchangedPolls = 0;
    
    while (Date.now() - startTime < timeout) {
      // Use connection pool for API calls
      const project = await this.getProjectFromPool(projectId);
      
      // Update progress
      nodeState.progress = project.progress;
      if (options.onProgress) {
        options.onProgress(project.progress);
      }
      
      // Check if progress changed
      if (project.progress === lastProgress) {
        unchangedPolls++;
      } else {
        unchangedPolls = 0;
        lastProgress = project.progress;
      }
      
      // Check status
      if (project.status === 'completed') {
        return project.result || project.output;
      } else if (project.status === 'failed') {
        throw new Error(project.error || 'Agent execution failed');
      }
      
      // Adaptive polling interval
      if (unchangedPolls > 3) {
        // Increase interval with backoff
        pollInterval = Math.min(pollInterval * backoffMultiplier, maxInterval);
      } else if (project.progress > 80) {
        // Poll more frequently when near completion
        pollInterval = 500;
      }
      
      // Add jitter to prevent thundering herd
      const jitter = Math.random() * 200 - 100; // ±100ms
      
      // Wait before next poll
      await new Promise(resolve => setTimeout(resolve, pollInterval + jitter));
    }
    
    throw new Error('Agent execution timed out');
  }
  
  /**
   * Get project status using connection pool
   */
  private async getProjectFromPool(projectId: string): Promise<any> {
    // Reuse connection from pool if available
    const poolKey = 'nexusforge_api';
    if (!this.connectionPool.has(poolKey)) {
      this.connectionPool.set(poolKey, {
        lastUsed: Date.now(),
        requests: 0
      });
    }
    
    const connection = this.connectionPool.get(poolKey);
    connection.requests++;
    connection.lastUsed = Date.now();
    
    return await nexusForgeApi.getProject(projectId);
  }
  
  /**
   * Poll batch execution status
   */
  private async pollBatchExecution(
    batchId: string,
    options: ExecutionOptions
  ): Promise<any[]> {
    const timeout = options.timeout || 600000;
    const startTime = Date.now();
    let pollInterval = 1000;
    
    while (Date.now() - startTime < timeout) {
      const batch = await nexusForgeApi.getBatchStatus(batchId);
      
      if (batch.status === 'completed') {
        return batch.results;
      } else if (batch.status === 'failed') {
        throw new Error(batch.error || 'Batch execution failed');
      }
      
      await new Promise(resolve => setTimeout(resolve, pollInterval));
      pollInterval = Math.min(pollInterval * 1.2, 3000); // Backoff to max 3s
    }
    
    throw new Error('Batch execution timed out');
  }
  
  /**
   * Execute transform node
   */
  private async executeTransformNode(
    node: WorkflowNode,
    inputs: any
  ): Promise<any> {
    const transform = node.data.config.transform;
    
    if (!transform) {
      return inputs;
    }
    
    // Apply transformation
    try {
      // Simple transformation logic - can be extended
      switch (transform.type) {
        case 'map':
          return this.applyMapTransform(inputs, transform.expression);
        case 'filter':
          return this.applyFilterTransform(inputs, transform.expression);
        case 'reduce':
          return this.applyReduceTransform(inputs, transform.expression);
        default:
          return inputs;
      }
    } catch (error) {
      throw new Error(`Transform failed: ${error.message}`);
    }
  }
  
  /**
   * Execute condition node
   */
  private async executeConditionNode(
    node: WorkflowNode,
    inputs: any
  ): Promise<boolean> {
    const condition = node.data.config.condition;
    
    if (!condition) {
      return true;
    }
    
    try {
      // Evaluate condition
      return this.evaluateCondition(inputs, condition);
    } catch (error) {
      throw new Error(`Condition evaluation failed: ${error.message}`);
    }
  }
  
  /**
   * Execute parallel node
   */
  private async executeParallelNode(
    node: WorkflowNode,
    inputs: any
  ): Promise<any> {
    // Parallel nodes simply pass through inputs to trigger parallel execution
    return inputs;
  }
  
  /**
   * Gather inputs for a node
   */
  private async gatherNodeInputs(
    node: WorkflowNode,
    workflow: WorkflowDefinition,
    execution: WorkflowExecution
  ): Promise<any> {
    const inputs: any = {};
    
    // Find connections to this node
    const incomingConnections = workflow.connections.filter(
      conn => conn.target.nodeId === node.id
    );
    
    // Gather outputs from connected nodes
    for (const conn of incomingConnections) {
      const sourceNode = workflow.nodes.find(n => n.id === conn.source.nodeId);
      if (!sourceNode) continue;
      
      const sourceState = execution.nodeStates.get(conn.source.nodeId);
      if (!sourceState || sourceState.status !== NodeExecutionStatus.COMPLETED) continue;
      
      // Map output to input
      const targetPort = node.data.inputs.find(p => p.id === conn.target.portId);
      if (targetPort) {
        inputs[targetPort.name] = sourceState.output;
      }
    }
    
    return inputs;
  }
  
  /**
   * Check if node should be skipped
   */
  private async shouldSkipNode(
    node: WorkflowNode,
    workflow: WorkflowDefinition,
    execution: WorkflowExecution
  ): Promise<boolean> {
    // Check if all required inputs are available
    const requiredInputs = node.data.inputs.filter(input => input.required);
    const incomingConnections = workflow.connections.filter(
      conn => conn.target.nodeId === node.id
    );
    
    for (const input of requiredInputs) {
      const hasConnection = incomingConnections.some(
        conn => conn.target.portId === input.id
      );
      
      if (!hasConnection) {
        return true; // Skip if required input not connected
      }
      
      // Check if source node completed successfully
      const sourceConn = incomingConnections.find(
        conn => conn.target.portId === input.id
      );
      
      if (sourceConn) {
        const sourceState = execution.nodeStates.get(sourceConn.source.nodeId);
        if (!sourceState || sourceState.status !== NodeExecutionStatus.COMPLETED) {
          return true; // Skip if source not completed
        }
      }
    }
    
    return false;
  }
  
  /**
   * Get execution order (topological sort)
   */
  private getExecutionOrder(workflow: WorkflowDefinition): string[] {
    const adjacencyList = new Map<string, string[]>();
    const inDegree = new Map<string, number>();
    
    // Initialize
    workflow.nodes.forEach(node => {
      adjacencyList.set(node.id, []);
      inDegree.set(node.id, 0);
    });
    
    // Build graph
    workflow.connections.forEach(conn => {
      adjacencyList.get(conn.source.nodeId)?.push(conn.target.nodeId);
      inDegree.set(conn.target.nodeId, (inDegree.get(conn.target.nodeId) || 0) + 1);
    });
    
    // Topological sort
    const queue: string[] = [];
    const sorted: string[] = [];
    
    // Find nodes with no dependencies
    inDegree.forEach((degree, nodeId) => {
      if (degree === 0) queue.push(nodeId);
    });
    
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
  
  /**
   * Get parallel execution groups
   */
  private getParallelExecutionGroups(workflow: WorkflowDefinition): string[][] {
    const groups: string[][] = [];
    const processed = new Set<string>();
    const dependencies = this.buildDependencyMap(workflow);
    
    while (processed.size < workflow.nodes.length) {
      const group: string[] = [];
      
      workflow.nodes.forEach(node => {
        if (!processed.has(node.id)) {
          const deps = dependencies.get(node.id) || new Set();
          const canExecute = Array.from(deps).every(dep => processed.has(dep));
          
          if (canExecute) {
            group.push(node.id);
          }
        }
      });
      
      if (group.length === 0) break; // Prevent infinite loop
      
      group.forEach(nodeId => processed.add(nodeId));
      groups.push(group);
    }
    
    return groups;
  }
  
  /**
   * Build dependency map
   */
  private buildDependencyMap(workflow: WorkflowDefinition): Map<string, Set<string>> {
    const dependencies = new Map<string, Set<string>>();
    
    workflow.nodes.forEach(node => {
      dependencies.set(node.id, new Set());
    });
    
    workflow.connections.forEach(conn => {
      const deps = dependencies.get(conn.target.nodeId) || new Set();
      deps.add(conn.source.nodeId);
      dependencies.set(conn.target.nodeId, deps);
    });
    
    return dependencies;
  }
  
  /**
   * Chunk array for concurrency control
   */
  private chunkArray<T>(array: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }
  
  /**
   * Simple transform implementations
   */
  private applyMapTransform(data: any, expression: string): any {
    // Simple implementation - can be extended with proper expression evaluation
    if (Array.isArray(data)) {
      return data.map((item, index) => {
        // Basic property extraction
        if (expression.startsWith('.')) {
          const prop = expression.substring(1);
          return item[prop];
        }
        return item;
      });
    }
    return data;
  }
  
  private applyFilterTransform(data: any, expression: string): any {
    if (Array.isArray(data)) {
      return data.filter((item, index) => {
        // Basic filtering - can be extended
        if (expression.includes('!=')) {
          const [prop, value] = expression.split('!=').map(s => s.trim());
          return item[prop] != value;
        } else if (expression.includes('==')) {
          const [prop, value] = expression.split('==').map(s => s.trim());
          return item[prop] == value;
        }
        return true;
      });
    }
    return data;
  }
  
  private applyReduceTransform(data: any, expression: string): any {
    if (Array.isArray(data)) {
      // Basic reduction - can be extended
      if (expression === 'sum') {
        return data.reduce((sum, item) => sum + (typeof item === 'number' ? item : 0), 0);
      } else if (expression === 'count') {
        return data.length;
      }
    }
    return data;
  }
  
  /**
   * Evaluate condition
   */
  private evaluateCondition(data: any, condition: any): boolean {
    // Simple condition evaluation - can be extended
    const { type, expression } = condition;
    
    try {
      if (type === 'value') {
        return !!data;
      } else if (type === 'expression') {
        // Basic expression evaluation
        if (expression.includes('==')) {
          const [left, right] = expression.split('==').map(s => s.trim());
          return data[left] == right;
        } else if (expression.includes('>')) {
          const [left, right] = expression.split('>').map(s => s.trim());
          return data[left] > parseFloat(right);
        }
      }
    } catch (error) {
      console.error('Condition evaluation error:', error);
    }
    
    return true;
  }
  
  /**
   * Add log entry with metrics collection
   */
  private log(
    execution: WorkflowExecution,
    level: 'debug' | 'info' | 'warning' | 'error',
    message: string,
    nodeId?: string
  ): void {
    execution.logs.push({
      timestamp: new Date(),
      level,
      message,
      nodeId
    });
    
    // Collect metrics
    if (nodeId) {
      this.collectNodeMetrics(nodeId, { level, message });
    }
  }
  
  /**
   * Generate cache key for node execution
   */
  private generateCacheKey(node: WorkflowNode, inputs: any): string {
    const nodeData = {
      id: node.id,
      type: node.type,
      config: node.data.config,
      inputs: JSON.stringify(inputs)
    };
    
    // Simple hash function
    let hash = 0;
    const str = JSON.stringify(nodeData);
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    
    return `node_${node.id}_${hash}`;
  }
  
  /**
   * Get result from cache
   */
  private getFromCache(key: string, ttl: number): any {
    const cached = this.resultCache.get(key);
    if (!cached) return null;
    
    const age = Date.now() - cached.timestamp;
    if (age > ttl) {
      this.resultCache.delete(key);
      return null;
    }
    
    return cached.result;
  }
  
  /**
   * Add result to cache
   */
  private addToCache(key: string, result: any): void {
    this.resultCache.set(key, {
      result,
      timestamp: Date.now()
    });
    
    // Limit cache size
    if (this.resultCache.size > 1000) {
      // Remove oldest entries
      const entries = Array.from(this.resultCache.entries());
      entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
      
      for (let i = 0; i < 100; i++) {
        this.resultCache.delete(entries[i][0]);
      }
    }
  }
  
  /**
   * Collect performance metrics for a node
   */
  private collectNodeMetrics(nodeId: string, data: any): void {
    if (!this.metricsCollector.has(nodeId)) {
      this.metricsCollector.set(nodeId, {
        executions: 0,
        totalTime: 0,
        errors: 0,
        cacheHits: 0,
        cacheMisses: 0
      });
    }
    
    const metrics = this.metricsCollector.get(nodeId);
    
    if (data.level === 'error') {
      metrics.errors++;
    }
    
    if (data.message.includes('Cache hit')) {
      metrics.cacheHits++;
    } else if (data.message.includes('Cache miss')) {
      metrics.cacheMisses++;
    }
  }
  
  /**
   * Get performance metrics
   */
  getPerformanceMetrics(): Map<string, any> {
    return new Map(this.metricsCollector);
  }
  
  /**
   * Clear caches and reset metrics
   */
  clearCachesAndMetrics(): void {
    this.resultCache.clear();
    this.metricsCollector.clear();
    this.connectionPool.clear();
    
    // Close WebSocket connections
    this.websocketConnections.forEach(ws => ws.close());
    this.websocketConnections.clear();
  }
}

// Export singleton instance
export const workflowExecutor = WorkflowExecutor.getInstance();