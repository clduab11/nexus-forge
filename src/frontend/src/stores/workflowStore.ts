/**
 * Workflow Store
 * Zustand store for workflow editor state management
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import {
  WorkflowDefinition,
  WorkflowNode,
  WorkflowConnection,
  CanvasState,
  Viewport,
  NodeExecutionState,
  WorkflowExecution,
  ExecutionStatus
} from '../types/workflow.types';

interface WorkflowState {
  // Workflow data
  workflow: WorkflowDefinition;
  
  // Canvas state
  canvasState: CanvasState;
  
  // Selection state
  selectedNodes: Set<string>;
  selectedConnections: Set<string>;
  
  // Execution state
  isExecuting: boolean;
  executionState: Map<string, NodeExecutionState>;
  currentExecution: WorkflowExecution | null;
  
  // History for undo/redo
  history: {
    past: WorkflowDefinition[];
    future: WorkflowDefinition[];
  };
  
  // Actions
  setWorkflow: (workflow: WorkflowDefinition) => void;
  
  // Node operations
  addNode: (node: WorkflowNode) => void;
  updateNode: (nodeId: string, updates: Partial<WorkflowNode>) => void;
  deleteNode: (nodeId: string) => void;
  
  // Connection operations
  addConnection: (connection: WorkflowConnection) => void;
  updateConnection: (connectionId: string, updates: Partial<WorkflowConnection>) => void;
  deleteConnection: (connectionId: string) => void;
  
  // Canvas operations
  setViewport: (viewport: Viewport) => void;
  resetViewport: () => void;
  
  // Selection operations
  selectNode: (nodeId: string, multi: boolean) => void;
  selectConnection: (connectionId: string, multi: boolean) => void;
  clearSelection: () => void;
  selectAll: () => void;
  
  // History operations
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  
  // Execution operations
  startExecution: (execution: WorkflowExecution) => void;
  updateNodeExecution: (nodeId: string, state: NodeExecutionState) => void;
  completeExecution: (status: ExecutionStatus, error?: Error) => void;
  
  // Utility operations
  validateWorkflow: () => { isValid: boolean; errors: string[] };
  getConnectedNodes: (nodeId: string) => { inputs: string[]; outputs: string[] };
  exportWorkflow: () => string;
  importWorkflow: (json: string) => void;
}

const createDefaultWorkflow = (): WorkflowDefinition => ({
  id: `wf_${Date.now()}`,
  version: '1.0.0',
  metadata: {
    name: 'New Workflow',
    description: '',
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
  settings: {
    executionMode: 'sequential',
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
  }
});

const createDefaultCanvasState = (): CanvasState => ({
  nodes: new Map(),
  connections: new Map(),
  viewport: {
    x: 0,
    y: 0,
    zoom: 1,
    minZoom: 0.1,
    maxZoom: 2
  },
  selection: {
    nodes: new Set(),
    connections: new Set()
  },
  history: {
    past: [],
    future: [],
    maxSize: 50
  }
});

export const useWorkflowStore = create<WorkflowState>()(
  devtools(
    immer((set, get) => ({
      // Initial state
      workflow: createDefaultWorkflow(),
      canvasState: createDefaultCanvasState(),
      selectedNodes: new Set(),
      selectedConnections: new Set(),
      isExecuting: false,
      executionState: new Map(),
      currentExecution: null,
      history: {
        past: [],
        future: []
      },
      canUndo: false,
      canRedo: false,
      
      // Actions
      setWorkflow: (workflow) => set((state) => {
        state.workflow = workflow;
        state.workflow.metadata.modified = new Date();
        state.history.past.push(state.workflow);
        state.history.future = [];
        state.canUndo = state.history.past.length > 0;
        state.canRedo = false;
      }),
      
      // Node operations
      addNode: (node) => set((state) => {
        state.workflow.nodes.push(node);
        state.workflow.metadata.modified = new Date();
        state.history.past.push(JSON.parse(JSON.stringify(state.workflow)));
        state.history.future = [];
        state.canUndo = true;
        state.canRedo = false;
      }),
      
      updateNode: (nodeId, updates) => set((state) => {
        const nodeIndex = state.workflow.nodes.findIndex(n => n.id === nodeId);
        if (nodeIndex !== -1) {
          Object.assign(state.workflow.nodes[nodeIndex], updates);
          state.workflow.metadata.modified = new Date();
        }
      }),
      
      deleteNode: (nodeId) => set((state) => {
        // Remove node
        state.workflow.nodes = state.workflow.nodes.filter(n => n.id !== nodeId);
        
        // Remove connections to/from this node
        state.workflow.connections = state.workflow.connections.filter(
          c => c.source.nodeId !== nodeId && c.target.nodeId !== nodeId
        );
        
        // Remove from selection
        state.selectedNodes.delete(nodeId);
        
        state.workflow.metadata.modified = new Date();
        state.history.past.push(JSON.parse(JSON.stringify(state.workflow)));
        state.history.future = [];
        state.canUndo = true;
        state.canRedo = false;
      }),
      
      // Connection operations
      addConnection: (connection) => set((state) => {
        // Check if connection already exists
        const exists = state.workflow.connections.some(
          c => c.source.nodeId === connection.source.nodeId &&
               c.source.portId === connection.source.portId &&
               c.target.nodeId === connection.target.nodeId &&
               c.target.portId === connection.target.portId
        );
        
        if (!exists) {
          state.workflow.connections.push(connection);
          state.workflow.metadata.modified = new Date();
          state.history.past.push(JSON.parse(JSON.stringify(state.workflow)));
          state.history.future = [];
          state.canUndo = true;
          state.canRedo = false;
        }
      }),
      
      updateConnection: (connectionId, updates) => set((state) => {
        const connIndex = state.workflow.connections.findIndex(c => c.id === connectionId);
        if (connIndex !== -1) {
          Object.assign(state.workflow.connections[connIndex], updates);
          state.workflow.metadata.modified = new Date();
        }
      }),
      
      deleteConnection: (connectionId) => set((state) => {
        state.workflow.connections = state.workflow.connections.filter(c => c.id !== connectionId);
        state.selectedConnections.delete(connectionId);
        state.workflow.metadata.modified = new Date();
        state.history.past.push(JSON.parse(JSON.stringify(state.workflow)));
        state.history.future = [];
        state.canUndo = true;
        state.canRedo = false;
      }),
      
      // Canvas operations
      setViewport: (viewport) => set((state) => {
        state.canvasState.viewport = viewport;
      }),
      
      resetViewport: () => set((state) => {
        state.canvasState.viewport = {
          x: 0,
          y: 0,
          zoom: 1,
          minZoom: 0.1,
          maxZoom: 2
        };
      }),
      
      // Selection operations
      selectNode: (nodeId, multi) => set((state) => {
        if (multi) {
          if (state.selectedNodes.has(nodeId)) {
            state.selectedNodes.delete(nodeId);
          } else {
            state.selectedNodes.add(nodeId);
          }
        } else {
          state.selectedNodes.clear();
          state.selectedNodes.add(nodeId);
          state.selectedConnections.clear();
        }
      }),
      
      selectConnection: (connectionId, multi) => set((state) => {
        if (multi) {
          if (state.selectedConnections.has(connectionId)) {
            state.selectedConnections.delete(connectionId);
          } else {
            state.selectedConnections.add(connectionId);
          }
        } else {
          state.selectedConnections.clear();
          state.selectedConnections.add(connectionId);
          state.selectedNodes.clear();
        }
      }),
      
      clearSelection: () => set((state) => {
        state.selectedNodes.clear();
        state.selectedConnections.clear();
      }),
      
      selectAll: () => set((state) => {
        state.workflow.nodes.forEach(node => state.selectedNodes.add(node.id));
        state.workflow.connections.forEach(conn => state.selectedConnections.add(conn.id));
      }),
      
      // History operations
      undo: () => set((state) => {
        if (state.history.past.length > 0) {
          const previous = state.history.past.pop()!;
          state.history.future.push(JSON.parse(JSON.stringify(state.workflow)));
          state.workflow = previous;
          state.canUndo = state.history.past.length > 0;
          state.canRedo = true;
        }
      }),
      
      redo: () => set((state) => {
        if (state.history.future.length > 0) {
          const next = state.history.future.pop()!;
          state.history.past.push(JSON.parse(JSON.stringify(state.workflow)));
          state.workflow = next;
          state.canUndo = true;
          state.canRedo = state.history.future.length > 0;
        }
      }),
      
      // Execution operations
      startExecution: (execution) => set((state) => {
        state.isExecuting = true;
        state.currentExecution = execution;
        state.executionState.clear();
      }),
      
      updateNodeExecution: (nodeId, nodeState) => set((state) => {
        state.executionState.set(nodeId, nodeState);
      }),
      
      completeExecution: (status, error) => set((state) => {
        state.isExecuting = false;
        if (state.currentExecution) {
          state.currentExecution.status = status;
          state.currentExecution.endTime = new Date();
          if (error) {
            state.currentExecution.error = {
              code: 'EXECUTION_ERROR',
              message: error.message,
              stack: error.stack
            };
          }
        }
      }),
      
      // Utility operations
      validateWorkflow: () => {
        const state = get();
        const errors: string[] = [];
        
        if (state.workflow.nodes.length === 0) {
          errors.push('Workflow must have at least one node');
        }
        
        // Check for orphan nodes
        const connectedNodes = new Set<string>();
        state.workflow.connections.forEach(conn => {
          connectedNodes.add(conn.source.nodeId);
          connectedNodes.add(conn.target.nodeId);
        });
        
        const orphanNodes = state.workflow.nodes.filter(
          node => !connectedNodes.has(node.id)
        );
        
        if (orphanNodes.length > 0 && state.workflow.nodes.length > 1) {
          errors.push(`Found ${orphanNodes.length} disconnected node(s)`);
        }
        
        return {
          isValid: errors.length === 0,
          errors
        };
      },
      
      getConnectedNodes: (nodeId) => {
        const state = get();
        const inputs = state.workflow.connections
          .filter(c => c.target.nodeId === nodeId)
          .map(c => c.source.nodeId);
        
        const outputs = state.workflow.connections
          .filter(c => c.source.nodeId === nodeId)
          .map(c => c.target.nodeId);
        
        return { inputs, outputs };
      },
      
      exportWorkflow: () => {
        const state = get();
        return JSON.stringify(state.workflow, null, 2);
      },
      
      importWorkflow: (json) => set((state) => {
        try {
          const workflow = JSON.parse(json);
          state.workflow = workflow;
          state.workflow.metadata.modified = new Date();
          state.clearSelection();
        } catch (error) {
          console.error('Failed to import workflow:', error);
        }
      })
    }))
  )
);