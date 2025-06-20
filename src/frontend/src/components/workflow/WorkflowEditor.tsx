/**
 * WorkflowEditor Component
 * Main visual workflow editor with drag-drop canvas
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Save,
  Download,
  Upload,
  Play,
  Pause,
  Stop,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Grid,
  Eye,
  EyeOff,
  Plus,
  Trash2,
  Copy,
  Settings
} from 'lucide-react';
import {
  WorkflowDefinition,
  WorkflowNode,
  WorkflowConnection,
  NodeType,
  Position,
  CanvasState,
  Viewport,
  Selection,
  AgentType
} from '../../types/workflow.types';
import { workflowSerializer } from '../../services/workflowSerializer';
import { WorkflowCanvas } from './WorkflowCanvas';
import { NodePalette } from './NodePalette';
import { WorkflowToolbar } from './WorkflowToolbar';
import { WorkflowSidebar } from './WorkflowSidebar';
import { useWorkflowStore } from '../../stores/workflowStore';
import { useWebSocket } from '../../contexts/WebSocketContext';

interface WorkflowEditorProps {
  initialWorkflow?: WorkflowDefinition;
  onSave?: (workflow: WorkflowDefinition) => void;
  onExecute?: (workflow: WorkflowDefinition) => void;
  readOnly?: boolean;
}

export const WorkflowEditor: React.FC<WorkflowEditorProps> = ({
  initialWorkflow,
  onSave,
  onExecute,
  readOnly = false
}) => {
  // State management
  const {
    workflow,
    canvasState,
    selectedNodes,
    selectedConnections,
    isExecuting,
    executionState,
    setWorkflow,
    addNode,
    updateNode,
    deleteNode,
    addConnection,
    deleteConnection,
    setViewport,
    selectNode,
    selectConnection,
    clearSelection,
    undo,
    redo,
    canUndo,
    canRedo
  } = useWorkflowStore();

  const { state: wsState } = useWebSocket();

  // Local state
  const [showGrid, setShowGrid] = useState(true);
  const [snapToGrid, setSnapToGrid] = useState(true);
  const [showMinimap, setShowMinimap] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<Position | null>(null);
  const [connectionPreview, setConnectionPreview] = useState<any>(null);

  // Refs
  const canvasRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Initialize workflow
  useEffect(() => {
    if (initialWorkflow) {
      setWorkflow(initialWorkflow);
    }
  }, [initialWorkflow, setWorkflow]);

  // Handle canvas mouse events
  const handleCanvasMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 0 && !e.ctrlKey && !e.metaKey) {
      setIsDragging(true);
      setDragStart({ x: e.clientX, y: e.clientY });
      clearSelection();
    }
  }, [clearSelection]);

  const handleCanvasMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging && dragStart && canvasRef.current) {
      const deltaX = e.clientX - dragStart.x;
      const deltaY = e.clientY - dragStart.y;
      
      // Pan the canvas
      setViewport({
        x: canvasState.viewport.x + deltaX,
        y: canvasState.viewport.y + deltaY,
        zoom: canvasState.viewport.zoom
      });
      
      setDragStart({ x: e.clientX, y: e.clientY });
    }
  }, [isDragging, dragStart, canvasState.viewport, setViewport]);

  const handleCanvasMouseUp = useCallback(() => {
    setIsDragging(false);
    setDragStart(null);
  }, []);

  // Handle zoom
  const handleZoom = useCallback((delta: number) => {
    const newZoom = Math.max(0.1, Math.min(2, canvasState.viewport.zoom + delta));
    setViewport({
      ...canvasState.viewport,
      zoom: newZoom
    });
  }, [canvasState.viewport, setViewport]);

  const handleZoomIn = () => handleZoom(0.1);
  const handleZoomOut = () => handleZoom(-0.1);
  const handleZoomReset = () => {
    setViewport({
      x: 0,
      y: 0,
      zoom: 1
    });
  };

  // Handle file operations
  const handleSave = useCallback(() => {
    if (onSave) {
      onSave(workflow);
    } else {
      // Download as file
      const blob = new Blob([workflowSerializer.serialize(workflow)], {
        type: 'application/json'
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${workflow.metadata.name || 'workflow'}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
  }, [workflow, onSave]);

  const handleLoad = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      workflowSerializer.importFromFile(file).then(loadedWorkflow => {
        setWorkflow(loadedWorkflow);
      }).catch(error => {
        console.error('Failed to load workflow:', error);
      });
    }
  }, [setWorkflow]);

  // Handle execution
  const handleExecute = useCallback(() => {
    if (onExecute) {
      // Validate workflow first
      const validation = workflowSerializer.validate(workflow);
      if (validation.isValid) {
        onExecute(workflow);
      } else {
        console.error('Workflow validation failed:', validation.errors);
      }
    }
  }, [workflow, onExecute]);

  // Handle node operations
  const handleAddNode = useCallback((nodeType: NodeType, agentType?: AgentType, position?: Position) => {
    const newNode: WorkflowNode = {
      id: `node_${Date.now()}`,
      type: nodeType,
      position: position || { x: 100, y: 100 },
      data: {
        label: agentType || nodeType,
        agentType,
        config: {},
        inputs: [],
        outputs: []
      },
      state: {
        enabled: true,
        locked: false,
        collapsed: false
      }
    };
    
    addNode(newNode);
  }, [addNode]);

  const handleDeleteSelected = useCallback(() => {
    selectedNodes.forEach(nodeId => deleteNode(nodeId));
    selectedConnections.forEach(connId => deleteConnection(connId));
    clearSelection();
  }, [selectedNodes, selectedConnections, deleteNode, deleteConnection, clearSelection]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (readOnly) return;
      
      // Delete
      if (e.key === 'Delete' || e.key === 'Backspace') {
        handleDeleteSelected();
      }
      
      // Undo/Redo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey && canUndo) {
        undo();
      }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey)) && canRedo) {
        redo();
      }
      
      // Save
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        handleSave();
      }
      
      // Select all
      if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
        e.preventDefault();
        // Select all nodes
        workflow.nodes.forEach(node => selectNode(node.id, false));
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [readOnly, handleDeleteSelected, undo, redo, canUndo, canRedo, handleSave, workflow.nodes, selectNode]);

  return (
    <div className="workflow-editor flex h-full bg-gray-50">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".json"
        onChange={handleLoad}
        className="hidden"
      />
      
      {/* Node Palette */}
      {!readOnly && (
        <NodePalette
          onAddNode={handleAddNode}
          className="w-64 bg-white border-r border-gray-200 shadow-sm"
        />
      )}
      
      {/* Main Canvas Area */}
      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <WorkflowToolbar
          workflow={workflow}
          canUndo={canUndo}
          canRedo={canRedo}
          isExecuting={isExecuting}
          showGrid={showGrid}
          snapToGrid={snapToGrid}
          onSave={handleSave}
          onLoad={() => fileInputRef.current?.click()}
          onExecute={handleExecute}
          onUndo={undo}
          onRedo={redo}
          onZoomIn={handleZoomIn}
          onZoomOut={handleZoomOut}
          onZoomReset={handleZoomReset}
          onToggleGrid={() => setShowGrid(!showGrid)}
          onToggleSnap={() => setSnapToGrid(!snapToGrid)}
          onToggleMinimap={() => setShowMinimap(!showMinimap)}
          onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
          readOnly={readOnly}
        />
        
        {/* Canvas */}
        <div
          ref={canvasRef}
          className="flex-1 relative overflow-hidden bg-gray-50"
          onMouseDown={handleCanvasMouseDown}
          onMouseMove={handleCanvasMouseMove}
          onMouseUp={handleCanvasMouseUp}
          onMouseLeave={handleCanvasMouseUp}
        >
          <WorkflowCanvas
            workflow={workflow}
            canvasState={canvasState}
            executionState={executionState}
            showGrid={showGrid}
            snapToGrid={snapToGrid}
            readOnly={readOnly}
            onNodeSelect={selectNode}
            onNodeUpdate={updateNode}
            onNodeDelete={deleteNode}
            onConnectionSelect={selectConnection}
            onConnectionCreate={addConnection}
            onConnectionDelete={deleteConnection}
            connectionPreview={connectionPreview}
            onConnectionPreviewUpdate={setConnectionPreview}
          />
          
          {/* Minimap */}
          {showMinimap && (
            <div className="absolute bottom-4 right-4 w-48 h-32 bg-white border border-gray-300 rounded-lg shadow-lg">
              {/* Minimap implementation */}
              <div className="p-2 text-xs text-gray-500">Minimap</div>
            </div>
          )}
          
          {/* Execution Status */}
          {isExecuting && (
            <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-blue-500 text-white px-4 py-2 rounded-lg shadow-lg flex items-center space-x-2">
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
              <span>Executing workflow...</span>
            </div>
          )}
        </div>
      </div>
      
      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: 320 }}
            exit={{ width: 0 }}
            className="bg-white border-l border-gray-200 shadow-sm overflow-hidden"
          >
            <WorkflowSidebar
              workflow={workflow}
              selectedNodes={selectedNodes}
              selectedConnections={selectedConnections}
              onNodeUpdate={updateNode}
              onWorkflowUpdate={setWorkflow}
              readOnly={readOnly}
            />
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Status Bar */}
      <div className="absolute bottom-0 left-0 right-0 bg-gray-800 text-white text-xs px-4 py-1 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <span>Nodes: {workflow.nodes.length}</span>
          <span>Connections: {workflow.connections.length}</span>
          <span>Zoom: {Math.round(canvasState.viewport.zoom * 100)}%</span>
        </div>
        <div className="flex items-center space-x-4">
          {wsState.isConnected ? (
            <span className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-400 rounded-full" />
              <span>Connected</span>
            </span>
          ) : (
            <span className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-red-400 rounded-full" />
              <span>Disconnected</span>
            </span>
          )}
        </div>
      </div>
    </div>
  );
};