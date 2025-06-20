/**
 * WorkflowCanvas Component
 * Renders the workflow nodes and connections with interaction handling
 */

import React, { useRef, useCallback, useState } from 'react';
import { motion } from 'framer-motion';
import {
  WorkflowDefinition,
  WorkflowNode,
  WorkflowConnection,
  CanvasState,
  Position,
  ConnectionType,
  NodeExecutionState
} from '../../types/workflow.types';
import { WorkflowNodeComponent } from './WorkflowNode';
import { WorkflowConnectionComponent } from './WorkflowConnection';

interface WorkflowCanvasProps {
  workflow: WorkflowDefinition;
  canvasState: CanvasState;
  executionState?: Map<string, NodeExecutionState>;
  showGrid: boolean;
  snapToGrid: boolean;
  readOnly: boolean;
  onNodeSelect: (nodeId: string, multi: boolean) => void;
  onNodeUpdate: (nodeId: string, updates: Partial<WorkflowNode>) => void;
  onNodeDelete: (nodeId: string) => void;
  onConnectionSelect: (connectionId: string, multi: boolean) => void;
  onConnectionCreate: (connection: WorkflowConnection) => void;
  onConnectionDelete: (connectionId: string) => void;
  connectionPreview: any;
  onConnectionPreviewUpdate: (preview: any) => void;
}

export const WorkflowCanvas: React.FC<WorkflowCanvasProps> = ({
  workflow,
  canvasState,
  executionState,
  showGrid,
  snapToGrid,
  readOnly,
  onNodeSelect,
  onNodeUpdate,
  onNodeDelete,
  onConnectionSelect,
  onConnectionCreate,
  onConnectionDelete,
  connectionPreview,
  onConnectionPreviewUpdate
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [draggingNode, setDraggingNode] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState<Position>({ x: 0, y: 0 });
  const [connectingFrom, setConnectingFrom] = useState<{ nodeId: string; portId: string } | null>(null);

  // Grid settings
  const gridSize = 20;
  
  // Transform coordinates from screen to canvas space
  const screenToCanvas = useCallback((x: number, y: number): Position => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    
    return {
      x: (x - rect.left - canvasState.viewport.x) / canvasState.viewport.zoom,
      y: (y - rect.top - canvasState.viewport.y) / canvasState.viewport.zoom
    };
  }, [canvasState.viewport]);

  // Snap position to grid
  const snapPosition = useCallback((position: Position): Position => {
    if (!snapToGrid) return position;
    
    return {
      x: Math.round(position.x / gridSize) * gridSize,
      y: Math.round(position.y / gridSize) * gridSize
    };
  }, [snapToGrid]);

  // Handle node dragging
  const handleNodeMouseDown = useCallback((e: React.MouseEvent, nodeId: string) => {
    if (readOnly) return;
    
    e.stopPropagation();
    const node = workflow.nodes.find(n => n.id === nodeId);
    if (!node) return;
    
    const canvasPos = screenToCanvas(e.clientX, e.clientY);
    setDraggingNode(nodeId);
    setDragOffset({
      x: canvasPos.x - node.position.x,
      y: canvasPos.y - node.position.y
    });
    
    // Select node
    onNodeSelect(nodeId, e.ctrlKey || e.metaKey);
  }, [workflow.nodes, readOnly, screenToCanvas, onNodeSelect]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (draggingNode) {
      const canvasPos = screenToCanvas(e.clientX, e.clientY);
      const newPosition = snapPosition({
        x: canvasPos.x - dragOffset.x,
        y: canvasPos.y - dragOffset.y
      });
      
      onNodeUpdate(draggingNode, { position: newPosition });
    } else if (connectingFrom && connectionPreview) {
      const canvasPos = screenToCanvas(e.clientX, e.clientY);
      onConnectionPreviewUpdate({
        ...connectionPreview,
        endPosition: canvasPos
      });
    }
  }, [draggingNode, connectingFrom, connectionPreview, screenToCanvas, snapPosition, dragOffset, onNodeUpdate, onConnectionPreviewUpdate]);

  const handleMouseUp = useCallback(() => {
    setDraggingNode(null);
    setConnectingFrom(null);
    onConnectionPreviewUpdate(null);
  }, [onConnectionPreviewUpdate]);

  // Handle port connections
  const handlePortClick = useCallback((nodeId: string, portId: string, isOutput: boolean, e: React.MouseEvent) => {
    e.stopPropagation();
    
    if (readOnly) return;
    
    if (!connectingFrom) {
      // Start connection
      if (isOutput) {
        const node = workflow.nodes.find(n => n.id === nodeId);
        const port = node?.data.outputs.find(p => p.id === portId);
        if (node && port) {
          setConnectingFrom({ nodeId, portId });
          const canvasPos = screenToCanvas(e.clientX, e.clientY);
          onConnectionPreviewUpdate({
            startNodeId: nodeId,
            startPortId: portId,
            startPosition: {
              x: node.position.x + 200, // Approximate port position
              y: node.position.y + 50
            },
            endPosition: canvasPos
          });
        }
      }
    } else {
      // Complete connection
      if (!isOutput && connectingFrom.nodeId !== nodeId) {
        const newConnection: WorkflowConnection = {
          id: `conn_${Date.now()}`,
          source: {
            nodeId: connectingFrom.nodeId,
            portId: connectingFrom.portId
          },
          target: {
            nodeId,
            portId
          },
          type: ConnectionType.DATA_FLOW
        };
        
        onConnectionCreate(newConnection);
      }
      
      // Reset connection state
      setConnectingFrom(null);
      onConnectionPreviewUpdate(null);
    }
  }, [readOnly, connectingFrom, workflow.nodes, screenToCanvas, onConnectionCreate, onConnectionPreviewUpdate]);

  // Get node position for connections
  const getNodePosition = (nodeId: string): Position | null => {
    const node = workflow.nodes.find(n => n.id === nodeId);
    return node?.position || null;
  };

  // Get port position relative to node
  const getPortPosition = (nodeId: string, portId: string, isOutput: boolean): Position | null => {
    const node = workflow.nodes.find(n => n.id === nodeId);
    if (!node) return null;
    
    const ports = isOutput ? node.data.outputs : node.data.inputs;
    const portIndex = ports.findIndex(p => p.id === portId);
    if (portIndex === -1) return null;
    
    // Calculate port position based on node dimensions and port index
    return {
      x: node.position.x + (isOutput ? 240 : 0), // Node width
      y: node.position.y + 40 + (portIndex * 30) // Header height + port spacing
    };
  };

  return (
    <div
      className="absolute inset-0"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Grid Background */}
      {showGrid && (
        <svg
          className="absolute inset-0 pointer-events-none"
          style={{
            transform: `translate(${canvasState.viewport.x}px, ${canvasState.viewport.y}px) scale(${canvasState.viewport.zoom})`
          }}
        >
          <defs>
            <pattern
              id="grid"
              width={gridSize}
              height={gridSize}
              patternUnits="userSpaceOnUse"
            >
              <circle cx={gridSize / 2} cy={gridSize / 2} r="1" fill="#e5e7eb" />
            </pattern>
          </defs>
          <rect width="10000" height="10000" fill="url(#grid)" />
        </svg>
      )}
      
      {/* Connections Layer */}
      <svg
        ref={svgRef}
        className="absolute inset-0"
        style={{
          transform: `translate(${canvasState.viewport.x}px, ${canvasState.viewport.y}px) scale(${canvasState.viewport.zoom})`,
          transformOrigin: '0 0'
        }}
      >
        {/* Render connections */}
        {workflow.connections.map(connection => {
          const sourcePos = getPortPosition(
            connection.source.nodeId,
            connection.source.portId,
            true
          );
          const targetPos = getPortPosition(
            connection.target.nodeId,
            connection.target.portId,
            false
          );
          
          if (!sourcePos || !targetPos) return null;
          
          return (
            <WorkflowConnectionComponent
              key={connection.id}
              connection={connection}
              sourcePosition={sourcePos}
              targetPosition={targetPos}
              isSelected={canvasState.selection.connections.has(connection.id)}
              onClick={(e) => {
                e.stopPropagation();
                onConnectionSelect(connection.id, e.ctrlKey || e.metaKey);
              }}
              onDelete={() => onConnectionDelete(connection.id)}
              readOnly={readOnly}
            />
          );
        })}
        
        {/* Connection preview */}
        {connectionPreview && (
          <WorkflowConnectionComponent
            connection={null}
            sourcePosition={connectionPreview.startPosition}
            targetPosition={connectionPreview.endPosition}
            isSelected={false}
            isPreview={true}
            onClick={() => {}}
            onDelete={() => {}}
            readOnly={true}
          />
        )}
      </svg>
      
      {/* Nodes Layer */}
      <div
        className="absolute inset-0"
        style={{
          transform: `translate(${canvasState.viewport.x}px, ${canvasState.viewport.y}px) scale(${canvasState.viewport.zoom})`,
          transformOrigin: '0 0'
        }}
      >
        {workflow.nodes.map(node => (
          <WorkflowNodeComponent
            key={node.id}
            node={node}
            isSelected={canvasState.selection.nodes.has(node.id)}
            executionState={executionState?.get(node.id)}
            onMouseDown={(e) => handleNodeMouseDown(e, node.id)}
            onPortClick={(portId, isOutput, e) => handlePortClick(node.id, portId, isOutput, e)}
            onDelete={() => onNodeDelete(node.id)}
            readOnly={readOnly}
          />
        ))}
      </div>
      
      {/* Selection Box */}
      {canvasState.selection.box && (
        <div
          className="absolute border-2 border-blue-500 bg-blue-500 bg-opacity-10"
          style={{
            left: Math.min(canvasState.selection.box.start.x, canvasState.selection.box.end.x),
            top: Math.min(canvasState.selection.box.start.y, canvasState.selection.box.end.y),
            width: Math.abs(canvasState.selection.box.end.x - canvasState.selection.box.start.x),
            height: Math.abs(canvasState.selection.box.end.y - canvasState.selection.box.start.y)
          }}
        />
      )}
    </div>
  );
};