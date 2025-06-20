/**
 * WorkflowConnection Component
 * Renders connections between workflow nodes
 */

import React, { memo, useMemo } from 'react';
import { WorkflowConnection, Position, ConnectionType } from '../../types/workflow.types';

interface WorkflowConnectionComponentProps {
  connection: WorkflowConnection | null;
  sourcePosition: Position;
  targetPosition: Position;
  isSelected: boolean;
  isPreview?: boolean;
  onClick: (e: React.MouseEvent) => void;
  onDelete: () => void;
  readOnly: boolean;
}

export const WorkflowConnectionComponent: React.FC<WorkflowConnectionComponentProps> = memo(({
  connection,
  sourcePosition,
  targetPosition,
  isSelected,
  isPreview = false,
  onClick,
  onDelete,
  readOnly
}) => {
  // Calculate path for bezier curve
  const path = useMemo(() => {
    const dx = targetPosition.x - sourcePosition.x;
    const dy = targetPosition.y - sourcePosition.y;
    
    // Control point offset for bezier curve
    const controlOffset = Math.min(Math.abs(dx) * 0.5, 100);
    
    return `M ${sourcePosition.x},${sourcePosition.y} 
            C ${sourcePosition.x + controlOffset},${sourcePosition.y} 
              ${targetPosition.x - controlOffset},${targetPosition.y} 
              ${targetPosition.x},${targetPosition.y}`;
  }, [sourcePosition, targetPosition]);
  
  // Calculate label position (middle of the curve)
  const labelPosition = useMemo(() => {
    return {
      x: (sourcePosition.x + targetPosition.x) / 2,
      y: (sourcePosition.y + targetPosition.y) / 2
    };
  }, [sourcePosition, targetPosition]);
  
  const getConnectionColor = () => {
    if (isPreview) return "#94a3b8"; // gray-400
    if (isSelected) return "#3b82f6"; // blue-500
    
    switch (connection?.type) {
      case ConnectionType.DATA_FLOW:
        return "#10b981"; // green-500
      case ConnectionType.CONTROL_FLOW:
        return "#8b5cf6"; // purple-500
      case ConnectionType.CONDITIONAL:
        return "#f59e0b"; // amber-500
      case ConnectionType.ERROR:
        return "#ef4444"; // red-500
      default:
        return "#6b7280"; // gray-500
    }
  };
  
  const strokeWidth = isSelected ? 3 : 2;
  const strokeDasharray = isPreview ? "5,5" : connection?.style?.strokeDasharray || "none";
  
  return (
    <g className="workflow-connection">
      {/* Invisible wider path for easier interaction */}
      <path
        d={path}
        fill="none"
        stroke="transparent"
        strokeWidth={20}
        className="cursor-pointer"
        onClick={onClick}
      />
      
      {/* Visible connection path */}
      <path
        d={path}
        fill="none"
        stroke={getConnectionColor()}
        strokeWidth={strokeWidth}
        strokeDasharray={strokeDasharray}
        className={`transition-all duration-200 ${
          isPreview ? "pointer-events-none" : "cursor-pointer"
        }`}
        onClick={onClick}
      />
      
      {/* Arrowhead */}
      <defs>
        <marker
          id={`arrowhead-${connection?.id || 'preview'}`}
          markerWidth="10"
          markerHeight="10"
          refX="9"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <polygon
            points="0 0, 10 3, 0 6"
            fill={getConnectionColor()}
          />
        </marker>
      </defs>
      
      <path
        d={path}
        fill="none"
        stroke="none"
        markerEnd={`url(#arrowhead-${connection?.id || 'preview'})`}
      />
      
      {/* Connection label/actions */}
      {!isPreview && isSelected && !readOnly && (
        <g transform={`translate(${labelPosition.x}, ${labelPosition.y})`}>
          {/* Background */}
          <rect
            x="-20"
            y="-10"
            width="40"
            height="20"
            rx="10"
            fill="white"
            stroke={getConnectionColor()}
            strokeWidth="1"
          />
          
          {/* Delete button */}
          <text
            x="0"
            y="0"
            textAnchor="middle"
            dominantBaseline="middle"
            className="cursor-pointer select-none"
            fill={getConnectionColor()}
            fontSize="12"
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
          >
            ×
          </text>
        </g>
      )}
      
      {/* Connection type indicator */}
      {connection?.type === ConnectionType.CONDITIONAL && (
        <g transform={`translate(${labelPosition.x}, ${labelPosition.y - 15})`}>
          <rect
            x="-15"
            y="-8"
            width="30"
            height="16"
            rx="8"
            fill="#fef3c7"
            stroke="#f59e0b"
            strokeWidth="1"
          />
          <text
            x="0"
            y="0"
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize="10"
            fill="#92400e"
          >
            IF
          </text>
        </g>
      )}
      
      {/* Data transform indicator */}
      {connection?.dataTransform && (
        <g transform={`translate(${labelPosition.x}, ${labelPosition.y + 15})`}>
          <rect
            x="-15"
            y="-8"
            width="30"
            height="16"
            rx="8"
            fill="#e0e7ff"
            stroke="#6366f1"
            strokeWidth="1"
          />
          <text
            x="0"
            y="0"
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize="10"
            fill="#4338ca"
          >
            fx
          </text>
        </g>
      )}
    </g>
  );
});