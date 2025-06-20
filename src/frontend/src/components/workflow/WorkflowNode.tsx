/**
 * WorkflowNode Component
 * Individual node representation in the workflow canvas
 */

import React, { memo } from 'react';
import { motion } from 'framer-motion';
import {
  Brain,
  Code,
  Search,
  GitBranch,
  Shuffle,
  Play,
  FileOutput,
  Clock,
  X,
  Circle,
  Square,
  ChevronDown,
  ChevronUp,
  Activity,
  CheckCircle,
  AlertCircle,
  Zap,
  Palette,
  TestTube,
  BarChart
} from 'lucide-react';
import {
  WorkflowNode,
  NodeType,
  AgentType,
  NodeExecutionState,
  NodeExecutionStatus
} from '../../types/workflow.types';

interface WorkflowNodeComponentProps {
  node: WorkflowNode;
  isSelected: boolean;
  executionState?: NodeExecutionState;
  onMouseDown: (e: React.MouseEvent) => void;
  onPortClick: (portId: string, isOutput: boolean, e: React.MouseEvent) => void;
  onDelete: () => void;
  readOnly: boolean;
}

// Agent icons mapping
const AGENT_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  [AgentType.STARRI]: Brain,
  [AgentType.JULES]: Code,
  [AgentType.GEMINI]: Zap,
  [AgentType.RESEARCHER]: Search,
  [AgentType.DEVELOPER]: Code,
  [AgentType.DESIGNER]: Palette,
  [AgentType.TESTER]: TestTube,
  [AgentType.ANALYST]: BarChart,
  [AgentType.OPTIMIZER]: Activity
};

// Node type icons
const NODE_TYPE_ICONS: Record<NodeType, React.ComponentType<{ className?: string }>> = {
  [NodeType.AGENT]: Brain,
  [NodeType.TRIGGER]: Play,
  [NodeType.CONDITION]: GitBranch,
  [NodeType.TRANSFORM]: Shuffle,
  [NodeType.OUTPUT]: FileOutput,
  [NodeType.SUBFLOW]: Square,
  [NodeType.PARALLEL]: Activity,
  [NodeType.LOOP]: Circle,
  [NodeType.DELAY]: Clock
};

// Agent colors
const AGENT_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  [AgentType.STARRI]: { bg: 'bg-purple-50', border: 'border-purple-400', text: 'text-purple-700' },
  [AgentType.JULES]: { bg: 'bg-blue-50', border: 'border-blue-400', text: 'text-blue-700' },
  [AgentType.GEMINI]: { bg: 'bg-green-50', border: 'border-green-400', text: 'text-green-700' },
  [AgentType.RESEARCHER]: { bg: 'bg-yellow-50', border: 'border-yellow-400', text: 'text-yellow-700' },
  [AgentType.DEVELOPER]: { bg: 'bg-indigo-50', border: 'border-indigo-400', text: 'text-indigo-700' },
  [AgentType.DESIGNER]: { bg: 'bg-pink-50', border: 'border-pink-400', text: 'text-pink-700' },
  [AgentType.TESTER]: { bg: 'bg-red-50', border: 'border-red-400', text: 'text-red-700' },
  [AgentType.ANALYST]: { bg: 'bg-cyan-50', border: 'border-cyan-400', text: 'text-cyan-700' },
  [AgentType.OPTIMIZER]: { bg: 'bg-orange-50', border: 'border-orange-400', text: 'text-orange-700' }
};

export const WorkflowNodeComponent: React.FC<WorkflowNodeComponentProps> = memo(({
  node,
  isSelected,
  executionState,
  onMouseDown,
  onPortClick,
  onDelete,
  readOnly
}) => {
  const Icon = node.data.agentType ? AGENT_ICONS[node.data.agentType] : NODE_TYPE_ICONS[node.type];
  const colors = node.data.agentType ? AGENT_COLORS[node.data.agentType] : null;
  
  const getExecutionStatusIcon = () => {
    if (!executionState) return null;
    
    switch (executionState.status) {
      case NodeExecutionStatus.RUNNING:
        return <Activity className="h-4 w-4 text-blue-500 animate-pulse" />;
      case NodeExecutionStatus.COMPLETED:
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case NodeExecutionStatus.FAILED:
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return null;
    }
  };
  
  const getNodeClasses = () => {
    const baseClasses = "absolute rounded-lg shadow-lg transition-all duration-200 cursor-move";
    const sizeClasses = node.state.collapsed ? "w-32" : "w-60";
    const selectedClasses = isSelected ? "ring-2 ring-blue-500 ring-offset-2" : "";
    const colorClasses = colors ? `${colors.bg} ${colors.border} border-2` : "bg-white border-2 border-gray-300";
    const executionClasses = executionState?.status === NodeExecutionStatus.RUNNING ? "animate-pulse" : "";
    
    return `${baseClasses} ${sizeClasses} ${selectedClasses} ${colorClasses} ${executionClasses}`;
  };
  
  return (
    <motion.div
      className={getNodeClasses()}
      style={{
        left: node.position.x,
        top: node.position.y
      }}
      onMouseDown={onMouseDown}
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      exit={{ scale: 0.8, opacity: 0 }}
      whileHover={{ boxShadow: "0 10px 25px rgba(0, 0, 0, 0.1)" }}
    >
      {/* Header */}
      <div className={`p-3 rounded-t-lg flex items-center justify-between ${
        colors ? `${colors.bg} ${colors.text}` : "bg-gray-100"
      }`}>
        <div className="flex items-center space-x-2">
          <Icon className="h-5 w-5" />
          <span className="font-medium text-sm truncate">{node.data.label}</span>
        </div>
        <div className="flex items-center space-x-1">
          {getExecutionStatusIcon()}
          {!readOnly && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete();
              }}
              className="p-1 rounded hover:bg-black hover:bg-opacity-10 transition-colors"
            >
              <X className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>
      
      {/* Body (when not collapsed) */}
      {!node.state.collapsed && (
        <div className="p-3">
          {/* Description */}
          {node.data.description && (
            <p className="text-xs text-gray-600 mb-3">{node.data.description}</p>
          )}
          
          {/* Input Ports */}
          {node.data.inputs.length > 0 && (
            <div className="mb-3">
              <div className="text-xs text-gray-500 mb-1">Inputs:</div>
              {node.data.inputs.map((port, index) => (
                <div
                  key={port.id}
                  className="flex items-center mb-1"
                  style={{ marginLeft: -12 }}
                >
                  <div
                    className="w-3 h-3 rounded-full bg-gray-400 hover:bg-blue-500 transition-colors cursor-pointer"
                    onClick={(e) => onPortClick(port.id, false, e)}
                  />
                  <span className="ml-3 text-xs text-gray-700">{port.name}</span>
                  {port.required && <span className="ml-1 text-red-500 text-xs">*</span>}
                </div>
              ))}
            </div>
          )}
          
          {/* Output Ports */}
          {node.data.outputs.length > 0 && (
            <div>
              <div className="text-xs text-gray-500 mb-1">Outputs:</div>
              {node.data.outputs.map((port, index) => (
                <div
                  key={port.id}
                  className="flex items-center justify-end mb-1"
                  style={{ marginRight: -12 }}
                >
                  <span className="mr-3 text-xs text-gray-700">{port.name}</span>
                  <div
                    className="w-3 h-3 rounded-full bg-gray-400 hover:bg-green-500 transition-colors cursor-pointer"
                    onClick={(e) => onPortClick(port.id, true, e)}
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      
      {/* Execution Progress */}
      {executionState && executionState.status === NodeExecutionStatus.RUNNING && (
        <div className="px-3 pb-2">
          <div className="w-full bg-gray-200 rounded-full h-1">
            <div
              className="bg-blue-500 h-1 rounded-full transition-all duration-300"
              style={{ width: `${executionState.progress}%` }}
            />
          </div>
          <div className="text-xs text-gray-500 mt-1 text-center">
            {executionState.progress}%
          </div>
        </div>
      )}
      
      {/* Error State */}
      {node.state.error && (
        <div className="px-3 pb-2">
          <div className="text-xs text-red-600 flex items-center">
            <AlertCircle className="h-3 w-3 mr-1" />
            {node.state.error}
          </div>
        </div>
      )}
    </motion.div>
  );
});