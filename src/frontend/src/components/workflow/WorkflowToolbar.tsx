/**
 * WorkflowToolbar Component
 * Toolbar for workflow editor actions
 */

import React from 'react';
import { motion } from 'framer-motion';
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
  Undo,
  Redo,
  Settings,
  HelpCircle,
  Share2,
  Copy,
  Trash2
} from 'lucide-react';
import { WorkflowDefinition } from '../../types/workflow.types';

interface WorkflowToolbarProps {
  workflow: WorkflowDefinition;
  canUndo: boolean;
  canRedo: boolean;
  isExecuting: boolean;
  showGrid: boolean;
  snapToGrid: boolean;
  onSave: () => void;
  onLoad: () => void;
  onExecute: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onZoomReset: () => void;
  onToggleGrid: () => void;
  onToggleSnap: () => void;
  onToggleMinimap: () => void;
  onToggleSidebar: () => void;
  readOnly: boolean;
}

export const WorkflowToolbar: React.FC<WorkflowToolbarProps> = ({
  workflow,
  canUndo,
  canRedo,
  isExecuting,
  showGrid,
  snapToGrid,
  onSave,
  onLoad,
  onExecute,
  onUndo,
  onRedo,
  onZoomIn,
  onZoomOut,
  onZoomReset,
  onToggleGrid,
  onToggleSnap,
  onToggleMinimap,
  onToggleSidebar,
  readOnly
}) => {
  const ToolbarButton: React.FC<{
    icon: React.ComponentType<{ className?: string }>;
    label: string;
    onClick: () => void;
    disabled?: boolean;
    active?: boolean;
    variant?: 'default' | 'primary' | 'danger';
  }> = ({ icon: Icon, label, onClick, disabled = false, active = false, variant = 'default' }) => {
    const getButtonClasses = () => {
      const base = "p-2 rounded-lg transition-all duration-200 relative group";
      
      if (disabled) {
        return `${base} opacity-50 cursor-not-allowed text-gray-400`;
      }
      
      switch (variant) {
        case 'primary':
          return `${base} ${active ? 'bg-blue-500 text-white' : 'hover:bg-blue-50 text-blue-600'}`;
        case 'danger':
          return `${base} hover:bg-red-50 text-red-600`;
        default:
          return `${base} ${
            active 
              ? 'bg-gray-200 text-gray-900' 
              : 'hover:bg-gray-100 text-gray-700'
          }`;
      }
    };
    
    return (
      <motion.button
        whileHover={!disabled ? { scale: 1.05 } : {}}
        whileTap={!disabled ? { scale: 0.95 } : {}}
        className={getButtonClasses()}
        onClick={onClick}
        disabled={disabled}
        title={label}
      >
        <Icon className="h-5 w-5" />
        <span className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 text-xs text-white bg-gray-800 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
          {label}
        </span>
      </motion.button>
    );
  };
  
  const ToolbarSeparator = () => (
    <div className="w-px h-8 bg-gray-300 mx-1" />
  );
  
  return (
    <div className="workflow-toolbar bg-white border-b border-gray-200 px-4 py-2">
      <div className="flex items-center justify-between">
        {/* Left section */}
        <div className="flex items-center space-x-2">
          {/* File operations */}
          {!readOnly && (
            <>
              <ToolbarButton
                icon={Save}
                label="Save Workflow (Ctrl+S)"
                onClick={onSave}
              />
              <ToolbarButton
                icon={Upload}
                label="Load Workflow"
                onClick={onLoad}
              />
              <ToolbarButton
                icon={Download}
                label="Export Workflow"
                onClick={onSave}
              />
              <ToolbarSeparator />
            </>
          )}
          
          {/* Execution controls */}
          <ToolbarButton
            icon={isExecuting ? Stop : Play}
            label={isExecuting ? "Stop Execution" : "Execute Workflow"}
            onClick={onExecute}
            variant="primary"
            active={isExecuting}
          />
          
          <ToolbarSeparator />
          
          {/* Edit operations */}
          {!readOnly && (
            <>
              <ToolbarButton
                icon={Undo}
                label="Undo (Ctrl+Z)"
                onClick={onUndo}
                disabled={!canUndo}
              />
              <ToolbarButton
                icon={Redo}
                label="Redo (Ctrl+Y)"
                onClick={onRedo}
                disabled={!canRedo}
              />
              <ToolbarSeparator />
            </>
          )}
          
          {/* View controls */}
          <ToolbarButton
            icon={ZoomOut}
            label="Zoom Out"
            onClick={onZoomOut}
          />
          <ToolbarButton
            icon={ZoomIn}
            label="Zoom In"
            onClick={onZoomIn}
          />
          <ToolbarButton
            icon={Maximize2}
            label="Reset Zoom"
            onClick={onZoomReset}
          />
          
          <ToolbarSeparator />
          
          {/* Toggle options */}
          <ToolbarButton
            icon={Grid}
            label={showGrid ? "Hide Grid" : "Show Grid"}
            onClick={onToggleGrid}
            active={showGrid}
          />
          <ToolbarButton
            icon={snapToGrid ? Eye : EyeOff}
            label={snapToGrid ? "Disable Snap to Grid" : "Enable Snap to Grid"}
            onClick={onToggleSnap}
            active={snapToGrid}
          />
        </div>
        
        {/* Center section - Workflow name */}
        <div className="flex-1 text-center">
          <h2 className="text-lg font-semibold text-gray-900">
            {workflow.metadata.name || 'Untitled Workflow'}
          </h2>
        </div>
        
        {/* Right section */}
        <div className="flex items-center space-x-2">
          <ToolbarButton
            icon={Share2}
            label="Share Workflow"
            onClick={() => {}}
            disabled={readOnly}
          />
          <ToolbarButton
            icon={Settings}
            label="Workflow Settings"
            onClick={onToggleSidebar}
          />
          <ToolbarButton
            icon={HelpCircle}
            label="Help"
            onClick={() => window.open('/docs/workflow-editor', '_blank')}
          />
        </div>
      </div>
      
      {/* Workflow info bar */}
      <div className="flex items-center justify-between mt-2 pt-2 border-t border-gray-100 text-xs text-gray-600">
        <div className="flex items-center space-x-4">
          <span>Version: {workflow.version}</span>
          <span>Created: {new Date(workflow.metadata.created).toLocaleDateString()}</span>
          <span>Modified: {new Date(workflow.metadata.modified).toLocaleDateString()}</span>
        </div>
        <div className="flex items-center space-x-4">
          <span>Category: {workflow.metadata.category || 'Uncategorized'}</span>
          {workflow.metadata.tags.length > 0 && (
            <span>Tags: {workflow.metadata.tags.join(', ')}</span>
          )}
        </div>
      </div>
    </div>
  );
};