/**
 * WorkflowSidebar Component
 * Properties panel for workflow and node configuration
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Settings,
  Info,
  Code,
  Database,
  Shield,
  Bell,
  ChevronDown,
  ChevronRight,
  X,
  Save,
  RefreshCw
} from 'lucide-react';
import {
  WorkflowDefinition,
  WorkflowNode,
  WorkflowConnection,
  NodeType,
  AgentType,
  DataType,
  ExecutionMode,
  WorkflowSettings
} from '../../types/workflow.types';

interface WorkflowSidebarProps {
  workflow: WorkflowDefinition;
  selectedNodes: Set<string>;
  selectedConnections: Set<string>;
  onNodeUpdate: (nodeId: string, updates: Partial<WorkflowNode>) => void;
  onWorkflowUpdate: (workflow: WorkflowDefinition) => void;
  readOnly: boolean;
}

export const WorkflowSidebar: React.FC<WorkflowSidebarProps> = ({
  workflow,
  selectedNodes,
  selectedConnections,
  onNodeUpdate,
  onWorkflowUpdate,
  readOnly
}) => {
  const [expandedSections, setExpandedSections] = useState({
    workflow: true,
    node: true,
    connection: true,
    variables: false,
    settings: false
  });
  
  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };
  
  const selectedNode = selectedNodes.size === 1 
    ? workflow.nodes.find(n => n.id === Array.from(selectedNodes)[0])
    : null;
    
  const selectedConnection = selectedConnections.size === 1
    ? workflow.connections.find(c => c.id === Array.from(selectedConnections)[0])
    : null;
  
  const SectionHeader: React.FC<{
    icon: React.ComponentType<{ className?: string }>;
    title: string;
    section: keyof typeof expandedSections;
    count?: number;
  }> = ({ icon: Icon, title, section, count }) => (
    <button
      onClick={() => toggleSection(section)}
      className="flex items-center justify-between w-full p-3 hover:bg-gray-50 transition-colors"
    >
      <div className="flex items-center space-x-2">
        <Icon className="h-4 w-4 text-gray-600" />
        <span className="text-sm font-medium text-gray-900">{title}</span>
        {count !== undefined && (
          <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full">
            {count}
          </span>
        )}
      </div>
      {expandedSections[section] ? (
        <ChevronDown className="h-4 w-4 text-gray-500" />
      ) : (
        <ChevronRight className="h-4 w-4 text-gray-500" />
      )}
    </button>
  );
  
  const FormField: React.FC<{
    label: string;
    value: any;
    onChange: (value: any) => void;
    type?: 'text' | 'number' | 'select' | 'textarea' | 'checkbox';
    options?: { value: string; label: string }[];
    disabled?: boolean;
  }> = ({ label, value, onChange, type = 'text', options, disabled = false }) => {
    const inputClasses = "mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm disabled:bg-gray-50 disabled:text-gray-500";
    
    return (
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-1">
          {label}
        </label>
        {type === 'select' ? (
          <select
            value={value}
            onChange={(e) => onChange(e.target.value)}
            disabled={disabled || readOnly}
            className={inputClasses}
          >
            {options?.map(opt => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        ) : type === 'textarea' ? (
          <textarea
            value={value}
            onChange={(e) => onChange(e.target.value)}
            disabled={disabled || readOnly}
            rows={3}
            className={inputClasses}
          />
        ) : type === 'checkbox' ? (
          <input
            type="checkbox"
            checked={value}
            onChange={(e) => onChange(e.target.checked)}
            disabled={disabled || readOnly}
            className="rounded border-gray-300 text-blue-600 shadow-sm focus:border-blue-500 focus:ring-blue-500"
          />
        ) : (
          <input
            type={type}
            value={value}
            onChange={(e) => onChange(type === 'number' ? Number(e.target.value) : e.target.value)}
            disabled={disabled || readOnly}
            className={inputClasses}
          />
        )}
      </div>
    );
  };
  
  return (
    <div className="workflow-sidebar h-full flex flex-col bg-white">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900">Properties</h3>
      </div>
      
      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {/* Workflow Properties */}
        <div className="border-b border-gray-200">
          <SectionHeader
            icon={Info}
            title="Workflow Properties"
            section="workflow"
          />
          <AnimatePresence>
            {expandedSections.workflow && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="px-4 pb-4"
              >
                <FormField
                  label="Name"
                  value={workflow.metadata.name}
                  onChange={(value) => onWorkflowUpdate({
                    ...workflow,
                    metadata: { ...workflow.metadata, name: value }
                  })}
                />
                <FormField
                  label="Description"
                  value={workflow.metadata.description}
                  onChange={(value) => onWorkflowUpdate({
                    ...workflow,
                    metadata: { ...workflow.metadata, description: value }
                  })}
                  type="textarea"
                />
                <FormField
                  label="Category"
                  value={workflow.metadata.category || ''}
                  onChange={(value) => onWorkflowUpdate({
                    ...workflow,
                    metadata: { ...workflow.metadata, category: value }
                  })}
                  type="select"
                  options={[
                    { value: '', label: 'Select category' },
                    { value: 'web_application', label: 'Web Application' },
                    { value: 'mobile_application', label: 'Mobile Application' },
                    { value: 'api_service', label: 'API Service' },
                    { value: 'data_pipeline', label: 'Data Pipeline' },
                    { value: 'ai_model', label: 'AI Model' },
                    { value: 'automation', label: 'Automation' }
                  ]}
                />
                <FormField
                  label="Tags (comma-separated)"
                  value={workflow.metadata.tags.join(', ')}
                  onChange={(value) => onWorkflowUpdate({
                    ...workflow,
                    metadata: {
                      ...workflow.metadata,
                      tags: value.split(',').map((t: string) => t.trim()).filter(Boolean)
                    }
                  })}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        
        {/* Selected Node Properties */}
        {selectedNode && (
          <div className="border-b border-gray-200">
            <SectionHeader
              icon={Settings}
              title="Node Properties"
              section="node"
            />
            <AnimatePresence>
              {expandedSections.node && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="px-4 pb-4"
                >
                  <div className="mb-3 p-2 bg-gray-50 rounded">
                    <p className="text-xs text-gray-600">
                      Node ID: <span className="font-mono">{selectedNode.id}</span>
                    </p>
                    <p className="text-xs text-gray-600">
                      Type: <span className="font-medium">{selectedNode.type}</span>
                    </p>
                  </div>
                  
                  <FormField
                    label="Label"
                    value={selectedNode.data.label}
                    onChange={(value) => onNodeUpdate(selectedNode.id, {
                      data: { ...selectedNode.data, label: value }
                    })}
                  />
                  
                  <FormField
                    label="Description"
                    value={selectedNode.data.description || ''}
                    onChange={(value) => onNodeUpdate(selectedNode.id, {
                      data: { ...selectedNode.data, description: value }
                    })}
                    type="textarea"
                  />
                  
                  {selectedNode.type === NodeType.AGENT && (
                    <>
                      <FormField
                        label="Agent Type"
                        value={selectedNode.data.agentType || ''}
                        onChange={(value) => onNodeUpdate(selectedNode.id, {
                          data: { ...selectedNode.data, agentType: value as AgentType }
                        })}
                        type="select"
                        options={[
                          { value: AgentType.STARRI, label: 'Starri AI' },
                          { value: AgentType.JULES, label: 'Jules Coder' },
                          { value: AgentType.GEMINI, label: 'Gemini AI' },
                          { value: AgentType.RESEARCHER, label: 'Researcher' },
                          { value: AgentType.DEVELOPER, label: 'Developer' },
                          { value: AgentType.DESIGNER, label: 'Designer' },
                          { value: AgentType.TESTER, label: 'Tester' },
                          { value: AgentType.ANALYST, label: 'Analyst' },
                          { value: AgentType.OPTIMIZER, label: 'Optimizer' }
                        ]}
                      />
                      
                      {/* Agent-specific configuration */}
                      <div className="mt-4 p-3 bg-gray-50 rounded">
                        <h4 className="text-sm font-medium text-gray-700 mb-2">
                          Agent Configuration
                        </h4>
                        <FormField
                          label="Max Tokens"
                          value={selectedNode.data.config.maxTokens || 4096}
                          onChange={(value) => onNodeUpdate(selectedNode.id, {
                            data: {
                              ...selectedNode.data,
                              config: { ...selectedNode.data.config, maxTokens: value }
                            }
                          })}
                          type="number"
                        />
                        <FormField
                          label="Temperature"
                          value={selectedNode.data.config.temperature || 0.7}
                          onChange={(value) => onNodeUpdate(selectedNode.id, {
                            data: {
                              ...selectedNode.data,
                              config: { ...selectedNode.data.config, temperature: value }
                            }
                          })}
                          type="number"
                        />
                      </div>
                    </>
                  )}
                  
                  {/* Node state */}
                  <div className="mt-4">
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Node State</h4>
                    <div className="space-y-2">
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          checked={selectedNode.state.enabled}
                          onChange={(e) => onNodeUpdate(selectedNode.id, {
                            state: { ...selectedNode.state, enabled: e.target.checked }
                          })}
                          disabled={readOnly}
                          className="rounded border-gray-300 text-blue-600"
                        />
                        <span className="ml-2 text-sm text-gray-700">Enabled</span>
                      </label>
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          checked={selectedNode.state.locked}
                          onChange={(e) => onNodeUpdate(selectedNode.id, {
                            state: { ...selectedNode.state, locked: e.target.checked }
                          })}
                          disabled={readOnly}
                          className="rounded border-gray-300 text-blue-600"
                        />
                        <span className="ml-2 text-sm text-gray-700">Locked</span>
                      </label>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
        
        {/* Selected Connection Properties */}
        {selectedConnection && (
          <div className="border-b border-gray-200">
            <SectionHeader
              icon={Code}
              title="Connection Properties"
              section="connection"
            />
            <AnimatePresence>
              {expandedSections.connection && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="px-4 pb-4"
                >
                  <div className="mb-3 p-2 bg-gray-50 rounded">
                    <p className="text-xs text-gray-600">
                      Connection ID: <span className="font-mono">{selectedConnection.id}</span>
                    </p>
                    <p className="text-xs text-gray-600">
                      From: {selectedConnection.source.nodeId} → {selectedConnection.target.nodeId}
                    </p>
                  </div>
                  
                  <FormField
                    label="Connection Type"
                    value={selectedConnection.type}
                    onChange={(value) => {
                      const updated = workflow.connections.map(c =>
                        c.id === selectedConnection.id ? { ...c, type: value } : c
                      );
                      onWorkflowUpdate({ ...workflow, connections: updated });
                    }}
                    type="select"
                    options={[
                      { value: 'data', label: 'Data Flow' },
                      { value: 'control', label: 'Control Flow' },
                      { value: 'conditional', label: 'Conditional' },
                      { value: 'parallel', label: 'Parallel' },
                      { value: 'sequential', label: 'Sequential' }
                    ]}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
        
        {/* Workflow Settings */}
        <div className="border-b border-gray-200">
          <SectionHeader
            icon={Shield}
            title="Workflow Settings"
            section="settings"
          />
          <AnimatePresence>
            {expandedSections.settings && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="px-4 pb-4"
              >
                <FormField
                  label="Execution Mode"
                  value={workflow.settings.executionMode}
                  onChange={(value) => onWorkflowUpdate({
                    ...workflow,
                    settings: { ...workflow.settings, executionMode: value as ExecutionMode }
                  })}
                  type="select"
                  options={[
                    { value: ExecutionMode.SEQUENTIAL, label: 'Sequential' },
                    { value: ExecutionMode.PARALLEL, label: 'Parallel' },
                    { value: ExecutionMode.HYBRID, label: 'Hybrid' }
                  ]}
                />
                
                <FormField
                  label="Timeout (seconds)"
                  value={workflow.settings.timeout}
                  onChange={(value) => onWorkflowUpdate({
                    ...workflow,
                    settings: { ...workflow.settings, timeout: value }
                  })}
                  type="number"
                />
                
                <FormField
                  label="Max Retries"
                  value={workflow.settings.maxRetries}
                  onChange={(value) => onWorkflowUpdate({
                    ...workflow,
                    settings: { ...workflow.settings, maxRetries: value }
                  })}
                  type="number"
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};