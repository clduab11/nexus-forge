/**
 * WorkflowBuilderPage Component
 * Main page for the visual workflow builder
 */

import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Sparkles, Play, Save, FileText, HelpCircle } from 'lucide-react';
import { WorkflowEditor } from '../components/workflow';
import { workflowTemplates, getPopularTemplates } from '../data/workflowTemplates';
import { workflowExecutor } from '../services/workflowExecutor';
import { nexusForgeApi } from '../services/nexusForgeApi';
import { useWebSocket } from '../contexts/WebSocketContext';
import { toast } from 'react-hot-toast';
import {
  WorkflowDefinition,
  WorkflowTemplate,
  WorkflowExecution
} from '../types/workflow.types';

export const WorkflowBuilderPage: React.FC = () => {
  const navigate = useNavigate();
  const { subscribeToWorkflow } = useWebSocket();
  
  const [selectedTemplate, setSelectedTemplate] = useState<WorkflowTemplate | null>(null);
  const [currentWorkflow, setCurrentWorkflow] = useState<WorkflowDefinition | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionProgress, setExecutionProgress] = useState(0);
  
  // Handle template selection
  const handleTemplateSelect = useCallback((template: WorkflowTemplate) => {
    setSelectedTemplate(template);
    setCurrentWorkflow(template.workflow);
    toast.success(`Loaded template: ${template.name}`);
  }, []);
  
  // Handle workflow save
  const handleSaveWorkflow = useCallback(async (workflow: WorkflowDefinition) => {
    try {
      // Save to backend
      const response = await nexusForgeApi.saveWorkflow(workflow);
      toast.success('Workflow saved successfully');
      return response;
    } catch (error) {
      toast.error('Failed to save workflow');
      console.error('Save error:', error);
    }
  }, []);
  
  // Handle workflow execution
  const handleExecuteWorkflow = useCallback(async (workflow: WorkflowDefinition) => {
    setIsExecuting(true);
    setExecutionProgress(0);
    
    try {
      // Subscribe to real-time updates
      const unsubscribe = subscribeToWorkflow(workflow.id, (update) => {
        if (update.type === 'progress') {
          setExecutionProgress(update.progress);
        }
      });
      
      // Execute workflow
      const execution = await workflowExecutor.execute(workflow, {
        parallel: true,
        onProgress: (progress) => setExecutionProgress(progress),
        onNodeStart: (nodeId) => {
          toast(`Executing: ${workflow.nodes.find(n => n.id === nodeId)?.data.label}`);
        },
        onNodeComplete: (nodeId, output) => {
          console.log(`Node ${nodeId} completed:`, output);
        },
        onNodeError: (nodeId, error) => {
          toast.error(`Node failed: ${error.message}`);
        }
      });
      
      // Handle completion
      if (execution.status === 'completed') {
        toast.success('Workflow executed successfully!');
        navigate(`/results/${execution.id}`);
      } else {
        toast.error('Workflow execution failed');
      }
      
      unsubscribe();
    } catch (error) {
      toast.error(`Execution error: ${error.message}`);
      console.error('Execution error:', error);
    } finally {
      setIsExecuting(false);
      setExecutionProgress(0);
    }
  }, [navigate, subscribeToWorkflow]);
  
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <Sparkles className="h-8 w-8 text-primary-600 mr-3" />
              <h1 className="text-xl font-semibold text-gray-900">
                Visual Workflow Builder
              </h1>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={() => window.open('/docs/workflow-builder', '_blank')}
                className="flex items-center px-3 py-2 text-sm text-gray-700 hover:text-gray-900"
              >
                <HelpCircle className="h-4 w-4 mr-1" />
                Help
              </button>
            </div>
          </div>
        </div>
      </div>
      
      {/* Main Content */}
      {!currentWorkflow ? (
        // Template Selection View
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Choose a Template or Start from Scratch
            </h2>
            <p className="text-lg text-gray-600">
              Select a pre-built workflow template or create your own custom workflow
            </p>
          </div>
          
          {/* Create New Button */}
          <div className="mb-8">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setCurrentWorkflow(createEmptyWorkflow())}
              className="w-full p-6 bg-gradient-to-r from-primary-600 to-primary-700 text-white rounded-lg shadow-lg hover:shadow-xl transition-shadow"
            >
              <div className="flex items-center justify-center">
                <Sparkles className="h-8 w-8 mr-3" />
                <span className="text-xl font-semibold">Create New Workflow</span>
              </div>
              <p className="mt-2 text-primary-100">
                Start with a blank canvas and build your workflow from scratch
              </p>
            </motion.button>
          </div>
          
          {/* Popular Templates */}
          <div>
            <h3 className="text-xl font-semibold text-gray-900 mb-4">
              Popular Templates
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {getPopularTemplates(6).map(template => (
                <motion.div
                  key={template.id}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => handleTemplateSelect(template)}
                  className="bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow cursor-pointer overflow-hidden"
                >
                  <div className="p-6">
                    <div className="flex items-center mb-3">
                      <FileText className="h-6 w-6 text-primary-600 mr-2" />
                      <h4 className="text-lg font-medium text-gray-900">
                        {template.name}
                      </h4>
                    </div>
                    <p className="text-sm text-gray-600 mb-4">
                      {template.description}
                    </p>
                    <div className="flex items-center justify-between text-xs text-gray-500">
                      <span>{template.requirements.estimatedTime}</span>
                      <span className="capitalize">{template.requirements.complexity}</span>
                    </div>
                    <div className="mt-3 flex flex-wrap gap-1">
                      {template.requirements.agents.map(agent => (
                        <span
                          key={agent}
                          className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs"
                        >
                          {agent}
                        </span>
                      ))}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        // Workflow Editor View
        <div className="h-[calc(100vh-4rem)]">
          <WorkflowEditor
            initialWorkflow={currentWorkflow}
            onSave={handleSaveWorkflow}
            onExecute={handleExecuteWorkflow}
            readOnly={false}
          />
          
          {/* Execution Progress Overlay */}
          {isExecuting && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="bg-white rounded-lg p-8 max-w-md w-full"
              >
                <div className="text-center">
                  <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4">
                    <Play className="h-8 w-8 text-blue-600 animate-pulse" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">
                    Executing Workflow
                  </h3>
                  <p className="text-gray-600 mb-6">
                    Your workflow is being executed by our AI agents...
                  </p>
                  
                  {/* Progress Bar */}
                  <div className="mb-4">
                    <div className="flex justify-between text-sm text-gray-600 mb-1">
                      <span>Progress</span>
                      <span>{executionProgress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <motion.div
                        className="bg-blue-600 h-2 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${executionProgress}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                  </div>
                  
                  <button
                    onClick={() => workflowExecutor.cancel(currentWorkflow.id)}
                    className="px-4 py-2 text-sm text-gray-700 bg-gray-200 rounded-lg hover:bg-gray-300 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </motion.div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Helper function to create empty workflow
const createEmptyWorkflow = (): WorkflowDefinition => ({
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
  triggers: [
    {
      id: 'trigger_manual',
      type: 'manual',
      name: 'Manual Start',
      config: {},
      enabled: true
    }
  ],
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