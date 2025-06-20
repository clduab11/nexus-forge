/**
 * Workflow Templates
 * Pre-built workflow templates for common use cases
 */

import {
  WorkflowTemplate,
  WorkflowDefinition,
  NodeType,
  AgentType,
  ConnectionType,
  DataType,
  ExecutionMode,
  WorkflowCategory
} from '../types/workflow.types';

export const workflowTemplates: WorkflowTemplate[] = [
  {
    id: 'tpl_fullstack_app',
    category: WorkflowCategory.WEB_APPLICATION,
    name: 'Full-Stack Web Application',
    description: 'Complete web application with React frontend and Node.js backend',
    thumbnail: '/templates/fullstack-app.png',
    requirements: {
      agents: [AgentType.STARRI, AgentType.JULES, AgentType.DESIGNER, AgentType.TESTER],
      minNodes: 8,
      estimatedTime: '30-45 minutes',
      complexity: 'intermediate',
      prerequisites: ['Basic web development knowledge']
    },
    popularity: 95,
    examples: [
      {
        name: 'E-commerce Platform',
        description: 'Online store with product catalog and checkout',
        input: { projectType: 'ecommerce', features: ['catalog', 'cart', 'payment'] },
        expectedOutput: { frontend: 'React app', backend: 'Node.js API', database: 'PostgreSQL' }
      }
    ],
    workflow: {
      id: 'wf_fullstack_template',
      version: '1.0.0',
      metadata: {
        name: 'Full-Stack Web Application',
        description: 'Automated full-stack application development workflow',
        author: 'Nexus Forge',
        created: new Date(),
        modified: new Date(),
        tags: ['web', 'fullstack', 'react', 'nodejs'],
        category: WorkflowCategory.WEB_APPLICATION,
        estimatedDuration: 45,
        requiredAgents: [AgentType.STARRI, AgentType.JULES, AgentType.DESIGNER, AgentType.TESTER]
      },
      nodes: [
        {
          id: 'node_trigger',
          type: NodeType.TRIGGER,
          position: { x: 100, y: 200 },
          data: {
            label: 'Start',
            config: { triggerType: 'manual' },
            inputs: [],
            outputs: [{ id: 'start', name: 'Start Signal', type: DataType.ANY, multiple: true }]
          },
          state: { enabled: true, locked: false, collapsed: false }
        },
        {
          id: 'node_requirements',
          type: NodeType.AGENT,
          position: { x: 300, y: 200 },
          data: {
            label: 'Requirements Analyzer',
            agentType: AgentType.STARRI,
            config: {
              mode: 'analyzer',
              prompt: 'Analyze project requirements and create detailed specifications'
            },
            inputs: [{ id: 'trigger', name: 'Trigger', type: DataType.ANY, required: true }],
            outputs: [{ id: 'specs', name: 'Specifications', type: DataType.OBJECT, multiple: false }]
          },
          state: { enabled: true, locked: false, collapsed: false }
        },
        {
          id: 'node_architecture',
          type: NodeType.AGENT,
          position: { x: 550, y: 100 },
          data: {
            label: 'Architecture Designer',
            agentType: AgentType.GEMINI,
            config: {
              model: 'gemini-2.5-flash-thinking',
              temperature: 0.7
            },
            inputs: [{ id: 'specs', name: 'Specifications', type: DataType.OBJECT, required: true }],
            outputs: [{ id: 'architecture', name: 'Architecture', type: DataType.OBJECT, multiple: false }]
          },
          state: { enabled: true, locked: false, collapsed: false }
        },
        {
          id: 'node_ui_design',
          type: NodeType.AGENT,
          position: { x: 550, y: 300 },
          data: {
            label: 'UI Designer',
            agentType: AgentType.DESIGNER,
            config: {
              style: 'modern',
              framework: 'tailwindcss'
            },
            inputs: [{ id: 'specs', name: 'Specifications', type: DataType.OBJECT, required: true }],
            outputs: [{ id: 'designs', name: 'UI Designs', type: DataType.OBJECT, multiple: false }]
          },
          state: { enabled: true, locked: false, collapsed: false }
        },
        {
          id: 'node_parallel',
          type: NodeType.PARALLEL,
          position: { x: 800, y: 200 },
          data: {
            label: 'Parallel Development',
            config: {},
            inputs: [
              { id: 'architecture', name: 'Architecture', type: DataType.OBJECT, required: true },
              { id: 'designs', name: 'Designs', type: DataType.OBJECT, required: true }
            ],
            outputs: [{ id: 'ready', name: 'Ready', type: DataType.ANY, multiple: true }]
          },
          state: { enabled: true, locked: false, collapsed: false }
        },
        {
          id: 'node_frontend',
          type: NodeType.AGENT,
          position: { x: 1000, y: 100 },
          data: {
            label: 'Frontend Developer',
            agentType: AgentType.JULES,
            config: {
              framework: 'react',
              language: 'typescript'
            },
            inputs: [
              { id: 'trigger', name: 'Trigger', type: DataType.ANY, required: true },
              { id: 'designs', name: 'Designs', type: DataType.OBJECT, required: true }
            ],
            outputs: [{ id: 'frontend_code', name: 'Frontend Code', type: DataType.CODE, multiple: false }]
          },
          state: { enabled: true, locked: false, collapsed: false }
        },
        {
          id: 'node_backend',
          type: NodeType.AGENT,
          position: { x: 1000, y: 300 },
          data: {
            label: 'Backend Developer',
            agentType: AgentType.JULES,
            config: {
              framework: 'express',
              language: 'nodejs'
            },
            inputs: [
              { id: 'trigger', name: 'Trigger', type: DataType.ANY, required: true },
              { id: 'architecture', name: 'Architecture', type: DataType.OBJECT, required: true }
            ],
            outputs: [{ id: 'backend_code', name: 'Backend Code', type: DataType.CODE, multiple: false }]
          },
          state: { enabled: true, locked: false, collapsed: false }
        },
        {
          id: 'node_testing',
          type: NodeType.AGENT,
          position: { x: 1250, y: 200 },
          data: {
            label: 'Automated Tester',
            agentType: AgentType.TESTER,
            config: {
              testTypes: ['unit', 'integration', 'e2e']
            },
            inputs: [
              { id: 'frontend', name: 'Frontend Code', type: DataType.CODE, required: true },
              { id: 'backend', name: 'Backend Code', type: DataType.CODE, required: true }
            ],
            outputs: [{ id: 'test_results', name: 'Test Results', type: DataType.OBJECT, multiple: false }]
          },
          state: { enabled: true, locked: false, collapsed: false }
        },
        {
          id: 'node_output',
          type: NodeType.OUTPUT,
          position: { x: 1500, y: 200 },
          data: {
            label: 'Project Output',
            config: {
              format: 'zip',
              includeDocumentation: true
            },
            inputs: [
              { id: 'frontend', name: 'Frontend', type: DataType.CODE, required: true },
              { id: 'backend', name: 'Backend', type: DataType.CODE, required: true },
              { id: 'tests', name: 'Tests', type: DataType.OBJECT, required: true }
            ],
            outputs: []
          },
          state: { enabled: true, locked: false, collapsed: false }
        }
      ],
      connections: [
        {
          id: 'conn_1',
          source: { nodeId: 'node_trigger', portId: 'start' },
          target: { nodeId: 'node_requirements', portId: 'trigger' },
          type: ConnectionType.CONTROL_FLOW
        },
        {
          id: 'conn_2',
          source: { nodeId: 'node_requirements', portId: 'specs' },
          target: { nodeId: 'node_architecture', portId: 'specs' },
          type: ConnectionType.DATA_FLOW
        },
        {
          id: 'conn_3',
          source: { nodeId: 'node_requirements', portId: 'specs' },
          target: { nodeId: 'node_ui_design', portId: 'specs' },
          type: ConnectionType.DATA_FLOW
        },
        {
          id: 'conn_4',
          source: { nodeId: 'node_architecture', portId: 'architecture' },
          target: { nodeId: 'node_parallel', portId: 'architecture' },
          type: ConnectionType.DATA_FLOW
        },
        {
          id: 'conn_5',
          source: { nodeId: 'node_ui_design', portId: 'designs' },
          target: { nodeId: 'node_parallel', portId: 'designs' },
          type: ConnectionType.DATA_FLOW
        },
        {
          id: 'conn_6',
          source: { nodeId: 'node_parallel', portId: 'ready' },
          target: { nodeId: 'node_frontend', portId: 'trigger' },
          type: ConnectionType.CONTROL_FLOW
        },
        {
          id: 'conn_7',
          source: { nodeId: 'node_parallel', portId: 'ready' },
          target: { nodeId: 'node_backend', portId: 'trigger' },
          type: ConnectionType.CONTROL_FLOW
        },
        {
          id: 'conn_8',
          source: { nodeId: 'node_ui_design', portId: 'designs' },
          target: { nodeId: 'node_frontend', portId: 'designs' },
          type: ConnectionType.DATA_FLOW
        },
        {
          id: 'conn_9',
          source: { nodeId: 'node_architecture', portId: 'architecture' },
          target: { nodeId: 'node_backend', portId: 'architecture' },
          type: ConnectionType.DATA_FLOW
        },
        {
          id: 'conn_10',
          source: { nodeId: 'node_frontend', portId: 'frontend_code' },
          target: { nodeId: 'node_testing', portId: 'frontend' },
          type: ConnectionType.DATA_FLOW
        },
        {
          id: 'conn_11',
          source: { nodeId: 'node_backend', portId: 'backend_code' },
          target: { nodeId: 'node_testing', portId: 'backend' },
          type: ConnectionType.DATA_FLOW
        },
        {
          id: 'conn_12',
          source: { nodeId: 'node_frontend', portId: 'frontend_code' },
          target: { nodeId: 'node_output', portId: 'frontend' },
          type: ConnectionType.DATA_FLOW
        },
        {
          id: 'conn_13',
          source: { nodeId: 'node_backend', portId: 'backend_code' },
          target: { nodeId: 'node_output', portId: 'backend' },
          type: ConnectionType.DATA_FLOW
        },
        {
          id: 'conn_14',
          source: { nodeId: 'node_testing', portId: 'test_results' },
          target: { nodeId: 'node_output', portId: 'tests' },
          type: ConnectionType.DATA_FLOW
        }
      ],
      variables: [
        {
          name: 'projectName',
          type: DataType.STRING,
          defaultValue: 'my-app',
          scope: 'global',
          description: 'Name of the project being created'
        },
        {
          name: 'framework',
          type: DataType.STRING,
          defaultValue: 'react',
          scope: 'global',
          description: 'Frontend framework to use'
        }
      ],
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
        executionMode: ExecutionMode.HYBRID,
        errorHandling: {
          onNodeError: 'retry',
          onConnectionError: 'skip',
          notifyOnError: true,
          errorTimeout: 300
        },
        timeout: 3600,
        maxRetries: 3,
        logging: {
          level: 'info',
          destinations: [{ type: 'console', config: {} }],
          includeData: true
        },
        notifications: {
          onStart: true,
          onComplete: true,
          onError: true,
          channels: [{ type: 'in_app', config: {} }]
        },
        security: {
          authentication: false,
          authorization: [],
          encryption: false,
          auditLog: true
        }
      }
    }
  },
  
  {
    id: 'tpl_ai_chatbot',
    category: WorkflowCategory.AI_MODEL,
    name: 'AI Chatbot Service',
    description: 'Intelligent chatbot with natural language processing and custom training',
    thumbnail: '/templates/ai-chatbot.png',
    requirements: {
      agents: [AgentType.STARRI, AgentType.GEMINI, AgentType.DEVELOPER, AgentType.TESTER],
      minNodes: 6,
      estimatedTime: '20-30 minutes',
      complexity: 'intermediate',
      prerequisites: ['Understanding of NLP concepts']
    },
    popularity: 88,
    examples: [
      {
        name: 'Customer Support Bot',
        description: 'AI assistant for handling customer inquiries',
        input: { domain: 'customer_support', intents: ['faq', 'complaints', 'orders'] },
        expectedOutput: { model: 'Fine-tuned LLM', api: 'REST API', interface: 'Web chat widget' }
      }
    ],
    workflow: {
      id: 'wf_chatbot_template',
      version: '1.0.0',
      metadata: {
        name: 'AI Chatbot Service',
        description: 'Build and deploy an intelligent chatbot',
        author: 'Nexus Forge',
        created: new Date(),
        modified: new Date(),
        tags: ['ai', 'chatbot', 'nlp', 'machine-learning'],
        category: WorkflowCategory.AI_MODEL,
        estimatedDuration: 25,
        requiredAgents: [AgentType.STARRI, AgentType.GEMINI, AgentType.DEVELOPER]
      },
      nodes: [
        // Simplified node structure for brevity
        {
          id: 'node_analyze',
          type: NodeType.AGENT,
          position: { x: 100, y: 200 },
          data: {
            label: 'Domain Analyzer',
            agentType: AgentType.STARRI,
            config: { mode: 'analyzer' },
            inputs: [],
            outputs: [{ id: 'analysis', name: 'Domain Analysis', type: DataType.OBJECT, multiple: false }]
          },
          state: { enabled: true, locked: false, collapsed: false }
        }
      ],
      connections: [],
      variables: [],
      triggers: [],
      settings: {
        executionMode: ExecutionMode.SEQUENTIAL,
        errorHandling: {
          onNodeError: 'stop',
          onConnectionError: 'stop',
          notifyOnError: true,
          errorTimeout: 300
        },
        timeout: 1800,
        maxRetries: 2,
        logging: {
          level: 'debug',
          destinations: [{ type: 'console', config: {} }],
          includeData: true
        },
        notifications: {
          onStart: false,
          onComplete: true,
          onError: true,
          channels: []
        },
        security: {
          authentication: true,
          authorization: ['admin'],
          encryption: true,
          auditLog: true
        }
      }
    }
  },
  
  {
    id: 'tpl_data_pipeline',
    category: WorkflowCategory.DATA_PIPELINE,
    name: 'ETL Data Pipeline',
    description: 'Extract, transform, and load data from multiple sources',
    thumbnail: '/templates/data-pipeline.png',
    requirements: {
      agents: [AgentType.ANALYST, AgentType.DEVELOPER, AgentType.OPTIMIZER],
      minNodes: 7,
      estimatedTime: '15-25 minutes',
      complexity: 'advanced',
      prerequisites: ['Data engineering knowledge', 'SQL basics']
    },
    popularity: 76,
    examples: [
      {
        name: 'Sales Analytics Pipeline',
        description: 'Process sales data from multiple sources',
        input: { sources: ['csv', 'api', 'database'], transformations: ['clean', 'aggregate', 'enrich'] },
        expectedOutput: { pipeline: 'Apache Airflow DAG', storage: 'Data warehouse', dashboard: 'Analytics dashboard' }
      }
    ],
    workflow: {
      id: 'wf_data_pipeline_template',
      version: '1.0.0',
      metadata: {
        name: 'ETL Data Pipeline',
        description: 'Automated data processing pipeline',
        author: 'Nexus Forge',
        created: new Date(),
        modified: new Date(),
        tags: ['data', 'etl', 'pipeline', 'analytics'],
        category: WorkflowCategory.DATA_PIPELINE,
        estimatedDuration: 20,
        requiredAgents: [AgentType.ANALYST, AgentType.DEVELOPER]
      },
      nodes: [],
      connections: [],
      variables: [],
      triggers: [
        {
          id: 'trigger_schedule',
          type: 'schedule',
          name: 'Daily Run',
          config: { cron: '0 2 * * *' },
          enabled: true
        }
      ],
      settings: {
        executionMode: ExecutionMode.SEQUENTIAL,
        errorHandling: {
          onNodeError: 'retry',
          onConnectionError: 'retry',
          notifyOnError: true,
          errorTimeout: 600
        },
        timeout: 7200,
        maxRetries: 5,
        logging: {
          level: 'info',
          destinations: [
            { type: 'console', config: {} },
            { type: 'file', config: { path: '/logs/pipeline.log' } }
          ],
          includeData: true
        },
        notifications: {
          onStart: true,
          onComplete: true,
          onError: true,
          channels: [
            { type: 'email', config: { to: 'admin@company.com' } },
            { type: 'slack', config: { channel: '#data-alerts' } }
          ]
        },
        security: {
          authentication: true,
          authorization: ['data-engineer', 'admin'],
          encryption: true,
          auditLog: true
        }
      }
    }
  }
];

// Helper function to get templates by category
export const getTemplatesByCategory = (category: WorkflowCategory): WorkflowTemplate[] => {
  return workflowTemplates.filter(template => template.category === category);
};

// Helper function to get popular templates
export const getPopularTemplates = (limit: number = 5): WorkflowTemplate[] => {
  return [...workflowTemplates]
    .sort((a, b) => b.popularity - a.popularity)
    .slice(0, limit);
};