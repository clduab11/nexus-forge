/**
 * NodePalette Component
 * Provides draggable nodes for the workflow editor
 */

import React, { useState } from 'react';
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
  Circle,
  Square,
  Zap,
  Palette,
  TestTube,
  BarChart,
  Activity,
  ChevronDown,
  ChevronRight,
  Plus
} from 'lucide-react';
import { NodeType, AgentType, Position } from '../../types/workflow.types';

interface NodePaletteProps {
  onAddNode: (nodeType: NodeType, agentType?: AgentType, position?: Position) => void;
  className?: string;
}

interface PaletteItem {
  type: NodeType;
  agentType?: AgentType;
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
}

const AGENT_NODES: PaletteItem[] = [
  {
    type: NodeType.AGENT,
    agentType: AgentType.STARRI,
    label: 'Starri AI',
    description: 'Master orchestrator and analyzer',
    icon: Brain,
    color: 'text-purple-600'
  },
  {
    type: NodeType.AGENT,
    agentType: AgentType.JULES,
    label: 'Jules Coder',
    description: 'Full-stack development agent',
    icon: Code,
    color: 'text-blue-600'
  },
  {
    type: NodeType.AGENT,
    agentType: AgentType.GEMINI,
    label: 'Gemini AI',
    description: 'Advanced reasoning and analysis',
    icon: Zap,
    color: 'text-green-600'
  },
  {
    type: NodeType.AGENT,
    agentType: AgentType.RESEARCHER,
    label: 'Researcher',
    description: 'Web research and data gathering',
    icon: Search,
    color: 'text-yellow-600'
  },
  {
    type: NodeType.AGENT,
    agentType: AgentType.DEVELOPER,
    label: 'Developer',
    description: 'Specialized coding tasks',
    icon: Code,
    color: 'text-indigo-600'
  },
  {
    type: NodeType.AGENT,
    agentType: AgentType.DESIGNER,
    label: 'Designer',
    description: 'UI/UX design and styling',
    icon: Palette,
    color: 'text-pink-600'
  },
  {
    type: NodeType.AGENT,
    agentType: AgentType.TESTER,
    label: 'Tester',
    description: 'Automated testing and QA',
    icon: TestTube,
    color: 'text-red-600'
  },
  {
    type: NodeType.AGENT,
    agentType: AgentType.ANALYST,
    label: 'Analyst',
    description: 'Data analysis and insights',
    icon: BarChart,
    color: 'text-cyan-600'
  },
  {
    type: NodeType.AGENT,
    agentType: AgentType.OPTIMIZER,
    label: 'Optimizer',
    description: 'Performance and cost optimization',
    icon: Activity,
    color: 'text-orange-600'
  }
];

const CONTROL_NODES: PaletteItem[] = [
  {
    type: NodeType.TRIGGER,
    label: 'Trigger',
    description: 'Start workflow execution',
    icon: Play,
    color: 'text-green-600'
  },
  {
    type: NodeType.CONDITION,
    label: 'Condition',
    description: 'Conditional branching',
    icon: GitBranch,
    color: 'text-amber-600'
  },
  {
    type: NodeType.TRANSFORM,
    label: 'Transform',
    description: 'Data transformation',
    icon: Shuffle,
    color: 'text-indigo-600'
  },
  {
    type: NodeType.PARALLEL,
    label: 'Parallel',
    description: 'Parallel execution',
    icon: Activity,
    color: 'text-purple-600'
  },
  {
    type: NodeType.LOOP,
    label: 'Loop',
    description: 'Iterative execution',
    icon: Circle,
    color: 'text-blue-600'
  },
  {
    type: NodeType.DELAY,
    label: 'Delay',
    description: 'Time-based delay',
    icon: Clock,
    color: 'text-gray-600'
  },
  {
    type: NodeType.SUBFLOW,
    label: 'Subflow',
    description: 'Nested workflow',
    icon: Square,
    color: 'text-cyan-600'
  },
  {
    type: NodeType.OUTPUT,
    label: 'Output',
    description: 'Workflow output',
    icon: FileOutput,
    color: 'text-green-600'
  }
];

export const NodePalette: React.FC<NodePaletteProps> = ({ onAddNode, className = '' }) => {
  const [expandedSections, setExpandedSections] = useState({
    agents: true,
    controls: true
  });
  
  const [draggingItem, setDraggingItem] = useState<PaletteItem | null>(null);
  
  const toggleSection = (section: 'agents' | 'controls') => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };
  
  const handleDragStart = (item: PaletteItem, e: React.DragEvent) => {
    setDraggingItem(item);
    e.dataTransfer.effectAllowed = 'copy';
  };
  
  const handleDragEnd = () => {
    setDraggingItem(null);
  };
  
  const handleClick = (item: PaletteItem) => {
    // Add node at default position when clicked
    onAddNode(item.type, item.agentType);
  };
  
  const renderPaletteItem = (item: PaletteItem) => {
    const Icon = item.icon;
    
    return (
      <motion.div
        key={`${item.type}-${item.agentType || item.label}`}
        className={`
          flex items-center p-3 bg-white border border-gray-200 rounded-lg
          cursor-pointer hover:shadow-md transition-all duration-200
          ${draggingItem === item ? 'opacity-50' : ''}
        `}
        draggable
        onDragStart={(e) => handleDragStart(item, e)}
        onDragEnd={handleDragEnd}
        onClick={() => handleClick(item)}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <Icon className={`h-5 w-5 ${item.color} mr-3`} />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-gray-900 truncate">
            {item.label}
          </div>
          <div className="text-xs text-gray-500 truncate">
            {item.description}
          </div>
        </div>
        <Plus className="h-4 w-4 text-gray-400 ml-2" />
      </motion.div>
    );
  };
  
  return (
    <div className={`node-palette overflow-y-auto ${className}`}>
      <div className="p-4">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Workflow Nodes</h3>
        
        {/* Agent Nodes Section */}
        <div className="mb-6">
          <button
            onClick={() => toggleSection('agents')}
            className="flex items-center justify-between w-full mb-3 text-left"
          >
            <h4 className="text-sm font-medium text-gray-700">AI Agents</h4>
            {expandedSections.agents ? (
              <ChevronDown className="h-4 w-4 text-gray-500" />
            ) : (
              <ChevronRight className="h-4 w-4 text-gray-500" />
            )}
          </button>
          
          <AnimatePresence>
            {expandedSections.agents && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="space-y-2"
              >
                {AGENT_NODES.map(renderPaletteItem)}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        
        {/* Control Nodes Section */}
        <div className="mb-6">
          <button
            onClick={() => toggleSection('controls')}
            className="flex items-center justify-between w-full mb-3 text-left"
          >
            <h4 className="text-sm font-medium text-gray-700">Control Flow</h4>
            {expandedSections.controls ? (
              <ChevronDown className="h-4 w-4 text-gray-500" />
            ) : (
              <ChevronRight className="h-4 w-4 text-gray-500" />
            )}
          </button>
          
          <AnimatePresence>
            {expandedSections.controls && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="space-y-2"
              >
                {CONTROL_NODES.map(renderPaletteItem)}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        
        {/* Instructions */}
        <div className="mt-6 p-3 bg-gray-50 rounded-lg">
          <p className="text-xs text-gray-600">
            <strong>Tip:</strong> Drag nodes to the canvas or click to add at default position.
            Connect nodes by clicking output ports and then input ports.
          </p>
        </div>
      </div>
    </div>
  );
};