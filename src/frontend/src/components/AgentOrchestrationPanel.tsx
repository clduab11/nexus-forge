import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Users, 
  Activity, 
  CheckCircle, 
  AlertCircle, 
  Clock, 
  Zap, 
  Brain,
  Code,
  Palette,
  Settings,
  RefreshCw
} from 'lucide-react';
import { AgentStatus } from '../services/nexusForgeApi';
import { useWebSocket } from '../contexts/WebSocketContext';
import { LoadingSpinner } from './common/LoadingSpinner';

interface AgentOrchestrationPanelProps {
  agentStatuses: AgentStatus[];
  isLoading: boolean;
}

const AGENT_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  'starri': Brain,
  'jules': Code,
  'gemini': Zap,
  'design': Palette,
  'orchestrator': Settings,
  'default': Activity,
};

const AGENT_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  'starri': { bg: 'bg-purple-50', text: 'text-purple-700', border: 'border-purple-200' },
  'jules': { bg: 'bg-blue-50', text: 'text-blue-700', border: 'border-blue-200' },
  'gemini': { bg: 'bg-green-50', text: 'text-green-700', border: 'border-green-200' },
  'design': { bg: 'bg-pink-50', text: 'text-pink-700', border: 'border-pink-200' },
  'orchestrator': { bg: 'bg-indigo-50', text: 'text-indigo-700', border: 'border-indigo-200' },
  'default': { bg: 'bg-gray-50', text: 'text-gray-700', border: 'border-gray-200' },
};

export const AgentOrchestrationPanel: React.FC<AgentOrchestrationPanelProps> = ({
  agentStatuses,
  isLoading,
}) => {
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const { state: wsState } = useWebSocket();

  // Real-time agent status updates from WebSocket
  const [liveAgentStatuses, setLiveAgentStatuses] = useState<Map<string, AgentStatus>>(new Map());

  useEffect(() => {
    // Update live statuses from WebSocket
    wsState.agentStatuses.forEach((agentStatus, agentId) => {
      setLiveAgentStatuses(prev => new Map(prev.set(agentId, agentStatus)));
    });
  }, [wsState.agentStatuses]);

  // Merge API data with live WebSocket updates
  const mergedAgentStatuses = agentStatuses.map(agent => {
    const liveStatus = liveAgentStatuses.get(agent.agent_id);
    return liveStatus || agent;
  });

  const getAgentIcon = (agentName: string) => {
    const agentType = agentName.toLowerCase();
    return AGENT_ICONS[agentType] || AGENT_ICONS.default;
  };

  const getAgentColors = (agentName: string) => {
    const agentType = agentName.toLowerCase();
    return AGENT_COLORS[agentType] || AGENT_COLORS.default;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'working':
        return <Activity className="h-4 w-4 text-blue-500 animate-pulse" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'idle':
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'working':
        return 'text-blue-600 bg-blue-50';
      case 'completed':
        return 'text-green-600 bg-green-50';
      case 'error':
        return 'text-red-600 bg-red-50';
      case 'idle':
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  const activeAgents = mergedAgentStatuses.filter(agent => agent.status === 'working');
  const idleAgents = mergedAgentStatuses.filter(agent => agent.status === 'idle');
  const errorAgents = mergedAgentStatuses.filter(agent => agent.status === 'error');

  return (
    <div className="bg-white rounded-lg shadow">
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900 flex items-center">
              <Users className="h-5 w-5 mr-2" />
              Agent Orchestration
            </h2>
            <p className="text-sm text-gray-600">Live agent coordination and status</p>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-sm">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
              <span className="text-blue-600 font-medium">{activeAgents.length} Active</span>
            </div>
            <div className="flex items-center space-x-2 text-sm">
              <div className="w-2 h-2 bg-gray-400 rounded-full" />
              <span className="text-gray-600">{idleAgents.length} Idle</span>
            </div>
            {errorAgents.length > 0 && (
              <div className="flex items-center space-x-2 text-sm">
                <div className="w-2 h-2 bg-red-400 rounded-full" />
                <span className="text-red-600">{errorAgents.length} Error</span>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="p-6">
        {isLoading ? (
          <div className="flex justify-center py-8">
            <LoadingSpinner size="md" text="Loading agent statuses..." />
          </div>
        ) : mergedAgentStatuses.length === 0 ? (
          <div className="text-center py-8">
            <Users className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No agents available</h3>
            <p className="mt-1 text-sm text-gray-500">
              Agents will appear here when they come online.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Active Agents */}
            {activeAgents.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-3">Active Agents</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {activeAgents.map((agent) => (
                    <AgentCard
                      key={agent.agent_id}
                      agent={agent}
                      onClick={() => setSelectedAgent(
                        selectedAgent === agent.agent_id ? null : agent.agent_id
                      )}
                      isSelected={selectedAgent === agent.agent_id}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Idle Agents */}
            {idleAgents.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-3">Idle Agents</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {idleAgents.map((agent) => (
                    <AgentCard
                      key={agent.agent_id}
                      agent={agent}
                      onClick={() => setSelectedAgent(
                        selectedAgent === agent.agent_id ? null : agent.agent_id
                      )}
                      isSelected={selectedAgent === agent.agent_id}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Error Agents */}
            {errorAgents.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-red-700 mb-3">Agents with Errors</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {errorAgents.map((agent) => (
                    <AgentCard
                      key={agent.agent_id}
                      agent={agent}
                      onClick={() => setSelectedAgent(
                        selectedAgent === agent.agent_id ? null : agent.agent_id
                      )}
                      isSelected={selectedAgent === agent.agent_id}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Agent Details Modal */}
        <AnimatePresence>
          {selectedAgent && (
            <AgentDetailsModal
              agent={mergedAgentStatuses.find(a => a.agent_id === selectedAgent)!}
              onClose={() => setSelectedAgent(null)}
            />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

interface AgentCardProps {
  agent: AgentStatus;
  onClick: () => void;
  isSelected: boolean;
}

const AgentCard: React.FC<AgentCardProps> = ({ agent, onClick, isSelected }) => {
  const Icon = AGENT_ICONS[agent.name.toLowerCase()] || AGENT_ICONS.default;
  const colors = AGENT_COLORS[agent.name.toLowerCase()] || AGENT_COLORS.default;

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={`border rounded-lg p-4 cursor-pointer transition-all duration-200 ${
        isSelected
          ? 'border-primary-500 bg-primary-50'
          : `border-gray-200 bg-white hover:${colors.border} hover:shadow-sm`
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className={`flex-shrink-0 p-2 rounded-lg ${colors.bg} ${colors.border} border`}>
            <Icon className={`h-5 w-5 ${colors.text}`} />
          </div>
          <div className="min-w-0 flex-1">
            <h3 className="text-sm font-medium text-gray-900 truncate">
              {agent.name}
            </h3>
            <p className="text-xs text-gray-500 truncate">
              {agent.current_task || 'No active task'}
            </p>
          </div>
        </div>
        
        <div className="flex flex-col items-end space-y-1">
          <div className="flex items-center space-x-1">
            {agent.status === 'working' && (
              <RefreshCw className="h-3 w-3 text-blue-500 animate-spin" />
            )}
            <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
              agent.status === 'working' ? 'text-blue-600 bg-blue-50' :
              agent.status === 'completed' ? 'text-green-600 bg-green-50' :
              agent.status === 'error' ? 'text-red-600 bg-red-50' :
              'text-gray-600 bg-gray-50'
            }`}>
              {agent.status}
            </span>
          </div>
          
          {agent.status === 'working' && (
            <div className="flex items-center space-x-1 text-xs text-gray-500">
              <span>{agent.progress}%</span>
              <div className="w-8 h-1 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-300"
                  style={{ width: `${agent.progress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>
      
      <div className="mt-2 text-xs text-gray-500">
        Last update: {new Date(agent.last_update).toLocaleTimeString()}
      </div>
    </motion.div>
  );
};

interface AgentDetailsModalProps {
  agent: AgentStatus;
  onClose: () => void;
}

const AgentDetailsModal: React.FC<AgentDetailsModalProps> = ({ agent, onClose }) => {
  const Icon = AGENT_ICONS[agent.name.toLowerCase()] || AGENT_ICONS.default;
  const colors = AGENT_COLORS[agent.name.toLowerCase()] || AGENT_COLORS.default;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
        className="bg-white rounded-lg p-6 max-w-md w-full"
      >
        <div className="flex items-center space-x-3 mb-4">
          <div className={`p-3 rounded-lg ${colors.bg} ${colors.border} border`}>
            <Icon className={`h-6 w-6 ${colors.text}`} />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">{agent.name}</h3>
            <p className="text-sm text-gray-600">Agent ID: {agent.agent_id}</p>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Status</label>
            <div className="mt-1 flex items-center space-x-2">
              {agent.status === 'working' && <Activity className="h-4 w-4 text-blue-500 animate-pulse" />}
              {agent.status === 'completed' && <CheckCircle className="h-4 w-4 text-green-500" />}
              {agent.status === 'error' && <AlertCircle className="h-4 w-4 text-red-500" />}
              {agent.status === 'idle' && <Clock className="h-4 w-4 text-gray-400" />}
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                agent.status === 'working' ? 'text-blue-600 bg-blue-50' :
                agent.status === 'completed' ? 'text-green-600 bg-green-50' :
                agent.status === 'error' ? 'text-red-600 bg-red-50' :
                'text-gray-600 bg-gray-50'
              }`}>
                {agent.status}
              </span>
            </div>
          </div>

          {agent.current_task && (
            <div>
              <label className="block text-sm font-medium text-gray-700">Current Task</label>
              <p className="mt-1 text-sm text-gray-900">{agent.current_task}</p>
            </div>
          )}

          {agent.status === 'working' && (
            <div>
              <label className="block text-sm font-medium text-gray-700">Progress</label>
              <div className="mt-1">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">{agent.progress}%</span>
                </div>
                <div className="mt-1 w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${agent.progress}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700">Last Update</label>
            <p className="mt-1 text-sm text-gray-900">
              {new Date(agent.last_update).toLocaleString()}
            </p>
          </div>
        </div>

        <div className="mt-6 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-200 border border-gray-300 rounded-md hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
          >
            Close
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
};