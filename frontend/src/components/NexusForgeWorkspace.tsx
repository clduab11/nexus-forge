import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Activity, 
  Clock, 
  CheckCircle, 
  AlertCircle, 
  Users, 
  Zap,
  TrendingUp
} from 'lucide-react';
import { useQuery } from 'react-query';
import { nexusForgeApi, ProjectResponse, AgentStatus } from '../services/nexusForgeApi';
import { useWebSocket } from '../contexts/WebSocketContext';
import { LoadingSpinner, ProgressBar } from './common/LoadingSpinner';
import { AgentOrchestrationPanel } from './AgentOrchestrationPanel';
import { TaskProgressTracker } from './TaskProgressTracker';

export const NexusForgeWorkspace: React.FC = () => {
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const { state: wsState, subscribeToAgents } = useWebSocket();

  // Fetch projects data
  const { data: projects = [], isLoading: projectsLoading, refetch: refetchProjects } = useQuery(
    'projects',
    nexusForgeApi.getProjects,
    {
      refetchInterval: 5000, // Refetch every 5 seconds
    }
  );

  // Fetch agent statuses
  const { data: agentStatuses = [], isLoading: agentsLoading } = useQuery(
    'agent-statuses',
    nexusForgeApi.getAgentStatuses,
    {
      refetchInterval: 2000, // Refetch every 2 seconds
    }
  );

  // Fetch metrics
  const { data: metrics } = useQuery(
    'metrics',
    nexusForgeApi.getMetrics,
    {
      refetchInterval: 10000, // Refetch every 10 seconds
    }
  );

  useEffect(() => {
    // Subscribe to real-time agent updates
    subscribeToAgents();
  }, [subscribeToAgents]);

  // Update projects when WebSocket receives updates
  useEffect(() => {
    if (wsState.projectUpdates.size > 0) {
      refetchProjects();
    }
  }, [wsState.projectUpdates, refetchProjects]);

  const activeProjects = projects.filter(p => p.status === 'in_progress');
  const completedProjects = projects.filter(p => p.status === 'completed');
  const failedProjects = projects.filter(p => p.status === 'failed');

  const activeAgents = agentStatuses.filter(a => a.status === 'working').length;
  const totalAgents = agentStatuses.length;

  const averageProgress = activeProjects.length > 0 
    ? activeProjects.reduce((sum, p) => sum + p.progress, 0) / activeProjects.length 
    : 0;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Nexus Forge Workspace</h1>
          <p className="mt-1 text-gray-600">
            Real-time AI-powered app development platform
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${wsState.isConnected ? 'bg-green-400' : 'bg-red-400'} status-indicator`} />
            <span className="text-sm font-medium text-gray-700">
              {wsState.isConnected ? 'Live' : 'Offline'}
            </span>
          </div>
          
          {wsState.latency > 0 && (
            <div className="text-xs text-gray-500">
              Latency: {wsState.latency}ms
            </div>
          )}
        </div>
      </div>

      {/* Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Active Projects"
          value={activeProjects.length}
          icon={Activity}
          color="text-blue-600"
          bgColor="bg-blue-50"
          trend={activeProjects.length > 0 ? 'up' : 'neutral'}
        />
        
        <MetricCard
          title="Completed Projects"
          value={completedProjects.length}
          icon={CheckCircle}
          color="text-green-600"
          bgColor="bg-green-50"
          trend={completedProjects.length > 0 ? 'up' : 'neutral'}
        />
        
        <MetricCard
          title="Active Agents"
          value={`${activeAgents}/${totalAgents}`}
          icon={Users}
          color="text-purple-600"
          bgColor="bg-purple-50"
          trend={activeAgents > 0 ? 'up' : 'neutral'}
        />
        
        <MetricCard
          title="Avg Progress"
          value={`${Math.round(averageProgress)}%`}
          icon={TrendingUp}
          color="text-indigo-600"
          bgColor="bg-indigo-50"
          trend={averageProgress > 50 ? 'up' : averageProgress > 0 ? 'neutral' : 'down'}
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Agent Orchestration Panel */}
        <div className="lg:col-span-1">
          <AgentOrchestrationPanel 
            agentStatuses={agentStatuses}
            isLoading={agentsLoading}
          />
        </div>

        {/* Task Progress Tracker */}
        <div className="lg:col-span-1">
          <TaskProgressTracker 
            projects={activeProjects}
            selectedProject={selectedProject}
            onProjectSelect={setSelectedProject}
            isLoading={projectsLoading}
          />
        </div>
      </div>

      {/* Active Projects List */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Active Projects</h2>
          <p className="text-sm text-gray-600">Real-time project generation status</p>
        </div>
        
        <div className="p-6">
          {projectsLoading ? (
            <div className="flex justify-center py-8">
              <LoadingSpinner size="lg" text="Loading projects..." />
            </div>
          ) : activeProjects.length === 0 ? (
            <div className="text-center py-8">
              <Activity className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No active projects</h3>
              <p className="mt-1 text-sm text-gray-500">
                Start building your first app with the Project Builder.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {activeProjects.map((project) => (
                <ProjectCard
                  key={project.id}
                  project={project}
                  onClick={() => setSelectedProject(project.id)}
                  isSelected={selectedProject === project.id}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Recent Messages */}
      {wsState.recentMessages.length > 0 && (
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">Recent Activity</h2>
            <p className="text-sm text-gray-600">Live updates from agents and tasks</p>
          </div>
          
          <div className="p-6">
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {wsState.recentMessages.slice(0, 10).map((message, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex items-start space-x-3 text-sm"
                >
                  <div className="flex-shrink-0 w-2 h-2 bg-blue-400 rounded-full mt-2" />
                  <div className="flex-1 min-w-0">
                    <p className="text-gray-900">
                      {getMessageDescription(message)}
                    </p>
                    <p className="text-gray-500 text-xs">
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Helper Components
interface MetricCardProps {
  title: string;
  value: string | number;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  bgColor: string;
  trend: 'up' | 'down' | 'neutral';
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, icon: Icon, color, bgColor, trend }) => {
  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className="bg-white rounded-lg shadow p-6"
    >
      <div className="flex items-center">
        <div className={`flex-shrink-0 p-3 rounded-lg ${bgColor}`}>
          <Icon className={`h-6 w-6 ${color}`} />
        </div>
        <div className="ml-4 flex-1">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <div className="flex items-center mt-1">
            <p className="text-2xl font-semibold text-gray-900">{value}</p>
            {trend !== 'neutral' && (
              <div className={`ml-2 flex items-center ${
                trend === 'up' ? 'text-green-600' : 'text-red-600'
              }`}>
                <TrendingUp className={`h-4 w-4 ${trend === 'down' ? 'transform rotate-180' : ''}`} />
              </div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

interface ProjectCardProps {
  project: ProjectResponse;
  onClick: () => void;
  isSelected: boolean;
}

const ProjectCard: React.FC<ProjectCardProps> = ({ project, onClick, isSelected }) => {
  const getStatusIcon = () => {
    switch (project.status) {
      case 'in_progress':
        return <Activity className="h-5 w-5 text-blue-500" />;
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'failed':
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      default:
        return <Clock className="h-5 w-5 text-gray-400" />;
    }
  };

  const getStatusColor = () => {
    switch (project.status) {
      case 'in_progress':
        return 'text-blue-600 bg-blue-50';
      case 'completed':
        return 'text-green-600 bg-green-50';
      case 'failed':
        return 'text-red-600 bg-red-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <motion.div
      whileHover={{ scale: 1.01 }}
      onClick={onClick}
      className={`border rounded-lg p-4 cursor-pointer transition-all duration-200 ${
        isSelected 
          ? 'border-primary-500 bg-primary-50' 
          : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm'
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="flex-shrink-0">
            {getStatusIcon()}
          </div>
          <div>
            <h3 className="text-sm font-medium text-gray-900">{project.name}</h3>
            <p className="text-xs text-gray-500">
              Created {new Date(project.created_at).toLocaleDateString()}
            </p>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor()}`}>
            {project.status.replace('_', ' ')}
          </span>
          <div className="text-right">
            <p className="text-sm font-medium text-gray-900">{project.progress}%</p>
          </div>
        </div>
      </div>
      
      {project.status === 'in_progress' && (
        <div className="mt-3">
          <ProgressBar progress={project.progress} size="sm" />
        </div>
      )}
    </motion.div>
  );
};

// Helper function to format WebSocket messages
const getMessageDescription = (message: any): string => {
  switch (message.type) {
    case 'project_update':
      return `Project ${message.payload.project_id} updated: ${message.payload.status} (${message.payload.progress}%)`;
    case 'agent_status':
      return `Agent ${message.payload.name} is now ${message.payload.status}`;
    case 'task_progress':
      return `Task "${message.payload.title}" is ${message.payload.status} (${message.payload.progress}%)`;
    case 'completion':
      return `Project generation completed successfully`;
    case 'error':
      return `Error: ${message.payload.message || 'Unknown error occurred'}`;
    default:
      return `Unknown event: ${message.type}`;
  }
};