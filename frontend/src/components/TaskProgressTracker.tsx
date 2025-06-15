import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CheckCircle, 
  Clock, 
  AlertCircle, 
  Play, 
  Pause,
  Timer,
  TrendingUp,
  Activity
} from 'lucide-react';
import { useQuery } from 'react-query';
import { ProjectResponse, TaskProgress, nexusForgeApi } from '../services/nexusForgeApi';
import { useWebSocket } from '../contexts/WebSocketContext';
import { LoadingSpinner, ProgressBar } from './common/LoadingSpinner';

interface TaskProgressTrackerProps {
  projects: ProjectResponse[];
  selectedProject: string | null;
  onProjectSelect: (projectId: string | null) => void;
  isLoading: boolean;
}

export const TaskProgressTracker: React.FC<TaskProgressTrackerProps> = ({
  projects,
  selectedProject,
  onProjectSelect,
  isLoading,
}) => {
  const [expandedTasks, setExpandedTasks] = useState<Set<string>>(new Set());
  const { state: wsState, subscribeToProject, unsubscribeFromProject } = useWebSocket();

  // Fetch tasks for selected project
  const { data: tasks = [], isLoading: tasksLoading } = useQuery(
    ['tasks', selectedProject],
    () => selectedProject ? nexusForgeApi.getTaskProgress(selectedProject) : Promise.resolve([]),
    {
      enabled: !!selectedProject,
      refetchInterval: 2000,
    }
  );

  // Real-time task updates from WebSocket
  const [liveTaskProgress, setLiveTaskProgress] = useState<Map<string, TaskProgress>>(new Map());

  useEffect(() => {
    // Subscribe to project updates when a project is selected
    if (selectedProject) {
      subscribeToProject(selectedProject);
    }

    return () => {
      if (selectedProject) {
        unsubscribeFromProject(selectedProject);
      }
    };
  }, [selectedProject, subscribeToProject, unsubscribeFromProject]);

  useEffect(() => {
    // Update live task progress from WebSocket
    wsState.taskProgress.forEach((taskProgress, taskId) => {
      setLiveTaskProgress(prev => new Map(prev.set(taskId, taskProgress)));
    });
  }, [wsState.taskProgress]);

  // Merge API data with live WebSocket updates
  const mergedTasks = tasks.map(task => {
    const liveTask = liveTaskProgress.get(task.task_id);
    return liveTask || task;
  });

  const toggleTaskExpansion = (taskId: string) => {
    setExpandedTasks(prev => {
      const newSet = new Set(prev);
      if (newSet.has(taskId)) {
        newSet.delete(taskId);
      } else {
        newSet.add(taskId);
      }
      return newSet;
    });
  };

  const getTaskIcon = (status: string) => {
    switch (status) {
      case 'in_progress':
        return <Activity className="h-4 w-4 text-blue-500 animate-pulse" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'pending':
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getTaskStatusColor = (status: string) => {
    switch (status) {
      case 'in_progress':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'completed':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'failed':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'pending':
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const activeTasks = mergedTasks.filter(task => task.status === 'in_progress');
  const completedTasks = mergedTasks.filter(task => task.status === 'completed');
  const pendingTasks = mergedTasks.filter(task => task.status === 'pending');
  const failedTasks = mergedTasks.filter(task => task.status === 'failed');

  const overallProgress = mergedTasks.length > 0 
    ? (completedTasks.length / mergedTasks.length) * 100 
    : 0;

  return (
    <div className="bg-white rounded-lg shadow">
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900 flex items-center">
              <Timer className="h-5 w-5 mr-2" />
              Task Progress Tracker
            </h2>
            <p className="text-sm text-gray-600">Real-time task execution monitoring</p>
          </div>
          
          {selectedProject && (
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900">Overall Progress</p>
                <p className="text-lg font-bold text-primary-600">{Math.round(overallProgress)}%</p>
              </div>
              <div className="w-16 h-16">
                <CircularProgress percentage={overallProgress} />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="p-6">
        {/* Project Selection */}
        {projects.length > 0 && (
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Project
            </label>
            <select
              value={selectedProject || ''}
              onChange={(e) => onProjectSelect(e.target.value || null)}
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="">Select a project...</option>
              {projects.map(project => (
                <option key={project.id} value={project.id}>
                  {project.name} - {project.status} ({project.progress}%)
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="flex justify-center py-8">
            <LoadingSpinner size="md" text="Loading projects..." />
          </div>
        )}

        {/* No Projects */}
        {!isLoading && projects.length === 0 && (
          <div className="text-center py-8">
            <Activity className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No active projects</h3>
            <p className="mt-1 text-sm text-gray-500">
              Start a new project to see task progress here.
            </p>
          </div>
        )}

        {/* No Project Selected */}
        {!isLoading && projects.length > 0 && !selectedProject && (
          <div className="text-center py-8">
            <Timer className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">Select a project</h3>
            <p className="mt-1 text-sm text-gray-500">
              Choose a project from the dropdown to view its task progress.
            </p>
          </div>
        )}

        {/* Tasks Loading */}
        {selectedProject && tasksLoading && (
          <div className="flex justify-center py-8">
            <LoadingSpinner size="md" text="Loading tasks..." />
          </div>
        )}

        {/* Tasks Content */}
        {selectedProject && !tasksLoading && (
          <div className="space-y-6">
            {/* Task Summary */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <TaskSummaryCard
                title="Active"
                count={activeTasks.length}
                color="text-blue-600"
                bgColor="bg-blue-50"
                icon={Play}
              />
              <TaskSummaryCard
                title="Completed"
                count={completedTasks.length}
                color="text-green-600"
                bgColor="bg-green-50"
                icon={CheckCircle}
              />
              <TaskSummaryCard
                title="Pending"
                count={pendingTasks.length}
                color="text-gray-600"
                bgColor="bg-gray-50"
                icon={Pause}
              />
              <TaskSummaryCard
                title="Failed"
                count={failedTasks.length}
                color="text-red-600"
                bgColor="bg-red-50"
                icon={AlertCircle}
              />
            </div>

            {/* Task Lists */}
            {mergedTasks.length === 0 ? (
              <div className="text-center py-8">
                <Timer className="mx-auto h-12 w-12 text-gray-400" />
                <h3 className="mt-2 text-sm font-medium text-gray-900">No tasks found</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Tasks will appear here when the project starts generating.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Active Tasks */}
                {activeTasks.length > 0 && (
                  <TaskSection
                    title="Active Tasks"
                    tasks={activeTasks}
                    expandedTasks={expandedTasks}
                    onToggleExpansion={toggleTaskExpansion}
                  />
                )}

                {/* Pending Tasks */}
                {pendingTasks.length > 0 && (
                  <TaskSection
                    title="Pending Tasks"
                    tasks={pendingTasks}
                    expandedTasks={expandedTasks}
                    onToggleExpansion={toggleTaskExpansion}
                  />
                )}

                {/* Completed Tasks */}
                {completedTasks.length > 0 && (
                  <TaskSection
                    title="Completed Tasks"
                    tasks={completedTasks}
                    expandedTasks={expandedTasks}
                    onToggleExpansion={toggleTaskExpansion}
                    collapsed={true}
                  />
                )}

                {/* Failed Tasks */}
                {failedTasks.length > 0 && (
                  <TaskSection
                    title="Failed Tasks"
                    tasks={failedTasks}
                    expandedTasks={expandedTasks}
                    onToggleExpansion={toggleTaskExpansion}
                  />
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// Helper Components
interface TaskSummaryCardProps {
  title: string;
  count: number;
  color: string;
  bgColor: string;
  icon: React.ComponentType<{ className?: string }>;
}

const TaskSummaryCard: React.FC<TaskSummaryCardProps> = ({ title, count, color, bgColor, icon: Icon }) => {
  return (
    <div className={`${bgColor} rounded-lg p-4`}>
      <div className="flex items-center">
        <Icon className={`h-5 w-5 ${color} mr-2`} />
        <div>
          <p className="text-lg font-semibold text-gray-900">{count}</p>
          <p className={`text-sm font-medium ${color}`}>{title}</p>
        </div>
      </div>
    </div>
  );
};

interface TaskSectionProps {
  title: string;
  tasks: TaskProgress[];
  expandedTasks: Set<string>;
  onToggleExpansion: (taskId: string) => void;
  collapsed?: boolean;
}

const TaskSection: React.FC<TaskSectionProps> = ({ 
  title, 
  tasks, 
  expandedTasks, 
  onToggleExpansion,
  collapsed = false 
}) => {
  const [sectionExpanded, setSectionExpanded] = useState(!collapsed);

  return (
    <div>
      <button
        onClick={() => setSectionExpanded(!sectionExpanded)}
        className="w-full flex items-center justify-between text-left"
      >
        <h3 className="text-sm font-medium text-gray-700">{title} ({tasks.length})</h3>
        <TrendingUp className={`h-4 w-4 text-gray-400 transition-transform duration-200 ${
          sectionExpanded ? 'transform rotate-90' : ''
        }`} />
      </button>
      
      <AnimatePresence>
        {sectionExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-3 space-y-2"
          >
            {tasks.map(task => (
              <TaskCard
                key={task.task_id}
                task={task}
                isExpanded={expandedTasks.has(task.task_id)}
                onToggleExpansion={() => onToggleExpansion(task.task_id)}
              />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

interface TaskCardProps {
  task: TaskProgress;
  isExpanded: boolean;
  onToggleExpansion: () => void;
}

const TaskCard: React.FC<TaskCardProps> = ({ task, isExpanded, onToggleExpansion }) => {
  const getTaskIcon = (status: string) => {
    switch (status) {
      case 'in_progress':
        return <Activity className="h-4 w-4 text-blue-500 animate-pulse" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'pending':
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getTaskStatusColor = (status: string) => {
    switch (status) {
      case 'in_progress':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'completed':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'failed':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'pending':
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  return (
    <motion.div
      whileHover={{ scale: 1.01 }}
      className="border rounded-lg p-3 bg-white hover:shadow-sm transition-all duration-200"
    >
      <div className="flex items-center justify-between cursor-pointer" onClick={onToggleExpansion}>
        <div className="flex items-center space-x-3 flex-1">
          <div className="flex-shrink-0">
            {getTaskIcon(task.status)}
          </div>
          <div className="min-w-0 flex-1">
            <h4 className="text-sm font-medium text-gray-900 truncate">{task.title}</h4>
            <p className="text-xs text-gray-500">Agent: {task.agent_id}</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getTaskStatusColor(task.status)}`}>
            {task.status.replace('_', ' ')}
          </span>
          {task.status === 'in_progress' && (
            <div className="text-xs text-gray-600">{task.progress}%</div>
          )}
        </div>
      </div>
      
      {task.status === 'in_progress' && (
        <div className="mt-2">
          <ProgressBar progress={task.progress} size="sm" />
        </div>
      )}
      
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-3 pt-3 border-t border-gray-100 text-xs text-gray-600 space-y-1"
          >
            <div>Task ID: {task.task_id}</div>
            {task.started_at && (
              <div>Started: {new Date(task.started_at).toLocaleString()}</div>
            )}
            {task.completed_at && (
              <div>Completed: {new Date(task.completed_at).toLocaleString()}</div>
            )}
            {task.estimated_duration && (
              <div>Estimated Duration: {Math.round(task.estimated_duration / 60)} minutes</div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

interface CircularProgressProps {
  percentage: number;
  size?: number;
}

const CircularProgress: React.FC<CircularProgressProps> = ({ percentage, size = 64 }) => {
  const radius = (size - 8) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDasharray = circumference;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg
        width={size}
        height={size}
        className="transform -rotate-90"
      >
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="transparent"
          stroke="#e5e7eb"
          strokeWidth="4"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="transparent"
          stroke="#3b82f6"
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={strokeDasharray}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-300 ease-in-out"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-xs font-semibold text-gray-700">
          {Math.round(percentage)}%
        </span>
      </div>
    </div>
  );
};