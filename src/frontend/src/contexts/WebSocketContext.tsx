import React, { createContext, useContext, useEffect, useReducer, ReactNode } from 'react';
import { webSocketService, WebSocketMessage, ProjectUpdatePayload, AgentStatusPayload, TaskProgressPayload } from '../services/websocketService';
import { useAuth } from './AuthContext';
import toast from 'react-hot-toast';

interface WebSocketState {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  latency: number;
  projectUpdates: Map<string, ProjectUpdatePayload>;
  agentStatuses: Map<string, AgentStatusPayload>;
  taskProgress: Map<string, TaskProgressPayload>;
  recentMessages: WebSocketMessage[];
}

type WebSocketAction =
  | { type: 'CONNECTING' }
  | { type: 'CONNECTED' }
  | { type: 'DISCONNECTED' }
  | { type: 'ERROR'; payload: string }
  | { type: 'UPDATE_LATENCY'; payload: number }
  | { type: 'PROJECT_UPDATE'; payload: ProjectUpdatePayload }
  | { type: 'AGENT_STATUS'; payload: AgentStatusPayload }
  | { type: 'TASK_PROGRESS'; payload: TaskProgressPayload }
  | { type: 'ADD_MESSAGE'; payload: WebSocketMessage }
  | { type: 'CLEAR_ERROR' };

const initialState: WebSocketState = {
  isConnected: false,
  isConnecting: false,
  error: null,
  latency: 0,
  projectUpdates: new Map(),
  agentStatuses: new Map(),
  taskProgress: new Map(),
  recentMessages: [],
};

const webSocketReducer = (state: WebSocketState, action: WebSocketAction): WebSocketState => {
  switch (action.type) {
    case 'CONNECTING':
      return {
        ...state,
        isConnecting: true,
        error: null,
      };
    case 'CONNECTED':
      return {
        ...state,
        isConnected: true,
        isConnecting: false,
        error: null,
      };
    case 'DISCONNECTED':
      return {
        ...state,
        isConnected: false,
        isConnecting: false,
      };
    case 'ERROR':
      return {
        ...state,
        isConnected: false,
        isConnecting: false,
        error: action.payload,
      };
    case 'UPDATE_LATENCY':
      return {
        ...state,
        latency: action.payload,
      };
    case 'PROJECT_UPDATE':
      const newProjectUpdates = new Map(state.projectUpdates);
      newProjectUpdates.set(action.payload.project_id, action.payload);
      return {
        ...state,
        projectUpdates: newProjectUpdates,
      };
    case 'AGENT_STATUS':
      const newAgentStatuses = new Map(state.agentStatuses);
      newAgentStatuses.set(action.payload.agent_id, action.payload);
      return {
        ...state,
        agentStatuses: newAgentStatuses,
      };
    case 'TASK_PROGRESS':
      const newTaskProgress = new Map(state.taskProgress);
      newTaskProgress.set(action.payload.task_id, action.payload);
      return {
        ...state,
        taskProgress: newTaskProgress,
      };
    case 'ADD_MESSAGE':
      return {
        ...state,
        recentMessages: [action.payload, ...state.recentMessages].slice(0, 100), // Keep last 100 messages
      };
    case 'CLEAR_ERROR':
      return {
        ...state,
        error: null,
      };
    default:
      return state;
  }
};

interface WebSocketContextType {
  state: WebSocketState;
  subscribeToProject: (projectId: string) => void;
  unsubscribeFromProject: (projectId: string) => void;
  subscribeToAgents: () => void;
  unsubscribeFromAgents: () => void;
  sendMessage: (event: string, data: any) => void;
  measureLatency: () => Promise<number>;
  clearError: () => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

interface WebSocketProviderProps {
  children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(webSocketReducer, initialState);
  const { state: authState } = useAuth();

  useEffect(() => {
    if (authState.isAuthenticated && !state.isConnected && !state.isConnecting) {
      connectWebSocket();
    }

    return () => {
      webSocketService.disconnect();
    };
  }, [authState.isAuthenticated]);

  useEffect(() => {
    // Set up WebSocket event listeners
    const handleProjectUpdate = (data: ProjectUpdatePayload) => {
      dispatch({ type: 'PROJECT_UPDATE', payload: data });
      
      // Add to recent messages
      dispatch({
        type: 'ADD_MESSAGE',
        payload: {
          type: 'project_update',
          payload: data,
          timestamp: new Date().toISOString(),
        },
      });
    };

    const handleAgentStatus = (data: AgentStatusPayload) => {
      dispatch({ type: 'AGENT_STATUS', payload: data });
      
      // Add to recent messages
      dispatch({
        type: 'ADD_MESSAGE',
        payload: {
          type: 'agent_status',
          payload: data,
          timestamp: new Date().toISOString(),
        },
      });
    };

    const handleTaskProgress = (data: TaskProgressPayload) => {
      dispatch({ type: 'TASK_PROGRESS', payload: data });
      
      // Add to recent messages
      dispatch({
        type: 'ADD_MESSAGE',
        payload: {
          type: 'task_progress',
          payload: data,
          timestamp: new Date().toISOString(),
        },
      });
    };

    const handleGenerationComplete = (data: any) => {
      dispatch({
        type: 'ADD_MESSAGE',
        payload: {
          type: 'completion',
          payload: data,
          timestamp: new Date().toISOString(),
        },
      });
    };

    const handleGenerationFailed = (data: any) => {
      dispatch({
        type: 'ADD_MESSAGE',
        payload: {
          type: 'error',
          payload: data,
          timestamp: new Date().toISOString(),
        },
      });
    };

    // Subscribe to WebSocket events
    webSocketService.on('project_update', handleProjectUpdate);
    webSocketService.on('agent_status', handleAgentStatus);
    webSocketService.on('task_progress', handleTaskProgress);
    webSocketService.on('generation_complete', handleGenerationComplete);
    webSocketService.on('generation_failed', handleGenerationFailed);

    return () => {
      // Cleanup event listeners
      webSocketService.off('project_update', handleProjectUpdate);
      webSocketService.off('agent_status', handleAgentStatus);
      webSocketService.off('task_progress', handleTaskProgress);
      webSocketService.off('generation_complete', handleGenerationComplete);
      webSocketService.off('generation_failed', handleGenerationFailed);
    };
  }, []);

  const connectWebSocket = async () => {
    if (state.isConnecting || state.isConnected) return;

    dispatch({ type: 'CONNECTING' });

    try {
      const token = localStorage.getItem('access_token');
      await webSocketService.connect(token || undefined);
      dispatch({ type: 'CONNECTED' });
      
      // Measure initial latency
      const latency = await webSocketService.ping();
      if (latency > 0) {
        dispatch({ type: 'UPDATE_LATENCY', payload: latency });
      }
    } catch (error: any) {
      const errorMessage = error.message || 'Failed to connect to WebSocket';
      dispatch({ type: 'ERROR', payload: errorMessage });
      console.error('WebSocket connection error:', error);
    }
  };

  const subscribeToProject = (projectId: string) => {
    if (state.isConnected) {
      webSocketService.subscribeToProject(projectId);
    }
  };

  const unsubscribeFromProject = (projectId: string) => {
    if (state.isConnected) {
      webSocketService.unsubscribeFromProject(projectId);
    }
  };

  const subscribeToAgents = () => {
    if (state.isConnected) {
      webSocketService.subscribeToAgents();
    }
  };

  const unsubscribeFromAgents = () => {
    if (state.isConnected) {
      webSocketService.unsubscribeFromAgents();
    }
  };

  const sendMessage = (event: string, data: any) => {
    if (state.isConnected) {
      webSocketService.send(event, data);
    } else {
      toast.error('WebSocket not connected. Please refresh the page.');
    }
  };

  const measureLatency = async (): Promise<number> => {
    if (!state.isConnected) return -1;

    const latency = await webSocketService.ping();
    if (latency > 0) {
      dispatch({ type: 'UPDATE_LATENCY', payload: latency });
    }
    return latency;
  };

  const clearError = () => {
    dispatch({ type: 'CLEAR_ERROR' });
  };

  const value: WebSocketContextType = {
    state,
    subscribeToProject,
    unsubscribeFromProject,
    subscribeToAgents,
    unsubscribeFromAgents,
    sendMessage,
    measureLatency,
    clearError,
  };

  return <WebSocketContext.Provider value={value}>{children}</WebSocketContext.Provider>;
};

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

export default WebSocketContext;