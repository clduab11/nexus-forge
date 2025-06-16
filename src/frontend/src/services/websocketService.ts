import { io, Socket } from 'socket.io-client';
import toast from 'react-hot-toast';

export interface WebSocketMessage {
  type: 'project_update' | 'agent_status' | 'task_progress' | 'error' | 'completion';
  payload: any;
  timestamp: string;
}

export interface ProjectUpdatePayload {
  project_id: string;
  status: string;
  progress: number;
  current_task?: string;
}

export interface AgentStatusPayload {
  agent_id: string;
  name: string;
  status: string;
  current_task?: string;
  progress: number;
}

export interface TaskProgressPayload {
  task_id: string;
  project_id: string;
  title: string;
  status: string;
  progress: number;
  agent_id: string;
}

class WebSocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 1000;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();

  connect(token?: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsUrl = process.env.REACT_APP_WS_URL || 'http://localhost:8000';
      
      this.socket = io(wsUrl, {
        auth: {
          token: token || localStorage.getItem('access_token'),
        },
        transports: ['websocket'],
        forceNew: true,
        reconnection: true,
        reconnectionAttempts: this.maxReconnectAttempts,
        reconnectionDelay: this.reconnectInterval,
      });

      this.socket.on('connect', () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        resolve();
      });

      this.socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        this.handleReconnection();
        reject(error);
      });

      this.socket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason);
        if (reason === 'io server disconnect') {
          // Server initiated disconnect, try to reconnect
          this.handleReconnection();
        }
      });

      this.socket.on('error', (error) => {
        console.error('WebSocket error:', error);
        toast.error('Connection error. Trying to reconnect...');
      });

      // Handle incoming messages
      this.socket.on('message', (message: WebSocketMessage) => {
        this.handleMessage(message);
      });

      // Specific event handlers
      this.socket.on('project_update', (data: ProjectUpdatePayload) => {
        this.emit('project_update', data);
      });

      this.socket.on('agent_status', (data: AgentStatusPayload) => {
        this.emit('agent_status', data);
      });

      this.socket.on('task_progress', (data: TaskProgressPayload) => {
        this.emit('task_progress', data);
      });

      this.socket.on('generation_complete', (data: any) => {
        this.emit('generation_complete', data);
        toast.success('Project generation completed!');
      });

      this.socket.on('generation_failed', (data: any) => {
        this.emit('generation_failed', data);
        toast.error('Project generation failed. Please try again.');
      });
    });
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.listeners.clear();
  }

  private handleReconnection(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      
      setTimeout(() => {
        if (!this.socket?.connected) {
          const token = localStorage.getItem('access_token');
          if (token) {
            this.connect(token).catch(() => {
              // Reconnection failed, will try again if attempts remaining
            });
          }
        }
      }, this.reconnectInterval * this.reconnectAttempts);
    } else {
      toast.error('Unable to maintain connection. Please refresh the page.');
    }
  }

  private handleMessage(message: WebSocketMessage): void {
    console.log('Received WebSocket message:', message);
    
    switch (message.type) {
      case 'project_update':
        this.emit('project_update', message.payload);
        break;
      case 'agent_status':
        this.emit('agent_status', message.payload);
        break;
      case 'task_progress':
        this.emit('task_progress', message.payload);
        break;
      case 'error':
        console.error('WebSocket error message:', message.payload);
        toast.error(message.payload.message || 'An error occurred');
        break;
      case 'completion':
        this.emit('completion', message.payload);
        break;
    }
  }

  // Event subscription methods
  on(event: string, callback: (data: any) => void): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  off(event: string, callback: (data: any) => void): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.delete(callback);
      if (eventListeners.size === 0) {
        this.listeners.delete(event);
      }
    }
  }

  private emit(event: string, data: any): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach(callback => callback(data));
    }
  }

  // Send messages to server
  send(event: string, data: any): void {
    if (this.socket?.connected) {
      this.socket.emit(event, data);
    } else {
      console.warn('WebSocket not connected. Cannot send message:', { event, data });
    }
  }

  // Project-specific methods
  subscribeToProject(projectId: string): void {
    this.send('subscribe_project', { project_id: projectId });
  }

  unsubscribeFromProject(projectId: string): void {
    this.send('unsubscribe_project', { project_id: projectId });
  }

  // Agent monitoring
  subscribeToAgents(): void {
    this.send('subscribe_agents', {});
  }

  unsubscribeFromAgents(): void {
    this.send('unsubscribe_agents', {});
  }

  // Connection status
  isConnected(): boolean {
    return this.socket?.connected || false;
  }

  // Ping for latency testing
  ping(): Promise<number> {
    return new Promise((resolve) => {
      if (!this.socket?.connected) {
        resolve(-1);
        return;
      }

      const startTime = Date.now();
      this.socket.emit('ping', startTime);
      
      this.socket.once('pong', () => {
        const latency = Date.now() - startTime;
        resolve(latency);
      });

      // Timeout after 5 seconds
      setTimeout(() => resolve(-1), 5000);
    });
  }
}

export const webSocketService = new WebSocketService();
export default webSocketService;