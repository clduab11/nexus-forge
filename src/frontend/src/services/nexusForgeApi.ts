import axios, { AxiosInstance, AxiosResponse } from 'axios';
import toast from 'react-hot-toast';

// Types
export interface ProjectRequest {
  name: string;
  description: string;
  platform: 'web' | 'mobile' | 'desktop';
  framework: string;
  features: string[];
  requirements: string;
}

export interface ProjectResponse {
  id: string;
  name: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  progress: number;
  created_at: string;
  updated_at: string;
  results?: {
    code_files: Array<{
      path: string;
      content: string;
      type: string;
    }>;
    assets: Array<{
      url: string;
      type: string;
      name: string;
    }>;
    documentation: string;
  };
}

export interface AgentStatus {
  agent_id: string;
  name: string;
  status: 'idle' | 'working' | 'completed' | 'error';
  current_task?: string;
  progress: number;
  last_update: string;
}

export interface TaskProgress {
  task_id: string;
  project_id: string;
  title: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  progress: number;
  agent_id: string;
  estimated_duration?: number;
  started_at?: string;
  completed_at?: string;
}

class NexusForgeAPI {
  private api: AxiosInstance;
  private baseURL: string;

  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
    
    this.api = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor to add auth token
    this.api.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('access_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor for error handling
    this.api.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('access_token');
          localStorage.removeItem('refresh_token');
          window.location.href = '/login';
        } else if (error.response?.status >= 500) {
          toast.error('Server error. Please try again later.');
        } else if (error.code === 'ECONNABORTED') {
          toast.error('Request timeout. Please try again.');
        }
        return Promise.reject(error);
      }
    );
  }

  // Authentication
  async login(email: string, password: string): Promise<{ access_token: string; refresh_token: string }> {
    const response = await this.api.post('/api/auth/login', { email, password });
    return response.data;
  }

  async register(email: string, password: string, name: string): Promise<{ access_token: string; refresh_token: string }> {
    const response = await this.api.post('/api/auth/register', { email, password, name });
    return response.data;
  }

  async refreshToken(refreshToken: string): Promise<{ access_token: string }> {
    const response = await this.api.post('/api/auth/refresh', { refresh_token: refreshToken });
    return response.data;
  }

  // Projects
  async createProject(projectData: ProjectRequest): Promise<ProjectResponse> {
    const response = await this.api.post('/api/nexus-forge/projects', projectData);
    return response.data;
  }

  async getProjects(): Promise<ProjectResponse[]> {
    const response = await this.api.get('/api/nexus-forge/projects');
    return response.data;
  }

  async getProject(projectId: string): Promise<ProjectResponse> {
    const response = await this.api.get(`/api/nexus-forge/projects/${projectId}`);
    return response.data;
  }

  async deleteProject(projectId: string): Promise<void> {
    await this.api.delete(`/api/nexus-forge/projects/${projectId}`);
  }

  // Agent Status
  async getAgentStatuses(): Promise<AgentStatus[]> {
    const response = await this.api.get('/api/nexus-forge/agents/status');
    return response.data;
  }

  async getAgentStatus(agentId: string): Promise<AgentStatus> {
    const response = await this.api.get(`/api/nexus-forge/agents/${agentId}/status`);
    return response.data;
  }

  // Task Progress
  async getTaskProgress(projectId: string): Promise<TaskProgress[]> {
    const response = await this.api.get(`/api/nexus-forge/projects/${projectId}/tasks`);
    return response.data;
  }

  async getTask(taskId: string): Promise<TaskProgress> {
    const response = await this.api.get(`/api/nexus-forge/tasks/${taskId}`);
    return response.data;
  }

  // Health Check
  async getHealth(): Promise<{ status: string; timestamp: string }> {
    const response = await this.api.get('/api/health');
    return response.data;
  }

  // Export Project Results
  async exportProject(projectId: string, format: 'zip' | 'git'): Promise<Blob> {
    const response = await this.api.get(`/api/nexus-forge/projects/${projectId}/export`, {
      params: { format },
      responseType: 'blob',
    });
    return response.data;
  }

  // Performance Metrics
  async getMetrics(): Promise<{
    active_projects: number;
    completed_projects: number;
    average_generation_time: number;
    agent_utilization: Record<string, number>;
  }> {
    const response = await this.api.get('/api/nexus-forge/metrics');
    return response.data;
  }
}

export const nexusForgeApi = new NexusForgeAPI();
export default nexusForgeApi;