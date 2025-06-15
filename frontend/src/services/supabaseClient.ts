import { createClient, SupabaseClient, RealtimeChannel } from '@supabase/supabase-js';
import toast from 'react-hot-toast';

// Database types
export interface ProjectCoordination {
  id: string;
  project_id: string;
  agent_assignments: Record<string, string[]>;
  task_dependencies: Record<string, string[]>;
  resource_allocation: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface AgentActivity {
  id: string;
  agent_id: string;
  project_id: string;
  task_id: string;
  status: 'active' | 'idle' | 'completed' | 'error';
  metadata: Record<string, any>;
  timestamp: string;
}

export interface UserPreferences {
  id: string;
  user_id: string;
  preferences: {
    theme: 'light' | 'dark';
    notifications: boolean;
    auto_refresh: boolean;
    default_framework: string;
    preferred_agents: string[];
  };
  created_at: string;
  updated_at: string;
}

class SupabaseService {
  private supabase: SupabaseClient;
  private channels: Map<string, RealtimeChannel> = new Map();

  constructor() {
    const supabaseUrl = process.env.REACT_APP_SUPABASE_URL!;
    const supabaseKey = process.env.REACT_APP_SUPABASE_ANON_KEY!;

    if (!supabaseUrl || !supabaseKey) {
      throw new Error('Supabase configuration missing');
    }

    this.supabase = createClient(supabaseUrl, supabaseKey, {
      auth: {
        autoRefreshToken: true,
        persistSession: true,
        detectSessionInUrl: false,
      },
      realtime: {
        params: {
          eventsPerSecond: 10,
        },
      },
    });
  }

  // Authentication methods
  async signIn(email: string, password: string) {
    const { data, error } = await this.supabase.auth.signInWithPassword({
      email,
      password,
    });

    if (error) {
      throw new Error(error.message);
    }

    return data;
  }

  async signUp(email: string, password: string, metadata?: Record<string, any>) {
    const { data, error } = await this.supabase.auth.signUp({
      email,
      password,
      options: {
        data: metadata,
      },
    });

    if (error) {
      throw new Error(error.message);
    }

    return data;
  }

  async signOut() {
    const { error } = await this.supabase.auth.signOut();
    if (error) {
      throw new Error(error.message);
    }
  }

  async getCurrentUser() {
    const { data: { user }, error } = await this.supabase.auth.getUser();
    if (error) {
      throw new Error(error.message);
    }
    return user;
  }

  // Project Coordination
  async getProjectCoordination(projectId: string): Promise<ProjectCoordination | null> {
    const { data, error } = await this.supabase
      .from('project_coordination')
      .select('*')
      .eq('project_id', projectId)
      .single();

    if (error && error.code !== 'PGRST116') {
      console.error('Error fetching project coordination:', error);
      return null;
    }

    return data;
  }

  async updateProjectCoordination(projectId: string, coordination: Partial<ProjectCoordination>): Promise<void> {
    const { error } = await this.supabase
      .from('project_coordination')
      .upsert({
        project_id: projectId,
        ...coordination,
        updated_at: new Date().toISOString(),
      });

    if (error) {
      console.error('Error updating project coordination:', error);
      throw new Error(error.message);
    }
  }

  // Agent Activity Tracking
  async getAgentActivities(projectId?: string): Promise<AgentActivity[]> {
    let query = this.supabase
      .from('agent_activity')
      .select('*')
      .order('timestamp', { ascending: false });

    if (projectId) {
      query = query.eq('project_id', projectId);
    }

    const { data, error } = await query;

    if (error) {
      console.error('Error fetching agent activities:', error);
      return [];
    }

    return data || [];
  }

  async recordAgentActivity(activity: Omit<AgentActivity, 'id' | 'timestamp'>): Promise<void> {
    const { error } = await this.supabase
      .from('agent_activity')
      .insert({
        ...activity,
        timestamp: new Date().toISOString(),
      });

    if (error) {
      console.error('Error recording agent activity:', error);
    }
  }

  // User Preferences
  async getUserPreferences(userId: string): Promise<UserPreferences | null> {
    const { data, error } = await this.supabase
      .from('user_preferences')
      .select('*')
      .eq('user_id', userId)
      .single();

    if (error && error.code !== 'PGRST116') {
      console.error('Error fetching user preferences:', error);
      return null;
    }

    return data;
  }

  async updateUserPreferences(userId: string, preferences: Partial<UserPreferences['preferences']>): Promise<void> {
    const { error } = await this.supabase
      .from('user_preferences')
      .upsert({
        user_id: userId,
        preferences,
        updated_at: new Date().toISOString(),
      });

    if (error) {
      console.error('Error updating user preferences:', error);
      throw new Error(error.message);
    }
  }

  // Real-time subscriptions
  subscribeToProjectCoordination(
    projectId: string,
    callback: (payload: any) => void
  ): RealtimeChannel {
    const channelName = `project-coordination-${projectId}`;
    
    if (this.channels.has(channelName)) {
      this.channels.get(channelName)?.unsubscribe();
    }

    const channel = this.supabase
      .channel(channelName)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'project_coordination',
          filter: `project_id=eq.${projectId}`,
        },
        callback
      )
      .subscribe();

    this.channels.set(channelName, channel);
    return channel;
  }

  subscribeToAgentActivity(
    projectId: string | null,
    callback: (payload: any) => void
  ): RealtimeChannel {
    const channelName = projectId ? `agent-activity-${projectId}` : 'agent-activity-global';
    
    if (this.channels.has(channelName)) {
      this.channels.get(channelName)?.unsubscribe();
    }

    let channelBuilder = this.supabase.channel(channelName);

    if (projectId) {
      channelBuilder = channelBuilder.on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'agent_activity',
          filter: `project_id=eq.${projectId}`,
        },
        callback
      );
    } else {
      channelBuilder = channelBuilder.on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'agent_activity',
        },
        callback
      );
    }

    const channel = channelBuilder.subscribe();
    this.channels.set(channelName, channel);
    return channel;
  }

  unsubscribeFromChannel(channelName: string): void {
    const channel = this.channels.get(channelName);
    if (channel) {
      channel.unsubscribe();
      this.channels.delete(channelName);
    }
  }

  unsubscribeAll(): void {
    this.channels.forEach((channel) => {
      channel.unsubscribe();
    });
    this.channels.clear();
  }

  // File storage
  async uploadFile(bucket: string, path: string, file: File): Promise<string> {
    const { data, error } = await this.supabase.storage
      .from(bucket)
      .upload(path, file, {
        cacheControl: '3600',
        upsert: false,
      });

    if (error) {
      console.error('Error uploading file:', error);
      throw new Error(error.message);
    }

    const { data: urlData } = this.supabase.storage
      .from(bucket)
      .getPublicUrl(data.path);

    return urlData.publicUrl;
  }

  async downloadFile(bucket: string, path: string): Promise<Blob> {
    const { data, error } = await this.supabase.storage
      .from(bucket)
      .download(path);

    if (error) {
      console.error('Error downloading file:', error);
      throw new Error(error.message);
    }

    return data;
  }

  async deleteFile(bucket: string, path: string): Promise<void> {
    const { error } = await this.supabase.storage
      .from(bucket)
      .remove([path]);

    if (error) {
      console.error('Error deleting file:', error);
      throw new Error(error.message);
    }
  }

  // Health check
  async healthCheck(): Promise<boolean> {
    try {
      const { error } = await this.supabase
        .from('health_check')
        .select('*')
        .limit(1);

      return !error;
    } catch {
      return false;
    }
  }
}

export const supabaseService = new SupabaseService();
export default supabaseService;