import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { motion } from 'framer-motion';
import {
  Activity,
  Cpu,
  HardDrive,
  Zap,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Clock,
} from 'lucide-react';

interface PerformanceMetrics {
  timestamp: number;
  throughput: number;
  latency: number;
  successRate: number;
  cpuUsage: number;
  memoryUsage: number;
  cacheHitRate: number;
  activeAgents: number;
  queueLength: number;
  parallelTasks: number;
}

interface AgentMetrics {
  agentId: string;
  completedTasks: number;
  failedTasks: number;
  avgExecutionTime: number;
  utilization: number;
}

interface PerformanceMonitorProps {
  executionId?: string;
  onOptimize?: () => void;
}

export const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({
  executionId,
  onOptimize,
}) => {
  const [metrics, setMetrics] = useState<PerformanceMetrics[]>([]);
  const [agentMetrics, setAgentMetrics] = useState<AgentMetrics[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState<PerformanceMetrics | null>(null);
  const [alerts, setAlerts] = useState<string[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    // Connect to WebSocket for real-time metrics
    const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:3000/ws';
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('Connected to performance monitoring');
      ws.send(JSON.stringify({
        type: 'subscribe_metrics',
        executionId,
      }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'performance_metrics') {
        const newMetric: PerformanceMetrics = {
          timestamp: Date.now(),
          throughput: data.throughput || 0,
          latency: data.latency || 0,
          successRate: data.successRate || 0,
          cpuUsage: data.cpuUsage || 0,
          memoryUsage: data.memoryUsage || 0,
          cacheHitRate: data.cacheHitRate || 0,
          activeAgents: data.activeAgents || 0,
          queueLength: data.queueLength || 0,
          parallelTasks: data.parallelTasks || 0,
        };
        
        setCurrentMetrics(newMetric);
        setMetrics(prev => [...prev.slice(-59), newMetric]); // Keep last 60 points
        
        // Check for alerts
        checkForAlerts(newMetric);
      } else if (data.type === 'agent_metrics') {
        setAgentMetrics(data.agents || []);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    wsRef.current = ws;

    // Fallback to polling if WebSocket fails
    intervalRef.current = setInterval(fetchMetrics, 2000);

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [executionId]);

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`/api/nexus-forge/metrics/performance${executionId ? `?executionId=${executionId}` : ''}`);
      const data = await response.json();
      
      const newMetric: PerformanceMetrics = {
        timestamp: Date.now(),
        throughput: data.throughput || 0,
        latency: data.latency || 0,
        successRate: data.successRate || 0,
        cpuUsage: data.cpuUsage || 0,
        memoryUsage: data.memoryUsage || 0,
        cacheHitRate: data.cacheHitRate || 0,
        activeAgents: data.activeAgents || 0,
        queueLength: data.queueLength || 0,
        parallelTasks: data.parallelTasks || 0,
      };
      
      setCurrentMetrics(newMetric);
      setMetrics(prev => [...prev.slice(-59), newMetric]);
      
      if (data.agents) {
        setAgentMetrics(data.agents);
      }
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  };

  const checkForAlerts = (metric: PerformanceMetrics) => {
    const newAlerts: string[] = [];
    
    if (metric.cpuUsage > 90) {
      newAlerts.push('High CPU usage detected');
    }
    if (metric.memoryUsage > 85) {
      newAlerts.push('High memory usage detected');
    }
    if (metric.successRate < 0.9) {
      newAlerts.push('Low success rate detected');
    }
    if (metric.queueLength > 1000) {
      newAlerts.push('Large task queue detected');
    }
    
    setAlerts(newAlerts);
  };

  const formatMetricsForChart = () => {
    return metrics.map((m, index) => ({
      time: index,
      throughput: m.throughput,
      latency: m.latency,
      successRate: m.successRate * 100,
      cpuUsage: m.cpuUsage,
      memoryUsage: m.memoryUsage,
      cacheHitRate: m.cacheHitRate * 100,
    }));
  };

  const getUtilizationData = () => {
    if (!currentMetrics) return [];
    
    return [
      { name: 'CPU', value: currentMetrics.cpuUsage, color: '#3B82F6' },
      { name: 'Memory', value: currentMetrics.memoryUsage, color: '#10B981' },
      { name: 'Cache Hit', value: currentMetrics.cacheHitRate * 100, color: '#F59E0B' },
      { name: 'Success Rate', value: currentMetrics.successRate * 100, color: '#6366F1' },
    ];
  };

  return (
    <div className="bg-gray-900 rounded-lg p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <Activity className="h-6 w-6 text-blue-500" />
          Performance Monitor
        </h2>
        {onOptimize && (
          <button
            onClick={onOptimize}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
          >
            <Zap className="h-4 w-4" />
            Optimize
          </button>
        )}
      </div>

      {/* Alerts */}
      {alerts.length > 0 && (
        <div className="space-y-2">
          {alerts.map((alert, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center gap-2 p-3 bg-yellow-500/20 border border-yellow-500/50 rounded-lg"
            >
              <AlertCircle className="h-5 w-5 text-yellow-500" />
              <span className="text-yellow-200">{alert}</span>
            </motion.div>
          ))}
        </div>
      )}

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Throughput"
          value={currentMetrics?.throughput.toFixed(2) || '0'}
          unit="tasks/sec"
          icon={TrendingUp}
          color="blue"
          change={calculateChange(metrics, 'throughput')}
        />
        <MetricCard
          title="Avg Latency"
          value={currentMetrics?.latency.toFixed(0) || '0'}
          unit="ms"
          icon={Clock}
          color="green"
          change={-calculateChange(metrics, 'latency')} // Negative because lower is better
        />
        <MetricCard
          title="Success Rate"
          value={((currentMetrics?.successRate || 0) * 100).toFixed(1)}
          unit="%"
          icon={CheckCircle}
          color="purple"
          change={calculateChange(metrics, 'successRate') * 100}
        />
        <MetricCard
          title="Cache Hit Rate"
          value={((currentMetrics?.cacheHitRate || 0) * 100).toFixed(1)}
          unit="%"
          icon={HardDrive}
          color="yellow"
          change={calculateChange(metrics, 'cacheHitRate') * 100}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Throughput & Latency Chart */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4">Throughput & Latency</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={formatMetricsForChart()}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: 'none' }}
                labelStyle={{ color: '#9CA3AF' }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="throughput"
                stroke="#3B82F6"
                strokeWidth={2}
                dot={false}
                name="Throughput (tasks/s)"
              />
              <Line
                type="monotone"
                dataKey="latency"
                stroke="#EF4444"
                strokeWidth={2}
                dot={false}
                name="Latency (ms)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Resource Utilization Chart */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4">Resource Utilization</h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={formatMetricsForChart()}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: 'none' }}
                labelStyle={{ color: '#9CA3AF' }}
              />
              <Legend />
              <Area
                type="monotone"
                dataKey="cpuUsage"
                stackId="1"
                stroke="#3B82F6"
                fill="#3B82F6"
                fillOpacity={0.6}
                name="CPU %"
              />
              <Area
                type="monotone"
                dataKey="memoryUsage"
                stackId="1"
                stroke="#10B981"
                fill="#10B981"
                fillOpacity={0.6}
                name="Memory %"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Success & Cache Hit Rates */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4">Performance Rates</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={formatMetricsForChart()}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" domain={[0, 100]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: 'none' }}
                labelStyle={{ color: '#9CA3AF' }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="successRate"
                stroke="#10B981"
                strokeWidth={2}
                dot={false}
                name="Success Rate %"
              />
              <Line
                type="monotone"
                dataKey="cacheHitRate"
                stroke="#F59E0B"
                strokeWidth={2}
                dot={false}
                name="Cache Hit Rate %"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Agent Performance */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4">Agent Performance</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={agentMetrics.slice(0, 10)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="agentId" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: 'none' }}
                labelStyle={{ color: '#9CA3AF' }}
              />
              <Legend />
              <Bar
                dataKey="completedTasks"
                fill="#10B981"
                name="Completed"
              />
              <Bar
                dataKey="failedTasks"
                fill="#EF4444"
                name="Failed"
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* System Status */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">System Status</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatusItem
            label="Active Agents"
            value={currentMetrics?.activeAgents || 0}
            icon={Cpu}
          />
          <StatusItem
            label="Queue Length"
            value={currentMetrics?.queueLength || 0}
            icon={Activity}
          />
          <StatusItem
            label="Parallel Tasks"
            value={currentMetrics?.parallelTasks || 0}
            icon={Zap}
          />
          <StatusItem
            label="Optimization"
            value={currentMetrics && currentMetrics.cacheHitRate > 0.5 ? 'Active' : 'Inactive'}
            icon={CheckCircle}
            isText
          />
        </div>
      </div>
    </div>
  );
};

interface MetricCardProps {
  title: string;
  value: string;
  unit: string;
  icon: React.ElementType;
  color: 'blue' | 'green' | 'purple' | 'yellow';
  change?: number;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit,
  icon: Icon,
  color,
  change,
}) => {
  const colorClasses = {
    blue: 'text-blue-500 bg-blue-500/20',
    green: 'text-green-500 bg-green-500/20',
    purple: 'text-purple-500 bg-purple-500/20',
    yellow: 'text-yellow-500 bg-yellow-500/20',
  };

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className="bg-gray-800 rounded-lg p-4 border border-gray-700"
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-gray-400 text-sm">{title}</span>
        <div className={`p-2 rounded-lg ${colorClasses[color]}`}>
          <Icon className="h-4 w-4" />
        </div>
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-bold text-white">{value}</span>
        <span className="text-gray-400 text-sm">{unit}</span>
      </div>
      {change !== undefined && (
        <div className={`text-sm mt-1 ${change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
          {change >= 0 ? '↑' : '↓'} {Math.abs(change).toFixed(1)}%
        </div>
      )}
    </motion.div>
  );
};

interface StatusItemProps {
  label: string;
  value: string | number;
  icon: React.ElementType;
  isText?: boolean;
}

const StatusItem: React.FC<StatusItemProps> = ({ label, value, icon: Icon, isText }) => {
  return (
    <div className="flex items-center gap-3">
      <Icon className="h-5 w-5 text-gray-400" />
      <div>
        <div className="text-gray-400 text-sm">{label}</div>
        <div className={`font-semibold ${isText ? 'text-green-400' : 'text-white'}`}>
          {value}
        </div>
      </div>
    </div>
  );
};

function calculateChange(metrics: PerformanceMetrics[], field: keyof PerformanceMetrics): number {
  if (metrics.length < 2) return 0;
  
  const current = metrics[metrics.length - 1][field] as number;
  const previous = metrics[Math.max(0, metrics.length - 10)][field] as number;
  
  if (previous === 0) return 0;
  
  return ((current - previous) / previous) * 100;
}

export default PerformanceMonitor;