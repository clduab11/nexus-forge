/**
 * Workflow System Type Definitions
 * Comprehensive type system for visual workflow composition
 */

// Core workflow types
export interface WorkflowDefinition {
  id: string;
  version: string;
  metadata: WorkflowMetadata;
  nodes: WorkflowNode[];
  connections: WorkflowConnection[];
  variables: WorkflowVariable[];
  triggers: WorkflowTrigger[];
  settings: WorkflowSettings;
}

export interface WorkflowMetadata {
  name: string;
  description: string;
  author: string;
  created: Date;
  modified: Date;
  tags: string[];
  category: WorkflowCategory;
  estimatedDuration: number; // in minutes
  requiredAgents: AgentType[];
}

export enum WorkflowCategory {
  WEB_APPLICATION = 'web_application',
  MOBILE_APPLICATION = 'mobile_application',
  API_SERVICE = 'api_service',
  DATA_PIPELINE = 'data_pipeline',
  AI_MODEL = 'ai_model',
  AUTOMATION = 'automation',
  ANALYSIS = 'analysis',
  TESTING = 'testing'
}

// Node system types
export interface WorkflowNode {
  id: string;
  type: NodeType;
  position: Position;
  data: NodeData;
  style?: NodeStyle;
  state: NodeState;
}

export interface Position {
  x: number;
  y: number;
}

export enum NodeType {
  AGENT = 'agent',
  TRIGGER = 'trigger',
  CONDITION = 'condition',
  TRANSFORM = 'transform',
  OUTPUT = 'output',
  SUBFLOW = 'subflow',
  PARALLEL = 'parallel',
  LOOP = 'loop',
  DELAY = 'delay'
}

export interface NodeData {
  label: string;
  description?: string;
  agentType?: AgentType;
  config: Record<string, any>;
  inputs: InputPort[];
  outputs: OutputPort[];
  validation?: NodeValidation;
}

export enum AgentType {
  STARRI = 'starri',
  JULES = 'jules',
  GEMINI = 'gemini',
  RESEARCHER = 'researcher',
  DEVELOPER = 'developer',
  DESIGNER = 'designer',
  TESTER = 'tester',
  ANALYST = 'analyst',
  OPTIMIZER = 'optimizer',
  CUSTOM = 'custom'
}

export interface Port {
  id: string;
  name: string;
  type: DataType;
  description?: string;
}

export interface InputPort extends Port {
  required: boolean;
  defaultValue?: any;
  validation?: PortValidation;
}

export interface OutputPort extends Port {
  multiple: boolean;
}

export enum DataType {
  STRING = 'string',
  NUMBER = 'number',
  BOOLEAN = 'boolean',
  OBJECT = 'object',
  ARRAY = 'array',
  FILE = 'file',
  CODE = 'code',
  ANY = 'any'
}

export interface NodeStyle {
  backgroundColor?: string;
  borderColor?: string;
  textColor?: string;
  icon?: string;
  shape?: 'rectangle' | 'rounded' | 'circle' | 'diamond';
}

export interface NodeState {
  enabled: boolean;
  locked: boolean;
  collapsed: boolean;
  error?: string;
  warning?: string;
}

// Connection system types
export interface WorkflowConnection {
  id: string;
  source: ConnectionEndpoint;
  target: ConnectionEndpoint;
  type: ConnectionType;
  dataTransform?: DataTransform;
  condition?: ConnectionCondition;
  style?: ConnectionStyle;
}

export interface ConnectionEndpoint {
  nodeId: string;
  portId: string;
}

export enum ConnectionType {
  DATA_FLOW = 'data',
  CONTROL_FLOW = 'control',
  CONDITIONAL = 'conditional',
  PARALLEL = 'parallel',
  SEQUENTIAL = 'sequential',
  ERROR = 'error'
}

export interface DataTransform {
  type: TransformType;
  expression: string;
  validation?: TransformValidation;
}

export enum TransformType {
  MAP = 'map',
  FILTER = 'filter',
  REDUCE = 'reduce',
  MERGE = 'merge',
  SPLIT = 'split',
  CUSTOM = 'custom'
}

export interface ConnectionCondition {
  type: 'expression' | 'value' | 'regex';
  expression: string;
  fallback?: string; // node ID for else branch
}

export interface ConnectionStyle {
  strokeColor?: string;
  strokeWidth?: number;
  strokeDasharray?: string;
  animated?: boolean;
  curvature?: number;
}

// Variables and triggers
export interface WorkflowVariable {
  name: string;
  type: DataType;
  defaultValue: any;
  scope: VariableScope;
  description?: string;
  validation?: VariableValidation;
}

export enum VariableScope {
  GLOBAL = 'global',
  LOCAL = 'local',
  NODE = 'node'
}

export interface WorkflowTrigger {
  id: string;
  type: TriggerType;
  name: string;
  config: TriggerConfig;
  enabled: boolean;
}

export enum TriggerType {
  MANUAL = 'manual',
  SCHEDULE = 'schedule',
  EVENT = 'event',
  WEBHOOK = 'webhook',
  FILE_WATCH = 'file_watch',
  API_CALL = 'api_call'
}

export interface TriggerConfig {
  [key: string]: any;
}

// Workflow settings
export interface WorkflowSettings {
  executionMode: ExecutionMode;
  errorHandling: ErrorHandlingStrategy;
  timeout: number; // in seconds
  maxRetries: number;
  logging: LoggingConfig;
  notifications: NotificationConfig;
  security: SecurityConfig;
}

export enum ExecutionMode {
  SEQUENTIAL = 'sequential',
  PARALLEL = 'parallel',
  HYBRID = 'hybrid'
}

export interface ErrorHandlingStrategy {
  onNodeError: 'stop' | 'continue' | 'retry' | 'fallback';
  onConnectionError: 'stop' | 'skip' | 'retry';
  notifyOnError: boolean;
  errorTimeout: number;
}

export interface LoggingConfig {
  level: 'debug' | 'info' | 'warning' | 'error';
  destinations: LogDestination[];
  includeData: boolean;
}

export interface LogDestination {
  type: 'console' | 'file' | 'database' | 'external';
  config: Record<string, any>;
}

export interface NotificationConfig {
  onStart: boolean;
  onComplete: boolean;
  onError: boolean;
  channels: NotificationChannel[];
}

export interface NotificationChannel {
  type: 'email' | 'slack' | 'webhook' | 'in_app';
  config: Record<string, any>;
}

export interface SecurityConfig {
  authentication: boolean;
  authorization: string[]; // role IDs
  encryption: boolean;
  auditLog: boolean;
}

// Validation types
export interface NodeValidation {
  rules: ValidationRule[];
  customValidator?: string; // function name or expression
}

export interface PortValidation {
  required: boolean;
  pattern?: string;
  min?: number;
  max?: number;
  customValidator?: string;
}

export interface VariableValidation {
  pattern?: string;
  min?: number;
  max?: number;
  enum?: any[];
  customValidator?: string;
}

export interface TransformValidation {
  inputSchema?: object;
  outputSchema?: object;
  testCases?: TestCase[];
}

export interface ValidationRule {
  type: 'required' | 'pattern' | 'range' | 'custom';
  config: Record<string, any>;
  message: string;
}

export interface TestCase {
  input: any;
  expectedOutput: any;
  description: string;
}

// Execution types
export interface WorkflowExecution {
  id: string;
  workflowId: string;
  status: ExecutionStatus;
  startTime: Date;
  endTime?: Date;
  nodeStates: Map<string, NodeExecutionState>;
  variables: Map<string, any>;
  logs: ExecutionLog[];
  metrics: ExecutionMetrics;
  error?: ExecutionError;
}

export enum ExecutionStatus {
  IDLE = 'idle',
  INITIALIZING = 'initializing',
  RUNNING = 'running',
  PAUSED = 'paused',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

export interface NodeExecutionState {
  nodeId: string;
  status: NodeExecutionStatus;
  progress: number;
  startTime?: Date;
  endTime?: Date;
  input?: any;
  output?: any;
  error?: NodeExecutionError;
  retries: number;
}

export enum NodeExecutionStatus {
  WAITING = 'waiting',
  READY = 'ready',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  SKIPPED = 'skipped',
  RETRYING = 'retrying'
}

export interface ExecutionLog {
  timestamp: Date;
  level: 'debug' | 'info' | 'warning' | 'error';
  nodeId?: string;
  message: string;
  data?: any;
}

export interface ExecutionMetrics {
  totalDuration: number;
  nodeDurations: Map<string, number>;
  resourceUsage: ResourceMetrics;
  dataProcessed: DataMetrics;
}

export interface ResourceMetrics {
  cpuUsage: number;
  memoryUsage: number;
  networkUsage: number;
  apiCalls: number;
}

export interface DataMetrics {
  inputSize: number;
  outputSize: number;
  recordsProcessed: number;
  transformations: number;
}

export interface ExecutionError {
  code: string;
  message: string;
  nodeId?: string;
  stack?: string;
  context?: Record<string, any>;
}

export interface NodeExecutionError extends ExecutionError {
  input?: any;
  partialOutput?: any;
}

// Template types
export interface WorkflowTemplate {
  id: string;
  category: WorkflowCategory;
  name: string;
  description: string;
  thumbnail: string;
  workflow: WorkflowDefinition;
  requirements: TemplateRequirements;
  examples: TemplateExample[];
  popularity: number;
}

export interface TemplateRequirements {
  agents: AgentType[];
  minNodes: number;
  estimatedTime: string;
  complexity: 'beginner' | 'intermediate' | 'advanced';
  prerequisites: string[];
}

export interface TemplateExample {
  name: string;
  description: string;
  input: any;
  expectedOutput: any;
}

// Canvas types
export interface CanvasState {
  nodes: Map<string, WorkflowNode>;
  connections: Map<string, WorkflowConnection>;
  viewport: Viewport;
  selection: Selection;
  history: HistoryState;
}

export interface Viewport {
  x: number;
  y: number;
  zoom: number;
  minZoom: number;
  maxZoom: number;
}

export interface Selection {
  nodes: Set<string>;
  connections: Set<string>;
  box?: SelectionBox;
}

export interface SelectionBox {
  start: Position;
  end: Position;
}

export interface HistoryState {
  past: CanvasState[];
  future: CanvasState[];
  maxSize: number;
}

// Drag and drop types
export interface DragItem {
  type: 'node' | 'connection' | 'port';
  data: any;
  offset?: Position;
}

export interface DropResult {
  position: Position;
  targetId?: string;
  targetType?: string;
}

// Export all types
export * from './workflow.types';