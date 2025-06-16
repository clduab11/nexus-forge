import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Download, 
  Eye, 
  Code, 
  FileText, 
  Image, 
  ExternalLink,
  CheckCircle,
  AlertCircle,
  Clock,
  Refresh,
  Share2,
  Copy,
  GitBranch
} from 'lucide-react';
import { useQuery } from 'react-query';
import { nexusForgeApi, ProjectResponse } from '../services/nexusForgeApi';
import { useWebSocket } from '../contexts/WebSocketContext';
import { LoadingSpinner, ProgressBar } from '../components/common/LoadingSpinner';
import { TaskProgressTracker } from '../components/TaskProgressTracker';
import toast from 'react-hot-toast';

const ResultsPage: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<'overview' | 'code' | 'assets' | 'docs'>('overview');
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const { subscribeToProject, unsubscribeFromProject } = useWebSocket();

  // Fetch project data
  const { data: project, isLoading, error, refetch } = useQuery(
    ['project', projectId],
    () => projectId ? nexusForgeApi.getProject(projectId) : Promise.reject('No project ID'),
    {
      enabled: !!projectId,
      refetchInterval: (data) => {
        // Refetch more frequently for in-progress projects
        return data?.status === 'in_progress' ? 2000 : 10000;
      },
    }
  );

  useEffect(() => {
    if (projectId) {
      subscribeToProject(projectId);
    }

    return () => {
      if (projectId) {
        unsubscribeFromProject(projectId);
      }
    };
  }, [projectId, subscribeToProject, unsubscribeFromProject]);

  const handleExport = async (format: 'zip' | 'git') => {
    if (!projectId) return;
    
    try {
      const blob = await nexusForgeApi.exportProject(projectId, format);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${project?.name || 'project'}.${format === 'zip' ? 'zip' : 'tar.gz'}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      toast.success(`Project exported as ${format.toUpperCase()}`);
    } catch (error) {
      toast.error('Failed to export project');
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success('Copied to clipboard');
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" text="Loading project..." />
      </div>
    );
  }

  if (error || !project) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <AlertCircle className="mx-auto h-12 w-12 text-red-500 mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Project Not Found</h2>
          <p className="text-gray-600 mb-4">The project you're looking for doesn't exist or has been deleted.</p>
          <button
            onClick={() => navigate('/dashboard')}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const getStatusIcon = () => {
    switch (project.status) {
      case 'completed':
        return <CheckCircle className="h-6 w-6 text-green-500" />;
      case 'failed':
        return <AlertCircle className="h-6 w-6 text-red-500" />;
      case 'in_progress':
        return <Clock className="h-6 w-6 text-blue-500 animate-pulse" />;
      default:
        return <Clock className="h-6 w-6 text-gray-400" />;
    }
  };

  const getStatusColor = () => {
    switch (project.status) {
      case 'completed':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'failed':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'in_progress':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const tabs = [
    { id: 'overview', name: 'Overview', icon: Eye },
    { id: 'code', name: 'Code Files', icon: Code },
    { id: 'assets', name: 'Assets', icon: Image },
    { id: 'docs', name: 'Documentation', icon: FileText },
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <button
            onClick={() => navigate('/dashboard')}
            className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100"
          >
            ←
          </button>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{project.name}</h1>
            <p className="text-gray-600">Project Results</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={() => refetch()}
            className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100"
          >
            <Refresh className="h-5 w-5" />
          </button>
          
          {project.status === 'completed' && (
            <div className="flex items-center space-x-2">
              <button
                onClick={() => handleExport('zip')}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center space-x-2"
              >
                <Download className="h-4 w-4" />
                <span>Download ZIP</span>
              </button>
              <button
                onClick={() => handleExport('git')}
                className="px-4 py-2 text-sm font-medium text-white bg-primary-600 border border-transparent rounded-lg hover:bg-primary-700 flex items-center space-x-2"
              >
                <GitBranch className="h-4 w-4" />
                <span>Export Git</span>
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Status Card */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex-shrink-0">
              {getStatusIcon()}
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h2 className="text-lg font-semibold text-gray-900">
                  {project.status.replace('_', ' ').toUpperCase()}
                </h2>
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getStatusColor()}`}>
                  {project.status.replace('_', ' ')}
                </span>
              </div>
              <p className="text-sm text-gray-600">
                Created: {new Date(project.created_at).toLocaleString()}
              </p>
              {project.updated_at !== project.created_at && (
                <p className="text-sm text-gray-600">
                  Updated: {new Date(project.updated_at).toLocaleString()}
                </p>
              )}
            </div>
          </div>
          
          <div className="text-right">
            <div className="text-2xl font-bold text-gray-900">{project.progress}%</div>
            <div className="text-sm text-gray-600">Complete</div>
          </div>
        </div>
        
        {project.status === 'in_progress' && (
          <div className="mt-4">
            <ProgressBar progress={project.progress} showPercentage={false} />
          </div>
        )}
      </div>

      {/* In-Progress Content */}
      {project.status === 'in_progress' && (
        <div className="grid grid-cols-1 lg:grid-cols-1 gap-6">
          <TaskProgressTracker
            projects={[project]}
            selectedProject={project.id}
            onProjectSelect={() => {}}
            isLoading={false}
          />
        </div>
      )}

      {/* Results Content (only for completed projects) */}
      {project.status === 'completed' && project.results && (
        <>
          {/* Tabs */}
          <div className="bg-white rounded-lg shadow">
            <div className="border-b border-gray-200">
              <nav className="flex space-x-8 px-6">
                {tabs.map((tab) => {
                  const Icon = tab.icon;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id as any)}
                      className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                        activeTab === tab.id
                          ? 'border-primary-500 text-primary-600'
                          : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                      }`}
                    >
                      <Icon className="h-4 w-4" />
                      <span>{tab.name}</span>
                    </button>
                  );
                })}
              </nav>
            </div>

            <div className="p-6">
              <AnimatePresence mode="wait">
                {activeTab === 'overview' && (
                  <OverviewTab key="overview" project={project} />
                )}
                {activeTab === 'code' && (
                  <CodeTab 
                    key="code" 
                    files={project.results.code_files} 
                    selectedFile={selectedFile}
                    onFileSelect={setSelectedFile}
                  />
                )}
                {activeTab === 'assets' && (
                  <AssetsTab key="assets" assets={project.results.assets} />
                )}
                {activeTab === 'docs' && (
                  <DocsTab key="docs" documentation={project.results.documentation} />
                )}
              </AnimatePresence>
            </div>
          </div>
        </>
      )}

      {/* Failed Project */}
      {project.status === 'failed' && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="text-center">
            <AlertCircle className="mx-auto h-12 w-12 text-red-500 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Generation Failed</h3>
            <p className="text-gray-600 mb-4">
              Unfortunately, we encountered an error while generating your project.
            </p>
            <div className="flex justify-center space-x-4">
              <button
                onClick={() => refetch()}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Retry
              </button>
              <button
                onClick={() => navigate('/builder')}
                className="px-4 py-2 text-sm font-medium text-white bg-primary-600 border border-transparent rounded-lg hover:bg-primary-700"
              >
                Create New Project
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Tab Components
interface OverviewTabProps {
  project: ProjectResponse;
}

const OverviewTab: React.FC<OverviewTabProps> = ({ project }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="space-y-6"
    >
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-medium text-gray-900 mb-1">Code Files</h4>
          <p className="text-2xl font-bold text-primary-600">
            {project.results?.code_files?.length || 0}
          </p>
        </div>
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-medium text-gray-900 mb-1">Assets</h4>
          <p className="text-2xl font-bold text-green-600">
            {project.results?.assets?.length || 0}
          </p>
        </div>
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-medium text-gray-900 mb-1">Documentation</h4>
          <p className="text-2xl font-bold text-blue-600">
            {project.results?.documentation ? '✓' : '✗'}
          </p>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-3">Project Summary</h3>
        <div className="bg-gray-50 rounded-lg p-4">
          <p className="text-gray-700">
            Your {project.name} project has been successfully generated with all requested features and components.
            The AI agents have created a complete, production-ready application following best practices and modern conventions.
          </p>
        </div>
      </div>
    </motion.div>
  );
};

interface CodeTabProps {
  files: Array<{ path: string; content: string; type: string }>;
  selectedFile: string | null;
  onFileSelect: (path: string | null) => void;
}

const CodeTab: React.FC<CodeTabProps> = ({ files, selectedFile, onFileSelect }) => {
  const selectedFileData = files.find(f => f.path === selectedFile);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="grid grid-cols-1 lg:grid-cols-2 gap-6"
    >
      {/* File Tree */}
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-3">Files</h3>
        <div className="border border-gray-200 rounded-lg max-h-96 overflow-y-auto">
          {files.map((file) => (
            <button
              key={file.path}
              onClick={() => onFileSelect(file.path)}
              className={`w-full text-left px-4 py-2 border-b border-gray-100 hover:bg-gray-50 flex items-center space-x-2 ${
                selectedFile === file.path ? 'bg-primary-50 text-primary-700' : ''
              }`}
            >
              <Code className="h-4 w-4 flex-shrink-0" />
              <span className="truncate text-sm">{file.path}</span>
              <span className="text-xs text-gray-500 ml-auto">{file.type}</span>
            </button>
          ))}
        </div>
      </div>

      {/* File Content */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-medium text-gray-900">
            {selectedFileData ? selectedFileData.path : 'Select a file'}
          </h3>
          {selectedFileData && (
            <button
              onClick={() => navigator.clipboard.writeText(selectedFileData.content)}
              className="flex items-center space-x-1 px-3 py-1 text-sm text-gray-600 hover:text-gray-900 border border-gray-300 rounded-md hover:bg-gray-50"
            >
              <Copy className="h-3 w-3" />
              <span>Copy</span>
            </button>
          )}
        </div>
        
        {selectedFileData ? (
          <div className="border border-gray-200 rounded-lg">
            <pre className="p-4 text-sm font-mono bg-gray-50 rounded-lg overflow-x-auto max-h-96 overflow-y-auto">
              <code>{selectedFileData.content}</code>
            </pre>
          </div>
        ) : (
          <div className="border border-gray-200 rounded-lg p-8 text-center text-gray-500">
            Select a file to view its contents
          </div>
        )}
      </div>
    </motion.div>
  );
};

interface AssetsTabProps {
  assets: Array<{ url: string; type: string; name: string }>;
}

const AssetsTab: React.FC<AssetsTabProps> = ({ assets }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
    >
      <h3 className="text-lg font-medium text-gray-900 mb-3">Generated Assets</h3>
      
      {assets.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          No assets generated for this project
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {assets.map((asset, index) => (
            <div key={index} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center space-x-3 mb-3">
                <Image className="h-5 w-5 text-gray-400" />
                <span className="font-medium text-gray-900 truncate">{asset.name}</span>
              </div>
              
              {asset.type.startsWith('image/') ? (
                <img
                  src={asset.url}
                  alt={asset.name}
                  className="w-full h-32 object-cover rounded-md mb-3"
                />
              ) : (
                <div className="w-full h-32 bg-gray-100 rounded-md mb-3 flex items-center justify-center">
                  <FileText className="h-8 w-8 text-gray-400" />
                </div>
              )}
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-500">{asset.type}</span>
                <a
                  href={asset.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary-600 hover:text-primary-700 flex items-center space-x-1 text-sm"
                >
                  <ExternalLink className="h-3 w-3" />
                  <span>View</span>
                </a>
              </div>
            </div>
          ))}
        </div>
      )}
    </motion.div>
  );
};

interface DocsTabProps {
  documentation: string;
}

const DocsTab: React.FC<DocsTabProps> = ({ documentation }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-medium text-gray-900">Documentation</h3>
        <button
          onClick={() => navigator.clipboard.writeText(documentation)}
          className="flex items-center space-x-1 px-3 py-1 text-sm text-gray-600 hover:text-gray-900 border border-gray-300 rounded-md hover:bg-gray-50"
        >
          <Copy className="h-3 w-3" />
          <span>Copy</span>
        </button>
      </div>
      
      <div className="border border-gray-200 rounded-lg p-6">
        <div className="prose max-w-none">
          <pre className="whitespace-pre-wrap text-sm text-gray-700">
            {documentation}
          </pre>
        </div>
      </div>
    </motion.div>
  );
};

export default ResultsPage;