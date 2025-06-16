import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Plus, 
  Smartphone, 
  Monitor, 
  Laptop, 
  Code, 
  Palette, 
  Database,
  Cloud,
  Zap,
  CheckCircle,
  ArrowRight,
  Sparkles
} from 'lucide-react';
import { useMutation, useQueryClient } from 'react-query';
import { useNavigate } from 'react-router-dom';
import { nexusForgeApi, ProjectRequest } from '../services/nexusForgeApi';
import { LoadingSpinner } from './common/LoadingSpinner';
import toast from 'react-hot-toast';

const PLATFORMS = [
  {
    id: 'web',
    name: 'Web Application',
    description: 'Modern responsive web apps',
    icon: Monitor,
    frameworks: ['React', 'Vue.js', 'Angular', 'Svelte', 'Next.js'],
    popular: true,
  },
  {
    id: 'mobile',
    name: 'Mobile App',
    description: 'Native and cross-platform mobile apps',
    icon: Smartphone,
    frameworks: ['React Native', 'Flutter', 'Ionic', 'Xamarin'],
    popular: true,
  },
  {
    id: 'desktop',
    name: 'Desktop Application',
    description: 'Cross-platform desktop applications',
    icon: Laptop,
    frameworks: ['Electron', 'Tauri', 'PyQt', 'WPF'],
    popular: false,
  },
];

const FEATURE_CATEGORIES = [
  {
    id: 'ui',
    name: 'User Interface',
    icon: Palette,
    features: [
      'Material Design',
      'Custom Theme',
      'Dark Mode',
      'Responsive Design',
      'Animations',
      'Accessibility',
    ],
  },
  {
    id: 'backend',
    name: 'Backend',
    icon: Database,
    features: [
      'REST API',
      'GraphQL',
      'Authentication',
      'Database Integration',
      'File Upload',
      'Real-time Updates',
    ],
  },
  {
    id: 'integrations',
    name: 'Integrations',
    icon: Cloud,
    features: [
      'Payment Processing',
      'Email Service',
      'Push Notifications',
      'Analytics',
      'Social Login',
      'Third-party APIs',
    ],
  },
  {
    id: 'advanced',
    name: 'Advanced Features',
    icon: Zap,
    features: [
      'AI/ML Integration',
      'Real-time Chat',
      'Video Calling',
      'Geolocation',
      'Offline Support',
      'Multi-language',
    ],
  },
];

export const ProjectBuilder: React.FC = () => {
  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState<ProjectRequest>({
    name: '',
    description: '',
    platform: 'web',
    framework: '',
    features: [],
    requirements: '',
  });

  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const createProjectMutation = useMutation(nexusForgeApi.createProject, {
    onSuccess: (project) => {
      queryClient.invalidateQueries('projects');
      toast.success('Project created successfully!');
      navigate(`/results/${project.id}`);
    },
    onError: (error: any) => {
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to create project';
      toast.error(errorMessage);
    },
  });

  const handleSubmit = () => {
    if (!formData.name.trim()) {
      toast.error('Please enter a project name');
      return;
    }
    if (!formData.framework) {
      toast.error('Please select a framework');
      return;
    }
    
    createProjectMutation.mutate(formData);
  };

  const nextStep = () => {
    if (step < 4) setStep(step + 1);
  };

  const prevStep = () => {
    if (step > 1) setStep(step - 1);
  };

  const selectedPlatform = PLATFORMS.find(p => p.id === formData.platform);

  return (
    <div className="max-w-4xl mx-auto p-6">
      {/* Header */}
      <div className="text-center mb-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-center mb-4"
        >
          <div className="p-3 bg-gradient-to-br from-primary-500 to-nexus-500 rounded-xl">
            <Sparkles className="h-8 w-8 text-white" />
          </div>
        </motion.div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Create New Project</h1>
        <p className="text-gray-600">AI-powered app development in minutes</p>
      </div>

      {/* Progress Steps */}
      <div className="flex justify-center mb-8">
        <div className="flex items-center space-x-4">
          {[1, 2, 3, 4].map((stepNumber) => (
            <React.Fragment key={stepNumber}>
              <div
                className={`flex items-center justify-center w-8 h-8 rounded-full border-2 transition-all duration-200 ${
                  step >= stepNumber
                    ? 'border-primary-500 bg-primary-500 text-white'
                    : 'border-gray-300 text-gray-400'
                }`}
              >
                {step > stepNumber ? (
                  <CheckCircle className="h-5 w-5" />
                ) : (
                  <span className="text-sm font-medium">{stepNumber}</span>
                )}
              </div>
              {stepNumber < 4 && (
                <div
                  className={`w-12 h-0.5 transition-all duration-200 ${
                    step > stepNumber ? 'bg-primary-500' : 'bg-gray-300'
                  }`}
                />
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Step Content */}
      <div className="bg-white rounded-lg shadow-lg p-8">
        <AnimatePresence mode="wait">
          {step === 1 && (
            <StepBasicInfo
              formData={formData}
              setFormData={setFormData}
              onNext={nextStep}
            />
          )}
          {step === 2 && (
            <StepPlatformSelection
              formData={formData}
              setFormData={setFormData}
              onNext={nextStep}
              onPrev={prevStep}
            />
          )}
          {step === 3 && (
            <StepFeatureSelection
              formData={formData}
              setFormData={setFormData}
              onNext={nextStep}
              onPrev={prevStep}
            />
          )}
          {step === 4 && (
            <StepReview
              formData={formData}
              setFormData={setFormData}
              onSubmit={handleSubmit}
              onPrev={prevStep}
              isLoading={createProjectMutation.isLoading}
            />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

// Step Components
interface StepProps {
  formData: ProjectRequest;
  setFormData: React.Dispatch<React.SetStateAction<ProjectRequest>>;
  onNext?: () => void;
  onPrev?: () => void;
  onSubmit?: () => void;
  isLoading?: boolean;
}

const StepBasicInfo: React.FC<StepProps> = ({ formData, setFormData, onNext }) => {
  return (
    <motion.div
      key="step1"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="space-y-6"
    >
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-2">Project Information</h2>
        <p className="text-gray-600">Tell us about your project idea</p>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Project Name *
          </label>
          <input
            type="text"
            value={formData.name}
            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            placeholder="My Awesome App"
            className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Description
          </label>
          <textarea
            value={formData.description}
            onChange={(e) => setFormData({ ...formData, description: e.target.value })}
            placeholder="Describe what your app should do..."
            rows={4}
            className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          />
        </div>
      </div>

      <div className="flex justify-end">
        <button
          onClick={onNext}
          disabled={!formData.name.trim()}
          className="px-6 py-3 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
        >
          <span>Next</span>
          <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </motion.div>
  );
};

const StepPlatformSelection: React.FC<StepProps> = ({ formData, setFormData, onNext, onPrev }) => {
  const selectedPlatform = PLATFORMS.find(p => p.id === formData.platform);

  return (
    <motion.div
      key="step2"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="space-y-6"
    >
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-2">Platform & Framework</h2>
        <p className="text-gray-600">Choose your target platform and framework</p>
      </div>

      <div className="space-y-6">
        {/* Platform Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-3">Platform</label>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {PLATFORMS.map((platform) => {
              const Icon = platform.icon;
              return (
                <motion.div
                  key={platform.id}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setFormData({ ...formData, platform: platform.id, framework: '' })}
                  className={`relative border-2 rounded-lg p-4 cursor-pointer transition-all duration-200 ${
                    formData.platform === platform.id
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-300 hover:border-gray-400'
                  }`}
                >
                  {platform.popular && (
                    <div className="absolute -top-2 -right-2 bg-primary-500 text-white text-xs px-2 py-1 rounded-full">
                      Popular
                    </div>
                  )}
                  <Icon className={`h-8 w-8 mb-3 ${
                    formData.platform === platform.id ? 'text-primary-600' : 'text-gray-600'
                  }`} />
                  <h3 className="font-medium text-gray-900 mb-1">{platform.name}</h3>
                  <p className="text-sm text-gray-600">{platform.description}</p>
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* Framework Selection */}
        {selectedPlatform && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-3">Framework</label>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {selectedPlatform.frameworks.map((framework) => (
                <motion.button
                  key={framework}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setFormData({ ...formData, framework })}
                  className={`p-3 border-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    formData.framework === framework
                      ? 'border-primary-500 bg-primary-50 text-primary-700'
                      : 'border-gray-300 text-gray-700 hover:border-gray-400'
                  }`}
                >
                  {framework}
                </motion.button>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="flex justify-between">
        <button
          onClick={onPrev}
          className="px-6 py-3 text-gray-700 border border-gray-300 rounded-lg font-medium hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
        >
          Previous
        </button>
        <button
          onClick={onNext}
          disabled={!formData.framework}
          className="px-6 py-3 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
        >
          <span>Next</span>
          <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </motion.div>
  );
};

const StepFeatureSelection: React.FC<StepProps> = ({ formData, setFormData, onNext, onPrev }) => {
  const toggleFeature = (feature: string) => {
    const updatedFeatures = formData.features.includes(feature)
      ? formData.features.filter(f => f !== feature)
      : [...formData.features, feature];
    setFormData({ ...formData, features: updatedFeatures });
  };

  return (
    <motion.div
      key="step3"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="space-y-6"
    >
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-2">Features & Requirements</h2>
        <p className="text-gray-600">Select features for your application</p>
      </div>

      <div className="space-y-6">
        {FEATURE_CATEGORIES.map((category) => {
          const Icon = category.icon;
          return (
            <div key={category.id} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center mb-3">
                <Icon className="h-5 w-5 text-gray-600 mr-2" />
                <h3 className="font-medium text-gray-900">{category.name}</h3>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {category.features.map((feature) => (
                  <motion.button
                    key={feature}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => toggleFeature(feature)}
                    className={`p-2 border rounded-md text-sm transition-all duration-200 ${
                      formData.features.includes(feature)
                        ? 'border-primary-500 bg-primary-50 text-primary-700'
                        : 'border-gray-300 text-gray-700 hover:border-gray-400'
                    }`}
                  >
                    {feature}
                  </motion.button>
                ))}
              </div>
            </div>
          );
        })}

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Additional Requirements
          </label>
          <textarea
            value={formData.requirements}
            onChange={(e) => setFormData({ ...formData, requirements: e.target.value })}
            placeholder="Any specific requirements, integrations, or custom features..."
            rows={4}
            className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          />
        </div>
      </div>

      <div className="flex justify-between">
        <button
          onClick={onPrev}
          className="px-6 py-3 text-gray-700 border border-gray-300 rounded-lg font-medium hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
        >
          Previous
        </button>
        <button
          onClick={onNext}
          className="px-6 py-3 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 flex items-center space-x-2"
        >
          <span>Next</span>
          <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </motion.div>
  );
};

const StepReview: React.FC<StepProps> = ({ formData, setFormData, onSubmit, onPrev, isLoading }) => {
  const selectedPlatform = PLATFORMS.find(p => p.id === formData.platform);

  return (
    <motion.div
      key="step4"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="space-y-6"
    >
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-2">Review & Create</h2>
        <p className="text-gray-600">Review your project configuration</p>
      </div>

      <div className="space-y-4">
        <div className="border border-gray-200 rounded-lg p-4">
          <h3 className="font-medium text-gray-900 mb-2">Project Details</h3>
          <div className="space-y-2 text-sm">
            <div><span className="font-medium">Name:</span> {formData.name}</div>
            {formData.description && (
              <div><span className="font-medium">Description:</span> {formData.description}</div>
            )}
            <div><span className="font-medium">Platform:</span> {selectedPlatform?.name}</div>
            <div><span className="font-medium">Framework:</span> {formData.framework}</div>
          </div>
        </div>

        {formData.features.length > 0 && (
          <div className="border border-gray-200 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-2">Selected Features</h3>
            <div className="flex flex-wrap gap-2">
              {formData.features.map((feature) => (
                <span
                  key={feature}
                  className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-800"
                >
                  {feature}
                </span>
              ))}
            </div>
          </div>
        )}

        {formData.requirements && (
          <div className="border border-gray-200 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-2">Additional Requirements</h3>
            <p className="text-sm text-gray-700">{formData.requirements}</p>
          </div>
        )}
      </div>

      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <div className="flex items-center">
          <Zap className="h-5 w-5 text-yellow-600 mr-2" />
          <div>
            <h4 className="text-sm font-medium text-yellow-800">AI Generation Process</h4>
            <p className="text-sm text-yellow-700 mt-1">
              Our AI agents will analyze your requirements and generate a complete application.
              This process typically takes 3-5 minutes.
            </p>
          </div>
        </div>
      </div>

      <div className="flex justify-between">
        <button
          onClick={onPrev}
          disabled={isLoading}
          className="px-6 py-3 text-gray-700 border border-gray-300 rounded-lg font-medium hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Previous
        </button>
        <button
          onClick={onSubmit}
          disabled={isLoading}
          className="px-6 py-3 bg-gradient-to-r from-primary-600 to-nexus-600 text-white rounded-lg font-medium hover:from-primary-700 hover:to-nexus-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
        >
          {isLoading ? (
            <>
              <LoadingSpinner size="sm" color="text-white" />
              <span>Creating Project...</span>
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4" />
              <span>Create Project</span>
            </>
          )}
        </button>
      </div>
    </motion.div>
  );
};