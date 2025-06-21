"""
AI Features data models and schemas
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class TrainingJobStatus(str, Enum):
    """Training job status"""

    PENDING = "pending"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJobType(str, Enum):
    """Training job types"""

    FINE_TUNING = "fine_tuning"
    TRANSFER_LEARNING = "transfer_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    FEW_SHOT_LEARNING = "few_shot_learning"
    CONTINUAL_LEARNING = "continual_learning"


class ModelArchitecture(str, Enum):
    """Supported model architectures"""

    TRANSFORMER = "transformer"
    LSTM = "lstm"
    GRU = "gru"
    CNN = "cnn"
    RESNET = "resnet"
    BERT = "bert"
    GPT = "gpt"
    T5 = "t5"
    CUSTOM = "custom"


class PredictionType(str, Enum):
    """Types of predictions"""

    COORDINATION = "coordination"
    PERFORMANCE = "performance"
    RESOURCE_USAGE = "resource_usage"
    ERROR_PREDICTION = "error_prediction"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    USER_BEHAVIOR = "user_behavior"


class QualityDimension(str, Enum):
    """Quality assessment dimensions"""

    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    SECURITY = "security"
    COMPLIANCE = "compliance"


class TrainingDataSource(BaseModel):
    """Training data source configuration"""

    source_type: str = Field(..., description="Type of data source")
    connection_string: str = Field(..., description="Connection details")
    query: Optional[str] = Field(None, description="Query to extract data")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Data filters")
    preprocessing: Dict[str, Any] = Field(
        default_factory=dict, description="Preprocessing config"
    )
    validation_split: float = Field(
        0.2, ge=0, le=0.5, description="Validation split ratio"
    )

    class Config:
        schema_extra = {
            "example": {
                "source_type": "supabase",
                "connection_string": "postgresql://...",
                "query": "SELECT * FROM agent_executions WHERE created_at > ?",
                "filters": {"status": "completed"},
                "preprocessing": {"normalize": True, "remove_outliers": True},
                "validation_split": 0.2,
            }
        }


class HyperParameters(BaseModel):
    """Training hyperparameters"""

    learning_rate: float = Field(0.001, gt=0, description="Learning rate")
    batch_size: int = Field(32, gt=0, description="Batch size")
    epochs: int = Field(100, gt=0, description="Number of epochs")
    dropout_rate: float = Field(0.1, ge=0, le=1, description="Dropout rate")
    weight_decay: float = Field(0.01, ge=0, description="Weight decay")

    # Model-specific parameters
    hidden_size: Optional[int] = Field(None, gt=0, description="Hidden layer size")
    num_layers: Optional[int] = Field(None, gt=0, description="Number of layers")
    attention_heads: Optional[int] = Field(
        None, gt=0, description="Number of attention heads"
    )
    sequence_length: Optional[int] = Field(None, gt=0, description="Sequence length")

    # Advanced parameters
    gradient_clip_norm: Optional[float] = Field(
        None, gt=0, description="Gradient clipping norm"
    )
    early_stopping_patience: int = Field(
        10, gt=0, description="Early stopping patience"
    )
    lr_scheduler: str = Field("cosine", description="Learning rate scheduler")
    optimizer: str = Field("adam", description="Optimizer type")

    # Custom parameters
    custom_params: Dict[str, Any] = Field(
        default_factory=dict, description="Custom parameters"
    )


class TrainingMetrics(BaseModel):
    """Training progress metrics"""

    epoch: int = Field(..., ge=0, description="Current epoch")
    train_loss: float = Field(..., description="Training loss")
    val_loss: Optional[float] = Field(None, description="Validation loss")
    train_accuracy: Optional[float] = Field(
        None, ge=0, le=1, description="Training accuracy"
    )
    val_accuracy: Optional[float] = Field(
        None, ge=0, le=1, description="Validation accuracy"
    )

    # Performance metrics
    learning_rate: float = Field(..., gt=0, description="Current learning rate")
    batch_time_ms: float = Field(..., gt=0, description="Time per batch in ms")
    memory_usage_mb: float = Field(..., ge=0, description="Memory usage in MB")
    gpu_utilization: Optional[float] = Field(
        None, ge=0, le=100, description="GPU utilization %"
    )

    # Custom metrics
    custom_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Custom metrics"
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelEvaluation(BaseModel):
    """Model evaluation results"""

    accuracy: float = Field(..., ge=0, le=1, description="Overall accuracy")
    precision: float = Field(..., ge=0, le=1, description="Precision")
    recall: float = Field(..., ge=0, le=1, description="Recall")
    f1_score: float = Field(..., ge=0, le=1, description="F1 score")

    # Performance metrics
    inference_time_ms: float = Field(..., gt=0, description="Average inference time")
    throughput_per_second: float = Field(..., gt=0, description="Inference throughput")
    memory_footprint_mb: float = Field(..., gt=0, description="Model memory footprint")

    # Confusion matrix and detailed metrics
    confusion_matrix: Optional[List[List[int]]] = Field(
        None, description="Confusion matrix"
    )
    class_metrics: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Per-class metrics"
    )

    # Cross-validation results
    cv_scores: Optional[List[float]] = Field(
        None, description="Cross-validation scores"
    )
    cv_mean: Optional[float] = Field(None, description="Cross-validation mean")
    cv_std: Optional[float] = Field(None, description="Cross-validation std")

    # Model-specific evaluations
    custom_evaluations: Dict[str, Any] = Field(
        default_factory=dict, description="Custom evaluations"
    )

    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class TrainingJob(BaseModel):
    """Training job configuration and status"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1, description="Job name")
    description: Optional[str] = Field(None, description="Job description")

    # Job configuration
    job_type: TrainingJobType = Field(..., description="Type of training job")
    architecture: ModelArchitecture = Field(..., description="Model architecture")
    base_model_id: Optional[str] = Field(
        None, description="Base model for transfer learning"
    )

    # Data configuration
    data_sources: List[TrainingDataSource] = Field(
        ..., description="Training data sources"
    )
    target_variable: str = Field(..., description="Target variable to predict")
    feature_columns: List[str] = Field(..., description="Feature columns")

    # Training configuration
    hyperparameters: HyperParameters = Field(
        ..., description="Training hyperparameters"
    )

    # Infrastructure configuration
    compute_requirements: Dict[str, Any] = Field(
        default_factory=dict, description="Compute requirements"
    )

    # Status and progress
    status: TrainingJobStatus = Field(default=TrainingJobStatus.PENDING)
    progress_percent: float = Field(0.0, ge=0, le=100, description="Training progress")
    current_epoch: int = Field(0, ge=0, description="Current epoch")

    # Metrics and results
    training_metrics: List[TrainingMetrics] = Field(
        default_factory=list, description="Training metrics"
    )
    evaluation_results: Optional[ModelEvaluation] = Field(
        None, description="Final evaluation"
    )

    # Resource tracking
    compute_time_minutes: float = Field(0.0, ge=0, description="Total compute time")
    cost_usd: float = Field(0.0, ge=0, description="Training cost in USD")

    # Metadata
    created_by: str = Field(..., description="User who created the job")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")
    tags: List[str] = Field(default_factory=list, description="Job tags")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None, description="Training start time")
    completed_at: Optional[datetime] = Field(
        None, description="Training completion time"
    )

    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if failed")
    retry_count: int = Field(0, ge=0, description="Number of retries")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PredictionModel(BaseModel):
    """Trained prediction model"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1, description="Model name")
    description: Optional[str] = Field(None, description="Model description")
    version: str = Field("1.0.0", description="Model version")

    # Model metadata
    architecture: ModelArchitecture = Field(..., description="Model architecture")
    prediction_type: PredictionType = Field(..., description="Type of predictions")
    training_job_id: str = Field(..., description="Source training job")

    # Model configuration
    input_schema: Dict[str, Any] = Field(..., description="Input data schema")
    output_schema: Dict[str, Any] = Field(..., description="Output data schema")
    preprocessing_config: Dict[str, Any] = Field(
        default_factory=dict, description="Preprocessing config"
    )
    postprocessing_config: Dict[str, Any] = Field(
        default_factory=dict, description="Postprocessing config"
    )

    # Performance metrics
    evaluation_metrics: ModelEvaluation = Field(
        ..., description="Model evaluation results"
    )
    benchmark_results: Optional[Dict[str, float]] = Field(
        None, description="Benchmark test results"
    )

    # Deployment configuration
    deployment_config: Dict[str, Any] = Field(
        default_factory=dict, description="Model deployment configuration"
    )

    # Model artifacts
    model_path: str = Field(..., description="Path to model files")
    weights_path: str = Field(..., description="Path to model weights")
    config_path: str = Field(..., description="Path to model config")

    # Status and versioning
    status: str = Field("active", description="Model status")
    is_deployed: bool = Field(default=False, description="Whether model is deployed")
    deployment_url: Optional[str] = Field(None, description="Model API endpoint")

    # Usage tracking
    prediction_count: int = Field(0, ge=0, description="Total predictions made")
    avg_inference_time_ms: float = Field(
        0.0, ge=0, description="Average inference time"
    )
    last_prediction: Optional[datetime] = Field(
        None, description="Last prediction timestamp"
    )

    # Metadata
    created_by: str = Field(..., description="User who created the model")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")
    tags: List[str] = Field(default_factory=list, description="Model tags")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deployed_at: Optional[datetime] = Field(None, description="Deployment timestamp")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PredictionRequest(BaseModel):
    """Prediction request"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = Field(..., description="Model to use for prediction")
    input_data: Dict[str, Any] = Field(..., description="Input data for prediction")

    # Request configuration
    batch_size: Optional[int] = Field(
        None, gt=0, description="Batch size for batch predictions"
    )
    confidence_threshold: Optional[float] = Field(
        None, ge=0, le=1, description="Confidence threshold"
    )
    return_probabilities: bool = Field(
        default=False, description="Return prediction probabilities"
    )

    # Metadata
    requested_by: str = Field(..., description="User making the request")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")
    request_source: str = Field("api", description="Source of the request")

    # Timestamps
    requested_at: datetime = Field(default_factory=datetime.utcnow)


class PredictionResponse(BaseModel):
    """Prediction response"""

    request_id: str = Field(..., description="Original request ID")
    model_id: str = Field(..., description="Model used for prediction")

    # Prediction results
    predictions: List[Dict[str, Any]] = Field(..., description="Prediction results")
    probabilities: Optional[List[Dict[str, float]]] = Field(
        None, description="Prediction probabilities"
    )
    confidence_scores: List[float] = Field(..., description="Confidence scores")

    # Performance metrics
    inference_time_ms: float = Field(..., gt=0, description="Inference time")
    total_time_ms: float = Field(..., gt=0, description="Total processing time")

    # Metadata
    model_version: str = Field(..., description="Model version used")
    prediction_timestamp: datetime = Field(default_factory=datetime.utcnow)


class QualityMetrics(BaseModel):
    """Quality assessment metrics"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    assessment_type: str = Field(..., description="Type of quality assessment")
    target_id: str = Field(..., description="ID of assessed entity")
    target_type: str = Field(..., description="Type of assessed entity")

    # Overall quality score
    overall_score: float = Field(..., ge=0, le=100, description="Overall quality score")
    quality_grade: str = Field(..., description="Quality grade (A-F)")

    # Dimensional scores
    dimensional_scores: Dict[QualityDimension, float] = Field(
        default_factory=dict, description="Scores by quality dimension"
    )

    # Detailed metrics
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Performance metrics"
    )
    reliability_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Reliability metrics"
    )
    efficiency_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Efficiency metrics"
    )

    # Issues and recommendations
    identified_issues: List[Dict[str, Any]] = Field(
        default_factory=list, description="Identified issues"
    )
    recommendations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Improvement recommendations"
    )

    # Assessment metadata
    assessment_method: str = Field(..., description="Method used for assessment")
    assessment_duration_ms: float = Field(..., gt=0, description="Assessment duration")
    confidence_level: float = Field(
        ..., ge=0, le=1, description="Assessment confidence"
    )

    # Comparison data
    baseline_score: Optional[float] = Field(
        None, description="Baseline score for comparison"
    )
    improvement_percent: Optional[float] = Field(
        None, description="Improvement from baseline"
    )

    # Metadata
    assessed_by: str = Field(..., description="System/user who performed assessment")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")

    # Timestamps
    assessed_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class CoordinationPrediction(BaseModel):
    """Prediction for agent coordination"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prediction_horizon_minutes: int = Field(..., gt=0, description="Prediction horizon")

    # Resource predictions
    predicted_cpu_usage: float = Field(
        ..., ge=0, le=100, description="Predicted CPU usage %"
    )
    predicted_memory_usage: float = Field(
        ..., ge=0, le=100, description="Predicted memory usage %"
    )
    predicted_request_volume: int = Field(
        ..., ge=0, description="Predicted request volume"
    )

    # Performance predictions
    predicted_response_time_ms: float = Field(
        ..., gt=0, description="Predicted response time"
    )
    predicted_error_rate: float = Field(
        ..., ge=0, le=1, description="Predicted error rate"
    )
    predicted_throughput: float = Field(..., gt=0, description="Predicted throughput")

    # Coordination recommendations
    recommended_agent_count: int = Field(
        ..., gt=0, description="Recommended agent count"
    )
    recommended_resource_allocation: Dict[str, float] = Field(
        default_factory=dict, description="Recommended resource allocation"
    )
    scaling_recommendations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Scaling recommendations"
    )

    # Confidence and uncertainty
    prediction_confidence: float = Field(
        ..., ge=0, le=1, description="Prediction confidence"
    )
    uncertainty_bounds: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict, description="Uncertainty bounds for predictions"
    )

    # Metadata
    model_version: str = Field(..., description="Prediction model version")
    features_used: List[str] = Field(..., description="Features used for prediction")
    prediction_method: str = Field(..., description="Prediction method used")

    # Timestamps
    predicted_at: datetime = Field(default_factory=datetime.utcnow)
    valid_until: datetime = Field(..., description="Prediction validity end time")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
