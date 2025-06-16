"""
AI Features Module for Nexus Forge
Advanced AI capabilities including training, prediction, and quality control
"""

from .custom_training import CustomTrainingEngine, TrainingPipeline
from .predictive_coordination import PredictiveCoordinator, CoordinationPredictor
from .quality_control import AutonomousQualityController, QualityAssessment
from .models import TrainingJob, PredictionModel, QualityMetrics

__all__ = [
    "CustomTrainingEngine",
    "TrainingPipeline",
    "PredictiveCoordinator", 
    "CoordinationPredictor",
    "AutonomousQualityController",
    "QualityAssessment",
    "TrainingJob",
    "PredictionModel",
    "QualityMetrics"
]