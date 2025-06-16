"""
AI Features Module for Nexus Forge
Advanced AI capabilities including training, prediction, and quality control
"""

from .custom_training import CustomTrainingEngine, TrainingPipeline
from .models import PredictionModel, QualityMetrics, TrainingJob
from .predictive_coordination import CoordinationPredictor, PredictiveCoordinator
from .quality_control import AutonomousQualityController, QualityAssessment

__all__ = [
    "CustomTrainingEngine",
    "TrainingPipeline",
    "PredictiveCoordinator",
    "CoordinationPredictor",
    "AutonomousQualityController",
    "QualityAssessment",
    "TrainingJob",
    "PredictionModel",
    "QualityMetrics",
]
