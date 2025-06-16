"""
Predictive Coordination
LSTM/Transformer-based prediction for agent coordination and resource optimization
"""

import asyncio
import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from ..core.cache import RedisCache
from ..core.exceptions import NotFoundError, ResourceError, ValidationError
from ..integrations.supabase.coordination_client import SupabaseCoordinationClient
from .models import (
    CoordinationPrediction,
    ModelArchitecture,
    PredictionModel,
    PredictionRequest,
    PredictionResponse,
    PredictionType,
)

logger = logging.getLogger(__name__)


class CoordinationLSTM(nn.Module):
    """LSTM model for coordination prediction"""

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(CoordinationLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, dropout=dropout, batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Apply self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)

        # Use last output
        last_output = attn_out[:, -1, :]

        # Final prediction layers
        x = self.dropout(last_output)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CoordinationTransformer(nn.Module):
    """Transformer model for coordination prediction"""

    def __init__(
        self, input_size, d_model, nhead, num_layers, output_size, dropout=0.1
    ):
        super(CoordinationTransformer, self).__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        seq_len = x.size(1)

        # Project input to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = x + pos_enc

        # Apply transformer
        x = self.transformer(x)
        x = self.layer_norm(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Final prediction layers
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class TimeSeriesPreprocessor:
    """Preprocessing for time series data"""

    def __init__(self, sequence_length=60, prediction_horizon=5):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scalers = {}
        self.feature_columns = []

    def fit_transform(
        self, data: pd.DataFrame, feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit scalers and transform data"""
        self.feature_columns = feature_columns

        # Fit scalers for each feature
        for col in feature_columns:
            scaler = MinMaxScaler()
            data[col] = scaler.fit_transform(data[[col]])
            self.scalers[col] = scaler

        # Create sequences
        X, y = self._create_sequences(data[feature_columns].values)

        return X, y

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted scalers"""
        # Apply scaling
        scaled_data = data.copy()
        for col in self.feature_columns:
            if col in self.scalers:
                scaled_data[col] = self.scalers[col].transform(data[[col]])

        # Create sequences (without targets)
        sequences = []
        for i in range(len(scaled_data) - self.sequence_length + 1):
            sequences.append(scaled_data.iloc[i : i + self.sequence_length].values)

        return np.array(sequences)

    def inverse_transform(
        self, predictions: np.ndarray, feature_name: str
    ) -> np.ndarray:
        """Inverse transform predictions"""
        if feature_name in self.scalers:
            return (
                self.scalers[feature_name]
                .inverse_transform(predictions.reshape(-1, 1))
                .flatten()
            )
        return predictions

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training"""
        X, y = [], []

        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            sequence = data[i : i + self.sequence_length]
            X.append(sequence)

            # Target (future values)
            target_start = i + self.sequence_length
            target_end = target_start + self.prediction_horizon
            target = data[target_start:target_end]
            y.append(target)

        return np.array(X), np.array(y)


class CoordinationPredictor:
    """Individual predictor for specific coordination aspect"""

    def __init__(self, model_id: str, model_type: str = "lstm"):
        self.model_id = model_id
        self.model_type = model_type
        self.model = None
        self.preprocessor = TimeSeriesPreprocessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False

        # Prediction configuration
        self.sequence_length = 60  # 60 time steps
        self.prediction_horizon = 5  # Predict 5 steps ahead
        self.confidence_threshold = 0.7

    async def load_model(self, model_path: str, config: Dict[str, Any]) -> None:
        """Load trained model"""
        try:
            # Build model architecture
            input_size = config.get("input_size", 10)
            output_size = config.get("output_size", 5)

            if self.model_type == "lstm":
                self.model = CoordinationLSTM(
                    input_size=input_size,
                    hidden_size=config.get("hidden_size", 128),
                    num_layers=config.get("num_layers", 2),
                    output_size=output_size,
                    dropout=config.get("dropout", 0.2),
                )
            else:  # transformer
                self.model = CoordinationTransformer(
                    input_size=input_size,
                    d_model=config.get("d_model", 256),
                    nhead=config.get("nhead", 8),
                    num_layers=config.get("num_layers", 6),
                    output_size=output_size,
                    dropout=config.get("dropout", 0.1),
                )

            # Load weights
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            logger.info(f"Loaded model {self.model_id}")

        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {e}")
            raise ResourceError(f"Model loading failed: {e}")

    async def predict(self, input_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make prediction with confidence"""
        if not self.is_loaded:
            raise ResourceError("Model not loaded")

        try:
            # Prepare input
            x = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                prediction = self.model(x)
                prediction = prediction.cpu().numpy().flatten()

            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(input_data, prediction)

            return prediction, confidence

        except Exception as e:
            logger.error(f"Prediction failed for model {self.model_id}: {e}")
            raise ResourceError(f"Prediction failed: {e}")

    async def predict_batch(
        self, input_batch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make batch predictions"""
        if not self.is_loaded:
            raise ResourceError("Model not loaded")

        try:
            x = torch.FloatTensor(input_batch).to(self.device)

            with torch.no_grad():
                predictions = self.model(x)
                predictions = predictions.cpu().numpy()

            # Calculate confidence for each prediction
            confidences = np.array(
                [
                    self._calculate_confidence(input_batch[i], predictions[i])
                    for i in range(len(input_batch))
                ]
            )

            return predictions, confidences

        except Exception as e:
            logger.error(f"Batch prediction failed for model {self.model_id}: {e}")
            raise ResourceError(f"Batch prediction failed: {e}")

    def _calculate_confidence(
        self, input_data: np.ndarray, prediction: np.ndarray
    ) -> float:
        """Calculate prediction confidence"""
        # Simple confidence based on input variance and prediction magnitude
        input_variance = np.var(input_data)
        prediction_magnitude = np.mean(np.abs(prediction))

        # Normalized confidence score
        confidence = 1.0 / (1.0 + input_variance + prediction_magnitude * 0.1)
        return min(max(confidence, 0.0), 1.0)


class PredictiveCoordinator:
    """Main predictive coordination system"""

    def __init__(self):
        self.cache = RedisCache()
        self.supabase = SupabaseCoordinationClient()

        # Predictors for different aspects
        self.predictors: Dict[str, CoordinationPredictor] = {}

        # Data collection
        self.metrics_buffer = deque(maxlen=1000)
        self.feature_columns = [
            "cpu_usage",
            "memory_usage",
            "request_rate",
            "response_time",
            "error_rate",
            "active_connections",
            "queue_length",
            "throughput",
            "latency_p95",
            "agent_count",
        ]

        # Prediction configuration
        self.prediction_interval_seconds = 30
        self.max_prediction_horizon_minutes = 60
        self.min_confidence_threshold = 0.6

        # Background tasks
        self._running = False
        self._data_collection_task = None
        self._prediction_task = None

    async def start(self):
        """Start predictive coordination service"""
        if self._running:
            return

        self._running = True

        # Load prediction models
        await self._load_prediction_models()

        # Start background tasks
        self._data_collection_task = asyncio.create_task(self._data_collection_loop())
        self._prediction_task = asyncio.create_task(self._prediction_loop())

        logger.info("Predictive coordinator started")

    async def stop(self):
        """Stop predictive coordination service"""
        self._running = False

        if self._data_collection_task:
            self._data_collection_task.cancel()

        if self._prediction_task:
            self._prediction_task.cancel()

        logger.info("Predictive coordinator stopped")

    async def predict_coordination_needs(
        self, horizon_minutes: int = 15, include_uncertainty: bool = True
    ) -> CoordinationPrediction:
        """Predict coordination needs for specified horizon"""

        if horizon_minutes > self.max_prediction_horizon_minutes:
            raise ValidationError(
                f"Prediction horizon cannot exceed {self.max_prediction_horizon_minutes} minutes"
            )

        # Get recent metrics data
        recent_metrics = await self._get_recent_metrics(hours=24)

        if len(recent_metrics) < 100:  # Need sufficient data
            raise ResourceError("Insufficient historical data for prediction")

        # Prepare data for prediction
        input_data = self._prepare_prediction_input(recent_metrics)

        # Make predictions using ensemble of models
        predictions = {}
        confidences = {}

        for predictor_name, predictor in self.predictors.items():
            try:
                pred, conf = await predictor.predict(input_data)
                predictions[predictor_name] = pred
                confidences[predictor_name] = conf
            except Exception as e:
                logger.warning(f"Prediction failed for {predictor_name}: {e}")

        # Aggregate predictions
        aggregated_predictions = self._aggregate_predictions(predictions, confidences)

        # Calculate uncertainty bounds if requested
        uncertainty_bounds = {}
        if include_uncertainty:
            uncertainty_bounds = self._calculate_uncertainty_bounds(
                predictions, confidences
            )

        # Generate coordination recommendations
        recommendations = await self._generate_coordination_recommendations(
            aggregated_predictions, horizon_minutes
        )

        # Create prediction result
        prediction = CoordinationPrediction(
            prediction_horizon_minutes=horizon_minutes,
            predicted_cpu_usage=aggregated_predictions.get("cpu_usage", 50.0),
            predicted_memory_usage=aggregated_predictions.get("memory_usage", 60.0),
            predicted_request_volume=int(
                aggregated_predictions.get("request_volume", 1000)
            ),
            predicted_response_time_ms=aggregated_predictions.get(
                "response_time", 150.0
            ),
            predicted_error_rate=aggregated_predictions.get("error_rate", 0.02),
            predicted_throughput=aggregated_predictions.get("throughput", 100.0),
            recommended_agent_count=recommendations["agent_count"],
            recommended_resource_allocation=recommendations["resource_allocation"],
            scaling_recommendations=recommendations["scaling_actions"],
            prediction_confidence=np.mean(list(confidences.values())),
            uncertainty_bounds=uncertainty_bounds,
            model_version="1.0.0",
            features_used=self.feature_columns,
            prediction_method="ensemble_lstm_transformer",
            valid_until=datetime.utcnow() + timedelta(minutes=horizon_minutes),
        )

        # Cache prediction
        await self._cache_prediction(prediction)

        # Log prediction
        await self._log_prediction(prediction)

        return prediction

    async def predict_resource_scaling(
        self, current_load: Dict[str, float], target_sla: Dict[str, float]
    ) -> Dict[str, Any]:
        """Predict optimal resource scaling"""

        # Analyze current state
        current_utilization = self._analyze_current_utilization(current_load)

        # Predict future load
        future_prediction = await self.predict_coordination_needs(horizon_minutes=30)

        # Calculate scaling requirements
        scaling_recommendations = {
            "cpu_scaling": self._calculate_cpu_scaling(
                current_utilization, future_prediction, target_sla
            ),
            "memory_scaling": self._calculate_memory_scaling(
                current_utilization, future_prediction, target_sla
            ),
            "horizontal_scaling": self._calculate_horizontal_scaling(
                current_utilization, future_prediction, target_sla
            ),
            "estimated_timeline": self._estimate_scaling_timeline(),
            "confidence": future_prediction.prediction_confidence,
            "sla_compliance_probability": self._calculate_sla_compliance_probability(
                future_prediction, target_sla
            ),
        }

        return scaling_recommendations

    async def get_prediction_accuracy(self, hours: int = 24) -> Dict[str, float]:
        """Calculate prediction accuracy over time window"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        # Get historical predictions and actual values
        predictions = await self._get_historical_predictions(start_time, end_time)
        actuals = await self._get_actual_metrics(start_time, end_time)

        if not predictions or not actuals:
            return {"error": "Insufficient data for accuracy calculation"}

        # Calculate accuracy metrics
        accuracy_metrics = {}

        for metric_name in ["cpu_usage", "memory_usage", "response_time", "error_rate"]:
            pred_values = [p.get(metric_name, 0) for p in predictions]
            actual_values = [a.get(metric_name, 0) for a in actuals]

            if pred_values and actual_values:
                # Mean Absolute Percentage Error
                mape = (
                    np.mean(
                        np.abs(
                            (np.array(actual_values) - np.array(pred_values))
                            / np.array(actual_values)
                        )
                    )
                    * 100
                )

                # Correlation coefficient
                correlation = (
                    np.corrcoef(pred_values, actual_values)[0, 1]
                    if len(pred_values) > 1
                    else 0
                )

                accuracy_metrics[metric_name] = {
                    "mape": min(mape, 100.0),  # Cap at 100%
                    "correlation": correlation if not np.isnan(correlation) else 0.0,
                    "sample_count": len(pred_values),
                }

        # Overall accuracy
        overall_mape = np.mean([m["mape"] for m in accuracy_metrics.values()])
        overall_correlation = np.mean(
            [m["correlation"] for m in accuracy_metrics.values()]
        )

        accuracy_metrics["overall"] = {
            "mape": overall_mape,
            "correlation": overall_correlation,
            "accuracy_grade": self._calculate_accuracy_grade(
                overall_mape, overall_correlation
            ),
        }

        return accuracy_metrics

    # Private helper methods

    async def _load_prediction_models(self):
        """Load prediction models for different coordination aspects"""
        # Get available prediction models
        result = (
            await self.supabase.client.table("prediction_models")
            .select("*")
            .eq("prediction_type", "coordination")
            .eq("status", "active")
            .execute()
        )

        models = result.data

        for model_data in models:
            model = PredictionModel(**model_data)

            # Create predictor
            predictor_type = (
                "lstm"
                if model.architecture == ModelArchitecture.LSTM
                else "transformer"
            )
            predictor = CoordinationPredictor(model.id, predictor_type)

            try:
                # Load model configuration
                config = {
                    "input_size": len(self.feature_columns),
                    "output_size": 5,  # Predict 5 metrics
                    "hidden_size": 128,
                    "num_layers": 2,
                }

                await predictor.load_model(model.model_path, config)

                # Add to predictors
                predictor_name = f"{model.architecture.value}_{model.id[:8]}"
                self.predictors[predictor_name] = predictor

                logger.info(f"Loaded predictor: {predictor_name}")

            except Exception as e:
                logger.error(f"Failed to load predictor for model {model.id}: {e}")

        if not self.predictors:
            logger.warning("No prediction models loaded, using fallback predictor")
            # Create a mock predictor for demonstration
            self.predictors["fallback"] = CoordinationPredictor("fallback", "lstm")

    async def _data_collection_loop(self):
        """Background data collection loop"""
        while self._running:
            try:
                # Collect current metrics
                metrics = await self._collect_current_metrics()

                # Add to buffer
                self.metrics_buffer.append(metrics)

                # Store in database
                await self._store_metrics(metrics)

                # Wait for next collection
                await asyncio.sleep(self.prediction_interval_seconds)

            except Exception as e:
                logger.error(f"Data collection error: {e}")
                await asyncio.sleep(10)  # Short delay on error

    async def _prediction_loop(self):
        """Background prediction loop"""
        while self._running:
            try:
                # Make periodic predictions
                prediction = await self.predict_coordination_needs(horizon_minutes=15)

                # Check if action is needed
                await self._check_prediction_alerts(prediction)

                # Wait for next prediction cycle
                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
                await asyncio.sleep(60)  # 1 minute delay on error

    async def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        # Mock metrics collection - would integrate with actual monitoring
        metrics = {
            "timestamp": datetime.utcnow().timestamp(),
            "cpu_usage": np.random.normal(50, 15),
            "memory_usage": np.random.normal(60, 20),
            "request_rate": np.random.normal(100, 30),
            "response_time": np.random.normal(150, 50),
            "error_rate": np.random.exponential(0.02),
            "active_connections": np.random.poisson(200),
            "queue_length": np.random.poisson(10),
            "throughput": np.random.normal(95, 25),
            "latency_p95": np.random.normal(300, 100),
            "agent_count": np.random.poisson(5),
        }

        # Ensure realistic bounds
        metrics["cpu_usage"] = max(0, min(100, metrics["cpu_usage"]))
        metrics["memory_usage"] = max(0, min(100, metrics["memory_usage"]))
        metrics["error_rate"] = max(0, min(1, metrics["error_rate"]))

        return metrics

    async def _get_recent_metrics(self, hours: int = 24) -> List[Dict[str, float]]:
        """Get recent metrics from database"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        # Get from buffer first (most recent)
        recent_metrics = list(self.metrics_buffer)

        # Supplement with database if needed
        if len(recent_metrics) < 100:
            # Mock historical data
            timestamps = pd.date_range(start=start_time, end=end_time, freq="30s")
            for ts in timestamps:
                if len(recent_metrics) >= 1000:
                    break
                recent_metrics.append(await self._collect_current_metrics())

        return recent_metrics[-1000:]  # Return up to 1000 most recent

    def _prepare_prediction_input(
        self, metrics_data: List[Dict[str, float]]
    ) -> np.ndarray:
        """Prepare input data for prediction"""
        # Convert to DataFrame
        df = pd.DataFrame(metrics_data)

        # Select feature columns
        feature_data = df[self.feature_columns].fillna(0).values

        # Get last sequence for prediction
        sequence_length = min(60, len(feature_data))
        input_sequence = feature_data[-sequence_length:]

        # Normalize
        input_sequence = (input_sequence - np.mean(input_sequence, axis=0)) / (
            np.std(input_sequence, axis=0) + 1e-8
        )

        return input_sequence

    def _aggregate_predictions(
        self, predictions: Dict[str, np.ndarray], confidences: Dict[str, float]
    ) -> Dict[str, float]:
        """Aggregate predictions from multiple models"""
        if not predictions:
            # Return default predictions
            return {
                "cpu_usage": 50.0,
                "memory_usage": 60.0,
                "request_volume": 1000.0,
                "response_time": 150.0,
                "error_rate": 0.02,
                "throughput": 100.0,
            }

        # Weighted average based on confidence
        total_weight = sum(confidences.values())

        aggregated = {}
        metric_names = [
            "cpu_usage",
            "memory_usage",
            "request_volume",
            "response_time",
            "error_rate",
            "throughput",
        ]

        for i, metric in enumerate(metric_names):
            weighted_sum = 0
            for model_name, pred in predictions.items():
                if i < len(pred):
                    weight = confidences.get(model_name, 0.5)
                    weighted_sum += pred[i] * weight

            aggregated[metric] = (
                weighted_sum / total_weight if total_weight > 0 else 50.0
            )

        return aggregated

    def _calculate_uncertainty_bounds(
        self, predictions: Dict[str, np.ndarray], confidences: Dict[str, float]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate uncertainty bounds for predictions"""
        uncertainty_bounds = {}

        if not predictions:
            return uncertainty_bounds

        # Calculate bounds based on prediction variance and confidence
        metric_names = ["cpu_usage", "memory_usage", "response_time", "error_rate"]

        for i, metric in enumerate(metric_names):
            values = []
            for model_name, pred in predictions.items():
                if i < len(pred):
                    values.append(pred[i])

            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                avg_confidence = np.mean(list(confidences.values()))

                # Wider bounds for lower confidence
                uncertainty_factor = 2.0 * (1.0 - avg_confidence)
                lower_bound = mean_val - std_val * uncertainty_factor
                upper_bound = mean_val + std_val * uncertainty_factor

                uncertainty_bounds[metric] = (float(lower_bound), float(upper_bound))

        return uncertainty_bounds

    async def _generate_coordination_recommendations(
        self, predictions: Dict[str, float], horizon_minutes: int
    ) -> Dict[str, Any]:
        """Generate coordination recommendations based on predictions"""

        predicted_cpu = predictions.get("cpu_usage", 50.0)
        predicted_memory = predictions.get("memory_usage", 60.0)
        predicted_response_time = predictions.get("response_time", 150.0)

        # Calculate recommended agent count
        base_agent_count = 3
        if predicted_cpu > 80 or predicted_memory > 85:
            recommended_agents = base_agent_count + 2
        elif predicted_cpu > 60 or predicted_memory > 70:
            recommended_agents = base_agent_count + 1
        elif predicted_cpu < 30 and predicted_memory < 40:
            recommended_agents = max(1, base_agent_count - 1)
        else:
            recommended_agents = base_agent_count

        # Resource allocation recommendations
        resource_allocation = {
            "cpu_per_agent": max(0.5, predicted_cpu / 100.0 * 2.0),
            "memory_per_agent_gb": max(1.0, predicted_memory / 100.0 * 4.0),
            "max_concurrent_requests": max(10, 50 - int(predicted_response_time / 10)),
        }

        # Scaling actions
        scaling_actions = []

        if predicted_cpu > 75:
            scaling_actions.append(
                {
                    "action": "scale_up_cpu",
                    "priority": "high",
                    "timeline_minutes": 5,
                    "reason": f"High CPU usage predicted: {predicted_cpu:.1f}%",
                }
            )

        if predicted_memory > 80:
            scaling_actions.append(
                {
                    "action": "scale_up_memory",
                    "priority": "high",
                    "timeline_minutes": 5,
                    "reason": f"High memory usage predicted: {predicted_memory:.1f}%",
                }
            )

        if predicted_response_time > 200:
            scaling_actions.append(
                {
                    "action": "add_agents",
                    "priority": "medium",
                    "timeline_minutes": 10,
                    "reason": f"High response time predicted: {predicted_response_time:.1f}ms",
                }
            )

        return {
            "agent_count": recommended_agents,
            "resource_allocation": resource_allocation,
            "scaling_actions": scaling_actions,
        }

    def _analyze_current_utilization(
        self, current_load: Dict[str, float]
    ) -> Dict[str, float]:
        """Analyze current resource utilization"""
        return {
            "cpu_utilization": current_load.get("cpu_usage", 50.0) / 100.0,
            "memory_utilization": current_load.get("memory_usage", 60.0) / 100.0,
            "network_utilization": current_load.get("network_usage", 30.0) / 100.0,
            "agent_efficiency": min(1.0, current_load.get("throughput", 100.0) / 150.0),
        }

    def _calculate_cpu_scaling(
        self,
        current_util: Dict[str, float],
        prediction: CoordinationPrediction,
        target_sla: Dict[str, float],
    ) -> Dict[str, Any]:
        """Calculate CPU scaling recommendations"""
        predicted_cpu = prediction.predicted_cpu_usage / 100.0
        target_cpu = target_sla.get("max_cpu_usage", 70.0) / 100.0

        if predicted_cpu > target_cpu:
            scale_factor = predicted_cpu / target_cpu
            return {
                "action": "scale_up",
                "scale_factor": scale_factor,
                "urgency": "high" if scale_factor > 1.5 else "medium",
            }
        elif predicted_cpu < target_cpu * 0.5:
            return {"action": "scale_down", "scale_factor": 0.7, "urgency": "low"}
        else:
            return {"action": "maintain", "scale_factor": 1.0, "urgency": "none"}

    def _calculate_memory_scaling(
        self,
        current_util: Dict[str, float],
        prediction: CoordinationPrediction,
        target_sla: Dict[str, float],
    ) -> Dict[str, Any]:
        """Calculate memory scaling recommendations"""
        predicted_memory = prediction.predicted_memory_usage / 100.0
        target_memory = target_sla.get("max_memory_usage", 80.0) / 100.0

        if predicted_memory > target_memory:
            scale_factor = predicted_memory / target_memory
            return {
                "action": "scale_up",
                "scale_factor": scale_factor,
                "urgency": "high" if scale_factor > 1.3 else "medium",
            }
        else:
            return {"action": "maintain", "scale_factor": 1.0, "urgency": "none"}

    def _calculate_horizontal_scaling(
        self,
        current_util: Dict[str, float],
        prediction: CoordinationPrediction,
        target_sla: Dict[str, float],
    ) -> Dict[str, Any]:
        """Calculate horizontal scaling recommendations"""
        predicted_response_time = prediction.predicted_response_time_ms
        target_response_time = target_sla.get("max_response_time_ms", 200.0)

        if predicted_response_time > target_response_time:
            additional_agents = max(
                1, int((predicted_response_time - target_response_time) / 50)
            )
            return {
                "action": "scale_out",
                "additional_instances": additional_agents,
                "urgency": "high" if additional_agents > 2 else "medium",
            }
        elif prediction.recommended_agent_count < 2:
            return {"action": "scale_in", "remove_instances": 1, "urgency": "low"}
        else:
            return {"action": "maintain", "urgency": "none"}

    def _estimate_scaling_timeline(self) -> Dict[str, int]:
        """Estimate scaling timeline"""
        return {
            "cpu_scaling_minutes": 3,
            "memory_scaling_minutes": 3,
            "horizontal_scaling_minutes": 8,
            "full_deployment_minutes": 15,
        }

    def _calculate_sla_compliance_probability(
        self, prediction: CoordinationPrediction, target_sla: Dict[str, float]
    ) -> float:
        """Calculate probability of SLA compliance"""
        compliance_scores = []

        # CPU compliance
        if "max_cpu_usage" in target_sla:
            cpu_compliance = 1.0 - max(
                0,
                (prediction.predicted_cpu_usage - target_sla["max_cpu_usage"]) / 100.0,
            )
            compliance_scores.append(cpu_compliance)

        # Memory compliance
        if "max_memory_usage" in target_sla:
            memory_compliance = 1.0 - max(
                0,
                (prediction.predicted_memory_usage - target_sla["max_memory_usage"])
                / 100.0,
            )
            compliance_scores.append(memory_compliance)

        # Response time compliance
        if "max_response_time_ms" in target_sla:
            response_compliance = 1.0 - max(
                0,
                (
                    prediction.predicted_response_time_ms
                    - target_sla["max_response_time_ms"]
                )
                / target_sla["max_response_time_ms"],
            )
            compliance_scores.append(response_compliance)

        return np.mean(compliance_scores) if compliance_scores else 0.8

    def _calculate_accuracy_grade(self, mape: float, correlation: float) -> str:
        """Calculate accuracy grade"""
        if mape < 10 and correlation > 0.9:
            return "A"
        elif mape < 20 and correlation > 0.8:
            return "B"
        elif mape < 30 and correlation > 0.6:
            return "C"
        elif mape < 50 and correlation > 0.4:
            return "D"
        else:
            return "F"

    async def _cache_prediction(self, prediction: CoordinationPrediction) -> None:
        """Cache prediction result"""
        cache_key = f"prediction:coordination:{prediction.id}"
        await self.cache.set(
            cache_key, prediction.dict(), ttl=prediction.prediction_horizon_minutes * 60
        )

    async def _log_prediction(self, prediction: CoordinationPrediction) -> None:
        """Log prediction to database"""
        try:
            await self.supabase.client.table("coordination_predictions").insert(
                prediction.dict()
            ).execute()
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")

    async def _store_metrics(self, metrics: Dict[str, float]) -> None:
        """Store metrics to database"""
        try:
            await self.supabase.client.table("system_metrics").insert(metrics).execute()
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")

    async def _check_prediction_alerts(
        self, prediction: CoordinationPrediction
    ) -> None:
        """Check if prediction triggers any alerts"""
        if prediction.predicted_cpu_usage > 90:
            logger.warning(
                f"High CPU usage predicted: {prediction.predicted_cpu_usage:.1f}%"
            )

        if prediction.predicted_memory_usage > 95:
            logger.warning(
                f"High memory usage predicted: {prediction.predicted_memory_usage:.1f}%"
            )

        if prediction.predicted_error_rate > 0.1:
            logger.warning(
                f"High error rate predicted: {prediction.predicted_error_rate:.3f}"
            )

    async def _get_historical_predictions(
        self, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get historical predictions for accuracy calculation"""
        # Mock historical predictions
        return []

    async def _get_actual_metrics(
        self, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get actual metrics for accuracy calculation"""
        # Mock actual metrics
        return []
