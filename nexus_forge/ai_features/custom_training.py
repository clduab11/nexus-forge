"""
Custom Training Engine
Handles agent training with transfer learning and advanced techniques
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import json
import numpy as np
import pickle
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .models import (
    TrainingJob, TrainingJobStatus, TrainingJobType,
    ModelArchitecture, HyperParameters, TrainingMetrics,
    ModelEvaluation, PredictionModel, TrainingDataSource
)
from ..core.cache import RedisCache
from ..integrations.supabase.coordination_client import SupabaseCoordinationClient
from ..core.exceptions import ValidationError, NotFoundError, ResourceError

logger = logging.getLogger(__name__)


class AgentDataset(Dataset):
    """Custom dataset for agent training data"""
    
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y.dtype in ['int32', 'int64'] else torch.FloatTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class TransformerModel(nn.Module):
    """Transformer model for agent coordination"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_heads, num_classes, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.randn(1000, hidden_size))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        seq_len = x.size(1)
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = x + pos_enc
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


class LSTMModel(nn.Module):
    """LSTM model for sequence prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.1):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Classification
        x = self.dropout(last_output)
        x = self.classifier(x)
        
        return x


class TrainingPipeline:
    """Training pipeline for custom models"""
    
    def __init__(self, job: TrainingJob):
        self.job = job
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize wandb if configured
        self.use_wandb = False
        try:
            wandb.init(
                project="nexus-forge-training",
                name=f"{job.name}_{job.id[:8]}",
                config=job.hyperparameters.dict()
            )
            self.use_wandb = True
        except Exception:
            logger.warning("Weights & Biases not configured, skipping logging")
    
    async def prepare_data(self, data_sources: List[TrainingDataSource]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from sources"""
        logger.info("Preparing training data")
        
        all_features = []
        all_targets = []
        
        for source in data_sources:
            # Load data from source
            data = await self._load_data_from_source(source)
            
            # Extract features and targets
            features = data[self.job.feature_columns].values
            targets = data[self.job.target_variable].values
            
            # Apply preprocessing
            if source.preprocessing:
                features = self._apply_preprocessing(features, source.preprocessing)
            
            all_features.append(features)
            all_targets.append(targets)
        
        # Combine all data
        X = np.vstack(all_features)
        y = np.hstack(all_targets)
        
        # Normalize features
        X = self.scaler.fit_transform(X)
        
        # Encode labels if necessary
        if y.dtype == 'object' or len(np.unique(y)) < len(y) * 0.1:
            y = self.label_encoder.fit_transform(y)
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def build_model(self, input_size: int, num_classes: int) -> nn.Module:
        """Build model based on architecture"""
        hp = self.job.hyperparameters
        
        if self.job.architecture == ModelArchitecture.TRANSFORMER:
            model = TransformerModel(
                input_size=input_size,
                hidden_size=hp.hidden_size or 256,
                num_layers=hp.num_layers or 6,
                num_heads=hp.attention_heads or 8,
                num_classes=num_classes,
                dropout=hp.dropout_rate
            )
        
        elif self.job.architecture == ModelArchitecture.LSTM:
            model = LSTMModel(
                input_size=input_size,
                hidden_size=hp.hidden_size or 128,
                num_layers=hp.num_layers or 2,
                num_classes=num_classes,
                dropout=hp.dropout_rate
            )
        
        else:
            # Simple feedforward network
            layers = []
            current_size = input_size
            
            for i in range(hp.num_layers or 3):
                hidden_size = hp.hidden_size or 256
                layers.extend([
                    nn.Linear(current_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(hp.dropout_rate)
                ])
                current_size = hidden_size
            
            layers.append(nn.Linear(current_size, num_classes))
            model = nn.Sequential(*layers)
        
        return model.to(self.device)
    
    def setup_optimizer(self) -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler]]:
        """Setup optimizer and scheduler"""
        hp = self.job.hyperparameters
        
        # Optimizer
        if hp.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=hp.learning_rate,
                weight_decay=hp.weight_decay
            )
        elif hp.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=hp.learning_rate,
                momentum=0.9,
                weight_decay=hp.weight_decay
            )
        else:
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=hp.learning_rate,
                weight_decay=hp.weight_decay
            )
        
        # Scheduler
        scheduler = None
        if hp.lr_scheduler.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=hp.epochs
            )
        elif hp.lr_scheduler.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=hp.epochs // 3, gamma=0.1
            )
        elif hp.lr_scheduler.lower() == "reduce":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=hp.early_stopping_patience // 2
            )
        
        return optimizer, scheduler
    
    async def train(self, X: np.ndarray, y: np.ndarray) -> ModelEvaluation:
        """Train the model"""
        logger.info("Starting model training")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Create datasets and dataloaders
        train_dataset = AgentDataset(X_train, y_train)
        val_dataset = AgentDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.job.hyperparameters.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.job.hyperparameters.batch_size,
            shuffle=False
        )
        
        # Build model
        input_size = X.shape[1]
        num_classes = len(np.unique(y))
        self.model = self.build_model(input_size, num_classes)
        
        # Setup training
        self.optimizer, self.scheduler = self.setup_optimizer()
        criterion = nn.CrossEntropyLoss() if num_classes > 1 else nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.job.hyperparameters.epochs):
            # Training phase
            train_metrics = await self._train_epoch(train_loader, criterion, epoch)
            
            # Validation phase
            val_metrics = await self._validate_epoch(val_loader, criterion, epoch)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping
            if val_metrics.val_loss < best_val_loss:
                best_val_loss = val_metrics.val_loss
                patience_counter = 0
                # Save best model
                await self._save_model_checkpoint(epoch, "best")
            else:
                patience_counter += 1
            
            if patience_counter >= self.job.hyperparameters.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_metrics.train_loss,
                    "val_loss": val_metrics.val_loss,
                    "train_accuracy": train_metrics.train_accuracy,
                    "val_accuracy": val_metrics.val_accuracy,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
        
        # Final evaluation
        evaluation = await self._evaluate_model(val_loader, y_val)
        
        if self.use_wandb:
            wandb.finish()
        
        return evaluation
    
    async def apply_transfer_learning(self, base_model_path: str) -> None:
        """Apply transfer learning from base model"""
        logger.info(f"Applying transfer learning from {base_model_path}")
        
        try:
            # Load base model
            base_model = torch.load(base_model_path, map_location=self.device)
            
            if self.model is None:
                raise ValueError("Target model not initialized")
            
            # Transfer compatible layers
            model_dict = self.model.state_dict()
            base_dict = base_model.state_dict()
            
            # Filter compatible layers
            compatible_dict = {
                k: v for k, v in base_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            
            # Update model
            model_dict.update(compatible_dict)
            self.model.load_state_dict(model_dict)
            
            # Freeze early layers if specified
            if self.job.job_type == TrainingJobType.TRANSFER_LEARNING:
                self._freeze_early_layers(freeze_ratio=0.7)
            
            logger.info(f"Transferred {len(compatible_dict)} layers")
            
        except Exception as e:
            logger.error(f"Transfer learning failed: {e}")
            raise ResourceError(f"Transfer learning failed: {e}")
    
    async def _train_epoch(self, train_loader: DataLoader, criterion, epoch: int) -> TrainingMetrics:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        batch_times = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            start_time = datetime.now()
            
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            if self.job.hyperparameters.gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.job.hyperparameters.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            if len(output.shape) > 1 and output.shape[1] > 1:
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Track batch time
            batch_time = (datetime.now() - start_time).total_seconds() * 1000
            batch_times.append(batch_time)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0
        avg_batch_time = np.mean(batch_times)
        
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=avg_loss,
            train_accuracy=accuracy,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            batch_time_ms=avg_batch_time,
            memory_usage_mb=torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        )
        
        return metrics
    
    async def _validate_epoch(self, val_loader: DataLoader, criterion, epoch: int) -> TrainingMetrics:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                if len(output.shape) > 1 and output.shape[1] > 1:
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0
        
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=0,  # Not applicable for validation
            val_loss=avg_loss,
            val_accuracy=accuracy,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            batch_time_ms=0,  # Not tracked for validation
            memory_usage_mb=torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        )
        
        return metrics
    
    async def _evaluate_model(self, val_loader: DataLoader, y_true: np.ndarray) -> ModelEvaluation:
        """Evaluate trained model"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        inference_times = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                start_time = datetime.now()
                output = self.model(data)
                inference_time = (datetime.now() - start_time).total_seconds() * 1000
                inference_times.append(inference_time)
                
                if len(output.shape) > 1 and output.shape[1] > 1:
                    pred = output.argmax(dim=1)
                else:
                    pred = output
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        y_pred = np.array(all_predictions)
        y_true = np.array(all_targets)
        
        accuracy = accuracy_score(y_true, y_pred)
        
        if len(np.unique(y_true)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted'
            )
        else:
            precision = recall = f1 = accuracy
        
        evaluation = ModelEvaluation(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            inference_time_ms=np.mean(inference_times),
            throughput_per_second=1000 / np.mean(inference_times),
            memory_footprint_mb=sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
        )
        
        return evaluation
    
    async def _save_model_checkpoint(self, epoch: int, checkpoint_type: str = "epoch") -> str:
        """Save model checkpoint"""
        checkpoint_dir = Path(f"checkpoints/{self.job.id}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{checkpoint_type}_{epoch}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state': pickle.dumps(self.scaler),
            'label_encoder_state': pickle.dumps(self.label_encoder)
        }, checkpoint_path)
        
        return str(checkpoint_path)
    
    def _freeze_early_layers(self, freeze_ratio: float = 0.7) -> None:
        """Freeze early layers for transfer learning"""
        layers = list(self.model.named_parameters())
        freeze_count = int(len(layers) * freeze_ratio)
        
        for i, (name, param) in enumerate(layers):
            if i < freeze_count:
                param.requires_grad = False
                logger.debug(f"Froze layer: {name}")
    
    async def _load_data_from_source(self, source: TrainingDataSource) -> Any:
        """Load data from training data source"""
        # This would implement actual data loading from various sources
        # For now, return mock data
        import pandas as pd
        
        # Mock data generation
        np.random.seed(42)
        n_samples = 1000
        n_features = len(self.job.feature_columns)
        
        data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=self.job.feature_columns
        )
        data[self.job.target_variable] = np.random.randint(0, 3, n_samples)
        
        return data
    
    def _apply_preprocessing(self, data: np.ndarray, preprocessing_config: Dict[str, Any]) -> np.ndarray:
        """Apply preprocessing to data"""
        if preprocessing_config.get("normalize"):
            data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        
        if preprocessing_config.get("remove_outliers"):
            # Simple outlier removal using IQR
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            outlier_mask = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
            data = np.where(outlier_mask, np.median(data, axis=0), data)
        
        return data


class CustomTrainingEngine:
    """Main engine for custom agent training"""
    
    def __init__(self):
        self.cache = RedisCache()
        self.supabase = SupabaseCoordinationClient()
        self.active_jobs: Dict[str, TrainingPipeline] = {}
        
        # Training configuration
        self.max_concurrent_jobs = 5
        self.model_storage_path = Path("models")
        self.model_storage_path.mkdir(exist_ok=True)
    
    async def create_training_job(self, job: TrainingJob) -> TrainingJob:
        """Create new training job"""
        logger.info(f"Creating training job: {job.name}")
        
        # Validate job configuration
        await self._validate_training_job(job)
        
        # Save job to database
        await self.supabase.client.table("training_jobs").insert(
            job.dict()
        ).execute()
        
        logger.info(f"Created training job: {job.id}")
        return job
    
    async def start_training_job(self, job_id: str) -> bool:
        """Start training job execution"""
        job = await self._get_training_job(job_id)
        
        if not job:
            raise NotFoundError(f"Training job {job_id} not found")
        
        if job.status != TrainingJobStatus.PENDING:
            raise ValidationError(f"Job {job_id} is not in pending status")
        
        # Check resource limits
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            raise ResourceError("Maximum concurrent training jobs reached")
        
        logger.info(f"Starting training job: {job_id}")
        
        # Update job status
        job.status = TrainingJobStatus.PREPARING
        await self._update_training_job(job)
        
        # Start training in background
        training_task = asyncio.create_task(self._execute_training_job(job))
        
        return True
    
    async def get_training_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get training job status and metrics"""
        job = await self._get_training_job(job_id)
        
        if not job:
            raise NotFoundError(f"Training job {job_id} not found")
        
        # Get latest metrics
        latest_metrics = job.training_metrics[-1] if job.training_metrics else None
        
        status = {
            "job_id": job_id,
            "name": job.name,
            "status": job.status.value,
            "progress_percent": job.progress_percent,
            "current_epoch": job.current_epoch,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "compute_time_minutes": job.compute_time_minutes,
            "cost_usd": job.cost_usd,
            "latest_metrics": latest_metrics.dict() if latest_metrics else None,
            "error_message": job.error_message
        }
        
        # Add real-time metrics if job is active
        if job_id in self.active_jobs:
            status["is_active"] = True
            # Add any real-time metrics here
        else:
            status["is_active"] = False
        
        return status
    
    async def cancel_training_job(self, job_id: str) -> bool:
        """Cancel running training job"""
        job = await self._get_training_job(job_id)
        
        if not job:
            raise NotFoundError(f"Training job {job_id} not found")
        
        if job.status not in [TrainingJobStatus.PREPARING, TrainingJobStatus.TRAINING]:
            raise ValidationError(f"Job {job_id} cannot be cancelled in current status")
        
        logger.info(f"Cancelling training job: {job_id}")
        
        # Remove from active jobs
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        
        # Update job status
        job.status = TrainingJobStatus.CANCELLED
        await self._update_training_job(job)
        
        return True
    
    async def list_training_jobs(
        self,
        status: Optional[TrainingJobStatus] = None,
        user_id: Optional[str] = None,
        limit: int = 50
    ) -> List[TrainingJob]:
        """List training jobs with optional filtering"""
        query = self.supabase.client.table("training_jobs").select("*")
        
        if status:
            query = query.eq("status", status.value)
        
        if user_id:
            query = query.eq("created_by", user_id)
        
        result = await query.order("created_at", desc=True).limit(limit).execute()
        
        return [TrainingJob(**job_data) for job_data in result.data]
    
    async def create_model_from_job(self, job_id: str) -> PredictionModel:
        """Create deployable model from completed training job"""
        job = await self._get_training_job(job_id)
        
        if not job:
            raise NotFoundError(f"Training job {job_id} not found")
        
        if job.status != TrainingJobStatus.COMPLETED:
            raise ValidationError(f"Job {job_id} is not completed")
        
        if not job.evaluation_results:
            raise ValidationError(f"Job {job_id} has no evaluation results")
        
        logger.info(f"Creating model from training job: {job_id}")
        
        # Create model record
        model = PredictionModel(
            name=f"{job.name}_model",
            description=f"Model created from training job {job.name}",
            architecture=job.architecture,
            prediction_type="coordination",  # Default type
            training_job_id=job_id,
            input_schema={"features": job.feature_columns},
            output_schema={"prediction": "string"},
            evaluation_metrics=job.evaluation_results,
            model_path=str(self.model_storage_path / f"{job_id}_model.pt"),
            weights_path=str(self.model_storage_path / f"{job_id}_weights.pt"),
            config_path=str(self.model_storage_path / f"{job_id}_config.json"),
            created_by=job.created_by,
            tenant_id=job.tenant_id
        )
        
        # Save model to database
        await self.supabase.client.table("prediction_models").insert(
            model.dict()
        ).execute()
        
        logger.info(f"Created model: {model.id}")
        return model
    
    # Private helper methods
    
    async def _execute_training_job(self, job: TrainingJob) -> None:
        """Execute training job"""
        pipeline = TrainingPipeline(job)
        self.active_jobs[job.id] = pipeline
        
        try:
            # Update status to training
            job.status = TrainingJobStatus.TRAINING
            job.started_at = datetime.utcnow()
            await self._update_training_job(job)
            
            # Prepare data
            X, y = await pipeline.prepare_data(job.data_sources)
            
            # Apply transfer learning if specified
            if job.base_model_id:
                base_model_path = await self._get_base_model_path(job.base_model_id)
                await pipeline.apply_transfer_learning(base_model_path)
            
            # Train model
            evaluation = await pipeline.train(X, y)
            
            # Save model artifacts
            model_path = await self._save_model_artifacts(job, pipeline)
            
            # Update job with results
            job.status = TrainingJobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.evaluation_results = evaluation
            job.progress_percent = 100.0
            
            await self._update_training_job(job)
            
            logger.info(f"Training job completed: {job.id}")
            
        except Exception as e:
            logger.error(f"Training job failed: {job.id}, error: {e}")
            
            # Update job with error
            job.status = TrainingJobStatus.FAILED
            job.error_message = str(e)
            job.retry_count += 1
            
            await self._update_training_job(job)
        
        finally:
            # Remove from active jobs
            if job.id in self.active_jobs:
                del self.active_jobs[job.id]
    
    async def _validate_training_job(self, job: TrainingJob) -> None:
        """Validate training job configuration"""
        # Validate data sources
        if not job.data_sources:
            raise ValidationError("At least one data source is required")
        
        # Validate feature columns
        if not job.feature_columns:
            raise ValidationError("Feature columns are required")
        
        # Validate target variable
        if not job.target_variable:
            raise ValidationError("Target variable is required")
        
        # Validate hyperparameters
        hp = job.hyperparameters
        if hp.learning_rate <= 0:
            raise ValidationError("Learning rate must be positive")
        
        if hp.batch_size <= 0:
            raise ValidationError("Batch size must be positive")
        
        if hp.epochs <= 0:
            raise ValidationError("Epochs must be positive")
    
    async def _get_training_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job by ID"""
        result = await self.supabase.client.table("training_jobs") \
            .select("*") \
            .eq("id", job_id) \
            .execute()
        
        if not result.data:
            return None
        
        return TrainingJob(**result.data[0])
    
    async def _update_training_job(self, job: TrainingJob) -> None:
        """Update training job in database"""
        await self.supabase.client.table("training_jobs") \
            .update(job.dict()) \
            .eq("id", job.id) \
            .execute()
    
    async def _get_base_model_path(self, base_model_id: str) -> str:
        """Get path to base model for transfer learning"""
        # Query for base model
        result = await self.supabase.client.table("prediction_models") \
            .select("model_path") \
            .eq("id", base_model_id) \
            .execute()
        
        if not result.data:
            raise NotFoundError(f"Base model {base_model_id} not found")
        
        return result.data[0]["model_path"]
    
    async def _save_model_artifacts(self, job: TrainingJob, pipeline: TrainingPipeline) -> str:
        """Save trained model artifacts"""
        model_dir = self.model_storage_path / job.id
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pt"
        torch.save(pipeline.model.state_dict(), model_path)
        
        # Save preprocessing artifacts
        scaler_path = model_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(pipeline.scaler, f)
        
        encoder_path = model_dir / "label_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(pipeline.label_encoder, f)
        
        # Save configuration
        config_path = model_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "job_id": job.id,
                "architecture": job.architecture.value,
                "hyperparameters": job.hyperparameters.dict(),
                "feature_columns": job.feature_columns,
                "target_variable": job.target_variable
            }, f, indent=2)
        
        return str(model_path)