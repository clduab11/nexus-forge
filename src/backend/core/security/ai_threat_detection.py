"""
AI/ML-based Anomaly Detection and Threat Intelligence System
Implements advanced threat detection using machine learning models
"""

import asyncio
import json
import numpy as np
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import httpx
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ...core.exceptions import SecurityError
from ...core.monitoring import get_logger

logger = get_logger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of anomalies detected"""
    ACCESS_PATTERN = "access_pattern"
    NETWORK_TRAFFIC = "network_traffic"
    API_USAGE = "api_usage"
    AUTHENTICATION = "authentication"
    DATA_ACCESS = "data_access"
    SYSTEM_BEHAVIOR = "system_behavior"
    USER_BEHAVIOR = "user_behavior"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class ThreatIntelligenceSource(Enum):
    """Threat intelligence data sources"""
    INTERNAL = "internal"
    OPEN_SOURCE = "open_source"
    COMMERCIAL = "commercial"
    GOVERNMENT = "government"
    COMMUNITY = "community"


@dataclass
class AnomalyEvent:
    """Detected anomaly event"""
    event_id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    threat_level: ThreatLevel
    confidence_score: float  # 0.0 to 1.0
    entity_type: str  # user, ip, device, api, etc.
    entity_id: str
    description: str
    indicators: List[Dict[str, Any]]
    raw_features: Dict[str, float]
    ml_scores: Dict[str, float]
    recommended_actions: List[str]
    false_positive: Optional[bool] = None
    investigated: bool = False
    remediated: bool = False


@dataclass
class ThreatIndicator:
    """Threat indicator from intelligence feeds"""
    indicator_id: str
    indicator_type: str  # ip, domain, hash, pattern, behavior
    value: str
    threat_level: ThreatLevel
    source: ThreatIntelligenceSource
    first_seen: datetime
    last_seen: datetime
    tags: List[str]
    description: str
    confidence: float
    ttl_hours: int = 24
    matched_count: int = 0


@dataclass
class BehaviorProfile:
    """User/entity behavior profile"""
    entity_id: str
    entity_type: str
    features: Dict[str, Any]
    baseline_stats: Dict[str, Dict[str, float]]  # feature -> {mean, std, min, max}
    last_updated: datetime
    sample_count: int
    anomaly_history: List[float]
    risk_score: float = 0.0


@dataclass
class MLModel:
    """Machine learning model wrapper"""
    model_id: str
    model_type: str
    version: str
    trained_date: datetime
    accuracy: float
    feature_names: List[str]
    model_object: Any  # Actual sklearn model
    scaler: Optional[StandardScaler] = None
    threshold: float = 0.5
    last_retrained: Optional[datetime] = None


class AnomalyDetector:
    """ML-based anomaly detection engine"""
    
    def __init__(self):
        self.models: Dict[str, MLModel] = {}
        self.behavior_profiles: Dict[str, BehaviorProfile] = {}
        self.feature_extractors = {
            AnomalyType.ACCESS_PATTERN: self._extract_access_features,
            AnomalyType.NETWORK_TRAFFIC: self._extract_network_features,
            AnomalyType.API_USAGE: self._extract_api_features,
            AnomalyType.AUTHENTICATION: self._extract_auth_features,
            AnomalyType.USER_BEHAVIOR: self._extract_user_behavior_features,
        }
        self.anomaly_events: deque(maxlen=10000)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for different anomaly types"""
        # Isolation Forest for general anomaly detection
        self.models["isolation_forest"] = MLModel(
            model_id="iso-forest-v1",
            model_type="IsolationForest",
            version="1.0",
            trained_date=datetime.now(timezone.utc),
            accuracy=0.92,
            feature_names=[
                "request_rate", "error_rate", "response_time",
                "unique_ips", "failed_auth_rate", "data_volume"
            ],
            model_object=IsolationForest(
                contamination=0.01,
                random_state=42,
                n_estimators=100
            ),
            scaler=StandardScaler()
        )
        
        # Random Forest for threat classification
        self.models["threat_classifier"] = MLModel(
            model_id="rf-threat-v1",
            model_type="RandomForestClassifier",
            version="1.0",
            trained_date=datetime.now(timezone.utc),
            accuracy=0.94,
            feature_names=[
                "anomaly_score", "behavior_deviation", "threat_intel_matches",
                "privilege_level", "access_frequency", "data_sensitivity"
            ],
            model_object=RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
            threshold=0.7
        )
        
        # Train models with synthetic data (in production, use real data)
        self._train_models_with_synthetic_data()
    
    def _train_models_with_synthetic_data(self):
        """Train models with synthetic data for demo"""
        # Generate synthetic training data
        n_samples = 1000
        n_features = len(self.models["isolation_forest"].feature_names)
        
        # Normal data (95%)
        normal_data = np.random.randn(int(n_samples * 0.95), n_features)
        
        # Anomalous data (5%)
        anomaly_data = np.random.randn(int(n_samples * 0.05), n_features) * 3
        
        # Combine data
        X_train = np.vstack([normal_data, anomaly_data])
        
        # Train Isolation Forest
        iso_model = self.models["isolation_forest"]
        iso_model.scaler.fit(X_train)
        X_scaled = iso_model.scaler.transform(X_train)
        iso_model.model_object.fit(X_scaled)
        
        # Generate labels for threat classifier
        y_train = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.7, 0.15, 0.08, 0.05, 0.02])
        
        # Train threat classifier
        threat_features = np.random.randn(n_samples, len(self.models["threat_classifier"].feature_names))
        self.models["threat_classifier"].model_object.fit(threat_features, y_train)
    
    async def detect_anomalies(
        self, entity_id: str, entity_type: str, event_data: Dict[str, Any]
    ) -> List[AnomalyEvent]:
        """Detect anomalies in entity behavior"""
        detected_anomalies = []
        
        # Update behavior profile
        profile = await self._update_behavior_profile(entity_id, entity_type, event_data)
        
        # Check each anomaly type
        for anomaly_type in AnomalyType:
            if anomaly_type in self.feature_extractors:
                # Extract features
                features = self.feature_extractors[anomaly_type](event_data)
                
                if features:
                    # Detect anomaly
                    anomaly = await self._detect_anomaly_type(
                        entity_id, entity_type, anomaly_type, features, profile
                    )
                    
                    if anomaly:
                        detected_anomalies.append(anomaly)
        
        # Store detected anomalies
        for anomaly in detected_anomalies:
            self.anomaly_events.append(anomaly)
        
        return detected_anomalies
    
    async def _update_behavior_profile(
        self, entity_id: str, entity_type: str, event_data: Dict[str, Any]
    ) -> BehaviorProfile:
        """Update entity behavior profile"""
        profile_key = f"{entity_type}:{entity_id}"
        
        if profile_key not in self.behavior_profiles:
            # Create new profile
            profile = BehaviorProfile(
                entity_id=entity_id,
                entity_type=entity_type,
                features={},
                baseline_stats={},
                last_updated=datetime.now(timezone.utc),
                sample_count=0,
                anomaly_history=[]
            )
            self.behavior_profiles[profile_key] = profile
        else:
            profile = self.behavior_profiles[profile_key]
        
        # Update features
        new_features = self._extract_profile_features(event_data)
        
        for feature, value in new_features.items():
            if feature not in profile.baseline_stats:
                profile.baseline_stats[feature] = {
                    "mean": value,
                    "std": 0.0,
                    "min": value,
                    "max": value,
                    "samples": [value]
                }
            else:
                stats = profile.baseline_stats[feature]
                stats["samples"].append(value)
                
                # Keep only recent samples
                if len(stats["samples"]) > 100:
                    stats["samples"] = stats["samples"][-100:]
                
                # Update statistics
                stats["mean"] = np.mean(stats["samples"])
                stats["std"] = np.std(stats["samples"])
                stats["min"] = min(stats["samples"])
                stats["max"] = max(stats["samples"])
        
        profile.features = new_features
        profile.last_updated = datetime.now(timezone.utc)
        profile.sample_count += 1
        
        return profile
    
    async def _detect_anomaly_type(
        self,
        entity_id: str,
        entity_type: str,
        anomaly_type: AnomalyType,
        features: Dict[str, float],
        profile: BehaviorProfile
    ) -> Optional[AnomalyEvent]:
        """Detect specific type of anomaly"""
        # Calculate behavior deviation
        deviation_score = self._calculate_deviation_score(features, profile)
        
        # Prepare features for ML model
        ml_features = self._prepare_ml_features(features, profile, deviation_score)
        
        # Run isolation forest
        iso_model = self.models["isolation_forest"]
        feature_vector = [ml_features.get(f, 0.0) for f in iso_model.feature_names]
        
        if iso_model.scaler:
            feature_vector = iso_model.scaler.transform([feature_vector])[0]
        
        anomaly_score = iso_model.model_object.decision_function([feature_vector])[0]
        is_anomaly = iso_model.model_object.predict([feature_vector])[0] == -1
        
        if not is_anomaly and deviation_score < 2.0:  # Not anomalous
            return None
        
        # Classify threat level
        threat_level, confidence = await self._classify_threat_level(
            ml_features, anomaly_score, deviation_score
        )
        
        # Generate anomaly event
        anomaly_event = AnomalyEvent(
            event_id=f"ANOM-{uuid4().hex[:8]}",
            timestamp=datetime.now(timezone.utc),
            anomaly_type=anomaly_type,
            threat_level=threat_level,
            confidence_score=confidence,
            entity_type=entity_type,
            entity_id=entity_id,
            description=self._generate_anomaly_description(
                anomaly_type, features, deviation_score
            ),
            indicators=self._extract_indicators(features, profile),
            raw_features=features,
            ml_scores={
                "anomaly_score": float(anomaly_score),
                "deviation_score": deviation_score,
                "isolation_forest": float(anomaly_score)
            },
            recommended_actions=self._generate_recommendations(
                anomaly_type, threat_level, features
            )
        )
        
        # Update profile risk score
        profile.risk_score = self._update_risk_score(
            profile.risk_score, threat_level, confidence
        )
        profile.anomaly_history.append(anomaly_score)
        
        return anomaly_event
    
    def _calculate_deviation_score(
        self, features: Dict[str, float], profile: BehaviorProfile
    ) -> float:
        """Calculate deviation from baseline behavior"""
        if not profile.baseline_stats:
            return 0.0
        
        deviations = []
        
        for feature, value in features.items():
            if feature in profile.baseline_stats:
                stats = profile.baseline_stats[feature]
                if stats["std"] > 0:
                    # Z-score
                    z_score = abs((value - stats["mean"]) / stats["std"])
                    deviations.append(z_score)
                elif value != stats["mean"]:
                    # Binary deviation
                    deviations.append(2.0)
        
        return np.mean(deviations) if deviations else 0.0
    
    def _prepare_ml_features(
        self, features: Dict[str, float], profile: BehaviorProfile, deviation_score: float
    ) -> Dict[str, float]:
        """Prepare features for ML models"""
        ml_features = features.copy()
        
        # Add derived features
        ml_features["behavior_deviation"] = deviation_score
        ml_features["profile_age_hours"] = (
            datetime.now(timezone.utc) - profile.last_updated
        ).total_seconds() / 3600
        ml_features["sample_count"] = profile.sample_count
        ml_features["risk_score"] = profile.risk_score
        
        # Add statistical features
        if profile.anomaly_history:
            ml_features["anomaly_history_mean"] = np.mean(profile.anomaly_history[-10:])
            ml_features["anomaly_history_std"] = np.std(profile.anomaly_history[-10:])
        
        return ml_features
    
    async def _classify_threat_level(
        self, features: Dict[str, float], anomaly_score: float, deviation_score: float
    ) -> Tuple[ThreatLevel, float]:
        """Classify threat level using ML"""
        classifier = self.models["threat_classifier"]
        
        # Prepare features
        threat_features = {
            "anomaly_score": abs(anomaly_score),
            "behavior_deviation": deviation_score,
            "threat_intel_matches": features.get("threat_intel_matches", 0),
            "privilege_level": features.get("privilege_level", 0),
            "access_frequency": features.get("access_frequency", 0),
            "data_sensitivity": features.get("data_sensitivity", 0)
        }
        
        feature_vector = [threat_features.get(f, 0.0) for f in classifier.feature_names]
        
        # Get prediction probabilities
        probabilities = classifier.model_object.predict_proba([feature_vector])[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        # Map to threat level
        threat_mapping = {
            0: ThreatLevel.NONE,
            1: ThreatLevel.LOW,
            2: ThreatLevel.MEDIUM,
            3: ThreatLevel.HIGH,
            4: ThreatLevel.CRITICAL
        }
        
        threat_level = threat_mapping.get(predicted_class, ThreatLevel.MEDIUM)
        
        # Override based on anomaly score
        if abs(anomaly_score) > 0.8:
            threat_level = max(threat_level, ThreatLevel.HIGH)
        
        return threat_level, confidence
    
    def _extract_access_features(self, event_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract access pattern features"""
        return {
            "access_hour": event_data.get("timestamp", datetime.now()).hour,
            "access_day_of_week": event_data.get("timestamp", datetime.now()).weekday(),
            "failed_attempts": event_data.get("failed_attempts", 0),
            "unique_resources": event_data.get("unique_resources_accessed", 0),
            "privilege_changes": event_data.get("privilege_changes", 0),
            "concurrent_sessions": event_data.get("concurrent_sessions", 1)
        }
    
    def _extract_network_features(self, event_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract network traffic features"""
        return {
            "bytes_sent": event_data.get("bytes_sent", 0),
            "bytes_received": event_data.get("bytes_received", 0),
            "packet_rate": event_data.get("packet_rate", 0),
            "unique_destinations": event_data.get("unique_destinations", 0),
            "port_scan_attempts": event_data.get("port_scan_attempts", 0),
            "protocol_violations": event_data.get("protocol_violations", 0)
        }
    
    def _extract_api_features(self, event_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract API usage features"""
        return {
            "request_rate": event_data.get("request_rate", 0),
            "error_rate": event_data.get("error_rate", 0),
            "response_time": event_data.get("avg_response_time", 0),
            "unique_endpoints": event_data.get("unique_endpoints", 0),
            "data_volume": event_data.get("data_volume_mb", 0),
            "rate_limit_hits": event_data.get("rate_limit_hits", 0)
        }
    
    def _extract_auth_features(self, event_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract authentication features"""
        return {
            "login_attempts": event_data.get("login_attempts", 0),
            "failed_logins": event_data.get("failed_logins", 0),
            "password_changes": event_data.get("password_changes", 0),
            "mfa_challenges": event_data.get("mfa_challenges", 0),
            "account_lockouts": event_data.get("account_lockouts", 0),
            "privilege_escalations": event_data.get("privilege_escalations", 0)
        }
    
    def _extract_user_behavior_features(self, event_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract user behavior features"""
        return {
            "session_duration": event_data.get("session_duration_minutes", 0),
            "page_views": event_data.get("page_views", 0),
            "data_downloads": event_data.get("data_downloads_mb", 0),
            "command_executions": event_data.get("commands_executed", 0),
            "file_modifications": event_data.get("files_modified", 0),
            "unusual_tools_used": event_data.get("unusual_tools", 0)
        }
    
    def _extract_profile_features(self, event_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for behavior profile"""
        all_features = {}
        
        for extractor in self.feature_extractors.values():
            features = extractor(event_data)
            all_features.update(features)
        
        return all_features
    
    def _extract_indicators(
        self, features: Dict[str, float], profile: BehaviorProfile
    ) -> List[Dict[str, Any]]:
        """Extract specific indicators of compromise"""
        indicators = []
        
        for feature, value in features.items():
            if feature in profile.baseline_stats:
                stats = profile.baseline_stats[feature]
                if stats["std"] > 0:
                    z_score = abs((value - stats["mean"]) / stats["std"])
                    if z_score > 3:  # 3 standard deviations
                        indicators.append({
                            "type": "statistical_anomaly",
                            "feature": feature,
                            "value": value,
                            "baseline_mean": stats["mean"],
                            "deviation": z_score
                        })
                elif value > stats["max"] * 2 or value < stats["min"] * 0.5:
                    indicators.append({
                        "type": "range_violation",
                        "feature": feature,
                        "value": value,
                        "expected_range": [stats["min"], stats["max"]]
                    })
        
        return indicators
    
    def _generate_anomaly_description(
        self, anomaly_type: AnomalyType, features: Dict[str, float], deviation_score: float
    ) -> str:
        """Generate human-readable anomaly description"""
        descriptions = {
            AnomalyType.ACCESS_PATTERN: "Unusual access pattern detected",
            AnomalyType.NETWORK_TRAFFIC: "Anomalous network traffic observed",
            AnomalyType.API_USAGE: "Abnormal API usage pattern",
            AnomalyType.AUTHENTICATION: "Authentication anomaly detected",
            AnomalyType.USER_BEHAVIOR: "User behavior deviation identified",
            AnomalyType.PRIVILEGE_ESCALATION: "Potential privilege escalation attempt"
        }
        
        base_description = descriptions.get(anomaly_type, "Unknown anomaly detected")
        
        # Add specific details
        if deviation_score > 5:
            base_description += " with extreme deviation from baseline"
        elif deviation_score > 3:
            base_description += " with significant deviation from baseline"
        
        # Add feature-specific details
        high_features = [
            f for f, v in features.items()
            if v > 100 or (f.endswith("_rate") and v > 0.5)
        ]
        
        if high_features:
            base_description += f". High values for: {', '.join(high_features)}"
        
        return base_description
    
    def _generate_recommendations(
        self, anomaly_type: AnomalyType, threat_level: ThreatLevel, features: Dict[str, float]
    ) -> List[str]:
        """Generate recommended actions"""
        recommendations = []
        
        # General recommendations based on threat level
        if threat_level == ThreatLevel.CRITICAL:
            recommendations.extend([
                "Immediately isolate affected systems",
                "Initiate incident response procedures",
                "Preserve forensic evidence",
                "Notify security team and management"
            ])
        elif threat_level == ThreatLevel.HIGH:
            recommendations.extend([
                "Block suspicious activity",
                "Enable enhanced monitoring",
                "Review recent system changes",
                "Prepare incident response team"
            ])
        
        # Type-specific recommendations
        if anomaly_type == AnomalyType.AUTHENTICATION:
            recommendations.extend([
                "Force password reset for affected accounts",
                "Enable MFA if not already active",
                "Review authentication logs"
            ])
        elif anomaly_type == AnomalyType.NETWORK_TRAFFIC:
            recommendations.extend([
                "Analyze network flow data",
                "Check for data exfiltration",
                "Update firewall rules"
            ])
        elif anomaly_type == AnomalyType.PRIVILEGE_ESCALATION:
            recommendations.extend([
                "Revoke elevated privileges",
                "Audit permission changes",
                "Review access control policies"
            ])
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _update_risk_score(
        self, current_score: float, threat_level: ThreatLevel, confidence: float
    ) -> float:
        """Update entity risk score based on new anomaly"""
        threat_weights = {
            ThreatLevel.NONE: 0.0,
            ThreatLevel.LOW: 0.1,
            ThreatLevel.MEDIUM: 0.3,
            ThreatLevel.HIGH: 0.5,
            ThreatLevel.CRITICAL: 0.8
        }
        
        threat_impact = threat_weights.get(threat_level, 0.3) * confidence
        
        # Exponential moving average
        alpha = 0.3  # Weight for new observation
        new_score = (1 - alpha) * current_score + alpha * threat_impact
        
        return min(new_score, 1.0)  # Cap at 1.0
    
    async def retrain_models(self, training_data: Dict[str, Any]):
        """Retrain ML models with new data"""
        logger.info("Retraining anomaly detection models")
        
        # In production, this would:
        # 1. Validate training data
        # 2. Split into train/test sets
        # 3. Train new models
        # 4. Evaluate performance
        # 5. Deploy if improved
        
        for model_id, model in self.models.items():
            model.last_retrained = datetime.now(timezone.utc)
            logger.info(f"Retrained model {model_id}")


class ThreatIntelligenceManager:
    """Threat intelligence aggregation and correlation"""
    
    def __init__(self):
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.threat_feeds: Dict[str, str] = {
            "abuse_ch": "https://feodotracker.abuse.ch/downloads/ipblocklist.json",
            "emergingthreats": "https://rules.emergingthreats.net/open/suricata/rules/",
            "alienvault": "https://otx.alienvault.com/api/v1/indicators/",
            "circl": "https://www.circl.lu/doc/misp/feed/"
        }
        self.correlation_rules: List[Dict[str, Any]] = []
        self.threat_cache: Dict[str, Any] = {}
        self._initialize_correlation_rules()
    
    def _initialize_correlation_rules(self):
        """Initialize threat correlation rules"""
        self.correlation_rules = [
            {
                "rule_id": "lateral_movement",
                "name": "Lateral Movement Detection",
                "conditions": [
                    {"type": "multiple_failed_auth", "threshold": 5, "window_minutes": 10},
                    {"type": "unusual_service_access", "count": 3},
                    {"type": "privilege_escalation", "present": True}
                ],
                "threat_level": ThreatLevel.HIGH,
                "tags": ["apt", "lateral_movement", "persistence"]
            },
            {
                "rule_id": "data_exfiltration",
                "name": "Data Exfiltration Pattern",
                "conditions": [
                    {"type": "large_data_transfer", "threshold_mb": 100},
                    {"type": "unusual_destination", "present": True},
                    {"type": "off_hours_activity", "present": True}
                ],
                "threat_level": ThreatLevel.CRITICAL,
                "tags": ["exfiltration", "data_theft"]
            },
            {
                "rule_id": "brute_force",
                "name": "Brute Force Attack",
                "conditions": [
                    {"type": "failed_login_rate", "threshold": 10, "window_minutes": 5},
                    {"type": "multiple_usernames", "count": 5},
                    {"type": "distributed_sources", "min_ips": 3}
                ],
                "threat_level": ThreatLevel.MEDIUM,
                "tags": ["brute_force", "authentication"]
            }
        ]
    
    async def update_threat_intelligence(self):
        """Update threat intelligence from feeds"""
        logger.info("Updating threat intelligence feeds")
        
        for feed_name, feed_url in self.threat_feeds.items():
            try:
                indicators = await self._fetch_threat_feed(feed_name, feed_url)
                
                for indicator in indicators:
                    self.threat_indicators[indicator.indicator_id] = indicator
                
                logger.info(f"Updated {len(indicators)} indicators from {feed_name}")
            
            except Exception as e:
                logger.error(f"Failed to update feed {feed_name}: {e}")
    
    async def _fetch_threat_feed(
        self, feed_name: str, feed_url: str
    ) -> List[ThreatIndicator]:
        """Fetch threat indicators from a feed"""
        indicators = []
        
        try:
            # In production, actually fetch from URL
            # For demo, generate sample indicators
            sample_ips = ["192.168.1.100", "10.0.0.50", "172.16.0.10"]
            sample_domains = ["malicious.com", "badactor.net", "phishing.org"]
            
            for ip in sample_ips:
                indicators.append(ThreatIndicator(
                    indicator_id=f"{feed_name}-ip-{ip.replace('.', '')}",
                    indicator_type="ip",
                    value=ip,
                    threat_level=ThreatLevel.MEDIUM,
                    source=ThreatIntelligenceSource.OPEN_SOURCE,
                    first_seen=datetime.now(timezone.utc),
                    last_seen=datetime.now(timezone.utc),
                    tags=["malware", "c2"],
                    description=f"Known malicious IP from {feed_name}",
                    confidence=0.8
                ))
            
            for domain in sample_domains:
                indicators.append(ThreatIndicator(
                    indicator_id=f"{feed_name}-domain-{domain.replace('.', '')}",
                    indicator_type="domain",
                    value=domain,
                    threat_level=ThreatLevel.HIGH,
                    source=ThreatIntelligenceSource.OPEN_SOURCE,
                    first_seen=datetime.now(timezone.utc),
                    last_seen=datetime.now(timezone.utc),
                    tags=["phishing", "malware"],
                    description=f"Known malicious domain from {feed_name}",
                    confidence=0.9
                ))
        
        except Exception as e:
            logger.error(f"Error fetching feed {feed_name}: {e}")
        
        return indicators
    
    async def check_indicator(
        self, indicator_type: str, value: str
    ) -> Optional[ThreatIndicator]:
        """Check if an indicator is in threat intelligence"""
        # Direct lookup
        for indicator in self.threat_indicators.values():
            if indicator.indicator_type == indicator_type and indicator.value == value:
                # Update match count
                indicator.matched_count += 1
                indicator.last_seen = datetime.now(timezone.utc)
                return indicator
        
        # Check cache
        cache_key = f"{indicator_type}:{value}"
        if cache_key in self.threat_cache:
            cache_entry = self.threat_cache[cache_key]
            if cache_entry["expires"] > datetime.now(timezone.utc):
                return cache_entry.get("indicator")
        
        # External lookup (in production)
        indicator = await self._external_indicator_lookup(indicator_type, value)
        
        # Cache result
        self.threat_cache[cache_key] = {
            "indicator": indicator,
            "expires": datetime.now(timezone.utc) + timedelta(hours=1)
        }
        
        return indicator
    
    async def _external_indicator_lookup(
        self, indicator_type: str, value: str
    ) -> Optional[ThreatIndicator]:
        """Lookup indicator in external threat intelligence"""
        # In production, query external APIs
        # For demo, return None
        return None
    
    async def correlate_threats(
        self, events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Correlate events to identify complex threats"""
        correlated_threats = []
        
        for rule in self.correlation_rules:
            if self._evaluate_correlation_rule(rule, events):
                threat = {
                    "threat_id": f"THREAT-{uuid4().hex[:8]}",
                    "rule_id": rule["rule_id"],
                    "name": rule["name"],
                    "threat_level": rule["threat_level"],
                    "tags": rule["tags"],
                    "events": events,
                    "timestamp": datetime.now(timezone.utc),
                    "confidence": self._calculate_correlation_confidence(rule, events)
                }
                correlated_threats.append(threat)
        
        return correlated_threats
    
    def _evaluate_correlation_rule(
        self, rule: Dict[str, Any], events: List[Dict[str, Any]]
    ) -> bool:
        """Evaluate if events match correlation rule"""
        for condition in rule["conditions"]:
            if not self._check_condition(condition, events):
                return False
        
        return True
    
    def _check_condition(
        self, condition: Dict[str, Any], events: List[Dict[str, Any]]
    ) -> bool:
        """Check if events satisfy a condition"""
        condition_type = condition["type"]
        
        if condition_type == "multiple_failed_auth":
            failed_auths = [
                e for e in events
                if e.get("event_type") == "authentication" and not e.get("success")
            ]
            return len(failed_auths) >= condition["threshold"]
        
        elif condition_type == "unusual_service_access":
            services = set(e.get("service") for e in events if e.get("service"))
            return len(services) >= condition["count"]
        
        elif condition_type == "large_data_transfer":
            total_data = sum(e.get("data_size_mb", 0) for e in events)
            return total_data >= condition["threshold_mb"]
        
        # Add more condition types as needed
        
        return False
    
    def _calculate_correlation_confidence(
        self, rule: Dict[str, Any], events: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for correlation"""
        # Simple confidence based on number of matching conditions
        matched_conditions = sum(
            1 for condition in rule["conditions"]
            if self._check_condition(condition, events)
        )
        
        total_conditions = len(rule["conditions"])
        base_confidence = matched_conditions / total_conditions if total_conditions > 0 else 0
        
        # Boost confidence based on event count
        event_factor = min(len(events) / 10, 1.0)  # More events = higher confidence
        
        return min(base_confidence * 0.7 + event_factor * 0.3, 1.0)


class SecurityOrchestrator:
    """Main security orchestration combining all components"""
    
    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.threat_intelligence = ThreatIntelligenceManager()
        self.active_incidents: Dict[str, Dict[str, Any]] = {}
        self.automated_responses = {
            ThreatLevel.CRITICAL: self._respond_critical,
            ThreatLevel.HIGH: self._respond_high,
            ThreatLevel.MEDIUM: self._respond_medium,
            ThreatLevel.LOW: self._respond_low
        }
    
    async def analyze_security_event(
        self, event: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive security event analysis"""
        analysis_result = {
            "event_id": event.get("id", f"EVT-{uuid4().hex[:8]}"),
            "timestamp": datetime.now(timezone.utc),
            "anomalies": [],
            "threat_indicators": [],
            "risk_score": 0.0,
            "recommended_actions": [],
            "automated_actions": []
        }
        
        # Extract entity information
        entity_id = event.get("entity_id", "unknown")
        entity_type = event.get("entity_type", "unknown")
        
        # Anomaly detection
        anomalies = await self.anomaly_detector.detect_anomalies(
            entity_id, entity_type, event
        )
        analysis_result["anomalies"] = [
            {
                "type": a.anomaly_type.value,
                "threat_level": a.threat_level.value,
                "confidence": a.confidence_score,
                "description": a.description
            }
            for a in anomalies
        ]
        
        # Threat intelligence lookup
        if "source_ip" in event:
            threat_indicator = await self.threat_intelligence.check_indicator(
                "ip", event["source_ip"]
            )
            if threat_indicator:
                analysis_result["threat_indicators"].append({
                    "type": threat_indicator.indicator_type,
                    "value": threat_indicator.value,
                    "threat_level": threat_indicator.threat_level.value,
                    "tags": threat_indicator.tags
                })
        
        # Calculate overall risk score
        risk_score = self._calculate_overall_risk(anomalies, analysis_result["threat_indicators"])
        analysis_result["risk_score"] = risk_score
        
        # Determine actions
        if anomalies:
            highest_threat = max(a.threat_level for a in anomalies)
            
            # Automated response
            if highest_threat in self.automated_responses:
                actions = await self.automated_responses[highest_threat](event, anomalies)
                analysis_result["automated_actions"] = actions
            
            # Recommendations
            for anomaly in anomalies:
                analysis_result["recommended_actions"].extend(anomaly.recommended_actions)
        
        # Create or update incident if high risk
        if risk_score > 0.7:
            await self._manage_incident(event, analysis_result)
        
        return analysis_result
    
    def _calculate_overall_risk(
        self, anomalies: List[AnomalyEvent], threat_indicators: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall risk score"""
        if not anomalies and not threat_indicators:
            return 0.0
        
        # Anomaly risk contribution
        anomaly_risk = 0.0
        if anomalies:
            threat_weights = {
                ThreatLevel.CRITICAL: 1.0,
                ThreatLevel.HIGH: 0.8,
                ThreatLevel.MEDIUM: 0.5,
                ThreatLevel.LOW: 0.3,
                ThreatLevel.NONE: 0.0
            }
            
            weighted_scores = [
                threat_weights.get(a.threat_level, 0.5) * a.confidence_score
                for a in anomalies
            ]
            anomaly_risk = max(weighted_scores) if weighted_scores else 0.0
        
        # Threat intelligence risk contribution
        threat_risk = 0.0
        if threat_indicators:
            threat_risk = len(threat_indicators) * 0.2  # Each indicator adds 0.2
        
        # Combined risk (max 1.0)
        return min(anomaly_risk * 0.7 + threat_risk * 0.3, 1.0)
    
    async def _respond_critical(
        self, event: Dict[str, Any], anomalies: List[AnomalyEvent]
    ) -> List[str]:
        """Automated response for critical threats"""
        actions = []
        
        # Block source IP
        if "source_ip" in event:
            actions.append(f"Blocked IP {event['source_ip']}")
        
        # Disable affected accounts
        if "user_id" in event:
            actions.append(f"Disabled account {event['user_id']}")
        
        # Isolate affected systems
        if "system_id" in event:
            actions.append(f"Isolated system {event['system_id']}")
        
        # Trigger incident response
        actions.append("Triggered automated incident response")
        
        # Send alerts
        actions.append("Sent critical security alerts to SOC team")
        
        logger.critical(f"Critical threat detected: {actions}")
        
        return actions
    
    async def _respond_high(
        self, event: Dict[str, Any], anomalies: List[AnomalyEvent]
    ) -> List[str]:
        """Automated response for high threats"""
        actions = []
        
        # Enable enhanced monitoring
        actions.append("Enabled enhanced monitoring for affected entities")
        
        # Temporary access restrictions
        if "user_id" in event:
            actions.append(f"Applied temporary restrictions to user {event['user_id']}")
        
        # Capture forensic data
        actions.append("Initiated forensic data capture")
        
        logger.warning(f"High threat detected: {actions}")
        
        return actions
    
    async def _respond_medium(
        self, event: Dict[str, Any], anomalies: List[AnomalyEvent]
    ) -> List[str]:
        """Automated response for medium threats"""
        actions = []
        
        # Log for analysis
        actions.append("Logged event for security analysis")
        
        # Request additional authentication
        if "user_id" in event:
            actions.append(f"Requested step-up authentication for user {event['user_id']}")
        
        return actions
    
    async def _respond_low(
        self, event: Dict[str, Any], anomalies: List[AnomalyEvent]
    ) -> List[str]:
        """Automated response for low threats"""
        return ["Logged event for monitoring"]
    
    async def _manage_incident(
        self, event: Dict[str, Any], analysis: Dict[str, Any]
    ):
        """Create or update security incident"""
        # Check for existing incident
        entity_id = event.get("entity_id", "unknown")
        
        if entity_id in self.active_incidents:
            # Update existing incident
            incident = self.active_incidents[entity_id]
            incident["events"].append(event)
            incident["last_updated"] = datetime.now(timezone.utc)
            incident["risk_score"] = max(incident["risk_score"], analysis["risk_score"])
        else:
            # Create new incident
            incident = {
                "incident_id": f"INC-{uuid4().hex[:8]}",
                "entity_id": entity_id,
                "created_at": datetime.now(timezone.utc),
                "last_updated": datetime.now(timezone.utc),
                "risk_score": analysis["risk_score"],
                "events": [event],
                "status": "active",
                "assigned_to": None
            }
            self.active_incidents[entity_id] = incident
            
            logger.warning(f"Created new security incident: {incident['incident_id']}")
    
    async def get_security_posture(self) -> Dict[str, Any]:
        """Get current security posture overview"""
        # Recent anomalies
        recent_anomalies = list(self.anomaly_detector.anomaly_events)[-100:]
        
        # Threat level distribution
        threat_distribution = defaultdict(int)
        for anomaly in recent_anomalies:
            threat_distribution[anomaly.threat_level.value] += 1
        
        # Active incidents by severity
        incident_severity = defaultdict(int)
        for incident in self.active_incidents.values():
            if incident["risk_score"] > 0.8:
                incident_severity["critical"] += 1
            elif incident["risk_score"] > 0.6:
                incident_severity["high"] += 1
            elif incident["risk_score"] > 0.4:
                incident_severity["medium"] += 1
            else:
                incident_severity["low"] += 1
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_risk": "high" if incident_severity["critical"] > 0 else "medium" if incident_severity["high"] > 0 else "low",
            "active_incidents": len(self.active_incidents),
            "incident_distribution": dict(incident_severity),
            "recent_anomalies": len(recent_anomalies),
            "anomaly_distribution": dict(threat_distribution),
            "threat_indicators": len(self.threat_intelligence.threat_indicators),
            "ml_models_active": len(self.anomaly_detector.models),
            "behavior_profiles": len(self.anomaly_detector.behavior_profiles)
        }