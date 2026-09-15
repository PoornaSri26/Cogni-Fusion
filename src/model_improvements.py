"""
Model Improvements Module
Provides advanced model enhancements including:
- Ensemble methods with multiple architectures
- Model interpretability (SHAP values, attention visualization)
- Confidence calibration
- Hyperparameter optimization
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional
import joblib

# Optional PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class AttentionFusionNetEnhanced(nn.Module):
        """Enhanced Attention Fusion Network with additional features."""
        
        def __init__(self, eeg_dim: int, eye_dim: int, hidden_dim: int = 128, num_classes: int = 3):
            super().__init__()
            
            # Enhanced EEG encoder with residual connections
            self.eeg_encoder = nn.Sequential(
                nn.Linear(eeg_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # Enhanced eye encoder
            self.eye_encoder = nn.Sequential(
                nn.Linear(eye_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # Cross-modal attention
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=0.2,
                batch_first=True
            )
            
            # Gated fusion with learnable weights
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 2),
                nn.Softmax(dim=-1)
            )
            
            # Enhanced classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(hidden_dim // 2, num_classes)
            )
            
            # Attention weights for interpretability
            self.attention_weights = None
        
        def forward(self, eeg_features, eye_features):
            # Encode modalities
            eeg_encoded = self.eeg_encoder(eeg_features)
            eye_encoded = self.eye_encoder(eye_features)
            
            # Expand dimensions for attention
            eeg_expanded = eeg_encoded.unsqueeze(1)  # (batch, 1, hidden)
            eye_expanded = eye_encoded.unsqueeze(1)  # (batch, 1, hidden)
            
            # Cross-modal attention
            combined = torch.cat([eeg_expanded, eye_expanded], dim=1)
            attended, attention_weights = self.cross_attention(combined, combined, combined)
            self.attention_weights = attention_weights.detach()
            
            # Gated fusion
            gate_weights = self.fusion_gate(torch.cat([eeg_encoded, eye_encoded], dim=-1))
            eeg_weight = gate_weights[:, 0:1]
            eye_weight = gate_weights[:, 1:2]
            
            fused = eeg_weight * eeg_encoded + eye_weight * eye_encoded
            
            # Classification
            logits = self.classifier(fused)
            return logits, gate_weights
else:
    class AttentionFusionNetEnhanced:
        """Placeholder class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for AttentionFusionNetEnhanced")


class EnsembleModel:
    """Ensemble of multiple models for robust predictions."""
    
    def __init__(self, models: List, weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        self.scalers = []
    
    def predict(self, eeg_features, eye_features):
        """Make ensemble predictions."""
        predictions = []
        
        for model in self.models:
            if TORCH_AVAILABLE and isinstance(model, nn.Module):
                with torch.no_grad():
                    model.eval()
                    logits, _ = model(eeg_features, eye_features)
                    probs = F.softmax(logits, dim=-1)
                    predictions.append(probs.cpu().numpy())
            else:
                # For sklearn models
                combined = np.concatenate([eeg_features, eye_features], axis=1)
                probs = model.predict_proba(combined)
                predictions.append(probs)
        
        # Weighted average
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        return weighted_pred
    
    def get_confidence(self, predictions: np.ndarray) -> float:
        """Get confidence score from predictions."""
        max_probs = np.max(predictions, axis=-1)
        return float(np.mean(max_probs))
    
    def add_scaler(self, scaler):
        """Add feature scaler to ensemble."""
        self.scalers.append(scaler)


class ModelInterpretability:
    """Tools for model interpretability and analysis."""
    
    def __init__(self, model):
        self.model = model
        self.feature_importance = {}
        self.shap_values = None
    
    def compute_feature_importance(self, eeg_features: np.ndarray, eye_features: np.ndarray, 
                                    feature_names: List[str]) -> Dict:
        """Compute feature importance using permutation importance."""
        from sklearn.metrics import accuracy_score
        
        # Get baseline predictions
        baseline_probs = self.model.predict(eeg_features, eye_features)
        baseline_preds = np.argmax(baseline_probs, axis=1)
        baseline_acc = accuracy_score(baseline_preds, baseline_preds)  # Self-consistency
        
        importance = {}
        combined_features = np.concatenate([eeg_features, eye_features], axis=1)
        
        for i, name in enumerate(feature_names):
            # Permute feature
            permuted_features = combined_features.copy()
            permuted_features[:, i] = np.random.permutation(permuted_features[:, i])
            
            eeg_perm = permuted_features[:, :eeg_features.shape[1]]
            eye_perm = permuted_features[:, eeg_features.shape[1]:]
            
            # Get predictions with permuted feature
            permuted_probs = self.model.predict(eeg_perm, eye_perm)
            permuted_preds = np.argmax(permuted_probs, axis=1)
            permuted_acc = accuracy_score(permuted_preds, baseline_preds)
            
            # Importance = decrease in accuracy
            importance[name] = baseline_acc - permuted_acc
        
        self.feature_importance = importance
        return importance
    
    def visualize_attention_weights(self, attention_weights: np.ndarray) -> Dict:
        """Analyze and return attention weight statistics."""
        if attention_weights is None:
            return {}
        
        attention_stats = {
            'mean_attention': float(np.mean(attention_weights)),
            'std_attention': float(np.std(attention_weights)),
            'max_attention': float(np.max(attention_weights)),
            'min_attention': float(np.min(attention_weights)),
            'attention_distribution': attention_weights.tolist()
        }
        
        return attention_stats
    
    def explain_prediction(self, eeg_features: np.ndarray, eye_features: np.ndarray, 
                          feature_names: List[str]) -> Dict:
        """Explain individual prediction using SHAP-like approach."""
        # Compute feature contributions
        explanation = {
            'eeg_contribution': {},
            'eye_contribution': {},
            'feature_importance': self.feature_importance
        }
        
        # Normalize features for contribution analysis
        eeg_norm = (eeg_features - np.mean(eeg_features, axis=0)) / (np.std(eeg_features, axis=0) + 1e-8)
        eye_norm = (eye_features - np.mean(eye_features, axis=0)) / (np.std(eye_features, axis=0) + 1e-8)
        
        # Simple contribution analysis (dot product with importance)
        for i, name in enumerate(feature_names[:len(eeg_norm[0])]):
            if name in self.feature_importance:
                explanation['eeg_contribution'][name] = float(
                    eeg_norm[0, i] * self.feature_importance[name]
                )
        
        for i, name in enumerate(feature_names[len(eeg_norm[0]):]):
            if name in self.feature_importance:
                explanation['eye_contribution'][name] = float(
                    eye_norm[0, i] * self.feature_importance[name]
                )
        
        return explanation


class ConfidenceCalibrator:
    """Calibrate model confidence scores."""
    
    def __init__(self, method: str = 'isotonic'):
        self.method = method
        self.calibrators = {}
        self.is_fitted = False
    
    def fit(self, predictions: np.ndarray, true_labels: np.ndarray):
        """Fit calibration on validation data."""
        n_classes = predictions.shape[1]
        
        for class_idx in range(n_classes):
            calibrator = CalibratedClassifierCV(method=self.method, cv='prefit')
            
            # Create dummy classifier for calibration
            from sklearn.dummy import DummyClassifier
            dummy = DummyClassifier(strategy='prior')
            dummy.fit(np.zeros((len(true_labels), 1)), true_labels)
            
            # Create probability array for this class
            class_probs = predictions[:, class_idx:class_idx+1]
            
            # Fit calibrator
            calibrator.fit(class_probs, true_labels)
            self.calibrators[class_idx] = calibrator
        
        self.is_fitted = True
    
    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """Calibrate prediction probabilities."""
        if not self.is_fitted:
            return predictions
        
        calibrated = predictions.copy()
        for class_idx in range(predictions.shape[1]):
            if class_idx in self.calibrators:
                class_probs = predictions[:, class_idx:class_idx+1]
                calibrated[:, class_idx] = self.calibrators[class_idx].predict_proba(class_probs)[:, 1]
        
        # Renormalize
        calibrated = calibrated / (calibrated.sum(axis=1, keepdims=True) + 1e-8)
        return calibrated
    
    def save(self, path: str):
        """Save calibrator to disk."""
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str):
        """Load calibrator from disk."""
        return joblib.load(path)


class HyperparameterOptimizer:
    """Optimize hyperparameters using cross-validation."""
    
    def __init__(self, param_grid: Dict, cv: int = 5):
        self.param_grid = param_grid
        self.cv = cv
        self.best_params = None
        self.best_score = None
    
    def optimize(self, model_class, X_train: np.ndarray, y_train: np.ndarray, 
                  scoring: str = 'accuracy') -> Dict:
        """Perform grid search optimization."""
        from sklearn.model_selection import GridSearchCV
        
        # Create base model
        base_model = model_class()
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            self.param_grid,
            cv=self.cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': grid_search.cv_results_
        }


def create_ensemble(neural_model, eeg_dim: int, eye_dim: int, 
                    X_train: np.ndarray, y_train: np.ndarray) -> EnsembleModel:
    """Create an ensemble of neural and traditional ML models."""
    
    # Prepare data for sklearn models
    X_combined = np.concatenate([X_train[:, :eeg_dim], X_train[:, eeg_dim:]], axis=1)
    
    # Train traditional models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_combined, y_train)
    
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_combined, y_train)
    
    # Create ensemble with available models
    models = [rf_model, gb_model]
    weights = [0.5, 0.5]
    
    if TORCH_AVAILABLE and neural_model is not None:
        models.insert(0, neural_model)
        weights = [0.5, 0.25, 0.25]  # Weight neural model higher
    
    ensemble = EnsembleModel(models=models, weights=weights)
    
    return ensemble