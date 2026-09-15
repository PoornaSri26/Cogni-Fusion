"""
Testing and Validation Framework
Provides comprehensive testing and validation including:
- Automated testing suite
- Real-time performance monitoring
- Benchmarking tools
- Validation metrics dashboard
"""

import unittest
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Callable
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

# Optional imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AutomatedTestSuite(unittest.TestCase):
    """Automated test suite for EEG eye project components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path("test_data")
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Generate synthetic test data
        self.test_eeg_data = np.random.randn(4, 256)  # 4 channels, 256 samples
        self.test_eye_data = pd.DataFrame({
            'gaze_x': np.random.randn(60),
            'gaze_y': np.random.randn(60),
            't': np.arange(60) / 60.0,
            'pupil_mm': np.random.uniform(2, 8, 60),
            'is_blink': np.random.choice([0, 1], 60, p=[0.95, 0.05]),
            'is_fixation': np.random.choice([0, 1], 60, p=[0.7, 0.3])
        })
    
    def test_eeg_feature_extraction(self):
        """Test EEG feature extraction."""
        from src.eeg_features import extract_eeg_features
        
        channel_names = ['AF7', 'AF8', 'TP9', 'TP10']
        features = extract_eeg_features(self.test_eeg_data, channel_names=channel_names)
        
        # Check that features were extracted
        self.assertGreater(len(features), 0)
        
        # Check for expected feature types
        self.assertIn('AF7_alpha_rel', features)
        self.assertIn('AF7_theta_alpha_ratio', features)
        self.assertIn('AF7_spectral_entropy', features)
        
        # Check for new advanced features
        self.assertIn('AF7_sample_entropy', features)
        self.assertIn('AF7_permutation_entropy', features)
        self.assertIn('AF7_fractal_dimension', features)
        
        # Check connectivity features
        self.assertIn('coh_AF7_AF8_alpha', features)
        self.assertIn('pli_AF7_AF8', features)
    
    def test_eye_feature_extraction(self):
        """Test eye feature extraction."""
        from src.eye_features import extract_eye_features
        
        features = extract_eye_features(self.test_eye_data)
        
        # Check that features were extracted
        self.assertGreater(len(features), 0)
        
        # Check for expected feature types
        self.assertIn('gaze_x_std', features)
        self.assertIn('gaze_speed_mean', features)
        self.assertIn('pupil_mean', features)
        
        # Check for new advanced features
        self.assertIn('microsaccade_rate_hz', features)
        self.assertIn('pupil_trend_slope', features)
        self.assertIn('pupil_variability_cv', features)
    
    def test_advanced_analytics(self):
        """Test advanced analytics module."""
        from src.advanced_analytics import AdvancedAnalytics
        
        analytics = AdvancedAnalytics()
        
        # Test statistical summary
        test_df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        
        summary = analytics.compute_statistical_summary(test_df)
        self.assertIn('mean', summary)
        self.assertIn('std', summary)
        
        # Test cross-frequency coupling
        coupling = analytics.compute_cross_frequency_coupling(self.test_eeg_data)
        self.assertGreater(len(coupling), 0)
        
        # Test temporal pattern analysis
        predictions = ['low', 'medium', 'high', 'low', 'medium']
        timestamps = [datetime.now().isoformat() for _ in range(5)]
        patterns = analytics.analyze_temporal_patterns(predictions, timestamps)
        self.assertIn('transition_probabilities', patterns)
    
    def test_model_improvements(self):
        """Test model improvements module."""
        try:
            from src.model_improvements import AttentionFusionNetEnhanced, EnsembleModel, TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                self.skipTest("PyTorch not available for model testing")
            
            import torch
            
            # Test enhanced model
            eeg_dim = 78
            eye_dim = 18
            model = AttentionFusionNetEnhanced(eeg_dim, eye_dim)
            
            # Test forward pass
            eeg_features = torch.randn(2, eeg_dim)
            eye_features = torch.randn(2, eye_dim)
            
            logits, gate_weights = model(eeg_features, eye_features)
            
            self.assertEqual(logits.shape[0], 2)
            self.assertEqual(logits.shape[1], 3)  # 3 classes
            self.assertEqual(gate_weights.shape[1], 2)  # 2 modalities
        except ImportError as e:
            self.skipTest(f"PyTorch not available for model testing: {e}")
    
    def test_dashboard_enhancements(self):
        """Test dashboard enhancements module."""
        from src.dashboard_enhancements import AlertSystem, SessionRecorder
        
        # Test alert system
        alert_system = AlertSystem()
        alerts = alert_system.check_alerts('high', 0.9, 10, 6)
        self.assertGreater(len(alerts), 0)
        
        # Test session recorder
        recorder = SessionRecorder()
        session_id = recorder.start_session('test_user')
        self.assertIsNotNone(session_id)
        
        recorder.record_epoch({'prediction': 'high', 'confidence': 0.8})
        session_summary = recorder.end_session()
        self.assertEqual(session_summary['epoch_count'], 1)
    
    def test_realtime_features(self):
        """Test real-time features module."""
        from src.realtime_features import RealTimeFeaturesManager
        
        manager = RealTimeFeaturesManager()
        
        # Test epoch processing
        summary = manager.process_epoch(
            prediction='high',
            confidence=0.8,
            eeg_trust=0.6,
            eye_trust=0.4,
            processing_time=0.1
        )
        
        self.assertIn('timestamp', summary)
        self.assertIn('break_recommendation', summary)
        self.assertIn('performance_metrics', summary)
        self.assertIn('stress_assessment', summary)
    
    def test_data_management(self):
        """Test data management module."""
        from src.data_management import DataLogger, SessionManager
        
        # Test data logger
        logger = DataLogger(log_dir="test_logs")
        logger.log_epoch({
            'prediction': 'high',
            'confidence': 0.8,
            'eeg_trust': 0.6,
            'eye_trust': 0.4,
            'processing_time': 0.1
        })
        
        stats = logger.get_statistics()
        self.assertGreater(stats['total_epochs'], 0)
        
        # Test session manager
        session_manager = SessionManager(storage_dir="test_sessions")
        session_id = session_manager.create_session('test_user')
        session_manager.add_epoch_to_session(session_id, {'test': 'data'})
        session = session_manager.close_session(session_id)
        self.assertEqual(session['epoch_count'], 1)
    
    def tearDown(self):
        """Clean up test artifacts."""
        # Clean up test directories
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
        
        test_logs = Path("test_logs")
        if test_logs.exists():
            shutil.rmtree(test_logs)
        
        test_sessions = Path("test_sessions")
        if test_sessions.exists():
            shutil.rmtree(test_sessions)
        
        test_db = Path("data/logs.db")
        if test_db.exists():
            test_db.unlink()


class PerformanceMonitor:
    """Real-time performance monitoring for the system."""
    
    def __init__(self):
        self.metrics = {
            'inference_time': [],
            'feature_extraction_time': [],
            'total_processing_time': [],
            'memory_usage': [],
            'cpu_usage': []
        }
        self.baseline_metrics = {}
        self.performance_alerts = []
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.process = psutil.Process() if self._check_psutil() else None
    
    def _check_psutil(self):
        """Check if psutil is available."""
        try:
            import psutil
            return True
        except ImportError:
            return False
    
    def record_inference_time(self, duration: float):
        """Record inference time."""
        self.metrics['inference_time'].append(duration)
    
    def record_feature_extraction_time(self, duration: float):
        """Record feature extraction time."""
        self.metrics['feature_extraction_time'].append(duration)
    
    def record_total_processing_time(self, duration: float):
        """Record total processing time."""
        self.metrics['total_processing_time'].append(duration)
    
    def record_system_metrics(self):
        """Record system metrics if psutil is available."""
        if self.process:
            try:
                import psutil
                self.metrics['memory_usage'].append(self.process.memory_info().rss / 1024 / 1024)  # MB
                self.metrics['cpu_usage'].append(self.process.cpu_percent())
            except:
                pass
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        
        return summary
    
    def detect_performance_issues(self, thresholds: Optional[Dict] = None) -> List[Dict]:
        """Detect performance issues based on thresholds."""
        thresholds = thresholds or {
            'max_inference_time': 1.0,  # seconds
            'max_memory_usage': 1000,  # MB
            'max_cpu_usage': 80  # percent
        }
        
        issues = []
        
        if self.metrics['inference_time']:
            avg_inference = np.mean(self.metrics['inference_time'])
            if avg_inference > thresholds['max_inference_time']:
                issues.append({
                    'type': 'slow_inference',
                    'severity': 'warning',
                    'message': f'Average inference time {avg_inference:.3f}s exceeds threshold {thresholds["max_inference_time"]}s',
                    'value': avg_inference
                })
        
        if self.metrics['memory_usage']:
            avg_memory = np.mean(self.metrics['memory_usage'])
            if avg_memory > thresholds['max_memory_usage']:
                issues.append({
                    'type': 'high_memory_usage',
                    'severity': 'warning',
                    'message': f'Average memory usage {avg_memory:.1f}MB exceeds threshold {thresholds["max_memory_usage"]}MB',
                    'value': avg_memory
                })
        
        if self.metrics['cpu_usage']:
            avg_cpu = np.mean(self.metrics['cpu_usage'])
            if avg_cpu > thresholds['max_cpu_usage']:
                issues.append({
                    'type': 'high_cpu_usage',
                    'severity': 'warning',
                    'message': f'Average CPU usage {avg_cpu:.1f}% exceeds threshold {thresholds["max_cpu_usage"]}%',
                    'value': avg_cpu
                })
        
        self.performance_alerts.extend(issues)
        return issues
    
    def set_baseline(self):
        """Set current metrics as baseline."""
        self.baseline_metrics = {k: np.mean(v) if v else 0.0 for k, v in self.metrics.items()}
    
    def compare_to_baseline(self) -> Dict:
        """Compare current performance to baseline."""
        if not self.baseline_metrics:
            return {'status': 'no_baseline'}
        
        comparison = {}
        for metric_name, baseline_value in self.baseline_metrics.items():
            if self.metrics[metric_name]:
                current_value = np.mean(self.metrics[metric_name])
                change = ((current_value - baseline_value) / baseline_value * 100) if baseline_value > 0 else 0
                comparison[metric_name] = {
                    'baseline': float(baseline_value),
                    'current': float(current_value),
                    'change_percent': float(change),
                    'status': 'degraded' if change > 20 else 'improved' if change < -20 else 'stable'
                }
        
        return comparison


class BenchmarkRunner:
    """Run benchmarking tests on the system."""
    
    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.benchmark_results = []
    
    def run_feature_extraction_benchmark(self, n_iterations: int = 100) -> Dict:
        """Benchmark feature extraction performance."""
        from src.eeg_features import extract_eeg_features
        from src.eye_features import extract_eye_features
        
        times = []
        
        for _ in range(n_iterations):
            # Generate test data
            eeg_data = np.random.randn(4, 256)
            eye_data = pd.DataFrame({
                'gaze_x': np.random.randn(60),
                'gaze_y': np.random.randn(60),
                't': np.arange(60) / 60.0,
                'pupil_mm': np.random.uniform(2, 8, 60)
            })
            
            # Time feature extraction
            start_time = time.time()
            
            eeg_features = extract_eeg_features(eeg_data, channel_names=['AF7', 'AF8', 'TP9', 'TP10'])
            eye_features = extract_eye_features(eye_data)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        result = {
            'benchmark_name': 'feature_extraction',
            'iterations': n_iterations,
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'throughput': float(n_iterations / sum(times))
        }
        
        self.benchmark_results.append(result)
        return result
    
    def run_inference_benchmark(self, model, n_iterations: int = 100) -> Dict:
        """Benchmark model inference performance."""
        try:
            import torch
            
            times = []
            model.eval()
            
            eeg_dim = 78
            eye_dim = 18
            
            with torch.no_grad():
                for _ in range(n_iterations):
                    eeg_features = torch.randn(1, eeg_dim)
                    eye_features = torch.randn(1, eye_dim)
                    
                    start_time = time.time()
                    logits, _ = model(eeg_features, eye_features)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
            
            result = {
                'benchmark_name': 'model_inference',
                'iterations': n_iterations,
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times)),
                'throughput': float(n_iterations / sum(times))
            }
            
            self.benchmark_results.append(result)
            return result
        except ImportError:
            return {'error': 'PyTorch not available for inference benchmark'}
    
    def run_end_to_end_benchmark(self, n_iterations: int = 50) -> Dict:
        """Benchmark end-to-end pipeline performance."""
        from src.eeg_features import extract_eeg_features
        from src.eye_features import extract_eye_features
        
        # Load model (placeholder - would use actual model)
        try:
            import torch
            model = torch.load('models/fusion_model.pt', map_location='cpu')
            model.eval()
        except:
            print("Model not found, skipping model inference in benchmark")
            model = None
        
        times = []
        
        for _ in range(n_iterations):
            # Generate test data
            eeg_data = np.random.randn(4, 256)
            eye_data = pd.DataFrame({
                'gaze_x': np.random.randn(60),
                'gaze_y': np.random.randn(60),
                't': np.arange(60) / 60.0,
                'pupil_mm': np.random.uniform(2, 8, 60)
            })
            
            start_time = time.time()
            
            # Feature extraction
            eeg_features = extract_eeg_features(eeg_data, channel_names=['AF7', 'AF8', 'TP9', 'TP10'])
            eye_features = extract_eye_features(eye_data)
            
            # Model inference (if model available)
            if model:
                try:
                    import torch
                    eeg_tensor = torch.randn(1, 78)  # Placeholder
                    eye_tensor = torch.randn(1, 18)  # Placeholder
                    with torch.no_grad():
                        logits, _ = model(eeg_tensor, eye_tensor)
                except:
                    pass  # Skip model inference if it fails
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        result = {
            'benchmark_name': 'end_to_end_pipeline',
            'iterations': n_iterations,
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'throughput': float(n_iterations / sum(times))
        }
        
        self.benchmark_results.append(result)
        return result
    
    def save_benchmark_results(self, filename: Optional[str] = None):
        """Save benchmark results to file."""
        if filename is None:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.benchmark_results, f, indent=2)
        
        return str(output_path)
    
    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report."""
        if not self.benchmark_results:
            return "No benchmark results available"
        
        report = ["# Benchmark Report", f"Generated: {datetime.now().isoformat()}", ""]
        
        for result in self.benchmark_results:
            report.append(f"## {result['benchmark_name']}")
            report.append(f"- Iterations: {result['iterations']}")
            report.append(f"- Mean Time: {result['mean_time']:.4f}s")
            report.append(f"- Std Time: {result['std_time']:.4f}s")
            report.append(f"- Min Time: {result['min_time']:.4f}s")
            report.append(f"- Max Time: {result['max_time']:.4f}s")
            report.append(f"- Throughput: {result['throughput']:.2f} ops/sec")
            report.append("")
        
        return "\n".join(report)


class ValidationMetrics:
    """Comprehensive validation metrics for model performance."""
    
    def __init__(self):
        self.validation_history = []
    
    def compute_classification_metrics(self, predictions: List[str], 
                                     true_labels: List[str],
                                     confidences: List[float]) -> Dict:
        """Compute comprehensive classification metrics."""
        from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                     confusion_matrix, roc_auc_score, classification_report)
        
        if len(predictions) != len(true_labels):
            return {'error': 'Predictions and true labels must have same length'}
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # Per-class metrics
        classes = ['low', 'medium', 'high']
        per_class_metrics = {}
        for i, class_name in enumerate(classes):
            per_class_metrics[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        # Overall metrics
        overall_metrics = {
            'accuracy': float(accuracy),
            'macro_precision': float(np.mean(precision)),
            'macro_recall': float(np.mean(recall)),
            'macro_f1': float(np.mean(f1)),
            'weighted_precision': float(np.average(precision, weights=support)),
            'weighted_recall': float(np.average(recall, weights=support)),
            'weighted_f1': float(np.average(f1, weights=support))
        }
        
        # Confidence calibration
        calibration_metrics = self._compute_calibration_metrics(predictions, true_labels, confidences)
        
        return {
            'overall_metrics': overall_metrics,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': conf_matrix.tolist(),
            'calibration_metrics': calibration_metrics,
            'classification_report': classification_report(true_labels, predictions, output_dict=True)
        }
    
    def _compute_calibration_metrics(self, predictions: List[str], 
                                    true_labels: List[str],
                                    confidences: List[float]) -> Dict:
        """Compute confidence calibration metrics."""
        # Bin confidences and compute accuracy per bin
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        calibration_data = []
        
        for bin_min, bin_max in bins:
            mask = [(c >= bin_min and c < bin_max) for c in confidences]
            if sum(mask) > 0:
                bin_predictions = [p for p, m in zip(predictions, mask) if m]
                bin_true = [t for t, m in zip(true_labels, mask) if m]
                bin_confidences = [c for c, m in zip(confidences, mask) if m]
                
                if bin_true:
                    bin_accuracy = sum(1 for p, t in zip(bin_predictions, bin_true) if p == t) / len(bin_true)
                    calibration_data.append({
                        'confidence_range': f"{bin_min}-{bin_max}",
                        'mean_confidence': float(np.mean(bin_confidences)),
                        'accuracy': float(bin_accuracy),
                        'count': len(bin_true),
                        'calibration_error': abs(float(np.mean(bin_confidences)) - float(bin_accuracy))
                    })
        
        # Expected Calibration Error (ECE)
        if calibration_data:
            ece = sum(d['calibration_error'] * d['count'] for d in calibration_data) / sum(d['count'] for d in calibration_data)
        else:
            ece = 0.0
        
        return {
            'calibration_data': calibration_data,
            'expected_calibration_error': float(ece)
        }
    
    def compute_temporal_validation_metrics(self, predictions: List[str], 
                                          timestamps: List[str]) -> Dict:
        """Compute temporal validation metrics."""
        if len(predictions) != len(timestamps):
            return {'error': 'Predictions and timestamps must have same length'}
        
        # Convert timestamps to datetime objects
        times = [datetime.fromisoformat(ts) for ts in timestamps]
        
        # Compute prediction stability
        state_changes = sum(1 for i in range(1, len(predictions)) if predictions[i] != predictions[i-1])
        stability_score = 1.0 - (state_changes / len(predictions))
        
        # Compute prediction distribution over time
        time_bins = {}
        for pred, time in zip(predictions, times):
            hour = time.hour
            if hour not in time_bins:
                time_bins[hour] = {'low': 0, 'medium': 0, 'high': 0}
            time_bins[hour][pred] += 1
        
        return {
            'state_changes': state_changes,
            'stability_score': float(stability_score),
            'temporal_distribution': time_bins,
            'prediction_duration': {
                'low': predictions.count('low'),
                'medium': predictions.count('medium'),
                'high': predictions.count('high')
            }
        }
    
    def run_validation_suite(self, model, test_data: pd.DataFrame, 
                           test_labels: List[str]) -> Dict:
        """Run comprehensive validation suite."""
        # Run predictions
        predictions = []
        confidences = []
        
        for _, row in test_data.iterrows():
            # Extract features (placeholder implementation)
            eeg_features = row.iloc[:78].values  # Assuming first 78 are EEG features
            eye_features = row.iloc[78:].values   # Remaining are eye features
            
            # Make prediction (placeholder)
            # In real implementation, this would use the actual model
            prediction = np.random.choice(['low', 'medium', 'high'])
            confidence = np.random.uniform(0.6, 0.95)
            
            predictions.append(prediction)
            confidences.append(confidence)
        
        # Compute metrics
        classification_metrics = self.compute_classification_metrics(predictions, test_labels, confidences)
        temporal_metrics = self.compute_temporal_validation_metrics(predictions, [datetime.now().isoformat() for _ in predictions])
        
        validation_result = {
            'timestamp': datetime.now().isoformat(),
            'classification_metrics': classification_metrics,
            'temporal_metrics': temporal_metrics,
            'overall_status': 'passed' if classification_metrics['overall_metrics']['accuracy'] > 0.7 else 'failed'
        }
        
        self.validation_history.append(validation_result)
        return validation_result


def run_all_tests():
    """Run all automated tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(AutomatedTestSuite)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


def run_performance_benchmarks():
    """Run performance benchmarks."""
    benchmark_runner = BenchmarkRunner()
    
    print("Running feature extraction benchmark...")
    feature_results = benchmark_runner.run_feature_extraction_benchmark(n_iterations=50)
    print(f"Feature extraction: {feature_results['mean_time']:.4f}s per iteration")
    
    print("Running end-to-end benchmark...")
    e2e_results = benchmark_runner.run_end_to_end_benchmark(n_iterations=20)
    if 'error' not in e2e_results:
        print(f"End-to-end: {e2e_results['mean_time']:.4f}s per iteration")
    else:
        print(f"End-to-end benchmark skipped: {e2e_results['error']}")
    
    # Save results
    benchmark_runner.save_benchmark_results()
    print("Benchmark results saved")
    
    return benchmark_runner