"""
Real-time Features Module
Provides real-time cognitive monitoring features including:
- Break recommendations based on sustained high load
- Performance metrics dashboard
- Stress detection and mitigation prompts
- Adaptive task difficulty suggestions
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import json


class BreakRecommender:
    """Intelligent break recommendation system."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'high_load_threshold': 0.7,
            'sustained_threshold': 5,  # epochs
            'break_duration': 300,  # seconds (5 minutes)
            'minimum_break_interval': 900,  # seconds (15 minutes)
            'recovery_threshold': 0.3
        }
        
        self.last_break_time = None
        self.sustained_high_count = 0
        self.break_history = []
    
    def evaluate_break_need(self, prediction: str, confidence: float, 
                          current_time: Optional[datetime] = None) -> Dict:
        """Evaluate if a break is needed."""
        current_time = current_time or datetime.now()
        
        # Check minimum interval since last break
        if self.last_break_time:
            time_since_break = (current_time - self.last_break_time).total_seconds()
            if time_since_break < self.config['minimum_break_interval']:
                return {
                    'recommend_break': False,
                    'reason': 'Insufficient time since last break',
                    'time_until_next_break': int(self.config['minimum_break_interval'] - time_since_break)
                }
        
        # Track sustained high load
        if prediction == 'high' and confidence > self.config['high_load_threshold']:
            self.sustained_high_count += 1
        else:
            self.sustained_high_count = 0
        
        # Evaluate break recommendation
        if self.sustained_high_count >= self.config['sustained_threshold']:
            return {
                'recommend_break': True,
                'reason': f'Sustained high cognitive load for {self.sustained_high_count} epochs',
                'sustained_count': self.sustained_high_count,
                'suggested_duration': self.config['break_duration'],
                'urgency': 'high' if self.sustained_high_count > self.config['sustained_threshold'] * 1.5 else 'medium'
            }
        
        return {
            'recommend_break': False,
            'reason': 'No sustained high load detected',
            'sustained_count': self.sustained_high_count
        }
    
    def record_break(self, duration: int, current_time: Optional[datetime] = None):
        """Record that a break was taken."""
        current_time = current_time or datetime.now()
        self.last_break_time = current_time
        self.sustained_high_count = 0
        
        self.break_history.append({
            'timestamp': current_time.isoformat(),
            'duration': duration,
            'sustained_count_before': self.sustained_high_count
        })
    
    def get_break_statistics(self) -> Dict:
        """Get break statistics."""
        if not self.break_history:
            return {
                'total_breaks': 0,
                'average_duration': 0,
                'total_break_time': 0
            }
        
        durations = [b['duration'] for b in self.break_history]
        
        return {
            'total_breaks': len(self.break_history),
            'average_duration': int(np.mean(durations)),
            'total_break_time': int(sum(durations)),
            'last_break': self.break_history[-1]['timestamp'] if self.break_history else None
        }


class PerformanceMonitor:
    """Real-time performance metrics monitoring."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size * 2)
        self.current_window = deque(maxlen=window_size)
        
        self.performance_metrics = {
            'prediction_accuracy': 0.0,
            'confidence_score': 0.0,
            'stability_score': 0.0,
            'eeg_trust_average': 0.0,
            'eye_trust_average': 0.0,
            'response_time': 0.0
        }
    
    def update_metrics(self, prediction: str, confidence: float, 
                      eeg_trust: float, eye_trust: float, 
                      processing_time: float, timestamp: Optional[datetime] = None):
        """Update performance metrics with new epoch."""
        timestamp = timestamp or datetime.now()
        
        metrics = {
            'timestamp': timestamp.isoformat(),
            'prediction': prediction,
            'confidence': confidence,
            'eeg_trust': eeg_trust,
            'eye_trust': eye_trust,
            'processing_time': processing_time
        }
        
        self.current_window.append(metrics)
        self.metrics_history.append(metrics)
        
        # Update aggregated metrics
        self._compute_aggregated_metrics()
    
    def _compute_aggregated_metrics(self):
        """Compute aggregated performance metrics."""
        if not self.current_window:
            return
        
        # Average confidence
        confidences = [m['confidence'] for m in self.current_window]
        self.performance_metrics['confidence_score'] = float(np.mean(confidences))
        
        # Average trust weights
        eeg_trusts = [m['eeg_trust'] for m in self.current_window]
        eye_trusts = [m['eye_trust'] for m in self.current_window]
        self.performance_metrics['eeg_trust_average'] = float(np.mean(eeg_trusts))
        self.performance_metrics['eye_trust_average'] = float(np.mean(eye_trusts))
        
        # Average processing time
        processing_times = [m['processing_time'] for m in self.current_window]
        self.performance_metrics['response_time'] = float(np.mean(processing_times))
        
        # Stability score (how consistent predictions are)
        predictions = [m['prediction'] for m in self.current_window]
        if len(predictions) > 1:
            changes = sum(1 for i in range(len(predictions)-1) if predictions[i] != predictions[i+1])
            stability = 1.0 - (changes / len(predictions))
            self.performance_metrics['stability_score'] = float(stability)
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary."""
        summary = self.performance_metrics.copy()
        summary['window_size'] = len(self.current_window)
        summary['total_epochs_processed'] = len(self.metrics_history)
        
        return summary
    
    def detect_performance_degradation(self, threshold: float = 0.7) -> Dict:
        """Detect if performance is degrading."""
        if len(self.current_window) < self.window_size:
            return {'degraded': False, 'reason': 'Insufficient data'}
        
        # Compare recent performance to historical average
        recent_confidence = np.mean([m['confidence'] for m in list(self.current_window)[-10:]])
        historical_confidence = np.mean([m['confidence'] for m in self.metrics_history])
        
        if recent_confidence < historical_confidence * threshold:
            return {
                'degraded': True,
                'reason': 'Confidence score dropped below threshold',
                'recent_confidence': float(recent_confidence),
                'historical_confidence': float(historical_confidence)
            }
        
        return {'degraded': False}


class StressDetector:
    """Advanced stress detection and mitigation system."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'stress_threshold': 0.75,
            'recovery_threshold': 0.4,
            'stress_indicators': {
                'high_confidence_high_load': 0.8,
                'rapid_state_changes': 3,  # changes per minute
                'sustained_high_eeg_trust': 0.7
            }
        }
        
        self.stress_history = []
        self.current_stress_level = 0.0
        self.state_change_count = 0
        self.last_state = None
        self.state_change_timestamps = deque(maxlen=60)
    
    def evaluate_stress(self, prediction: str, confidence: float, 
                       eeg_trust: float, timestamp: Optional[datetime] = None) -> Dict:
        """Evaluate current stress level."""
        timestamp = timestamp or datetime.now()
        
        stress_indicators = {
            'high_confidence_high_load': 0.0,
            'rapid_state_changes': 0.0,
            'sustained_high_eeg_trust': 0.0
        }
        
        # Indicator 1: High confidence high load
        if prediction == 'high' and confidence > self.config['stress_indicators']['high_confidence_high_load']:
            stress_indicators['high_confidence_high_load'] = confidence
        
        # Indicator 2: Rapid state changes
        if self.last_state and self.last_state != prediction:
            self.state_change_count += 1
            self.state_change_timestamps.append(timestamp)
        
        self.last_state = prediction
        
        # Calculate state change rate (changes per minute)
        recent_changes = [ts for ts in self.state_change_timestamps 
                         if (timestamp - ts).total_seconds() < 60]
        change_rate = len(recent_changes) / 60.0  # changes per second
        
        if change_rate > self.config['stress_indicators']['rapid_state_changes'] / 60.0:
            stress_indicators['rapid_state_changes'] = min(change_rate / (self.config['stress_indicators']['rapid_state_changes'] / 60.0), 1.0)
        
        # Indicator 3: Sustained high EEG trust
        if eeg_trust > self.config['stress_indicators']['sustained_high_eeg_trust']:
            stress_indicators['sustained_high_eeg_trust'] = eeg_trust
        
        # Calculate overall stress level
        stress_level = np.mean(list(stress_indicators.values()))
        self.current_stress_level = stress_level
        
        # Determine stress state
        is_stressed = stress_level > self.config['stress_threshold']
        is_recovering = stress_level < self.config['recovery_threshold']
        
        # Generate mitigation suggestions
        mitigation_suggestions = []
        if is_stressed:
            if stress_indicators['high_confidence_high_load'] > 0.6:
                mitigation_suggestions.append("Reduce task complexity")
            if stress_indicators['rapid_state_changes'] > 0.5:
                mitigation_suggestions.append("Focus on single task")
            if stress_indicators['sustained_high_eeg_trust'] > 0.6:
                mitigation_suggestions.append("Take a short breathing exercise")
        
        stress_assessment = {
            'stress_level': float(stress_level),
            'is_stressed': is_stressed,
            'is_recovering': is_recovering,
            'indicators': stress_indicators,
            'mitigation_suggestions': mitigation_suggestions,
            'timestamp': timestamp.isoformat()
        }
        
        self.stress_history.append(stress_assessment)
        
        return stress_assessment
    
    def get_stress_trend(self, window_minutes: int = 5) -> Dict:
        """Analyze stress trend over time window."""
        if not self.stress_history:
            return {'trend': 'unknown', 'average_stress': 0.0}
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_stress = [s for s in self.stress_history 
                        if datetime.fromisoformat(s['timestamp']) > cutoff_time]
        
        if not recent_stress:
            return {'trend': 'unknown', 'average_stress': 0.0}
        
        stress_levels = [s['stress_level'] for s in recent_stress]
        avg_stress = np.mean(stress_levels)
        
        # Determine trend
        if len(stress_levels) >= 3:
            recent_avg = np.mean(stress_levels[-3:])
            earlier_avg = np.mean(stress_levels[:-3])
            
            if recent_avg > earlier_avg * 1.1:
                trend = 'increasing'
            elif recent_avg < earlier_avg * 0.9:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'average_stress': float(avg_stress),
            'peak_stress': float(np.max(stress_levels)),
            'min_stress': float(np.min(stress_levels)),
            'data_points': len(recent_stress)
        }


class AdaptiveDifficultyManager:
    """Suggest adaptive task difficulty based on cognitive state."""
    
    def __init__(self, difficulty_levels: List[str] = None):
        self.difficulty_levels = difficulty_levels or ['easy', 'medium', 'hard', 'expert']
        self.current_difficulty = 'medium'
        self.difficulty_history = []
    
    def recommend_difficulty(self, prediction: str, confidence: float, 
                            stress_level: float) -> Dict:
        """Recommend task difficulty based on cognitive state."""
        recommendation = {
            'current_difficulty': self.current_difficulty,
            'recommended_difficulty': self.current_difficulty,
            'reason': 'No change needed',
            'confidence_adjustment': 0.0
        }
        
        # Logic for difficulty adjustment
        if prediction == 'low' and confidence > 0.7 and stress_level < 0.3:
            # User is under-stimulated, increase difficulty
            current_idx = self.difficulty_levels.index(self.current_difficulty)
            if current_idx < len(self.difficulty_levels) - 1:
                new_difficulty = self.difficulty_levels[current_idx + 1]
                recommendation['recommended_difficulty'] = new_difficulty
                recommendation['reason'] = 'User appears under-stimulated, increase difficulty'
                recommendation['confidence_adjustment'] = 0.1
        
        elif prediction == 'high' and confidence > 0.7 or stress_level > 0.7:
            # User is over-stimulated or stressed, decrease difficulty
            current_idx = self.difficulty_levels.index(self.current_difficulty)
            if current_idx > 0:
                new_difficulty = self.difficulty_levels[current_idx - 1]
                recommendation['recommended_difficulty'] = new_difficulty
                recommendation['reason'] = 'User appears over-stimulated or stressed, decrease difficulty'
                recommendation['confidence_adjustment'] = -0.1
        
        self.difficulty_history.append({
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'confidence': confidence,
            'stress_level': stress_level,
            'recommended_difficulty': recommendation['recommended_difficulty']
        })
        
        return recommendation
    
    def apply_difficulty_change(self, new_difficulty: str):
        """Apply the recommended difficulty change."""
        if new_difficulty in self.difficulty_levels:
            self.current_difficulty = new_difficulty
            return True
        return False
    
    def get_difficulty_statistics(self) -> Dict:
        """Get statistics about difficulty changes."""
        if not self.difficulty_history:
            return {'total_changes': 0, 'current_difficulty': self.current_difficulty}
        
        difficulty_counts = {}
        for entry in self.difficulty_history:
            diff = entry['recommended_difficulty']
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        return {
            'total_changes': len(self.difficulty_history),
            'current_difficulty': self.current_difficulty,
            'difficulty_distribution': difficulty_counts,
            'most_common_difficulty': max(difficulty_counts, key=difficulty_counts.get) if difficulty_counts else self.current_difficulty
        }


class RealTimeFeaturesManager:
    """Main manager for all real-time features."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.break_recommender = BreakRecommender(config.get('break_recommender') if config else None)
        self.performance_monitor = PerformanceMonitor()
        self.stress_detector = StressDetector(config.get('stress_detector') if config else None)
        self.difficulty_manager = AdaptiveDifficultyManager()
        
        self.feature_history = []
    
    def process_epoch(self, prediction: str, confidence: float, 
                     eeg_trust: float, eye_trust: float, 
                     processing_time: float) -> Dict:
        """Process a single epoch through all real-time features."""
        current_time = datetime.now()
        
        # Update performance monitor
        self.performance_monitor.update_metrics(
            prediction, confidence, eeg_trust, eye_trust, processing_time, current_time
        )
        
        # Evaluate break need
        break_recommendation = self.break_recommender.evaluate_break_need(
            prediction, confidence, current_time
        )
        
        # Evaluate stress
        stress_assessment = self.stress_detector.evaluate_stress(
            prediction, confidence, eeg_trust, current_time
        )
        
        # Recommend difficulty
        difficulty_recommendation = self.difficulty_manager.recommend_difficulty(
            prediction, confidence, stress_assessment['stress_level']
        )
        
        # Combine all features
        feature_summary = {
            'timestamp': current_time.isoformat(),
            'prediction': prediction,
            'confidence': confidence,
            'break_recommendation': break_recommendation,
            'performance_metrics': self.performance_monitor.get_performance_summary(),
            'stress_assessment': stress_assessment,
            'difficulty_recommendation': difficulty_recommendation
        }
        
        self.feature_history.append(feature_summary)
        
        return feature_summary
    
    def get_comprehensive_summary(self) -> Dict:
        """Get comprehensive summary of all real-time features."""
        return {
            'break_statistics': self.break_recommender.get_break_statistics(),
            'performance_summary': self.performance_monitor.get_performance_summary(),
            'stress_trend': self.stress_detector.get_stress_trend(),
            'difficulty_statistics': self.difficulty_manager.get_difficulty_statistics(),
            'total_epochs_processed': len(self.feature_history)
        }
    
    def record_break_taken(self, duration: int):
        """Record that a break was taken."""
        self.break_recommender.record_break(duration)
    
    def apply_difficulty_change(self, new_difficulty: str):
        """Apply difficulty change."""
        return self.difficulty_manager.apply_difficulty_change(new_difficulty)