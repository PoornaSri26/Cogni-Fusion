"""
Data Management Module
Provides comprehensive data management including:
- Comprehensive data logging system
- Session management and organization
- Statistical analysis tools
- Data quality monitoring
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from collections import defaultdict
import sqlite3


def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class DataLogger:
    """Comprehensive data logging system."""
    
    def __init__(self, log_dir: str = "logs", db_path: str = "data/logs.db"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.db_path = db_path
        
        # Initialize database
        self._init_database()
        
        # Setup logging
        self._setup_logging()
    
    def _init_database(self):
        """Initialize SQLite database for structured logging."""
        Path(self.db_path).parent.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS epochs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                eeg_trust REAL NOT NULL,
                eye_trust REAL NOT NULL,
                processing_time REAL NOT NULL,
                features_json TEXT,
                metadata_json TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                epoch_count INTEGER DEFAULT 0,
                metadata_json TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                session_id TEXT,
                metadata_json TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _setup_logging(self):
        """Setup file logging."""
        log_file = self.log_dir / f"system_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('DataLogger')
    
    def log_epoch(self, epoch_data: Dict, session_id: Optional[str] = None):
        """Log epoch data to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO epochs 
                (timestamp, prediction, confidence, eeg_trust, eye_trust, processing_time, features_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                epoch_data.get('timestamp', datetime.now().isoformat()),
                epoch_data.get('prediction'),
                epoch_data.get('confidence'),
                epoch_data.get('eeg_trust'),
                epoch_data.get('eye_trust'),
                epoch_data.get('processing_time'),
                json.dumps(convert_numpy_types(epoch_data.get('features', {}))),
                json.dumps(convert_numpy_types({'session_id': session_id}))
            ))
            
            conn.commit()
            self.logger.info(f"Logged epoch: {epoch_data.get('prediction')} with confidence {epoch_data.get('confidence'):.2f}")
            
        except Exception as e:
            self.logger.error(f"Error logging epoch: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def log_session(self, session_data: Dict):
        """Log session data to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO sessions 
                (session_id, user_id, start_time, end_time, epoch_count, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                session_data.get('session_id'),
                session_data.get('user_id'),
                session_data.get('start_time'),
                session_data.get('end_time'),
                session_data.get('epoch_count'),
                json.dumps(convert_numpy_types(session_data.get('metadata', {})))
            ))
            
            conn.commit()
            self.logger.info(f"Logged session: {session_data.get('session_id')}")
            
        except Exception as e:
            self.logger.error(f"Error logging session: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def log_alert(self, alert_data: Dict, session_id: Optional[str] = None):
        """Log alert data to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO alerts 
                (timestamp, alert_type, severity, message, session_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                alert_data.get('timestamp', datetime.now().isoformat()),
                alert_data.get('type'),
                alert_data.get('severity'),
                alert_data.get('message'),
                session_id,
                json.dumps(convert_numpy_types(alert_data))
            ))
            
            conn.commit()
            self.logger.info(f"Logged alert: {alert_data.get('type')} - {alert_data.get('message')}")
            
        except Exception as e:
            self.logger.error(f"Error logging alert: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def query_epochs(self, session_id: Optional[str] = None, 
                    start_time: Optional[str] = None, 
                    end_time: Optional[str] = None,
                    limit: int = 1000) -> pd.DataFrame:
        """Query epoch data from database."""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM epochs WHERE 1=1"
        params = []
        
        if session_id:
            query += " AND metadata_json LIKE ?"
            params.append(f'%"session_id": "{session_id}"%')
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        try:
            df = pd.read_sql_query(query, conn, params=params)
            return df
        except Exception as e:
            self.logger.error(f"Error querying epochs: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM epochs")
            epoch_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM sessions")
            session_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM alerts")
            alert_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(confidence) FROM epochs")
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            return {
                'total_epochs': epoch_count,
                'total_sessions': session_count,
                'total_alerts': alert_count,
                'average_confidence': avg_confidence
            }
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
        finally:
            conn.close()


class SessionManager:
    """Advanced session management and organization."""
    
    def __init__(self, storage_dir: str = "sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.active_sessions = {}
    
    def create_session(self, user_id: str, metadata: Optional[Dict] = None) -> str:
        """Create a new session."""
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        session = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'status': 'active',
            'metadata': metadata or {},
            'epoch_count': 0,
            'epoch_data': []
        }
        
        self.active_sessions[session_id] = session
        return session_id
    
    def add_epoch_to_session(self, session_id: str, epoch_data: Dict):
        """Add epoch data to active session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['epoch_data'].append(epoch_data)
            self.active_sessions[session_id]['epoch_count'] += 1
            self.active_sessions[session_id]['last_updated'] = datetime.now().isoformat()
    
    def close_session(self, session_id: str) -> Dict:
        """Close and save session."""
        if session_id not in self.active_sessions:
            return {}
        
        session = self.active_sessions[session_id]
        session['status'] = 'completed'
        session['completed_at'] = datetime.now().isoformat()
        
        # Save to disk
        session_file = self.storage_dir / f"{session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(convert_numpy_types(session), f, indent=2)
        
        # Also save epoch data as CSV
        if session['epoch_data']:
            df = pd.DataFrame(session['epoch_data'])
            csv_file = self.storage_dir / f"{session_id}_epochs.csv"
            df.to_csv(csv_file, index=False)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data."""
        # Check active sessions first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Check stored sessions
        session_file = self.storage_dir / f"{session_id}.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                return json.load(f)
        
        return None
    
    def list_sessions(self, user_id: Optional[str] = None, 
                     status: Optional[str] = None) -> List[Dict]:
        """List sessions with optional filtering."""
        sessions = []
        
        # Add active sessions
        for session_id, session in self.active_sessions.items():
            if user_id and session['user_id'] != user_id:
                continue
            if status and session['status'] != status:
                continue
            
            sessions.append({
                'session_id': session_id,
                'user_id': session['user_id'],
                'created_at': session['created_at'],
                'status': session['status'],
                'epoch_count': session['epoch_count']
            })
        
        # Add stored sessions
        for session_file in self.storage_dir.glob("*.json"):
            if session_file.stem.endswith('_epochs'):
                continue  # Skip epoch CSV files
            
            with open(session_file, 'r') as f:
                session = json.load(f)
                
                if user_id and session['user_id'] != user_id:
                    continue
                if status and session['status'] != status:
                    continue
                
                sessions.append({
                    'session_id': session['session_id'],
                    'user_id': session['user_id'],
                    'created_at': session['created_at'],
                    'status': session['status'],
                    'epoch_count': session.get('epoch_count', 0)
                })
        
        return sorted(sessions, key=lambda x: x['created_at'], reverse=True)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session and associated files."""
        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        # Remove files
        session_file = self.storage_dir / f"{session_id}.json"
        csv_file = self.storage_dir / f"{session_id}_epochs.csv"
        
        deleted = False
        if session_file.exists():
            session_file.unlink()
            deleted = True
        
        if csv_file.exists():
            csv_file.unlink()
            deleted = True
        
        return deleted


class StatisticalAnalyzer:
    """Comprehensive statistical analysis tools."""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_distribution(self, data: np.ndarray, feature_name: str) -> Dict:
        """Analyze statistical distribution of data."""
        if len(data) == 0:
            return {}
        
        from scipy import stats
        
        analysis = {
            'feature_name': feature_name,
            'count': len(data),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'median': float(np.median(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data)),
            'normality_test': {
                'statistic': float(stats.normaltest(data).statistic),
                'p_value': float(stats.normaltest(data).pvalue)
            }
        }
        
        return analysis
    
    def compare_groups(self, group1: np.ndarray, group2: np.ndarray, 
                      group1_name: str = "Group 1", group2_name: str = "Group 2") -> Dict:
        """Statistical comparison between two groups."""
        from scipy import stats
        
        comparison = {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'group1_stats': self.analyze_distribution(group1, group1_name),
            'group2_stats': self.analyze_distribution(group2, group2_name),
            't_test': {
                'statistic': float(stats.ttest_ind(group1, group2).statistic),
                'p_value': float(stats.ttest_ind(group1, group2).pvalue)
            },
            'mann_whitney': {
                'statistic': float(stats.mannwhitneyu(group1, group2).statistic),
                'p_value': float(stats.mannwhitneyu(group1, group2).pvalue)
            }
        }
        
        return comparison
    
    def time_series_analysis(self, timestamps: List[str], values: List[float]) -> Dict:
        """Analyze time series data."""
        if len(timestamps) != len(values) or len(values) < 2:
            return {}
        
        # Convert timestamps to datetime objects
        times = [datetime.fromisoformat(ts) for ts in timestamps]
        values = np.array(values)
        
        # Calculate time differences
        time_diffs = [(times[i] - times[i-1]).total_seconds() for i in range(1, len(times))]
        
        # Calculate trends
        from scipy import stats
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        analysis = {
            'trend': {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
            },
            'sampling': {
                'mean_interval': float(np.mean(time_diffs)),
                'std_interval': float(np.std(time_diffs)),
                'total_duration': float((times[-1] - times[0]).total_seconds())
            },
            'volatility': {
                'std': float(np.std(values)),
                'coefficient_of_variation': float(np.std(values) / (np.mean(values) + 1e-8))
            }
        }
        
        return analysis
    
    def correlation_analysis(self, features: pd.DataFrame) -> Dict:
        """Analyze correlations between features."""
        correlation_matrix = features.corr()
        
        # Find high correlations
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.7:  # High correlation threshold
                    high_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': float(corr)
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': high_correlations,
            'strongest_positive': max(high_correlations, key=lambda x: x['correlation']) if high_correlations else None,
            'strongest_negative': min(high_correlations, key=lambda x: x['correlation']) if high_correlations else None
        }
    
    def prediction_accuracy_analysis(self, predictions: List[str], 
                                   true_labels: List[str],
                                   confidences: List[float]) -> Dict:
        """Analyze prediction accuracy with confidence calibration."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        if len(predictions) != len(true_labels):
            return {'error': 'Predictions and true labels must have same length'}
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # Confidence calibration analysis
        confidence_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        calibration_data = []
        
        for bin_min, bin_max in confidence_bins:
            mask = [(c >= bin_min and c < bin_max) for c in confidences]
            if sum(mask) > 0:
                bin_predictions = [p for p, m in zip(predictions, mask) if m]
                bin_true = [t for t, m in zip(true_labels, mask) if m]
                bin_accuracy = accuracy_score(bin_true, bin_predictions)
                
                calibration_data.append({
                    'confidence_range': f"{bin_min}-{bin_max}",
                    'mean_confidence': float(np.mean([c for c, m in zip(confidences, mask) if m])),
                    'accuracy': float(bin_accuracy),
                    'count': sum(mask)
                })
        
        return {
            'overall_accuracy': float(accuracy),
            'precision_by_class': {label: float(p) for label, p in zip(['low', 'medium', 'high'], precision)},
            'recall_by_class': {label: float(r) for label, r in zip(['low', 'medium', 'high'], recall)},
            'f1_by_class': {label: float(f) for label, f in zip(['low', 'medium', 'high'], f1)},
            'confusion_matrix': conf_matrix.tolist(),
            'confidence_calibration': calibration_data
        }


class DataQualityMonitor:
    """Monitor data quality in real-time."""
    
    def __init__(self):
        self.quality_metrics = {
            'missing_data_count': 0,
            'outlier_count': 0,
            'quality_score': 1.0,
            'last_check': None
        }
        self.thresholds = {
            'missing_data_threshold': 0.1,  # 10% missing data
            'outlier_threshold': 3.0,  # 3 standard deviations
            'min_quality_score': 0.7
        }
    
    def check_data_quality(self, features: Dict) -> Dict:
        """Check quality of feature data."""
        quality_report = {
            'is_valid': True,
            'issues': [],
            'quality_score': 1.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Check for missing data
        missing_count = sum(1 for v in features.values() if (v is None) or (isinstance(v, (int, float)) and np.isnan(v)))
        total_features = len(features)
        missing_ratio = missing_count / total_features if total_features > 0 else 0
        
        if missing_ratio > self.thresholds['missing_data_threshold']:
            quality_report['is_valid'] = False
            quality_report['issues'].append(f"High missing data ratio: {missing_ratio:.2%}")
            quality_report['quality_score'] -= 0.3
        
        # Check for outliers
        numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
        if numeric_features:
            values = np.array(list(numeric_features.values()))
            mean = np.mean(values)
            std = np.std(values)
            
            outliers = [k for k, v in numeric_features.items() if abs(v - mean) > self.thresholds['outlier_threshold'] * std]
            
            if outliers:
                quality_report['issues'].append(f"Found {len(outliers)} potential outliers")
                quality_report['quality_score'] -= 0.1 * min(len(outliers) / len(numeric_features), 1.0)
        
        # Check for extreme values
        extreme_values = [k for k, v in numeric_features.items() if abs(v) > 1e6 or (abs(v) < 1e-6 and v != 0)]
        if extreme_values:
            quality_report['issues'].append(f"Found {len(extreme_values)} extreme values")
            quality_report['quality_score'] -= 0.1
        
        # Update metrics
        self.quality_metrics['missing_data_count'] = missing_count
        self.quality_metrics['outlier_count'] = len(outliers) if 'outliers' in locals() else 0
        self.quality_metrics['quality_score'] = max(quality_report['quality_score'], 0.0)
        self.quality_metrics['last_check'] = datetime.now().isoformat()
        
        quality_report['quality_metrics'] = self.quality_metrics
        
        return quality_report