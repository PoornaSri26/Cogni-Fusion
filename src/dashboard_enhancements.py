"""
Dashboard Enhancements Module
Provides advanced dashboard features including:
- Real-time alerting system
- Session recording and playback
- User profiles and historical tracking
- Export capabilities (CSV, JSON, PDF reports)
- Advanced visualizations
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from pathlib import Path


class AlertSystem:
    """Real-time alerting system for cognitive load monitoring."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.alerts = []
        self.alert_history = []
        self.config = self._load_config(config_path)
        self.active_alerts = {}
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load alert configuration."""
        default_config = {
            'high_load_threshold': 0.8,
            'sustained_high_load_count': 5,
            'low_load_threshold': 0.2,
            'alert_cooldown': 60,  # seconds
            'break_recommendation_threshold': 0.7,
            'stress_detection_threshold': 0.75
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                default_config.update(custom_config)
        
        return default_config
    
    def check_alerts(self, prediction: str, confidence: float, 
                    epoch_count: int, sustained_high_count: int) -> List[Dict]:
        """Check and generate alerts based on current state."""
        new_alerts = []
        current_time = datetime.now()
        
        # High cognitive load alert
        if prediction == 'high' and confidence > self.config['high_load_threshold']:
            alert = {
                'type': 'high_load',
                'severity': 'warning',
                'message': f'High cognitive load detected (confidence: {confidence:.2f})',
                'timestamp': current_time.isoformat(),
                'confidence': confidence,
                'prediction': prediction
            }
            new_alerts.append(alert)
        
        # Sustained high load alert
        if sustained_high_count >= self.config['sustained_high_load_count']:
            alert = {
                'type': 'sustained_high_load',
                'severity': 'critical',
                'message': f'Sustained high cognitive load for {sustained_high_count} epochs',
                'timestamp': current_time.isoformat(),
                'sustained_count': sustained_high_count,
                'recommendation': 'Consider taking a break'
            }
            new_alerts.append(alert)
        
        # Break recommendation
        if prediction == 'high' and confidence > self.config['break_recommendation_threshold']:
            alert = {
                'type': 'break_recommendation',
                'severity': 'info',
                'message': 'Break recommended due to sustained high cognitive load',
                'timestamp': current_time.isoformat(),
                'confidence': confidence
            }
            new_alerts.append(alert)
        
        # Stress detection alert
        if prediction == 'high' and confidence > self.config['stress_detection_threshold']:
            alert = {
                'type': 'stress_detected',
                'severity': 'warning',
                'message': 'Potential stress state detected',
                'timestamp': current_time.isoformat(),
                'confidence': confidence
            }
            new_alerts.append(alert)
        
        # Filter alerts based on cooldown
        filtered_alerts = self._filter_by_cooldown(new_alerts)
        
        # Update active alerts
        for alert in filtered_alerts:
            self.active_alerts[alert['type']] = alert
            self.alert_history.append(alert)
        
        return filtered_alerts
    
    def _filter_by_cooldown(self, new_alerts: List[Dict]) -> List[Dict]:
        """Filter alerts based on cooldown period."""
        filtered = []
        current_time = datetime.now()
        
        for alert in new_alerts:
            alert_type = alert['type']
            
            if alert_type in self.active_alerts:
                last_alert = self.active_alerts[alert_type]
                last_time = datetime.fromisoformat(last_alert['timestamp'])
                time_diff = (current_time - last_time).total_seconds()
                
                if time_diff >= self.config['alert_cooldown']:
                    filtered.append(alert)
            else:
                filtered.append(alert)
        
        return filtered
    
    def get_active_alerts(self) -> List[Dict]:
        """Get currently active alerts."""
        return list(self.active_alerts.values())
    
    def clear_alert(self, alert_type: str):
        """Clear a specific alert."""
        if alert_type in self.active_alerts:
            del self.active_alerts[alert_type]
    
    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """Get alert history."""
        return self.alert_history[-limit:]


class SessionRecorder:
    """Record and manage session data."""
    
    def __init__(self, storage_dir: str = "sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.current_session = None
        self.session_data = []
    
    def start_session(self, user_id: str, session_metadata: Optional[Dict] = None) -> str:
        """Start a new recording session."""
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = {
            'session_id': session_id,
            'user_id': user_id,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'metadata': session_metadata or {},
            'epochs': []
        }
        
        self.session_data = []
        return session_id
    
    def record_epoch(self, epoch_data: Dict):
        """Record a single epoch of data."""
        if self.current_session:
            epoch_record = {
                'timestamp': datetime.now().isoformat(),
                'data': epoch_data
            }
            self.session_data.append(epoch_record)
            self.current_session['epochs'].append(epoch_record)
    
    def end_session(self) -> Dict:
        """End current session and save to disk."""
        if not self.current_session:
            return {}
        
        self.current_session['end_time'] = datetime.now().isoformat()
        self.current_session['epoch_count'] = len(self.session_data)
        
        # Save session to disk
        session_file = self.storage_dir / f"{self.current_session['session_id']}.json"
        with open(session_file, 'w') as f:
            json.dump(self.current_session, f, indent=2)
        
        # Also save as CSV for easier analysis
        if self.session_data:
            df = pd.DataFrame([epoch['data'] for epoch in self.session_data])
            csv_file = self.storage_dir / f"{self.current_session['session_id']}.csv"
            df.to_csv(csv_file, index=False)
        
        session_summary = self.current_session.copy()
        self.current_session = None
        self.session_data = []
        
        return session_summary
    
    def get_session_list(self, user_id: Optional[str] = None) -> List[Dict]:
        """Get list of available sessions."""
        sessions = []
        
        for session_file in self.storage_dir.glob("*.json"):
            with open(session_file, 'r') as f:
                session = json.load(f)
                
                if user_id is None or session['user_id'] == user_id:
                    sessions.append({
                        'session_id': session['session_id'],
                        'user_id': session['user_id'],
                        'start_time': session['start_time'],
                        'end_time': session.get('end_time'),
                        'epoch_count': session.get('epoch_count', 0),
                        'metadata': session.get('metadata', {})
                    })
        
        return sorted(sessions, key=lambda x: x['start_time'], reverse=True)
    
    def load_session(self, session_id: str) -> Dict:
        """Load a specific session."""
        session_file = self.storage_dir / f"{session_id}.json"
        
        if session_file.exists():
            with open(session_file, 'r') as f:
                return json.load(f)
        
        return {}


class UserProfileManager:
    """Manage user profiles and historical data."""
    
    def __init__(self, storage_dir: str = "user_profiles"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
    
    def create_profile(self, user_id: str, metadata: Optional[Dict] = None) -> Dict:
        """Create a new user profile."""
        profile = {
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {},
            'baselines': {},
            'preferences': {},
            'session_history': []
        }
        
        profile_file = self.storage_dir / f"{user_id}.json"
        with open(profile_file, 'w') as f:
            json.dump(profile, f, indent=2)
        
        return profile
    
    def get_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile."""
        profile_file = self.storage_dir / f"{user_id}.json"
        
        if profile_file.exists():
            with open(profile_file, 'r') as f:
                return json.load(f)
        
        return None
    
    def update_profile(self, user_id: str, updates: Dict) -> Dict:
        """Update user profile."""
        profile = self.get_profile(user_id)
        
        if profile:
            profile.update(updates)
            profile['last_updated'] = datetime.now().isoformat()
            
            profile_file = self.storage_dir / f"{user_id}.json"
            with open(profile_file, 'w') as f:
                json.dump(profile, f, indent=2)
            
            return profile
        
        return {}
    
    def add_session_to_history(self, user_id: str, session_id: str, session_summary: Dict):
        """Add session to user's history."""
        profile = self.get_profile(user_id)
        
        if profile:
            if 'session_history' not in profile:
                profile['session_history'] = []
            
            profile['session_history'].append({
                'session_id': session_id,
                'timestamp': session_summary.get('start_time'),
                'epoch_count': session_summary.get('epoch_count', 0)
            })
            
            # Keep only last 50 sessions
            if len(profile['session_history']) > 50:
                profile['session_history'] = profile['session_history'][-50:]
            
            self.update_profile(user_id, profile)
    
    def update_baselines(self, user_id: str, baselines: Dict):
        """Update user's baseline metrics."""
        profile = self.get_profile(user_id)
        
        if profile:
            profile['baselines'] = baselines
            self.update_profile(user_id, profile)


class ExportManager:
    """Manage data export in various formats."""
    
    def __init__(self, export_dir: str = "exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
    
    def export_to_csv(self, data: pd.DataFrame, filename: str) -> str:
        """Export data to CSV format."""
        export_path = self.export_dir / f"{filename}.csv"
        data.to_csv(export_path, index=False)
        return str(export_path)
    
    def export_to_json(self, data: Dict, filename: str) -> str:
        """Export data to JSON format."""
        export_path = self.export_dir / f"{filename}.json"
        with open(export_path, 'w') as f:
            json.dump(data, f, indent=2)
        return str(export_path)
    
    def export_to_pdf_report(self, session_data: Dict, filename: str) -> str:
        """Generate PDF report from session data."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            
            export_path = self.export_dir / f"{filename}.pdf"
            doc = SimpleDocTemplate(str(export_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title = Paragraph(f"Cognitive Load Report - {session_data.get('session_id', 'Unknown')}", 
                            styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Session info
            info_data = [
                ['User ID', session_data.get('user_id', 'N/A')],
                ['Start Time', session_data.get('start_time', 'N/A')],
                ['End Time', session_data.get('end_time', 'N/A')],
                ['Total Epochs', str(session_data.get('epoch_count', 0))]
            ]
            
            info_table = Table(info_data, colWidths=[2*72, 4*72])
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(info_table)
            story.append(Spacer(1, 12))
            
            # Build PDF
            doc.build(story)
            return str(export_path)
            
        except ImportError:
            # Fallback if reportlab not available
            print("PDF export requires reportlab. Exporting to JSON instead.")
            return self.export_to_json(session_data, filename)
    
    def export_session_summary(self, session_data: Dict, formats: List[str] = ['csv', 'json']) -> Dict[str, str]:
        """Export session in multiple formats."""
        exported_files = {}
        
        if 'csv' in formats and session_data.get('epochs'):
            df = pd.DataFrame([epoch['data'] for epoch in session_data['epochs']])
            csv_path = self.export_to_csv(df, f"{session_data['session_id']}_summary")
            exported_files['csv'] = csv_path
        
        if 'json' in formats:
            json_path = self.export_to_json(session_data, f"{session_data['session_id']}_summary")
            exported_files['json'] = json_path
        
        if 'pdf' in formats:
            pdf_path = self.export_to_pdf_report(session_data, f"{session_data['session_id']}_report")
            exported_files['pdf'] = pdf_path
        
        return exported_files


class AdvancedVisualization:
    """Advanced visualization tools for dashboard."""
    
    @staticmethod
    def create_state_transition_matrix(predictions: List[str]) -> pd.DataFrame:
        """Create state transition matrix."""
        states = ['low', 'medium', 'high']
        matrix = pd.DataFrame(0, index=states, columns=states)
        
        for i in range(len(predictions) - 1):
            from_state = predictions[i]
            to_state = predictions[i + 1]
            if from_state in states and to_state in states:
                matrix.loc[from_state, to_state] += 1
        
        # Normalize by row
        row_sums = matrix.sum(axis=1)
        matrix = matrix.div(row_sums, axis=0).fillna(0)
        
        return matrix
    
    @staticmethod
    def compute_performance_metrics(predictions: List[str], true_labels: List[str]) -> Dict:
        """Compute comprehensive performance metrics."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': {},
            'recall': {},
            'f1': {},
            'confusion_matrix': confusion_matrix(true_labels, predictions).tolist()
        }
        
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        
        for i, state in enumerate(['low', 'medium', 'high']):
            metrics['precision'][state] = float(precision[i])
            metrics['recall'][state] = float(recall[i])
            metrics['f1'][state] = float(f1[i])
        
        return metrics
    
    @staticmethod
    def create_time_series_data(data: List[Dict], keys: List[str]) -> Dict[str, List]:
        """Create time series data for visualization."""
        time_series = {key: [] for key in keys}
        
        for epoch in data:
            for key in keys:
                time_series[key].append(epoch.get(key, 0))
        
        return time_series