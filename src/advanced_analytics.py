"""
Advanced Analytics Module
Provides comprehensive analysis tools for EEG and eye-tracking data including:
- Statistical analysis tools
- Cross-frequency coupling analysis
- Advanced connectivity measures
- Time-frequency analysis
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy, pearsonr
from typing import Dict, List, Tuple, Optional


class AdvancedAnalytics:
    """Advanced analytics for cognitive state monitoring."""
    
    def __init__(self):
        self.session_history = []
        self.baseline_metrics = {}
    
    def compute_statistical_summary(self, features: pd.DataFrame) -> Dict:
        """Compute comprehensive statistical summary of features."""
        summary = {
            'mean': features.mean().to_dict(),
            'std': features.std().to_dict(),
            'median': features.median().to_dict(),
            'min': features.min().to_dict(),
            'max': features.max().to_dict(),
            'skewness': features.skew().to_dict(),
            'kurtosis': features.kurtosis().to_dict()
        }
        return summary
    
    def compute_cross_frequency_coupling(self, eeg_data: np.ndarray, fs: int = 256) -> Dict:
        """Compute cross-frequency coupling (phase-amplitude coupling)."""
        n_channels = eeg_data.shape[0]
        coupling_metrics = {}
        
        for ch in range(n_channels):
            sig = eeg_data[ch]
            
            # Phase of low frequency (theta: 4-8 Hz)
            theta_phase = self._extract_phase(sig, fs, 4, 8)
            
            # Amplitude of high frequency (gamma: 30-45 Hz)
            gamma_amp = self._extract_amplitude(sig, fs, 30, 45)
            
            # Compute modulation index
            mi = self._compute_modulation_index(theta_phase, gamma_amp)
            coupling_metrics[f'ch{ch}_theta_gamma_coupling'] = mi
            
            # Other frequency pairs
            alpha_phase = self._extract_phase(sig, fs, 8, 13)
            beta_amp = self._extract_amplitude(sig, fs, 13, 30)
            coupling_metrics[f'ch{ch}_alpha_beta_coupling'] = self._compute_modulation_index(alpha_phase, beta_amp)
        
        return coupling_metrics
    
    def _extract_phase(self, sig: np.ndarray, fs: int, fmin: float, fmax: float) -> np.ndarray:
        """Extract phase of signal in frequency band using Hilbert transform."""
        try:
            # Bandpass filter
            b, a = signal.butter(4, [fmin/(fs/2), fmax/(fs/2)], btype='band')
            filtered = signal.filtfilt(b, a, sig)
            
            # Hilbert transform to get instantaneous phase
            analytic = signal.hilbert(filtered)
            phase = np.angle(analytic)
            return phase
        except Exception as e:
            # Fallback: return zeros if hilbert fails
            print(f"Warning: Hilbert transform failed: {e}")
            return np.zeros_like(sig)
    
    def _extract_amplitude(self, sig: np.ndarray, fs: int, fmin: float, fmax: float) -> np.ndarray:
        """Extract amplitude envelope of signal in frequency band."""
        try:
            b, a = signal.butter(4, [fmin/(fs/2), fmax/(fs/2)], btype='band')
            filtered = signal.filtfilt(b, a, sig)
            analytic = signal.hilbert(filtered)
            amplitude = np.abs(analytic)
            return amplitude
        except Exception as e:
            # Fallback: return signal absolute value if hilbert fails
            print(f"Warning: Amplitude extraction failed: {e}")
            return np.abs(sig)
    
    def _compute_modulation_index(self, phase: np.ndarray, amplitude: np.ndarray, n_bins: int = 18) -> float:
        """Compute modulation index for phase-amplitude coupling."""
        # Bin phase
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        binned_amplitude = []
        
        for i in range(n_bins):
            mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
            if np.sum(mask) > 0:
                binned_amplitude.append(np.mean(amplitude[mask]))
            else:
                binned_amplitude.append(0)
        
        binned_amplitude = np.array(binned_amplitude)
        # Normalize
        binned_amplitude = binned_amplitude / (np.sum(binned_amplitude) + 1e-12)
        
        # Compute KL divergence
        uniform = np.ones(n_bins) / n_bins
        kl_div = np.sum(binned_amplitude * np.log((binned_amplitude + 1e-12) / (uniform + 1e-12)))
        
        return float(kl_div / np.log(n_bins))
    
    def compute_time_frequency_analysis(self, eeg_data: np.ndarray, fs: int = 256) -> Dict:
        """Compute time-frequency representation using wavelet transform."""
        n_channels = eeg_data.shape[0]
        tf_features = {}
        
        for ch in range(n_channels):
            sig = eeg_data[ch]
            
            # Compute wavelet power spectrum
            freqs = np.arange(1, 46, 1)  # 1-45 Hz
            powers = []
            
            for freq in freqs:
                # Morlet wavelet
                wavelet = self._morlet_wavelet(len(sig), freq, fs)
                convolved = signal.convolve(sig, wavelet, mode='same')
                power = np.abs(convolved) ** 2
                powers.append(np.mean(power))
            
            # Store band powers
            tf_features[f'ch{ch}_wavelet_delta'] = float(np.mean(powers[0:4]))
            tf_features[f'ch{ch}_wavelet_theta'] = float(np.mean(powers[4:8]))
            tf_features[f'ch{ch}_wavelet_alpha'] = float(np.mean(powers[8:13]))
            tf_features[f'ch{ch}_wavelet_beta'] = float(np.mean(powers[13:30]))
            tf_features[f'ch{ch}_wavelet_gamma'] = float(np.mean(powers[30:45]))
        
        return tf_features
    
    def _morlet_wavelet(self, n_samples: int, freq: float, fs: int, width: float = 7.0) -> np.ndarray:
        """Generate Morlet wavelet."""
        t = np.arange(n_samples) / fs
        wavelet = np.exp(1j * 2 * np.pi * freq * t) * np.exp(-t**2 / (2 * (width / (2 * np.pi * freq))**2))
        return wavelet
    
    def compute_graph_metrics(self, connectivity_matrix: np.ndarray) -> Dict:
        """Compute graph theory metrics from connectivity matrix."""
        # Node degree
        degree = np.sum(connectivity_matrix, axis=1)
        
        # Clustering coefficient
        clustering = []
        for i in range(len(connectivity_matrix)):
            neighbors = np.where(connectivity_matrix[i] > 0.5)[0]
            if len(neighbors) < 2:
                clustering.append(0.0)
                continue
            
            subgraph = connectivity_matrix[np.ix_(neighbors, neighbors)]
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            actual_edges = np.sum(subgraph) / 2  # Remove diagonal
            clustering.append(actual_edges / possible_edges if possible_edges > 0 else 0.0)
        
        # Betweenness centrality (simplified)
        betweenness = np.zeros(len(connectivity_matrix))
        for i in range(len(connectivity_matrix)):
            for j in range(len(connectivity_matrix)):
                if i != j:
                    # Check if i is on shortest path
                    paths = self._find_shortest_paths(connectivity_matrix, j, i)
                    for k in range(len(connectivity_matrix)):
                        if k != i and k != j:
                            paths_to_k = self._find_shortest_paths(connectivity_matrix, j, k)
                            for path in paths_to_k:
                                if i in path:
                                    betweenness[i] += 1
        
        return {
            'mean_degree': float(np.mean(degree)),
            'std_degree': float(np.std(degree)),
            'mean_clustering': float(np.mean(clustering)),
            'std_clustering': float(np.std(clustering)),
            'mean_betweenness': float(np.mean(betweenness)),
            'network_density': float(np.sum(connectivity_matrix) / (len(connectivity_matrix) ** 2))
        }
    
    def _find_shortest_paths(self, matrix: np.ndarray, start: int, end: int) -> List[List[int]]:
        """Find shortest paths using BFS (simplified)."""
        from collections import deque
        
        n = len(matrix)
        visited = [[False] * n for _ in range(n)]
        paths = []
        
        def bfs(current_path: List[int], visited_local: List[bool]):
            if current_path[-1] == end:
                paths.append(current_path.copy())
                return
            
            current = current_path[-1]
            for neighbor in range(n):
                if matrix[current][neighbor] > 0.5 and not visited_local[neighbor]:
                    visited_local[neighbor] = True
                    current_path.append(neighbor)
                    bfs(current_path, visited_local)
                    current_path.pop()
                    visited_local[neighbor] = False
        
        visited[start] = [True] * n
        bfs([start], visited[start])
        return paths
    
    def analyze_temporal_patterns(self, predictions: List[str], timestamps: List[float]) -> Dict:
        """Analyze temporal patterns in predictions."""
        if len(predictions) < 2:
            return {}
        
        # State transition analysis
        transitions = {}
        for i in range(len(predictions) - 1):
            transition = f"{predictions[i]}->{predictions[i+1]}"
            transitions[transition] = transitions.get(transition, 0) + 1
        
        total_transitions = sum(transitions.values())
        transition_probs = {k: v / total_transitions for k, v in transitions.items()}
        
        # State duration analysis
        state_durations = {}
        current_state = predictions[0]
        duration = 1
        
        for i in range(1, len(predictions)):
            if predictions[i] == current_state:
                duration += 1
            else:
                state_durations[current_state] = state_durations.get(current_state, []) + [duration]
                current_state = predictions[i]
                duration = 1
        
        state_durations[current_state] = state_durations.get(current_state, []) + [duration]
        
        # Compute duration statistics
        duration_stats = {}
        for state, durations in state_durations.items():
            duration_stats[state] = {
                'mean': float(np.mean(durations)),
                'std': float(np.std(durations)),
                'max': int(np.max(durations)),
                'min': int(np.min(durations))
            }
        
        return {
            'transition_probabilities': transition_probs,
            'state_duration_stats': duration_stats,
            'total_epochs': len(predictions)
        }
    
    def compute_baselines(self, features: pd.DataFrame, n_samples: int = 30) -> Dict:
        """Compute baseline metrics from initial samples."""
        baseline = features.head(n_samples).mean().to_dict()
        self.baseline_metrics = baseline
        return baseline
    
    def detect_anomalies(self, current_features: Dict, threshold: float = 2.0) -> Dict:
        """Detect anomalies compared to baseline."""
        if not self.baseline_metrics:
            return {}
        
        anomalies = {}
        for key, value in current_features.items():
            if key in self.baseline_metrics:
                baseline = self.baseline_metrics[key]
                if baseline > 0:
                    z_score = abs((value - baseline) / baseline)
                    if z_score > threshold:
                        anomalies[key] = {
                            'current': value,
                            'baseline': baseline,
                            'z_score': z_score
                        }
        
        return anomalies