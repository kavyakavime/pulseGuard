# ml/features.py

import numpy as np
import pandas as pd
from signal_processing import PPGSignalProcessor

class FeatureExtractor:
    """
    Extract features from PPG signals
    - Heart Rate (BPM)
    - HRV (RMSSD)
    - Strain Index
    """
    
    def __init__(self, sample_rate=100):
        self.fs = sample_rate
    
    def compute_hr(self, peaks):
        """Compute heart rate from peaks"""
        if len(peaks) < 2:
            return 0
        
        # Inter-beat intervals in seconds
        ibi_sec = np.diff(peaks) / self.fs
        
        # HR in BPM
        hr_bpm = 60 / np.mean(ibi_sec)
        
        return hr_bpm
    
    def compute_hrv(self, peaks):
        """Compute HRV (RMSSD method)"""
        if len(peaks) < 3:
            return 0
        
        # Inter-beat intervals in ms
        ibi_ms = np.diff(peaks) / self.fs * 1000
        
        # Successive differences
        diff_ibi = np.diff(ibi_ms)
        
        # RMSSD
        rmssd = np.sqrt(np.mean(diff_ibi ** 2))
        
        return rmssd
    
    def compute_strain_index(self, hr, hrv, baseline_hr=70, baseline_hrv=50):
        """
        Strain index (0-1)
        Higher = more strain
        """
        # HR component
        hr_strain = max(0, (hr - baseline_hr) / 50)
        
        # HRV component
        hrv_strain = max(0, (baseline_hrv - hrv) / baseline_hrv)
        
        # Combine
        strain = 0.6 * hr_strain + 0.4 * hrv_strain
        
        return min(1.0, max(0.0, strain))
    
    def extract_windowed_features(self, peaks, window_size=10):
        """
        Extract features over sliding windows
        
        Args:
            peaks: Array of peak indices
            window_size: Number of peaks per window
        
        Returns:
            DataFrame with HR, HRV per window
        """
        features = []
        
        # Slide window
        for i in range(0, len(peaks) - window_size, 5):
            window_peaks = peaks[i:i+window_size]
            
            hr = self.compute_hr(window_peaks)
            hrv = self.compute_hrv(window_peaks)
            strain = self.compute_strain_index(hr, hrv)
            
            features.append({
                'window_idx': i,
                'hr': hr,
                'hrv': hrv,
                'strain': strain
            })
        
        return pd.DataFrame(features)

if __name__ == "__main__":
    processor = PPGSignalProcessor()
    extractor = FeatureExtractor()
    
    try:
        df, filtered, peaks = processor.process_csv('data/mock_baseline.csv')
        features = extractor.extract_windowed_features(peaks)
        
        print("\n✅ Features extracted:")
        print(f"   Mean HR: {features['hr'].mean():.1f} BPM")
        print(f"   Mean HRV: {features['hrv'].mean():.1f} ms")
        print(f"   Mean Strain: {features['strain'].mean():.3f}")
    except FileNotFoundError:
        print("⚠️ Run 'python data/generate_mock_data.py' first!")