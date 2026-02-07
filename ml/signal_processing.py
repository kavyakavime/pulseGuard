# ml/signal.py

import numpy as np
from scipy import signal
import pandas as pd

class PPGSignalProcessor:
    """
    Signal processing for PPG data
    - DC removal
    - Band-pass filtering (0.5-5 Hz)
    - Peak detection
    """
    
    def __init__(self, sample_rate=100):
        self.fs = sample_rate
        
        # Design band-pass filter for heart rate (0.5-5 Hz)
        nyquist = self.fs / 2
        low = 0.5 / nyquist
        high = 5.0 / nyquist
        self.b, self.a = signal.butter(4, [low, high], btype='band')
    
    def remove_dc(self, data):
        """Remove DC offset (baseline)"""
        return data - np.mean(data)
    
    def filter_signal(self, data):
        """Apply band-pass filter"""
        if len(data) < 20:
            return data
        
        # Remove DC first
        data_no_dc = self.remove_dc(data)
        
        # Apply filter
        try:
            filtered = signal.filtfilt(self.b, self.a, data_no_dc)
        except:
            filtered = data_no_dc
        
        return filtered
    
    def detect_peaks(self, data):
        """Detect heartbeat peaks"""
        if len(data) < 50:
            return np.array([])
        
        # Find peaks
        peaks, _ = signal.find_peaks(
            data,
            distance=int(self.fs * 0.4),  # Min 400ms between beats
            prominence=np.std(data) * 0.3,
            height=np.mean(data)
        )
        
        return peaks
    
    def process_csv(self, csv_path):
        """
        Process CSV file
        
        Returns:
            df: Original dataframe
            filtered_ir: Filtered IR signal
            peaks: Peak indices
        """
        df = pd.read_csv(csv_path)
        
        # Get IR signal
        ir_signal = df['ir'].values
        
        # Filter
        filtered_ir = self.filter_signal(ir_signal)
        
        # Detect peaks
        peaks = self.detect_peaks(filtered_ir)
        
        print(f"✅ Processed {csv_path}")
        print(f"   Samples: {len(df)}")
        print(f"   Peaks: {len(peaks)}")
        print(f"   Duration: {len(df) / self.fs:.1f}s")
        
        return df, filtered_ir, peaks

if __name__ == "__main__":
    processor = PPGSignalProcessor()
    
    # Test on mock data
    try:
        df, filtered, peaks = processor.process_csv('data/mock_baseline.csv')
        print("\n✅ Signal processing working!")
    except FileNotFoundError:
        print("⚠️ Run 'python data/generate_mock_data.py' first!")