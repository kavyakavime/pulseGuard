# ml/ingest.py

import serial
import pandas as pd
import time
from collections import deque

class DataIngestor:
    """
    Read PPG data from ESP32 or CSV files
    Supports both live serial and offline replay
    """
    
    def __init__(self, mode='csv', serial_port=None, csv_file=None):
        """
        Args:
            mode: 'serial' or 'csv'
            serial_port: e.g., '/dev/ttyUSB0' or 'COM3'
            csv_file: Path to CSV file
        """
        self.mode = mode
        self.serial_port = serial_port
        self.csv_file = csv_file
        self.buffer = deque(maxlen=6000)  # 60 seconds at 100Hz
        
        if mode == 'serial' and serial_port:
            self.ser = serial.Serial(serial_port, 115200, timeout=1)
            time.sleep(2)  # Wait for connection
            self.ser.readline()  # Skip header
            print(f"✅ Connected to {serial_port}")
        
        elif mode == 'csv' and csv_file:
            self.df = pd.read_csv(csv_file)
            self.current_idx = 0
            print(f"✅ Loaded {csv_file} ({len(self.df)} samples)")
    
    def read_sample(self):
        """
        Read one sample
        
        Returns:
            dict with time, ir, red, bpm, hrv, etc.
        """
        if self.mode == 'serial':
            try:
                line = self.ser.readline().decode('utf-8').strip()
                values = line.split(',')
                
                if len(values) >= 9:
                    sample = {
                        'time': int(values[0]),
                        'ir': int(values[1]),
                        'red': int(values[2]),
                        'bpm': float(values[3]),
                        'hrv': float(values[4]),
                        'spo2': float(values[5]),
                        'fingerDetected': int(values[6]),
                        'hrvReady': int(values[7]),
                        'beatQuality': float(values[8])
                    }
                    self.buffer.append(sample)
                    return sample
            except:
                return None
        
        elif self.mode == 'csv':
            if self.current_idx < len(self.df):
                sample = self.df.iloc[self.current_idx].to_dict()
                self.buffer.append(sample)
                self.current_idx += 1
                return sample
            return None
    
    def get_buffer(self):
        """Get recent samples as DataFrame"""
        if len(self.buffer) == 0:
            return pd.DataFrame()
        return pd.DataFrame(list(self.buffer))

if __name__ == "__main__":
    # Test CSV mode
    ingestor = DataIngestor(mode='csv', csv_file='data/mock_demo.csv')
    
    print("\n📥 Reading first 10 samples:")
    for i in range(10):
        sample = ingestor.read_sample()
        if sample:
            print(f"   IR={sample['ir']}, HR={sample['bpm']:.0f}")
    
    print(f"\n✅ Buffer contains {len(ingestor.get_buffer())} samples")
