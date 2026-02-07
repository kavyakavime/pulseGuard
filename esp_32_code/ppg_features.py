"""
PPG Feature Extraction for Stress/Strain Detection
==================================================
Computes features from HR, HRV, and IBI over rolling windows.

Features:
- HR mean (average over window)
- HRV drop (relative to baseline)
- Pulse irregularity (IBI coefficient of variation)
- Strain Index (0-1) - composite stress metric

Strain increases with: higher HR, lower HRV, more irregular pulse
"""

import numpy as np
from collections import deque


class StrainMonitor:
    """Monitor strain/stress using rolling window features."""
    
    def __init__(self, window_sec=60, baseline_sec=30):
        self.window_sec = window_sec
        self.baseline_sec = baseline_sec
        
        # Rolling window (at 10 Hz vitals: 600 = 60 sec)
        self.hr_window = deque(maxlen=600)
        self.hrv_window = deque(maxlen=600)
        self.ibi_window = deque(maxlen=600)
        self.time_window = deque(maxlen=600)
        
        # Baseline (first 30-60s)
        self.hr_baseline = None
        self.hrv_baseline = None
        self.baseline_computed = False
        self.baseline_start_time = None
    
    def add_sample(self, timestamp, hr, hrv, ibi):
        """Add a sample to the rolling window."""
        if hr > 0:
            self.hr_window.append(hr)
            self.hrv_window.append(hrv)
            self.ibi_window.append(ibi)
            self.time_window.append(timestamp)
            
            if self.baseline_start_time is None:
                self.baseline_start_time = timestamp
    
    def compute_baseline(self):
        """Compute baseline from initial data."""
        if self.baseline_computed or self.baseline_start_time is None:
            return
        
        if not self.time_window:
            return
        
        duration = self.time_window[-1] - self.baseline_start_time
        if duration >= self.baseline_sec:
            hrs = [h for h in self.hr_window if h > 30]  # Filter out bad values
            hrvs = [h for h in self.hrv_window if h > 5]  # HRV > 5ms
            
            if len(hrs) >= 5 and len(hrvs) >= 3:  # Relaxed requirements
                self.hr_baseline = np.median(hrs)
                self.hrv_baseline = np.median(hrvs)
                self.baseline_computed = True
                print(f"\n✅ Baseline set: HR={self.hr_baseline:.0f} BPM, HRV={self.hrv_baseline:.0f}ms")
    
    def get_features(self):
        """
        Compute features from rolling window.
        
        Returns dict:
            - hr_mean: average HR over window
            - hrv_mean: average HRV over window
            - hrv_drop: HRV relative to baseline (0-1, 1=large drop)
            - irregularity: pulse irregularity (CV of IBI)
            - strain_index: composite 0-1 (0=relaxed, 1=strained)
        """
        if not self.baseline_computed:
            self.compute_baseline()
        
        # Always compute features, even if baseline not ready
        if len(self.hr_window) < 3:
            return {
                "hr_mean": 0,
                "hrv_mean": 0,
                "hrv_drop": 0,
                "irregularity": 0,
                "strain_index": 0,
                "baseline_ready": False,
            }
        
        # Get window data (last 30-60s)
        duration = self.time_window[-1] - self.time_window[0] if len(self.time_window) > 1 else 0
        keep_n = len(self.hr_window)
        if duration > self.window_sec:
            # Keep only last window_sec seconds
            cutoff_time = self.time_window[-1] - self.window_sec
            keep_idx = next((i for i, t in enumerate(self.time_window) if t >= cutoff_time), 0)
            hrs = list(self.hr_window)[keep_idx:]
            hrvs = list(self.hrv_window)[keep_idx:]
            ibis = list(self.ibi_window)[keep_idx:]
        else:
            hrs = list(self.hr_window)
            hrvs = list(self.hrv_window)
            ibis = list(self.ibi_window)
        
        # HR mean
        hr_mean = np.mean(hrs) if hrs else 0
        
        # HRV mean
        hrvs_valid = [h for h in hrvs if h > 5]  # HRV must be > 5ms
        hrv_mean = np.mean(hrvs_valid) if hrvs_valid else 0
        
        # HRV drop (relative to baseline)
        hrv_drop = 0
        if self.baseline_computed and self.hrv_baseline > 0 and hrv_mean > 0:
            hrv_drop = max(0, (self.hrv_baseline - hrv_mean) / self.hrv_baseline)
        
        # Pulse irregularity (CV of IBI)
        ibis_valid = [i for i in ibis if 400 < i < 1500]
        irregularity = 0
        if len(ibis_valid) >= 2:
            cv = np.std(ibis_valid) / np.mean(ibis_valid) if np.mean(ibis_valid) > 0 else 0
            irregularity = min(1.0, cv / 0.15)  # CV > 0.15 = very irregular
        
        # Strain Index (0-1)
        # Components: HR elevation, HRV drop, irregularity
        strain = 0
        
        if self.baseline_computed and self.hr_baseline > 0:
            # HR elevation (normalized)
            hr_norm = max(0, (hr_mean - self.hr_baseline) / 30.0)  # +30 BPM = max strain
            hr_norm = min(1.0, hr_norm)
            
            # Combine: 40% HR, 40% HRV drop, 20% irregularity
            strain = (hr_norm * 0.4) + (hrv_drop * 0.4) + (irregularity * 0.2)
            strain = min(1.0, max(0.0, strain))
        else:
            # No baseline yet - use simple heuristic
            # HR 60-80 = relaxed, 80-110 = moderate, 110+ = strained
            # HRV 50+ = good, 30-50 = ok, <30 = stressed
            if hr_mean > 30:
                hr_simple = min(1.0, max(0, (hr_mean - 60) / 50.0))  # 60-110 BPM range
                hrv_simple = 0 if hrv_mean < 10 else (1.0 - min(1.0, hrv_mean / 80.0))  # lower HRV = higher strain
                strain = (hr_simple * 0.5) + (hrv_simple * 0.3) + (irregularity * 0.2)
                strain = min(1.0, max(0.0, strain))
        
        return {
            "hr_mean": hr_mean,
            "hrv_mean": hrv_mean,
            "hrv_drop": hrv_drop,
            "irregularity": irregularity,
            "strain_index": strain,
            "baseline_ready": self.baseline_computed,
        }
    
    def get_baseline_str(self):
        """Get baseline as string for display."""
        if not self.baseline_computed:
            return "Computing baseline..."
        return f"Baseline: HR={self.hr_baseline:.0f}, HRV={self.hrv_baseline:.0f}ms"
