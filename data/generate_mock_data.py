# data/generate_mock_data.py

import numpy as np
import pandas as pd
import os

def generate_realistic_ppg(duration_sec, base_hr=72, base_hrv=45, stress=False):
    """
    Generate highly realistic PPG data
    
    Normal ranges:
    - Resting HR: 60-100 BPM (avg 72)
    - Normal HRV: 20-100 ms (avg 45)
    
    Stress response:
    - HR increases 20-30 BPM
    - HRV decreases 50-70%
    """
    
    sample_rate = 100
    n_samples = duration_sec * sample_rate
    
    if stress:
        # Stress: HR rises, HRV drops
        hr = base_hr + 22 + np.random.normal(0, 3, n_samples)
        hrv = base_hrv * 0.35 + np.random.normal(0, 4, n_samples)
    else:
        # Normal: slight variations
        hr = base_hr + np.random.normal(0, 2.5, n_samples)
        hrv = base_hrv + np.random.normal(0, 7, n_samples)
    
    # Ensure realistic bounds
    hr = np.clip(hr, 50, 120)
    hrv = np.clip(hrv, 10, 150)
    
    # Generate PPG waveform
    time_ms = np.arange(0, n_samples * 10, 10)
    ir_values = []
    red_values = []
    
    for i, t_ms in enumerate(time_ms):
        t_sec = t_ms / 1000.0
        freq = hr[i] / 60.0
        
        # Realistic PPG shape
        phase = 2 * np.pi * freq * t_sec
        systolic = np.sin(phase)
        dicrotic = 0.25 * np.sin(phase + 0.6)
        
        # Respiratory modulation
        resp = 0.08 * np.sin(2 * np.pi * 0.25 * t_sec)
        
        # Noise
        noise = np.random.normal(0, 0.04 if stress else 0.02)
        
        pulse = systolic + dicrotic + resp + noise
        
        # Convert to sensor values
        ir_base = 85000 if not stress else 88000
        ir = ir_base + 16000 * pulse + np.random.normal(0, 500)
        red = ir * 0.48 + np.random.normal(0, 300)
        
        ir_values.append(max(0, int(ir)))
        red_values.append(max(0, int(red)))
    
    df = pd.DataFrame({
        'time': time_ms,
        'ir': ir_values,
        'red': red_values,
        'bpm': hr.astype(int),
        'hrv': hrv.astype(int),
        'spo2': 97 if not stress else 95,
        'fingerDetected': 1,
        'hrvReady': [0] * 100 + [1] * (n_samples - 100),
        'beatQuality': 22 + np.random.normal(0, 3, n_samples)
    })
    
    return df

if __name__ == "__main__":
    print("🧬 Generating realistic PPG data...\n")
    
    os.makedirs('data', exist_ok=True)
    
    # Training baseline (3 min, normal)
    print("📊 Training baseline (180s, HR~72, HRV~45)...")
    baseline = generate_realistic_ppg(180, base_hr=72, base_hrv=45, stress=False)
    baseline.to_csv('data/mock_baseline.csv', index=False)
    
    # Stress scenario (1 min)
    print("😰 Stress scenario (60s, HR~94, HRV~16)...")
    stress = generate_realistic_ppg(60, base_hr=72, base_hrv=45, stress=True)
    stress.to_csv('data/mock_stress.csv', index=False)
    
    # Demo sequence: Normal → Stress → Recovery
    print("🎬 Demo sequence (90s)...")
    normal = generate_realistic_ppg(30, base_hr=72, base_hrv=45, stress=False)
    stressed = generate_realistic_ppg(30, base_hr=72, base_hrv=45, stress=True)
    recovery = generate_realistic_ppg(30, base_hr=78, base_hrv=32, stress=False)
    
    demo = pd.concat([normal, stressed, recovery], ignore_index=True)
    demo['time'] = np.arange(0, len(demo) * 10, 10)
    demo.to_csv('data/mock_demo.csv', index=False)
    
    print("\n✅ Data generated!")
    print(f"   📁 data/mock_baseline.csv")
    print(f"   📁 data/mock_stress.csv")
    print(f"   📁 data/mock_demo.csv")
    
    # Show stats
    print("\n📊 Baseline Stats:")
    print(f"   HR:  {baseline['bpm'].mean():.1f} ± {baseline['bpm'].std():.1f} BPM")
    print(f"   HRV: {baseline['hrv'].mean():.1f} ± {baseline['hrv'].std():.1f} ms")
    
    print("\n😰 Stress Stats:")
    print(f"   HR:  {stress['bpm'].mean():.1f} ± {stress['bpm'].std():.1f} BPM")
    print(f"   HRV: {stress['hrv'].mean():.1f} ± {stress['hrv'].std():.1f} ms")