# ml/train_on_real_data.py

import pandas as pd
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from signal_processing import PPGSignalProcessor
from features import FeatureExtractor
from model import StrainDetector

print("🫀 PULSEGUARD - TRAINING ON YOUR REAL DATA")
print("=" * 80)

# Get file path
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    csv_path = 'data/my_baseline.csv'

print(f"\n📥 Loading: {csv_path}")

# Load data
try:
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} samples")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    sys.exit(1)

# Show basic stats
print(f"\n📊 DATA OVERVIEW:")
print(f"   Duration: {(df['time'].max() - df['time'].min()) / 1000:.1f} seconds")
print(f"   IR range: {df['ir'].min()} - {df['ir'].max()}")
print(f"   Finger detected: {(df['fingerDetected']==1).sum()} / {len(df)} samples")

# Filter to finger-detected only
df_valid = df[df['fingerDetected'] == 1].copy()
print(f"\n✅ Using {len(df_valid)} samples with finger detected")

if len(df_valid) < 100:
    print("❌ Not enough data! Need at least 100 samples with finger on sensor")
    sys.exit(1)

# Process signal
print("\n🔬 PROCESSING SIGNAL...")
processor = PPGSignalProcessor(sample_rate=100)

ir_signal = df_valid['ir'].values
print(f"   Filtering {len(ir_signal)} IR samples...")
filtered = processor.filter_signal(ir_signal)

print(f"   Detecting heartbeats...")
peaks = processor.detect_peaks(filtered)
print(f"   ✅ Found {len(peaks)} heartbeats!")

if len(peaks) < 10:
    print("❌ Not enough heartbeats detected!")
    sys.exit(1)

# Extract features
print("\n📈 EXTRACTING FEATURES...")
extractor = FeatureExtractor(sample_rate=100)
features = extractor.extract_windowed_features(peaks, window_size=10)

print(f"   ✅ Extracted {len(features)} windows")
print(f"\n💓 YOUR HEART METRICS:")
print(f"   HR:  {features['hr'].mean():.1f} ± {features['hr'].std():.1f} BPM")
print(f"   HRV: {features['hrv'].mean():.1f} ± {features['hrv'].std():.1f} ms")

# Train model
print("\n🧠 TRAINING YOUR PERSONALIZED MODEL...")
detector = StrainDetector()
detector.train(features['hr'].values, features['hrv'].values)

# Save
model_path = 'ml/strain_model_REAL.pkl'
detector.save(model_path)

print("\n" + "=" * 80)
print("✅ SUCCESS! YOUR MODEL IS TRAINED!")
print("=" * 80)
print(f"\n📋 YOUR PERSONAL BASELINE:")
print(f"   Heart Rate: {detector.baseline_hr:.1f} BPM")
print(f"   HRV: {detector.baseline_hrv:.1f} ms")
print(f"\n💾 Model saved to: {model_path}")
print("\n🎯 NEXT STEPS:")
print("   1. Collect stress data (hold breath, exercise)")
print("   2. Test: python ml/test_real_model.py data/stress.csv")
print("   3. Dashboard: streamlit run ml/dashboard.py")