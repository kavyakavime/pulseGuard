# ml/test_real_model.py

import pandas as pd
from signal_processing import PPGSignalProcessor
from features import FeatureExtractor
from model import StrainDetector
import sys

def test_model_on_data(csv_path, model_path='ml/strain_model_REAL.pkl'):
    """
    Test your trained model on new data
    """
    print("="*80)
    print("🧪 TESTING PULSEGUARD MODEL")
    print("="*80)
    
    # Load model
    print(f"\n📂 Loading model: {model_path}")
    try:
        detector = StrainDetector.load(model_path)
    except FileNotFoundError:
        print("❌ Model not found! Train first:")
        print("   python ml/train_on_real_data.py <baseline_data.csv>")
        return
    
    # Load test data
    print(f"\n📥 Loading test data: {csv_path}")
    df = pd.read_csv(csv_path)
    
    finger_on = df[df['fingerDetected'] == 1]
    print(f"   Samples with finger: {len(finger_on)}")
    
    # Process
    processor = PPGSignalProcessor()
    extractor = FeatureExtractor()
    
    ir_signal = finger_on['ir'].values
    filtered = processor.filter_signal(ir_signal)
    peaks = processor.detect_peaks(filtered)
    
    print(f"   Heartbeats detected: {len(peaks)}")
    
    if len(peaks) < 10:
        print("❌ Not enough heartbeats!")
        return
    
    features_df = extractor.extract_windowed_features(peaks)
    
    print(f"\n📊 RESULTS:")
    print(f"{'Window':<8} {'HR':<8} {'HRV':<8} {'Status':<20} {'Risk':<8}")
    print("-"*60)
    
    strain_count = 0
    
    for i, row in features_df.iterrows():
        hr, hrv = row['hr'], row['hrv']
        is_strain, risk = detector.detect_strain(hr, hrv)
        
        if is_strain:
            status = "🚨 STRAIN EVENT"
            strain_count += 1
        elif risk > 50:
            status = "🟡 Rising Strain"
        else:
            status = "🟢 Normal"
        
        print(f"{i:<8} {hr:<8.1f} {hrv:<8.1f} {status:<20} {risk:<8}/100")
    
    print("\n" + "="*80)
    print(f"SUMMARY:")
    print(f"   Total windows: {len(features_df)}")
    print(f"   Strain events: {strain_count} ({strain_count/len(features_df)*100:.1f}%)")
    print(f"   Average risk: {features_df.apply(lambda r: detector.detect_strain(r['hr'], r['hrv'])[1], axis=1).mean():.1f}/100")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ml/test_real_model.py <test_data.csv>")
        sys.exit(1)
    
    test_model_on_data(sys.argv[1])