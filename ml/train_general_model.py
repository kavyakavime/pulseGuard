# ml/train_general_model.py

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))

from model import GeneralStrainClassifier

DEFAULT_FILES = [
    ("data/my_baseline.csv", 0),
    ("data/my_stress.csv", 1),
    ("data/mock_baseline.csv", 0),
    ("data/mock_stress.csv", 1),
]

FEATURES = ["hr", "hrv", "spo2", "beat_quality"]


def _safe_median(series):
    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) == 0:
        return np.nan
    return float(series.median())


def build_windowed_features(df, label, step_ms=2000, hr_window_ms=5000, hrv_window_ms=15000):
    if "time" in df.columns:
        df = df.sort_values("time")
        times = df["time"].astype(float).values
    else:
        df = df.reset_index(drop=True)
        times = df.index.values.astype(float) * 10.0

    rows = []
    start_time = times[0]
    end_time = times[-1]

    for t in np.arange(start_time + hr_window_ms, end_time + 1, step_ms):
        hr_mask = (times >= (t - hr_window_ms)) & (times <= t)
        hrv_mask = (times >= (t - hrv_window_ms)) & (times <= t)

        df_hr = df.loc[hr_mask]
        df_hrv = df.loc[hrv_mask]

        hr = _safe_median(df_hr["bpm"])
        hrv = _safe_median(df_hrv.loc[df_hrv["hrvReady"] == 1, "hrv"])
        beat_quality = _safe_median(df_hr["beatQuality"])
        spo2 = _safe_median(df_hr["spo2"])

        rows.append({
            "hr": hr,
            "hrv": hrv,
            "spo2": spo2,
            "beat_quality": beat_quality,
            "label": label,
        })

    return pd.DataFrame(rows)


def load_and_filter(csv_path):
    df = pd.read_csv(csv_path)
    required = ["bpm", "hrv", "spo2", "fingerDetected", "hrvReady", "beatQuality"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df = df.copy()
    df = df[df["fingerDetected"] == 1]
    df = df[df["beatQuality"] >= 40]
    df = df[df["bpm"] > 30]
    return df


def build_dataset(file_label_pairs):
    frames = []
    for path, label in file_label_pairs:
        if not os.path.exists(path):
            print(f"⚠️  Skipping missing: {path}")
            continue
        print(f"📥 Loading {path} (label={label})")
        df = load_and_filter(path)
        if len(df) < 100:
            print(f"⚠️  Not enough valid samples in {path} ({len(df)})")
            continue
        feats = build_windowed_features(df, label)
        feats = feats.dropna(subset=["hr", "beat_quality"])
        frames.append(feats)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    if len(sys.argv) > 1:
        pairs = []
        for arg in sys.argv[1:]:
            if ":" not in arg:
                print("Expected args like path:label")
                sys.exit(1)
            path, label = arg.split(":", 1)
            pairs.append((path, int(label)))
    else:
        pairs = DEFAULT_FILES

    dataset = build_dataset(pairs)
    if dataset.empty:
        print("❌ No training data built. Check file paths and columns.")
        sys.exit(1)

    X = dataset[FEATURES]
    y = dataset["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = GeneralStrainClassifier()
    clf.train(X_train, y_train)

    preds = clf.model.predict(X_test.fillna(clf.feature_medians))
    acc = accuracy_score(y_test, preds)

    print("\n📊 QUICK EVAL")
    print(f"   Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, digits=3))

    clf.save("ml/strain_model_general.pkl")
    print("✅ Done")


if __name__ == "__main__":
    main()
