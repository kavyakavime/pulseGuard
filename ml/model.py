# ml/model.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os


class GeneralStrainClassifier:
    """
    Supervised strain classifier
    - Trains on labeled normal vs strain data
    - Outputs probability of strain
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.feature_medians = None
        self.is_trained = False

    def train(self, X_df, y):
        """
        Train on labeled data.

        Args:
            X_df: pandas DataFrame of features
            y: array-like labels (0=normal, 1=strain)
        """
        self.feature_names = list(X_df.columns)
        self.feature_medians = X_df.median(numeric_only=True).to_dict()
        X_filled = X_df.fillna(self.feature_medians)

        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
        self.model.fit(X_filled.values, y)
        self.is_trained = True

        print("\n✅ MODEL TRAINED!")
        print(f"   Samples: {len(X_df)}")
        print(f"   Features: {', '.join(self.feature_names)}")

    def predict_proba(self, features):
        """
        Predict strain probability.

        Args:
            features: dict or list/np array ordered by feature_names
        Returns:
            p_strain (float)
        """
        if not self.is_trained:
            return 0.0

        if isinstance(features, dict):
            row = []
            for name in self.feature_names:
                value = features.get(name, None)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    value = self.feature_medians.get(name, 0.0)
                row.append(value)
            X = np.array([row], dtype=float)
        else:
            X = np.array([features], dtype=float)

        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        if len(classes) == 1:
            return 1.0 if classes[0] == 1 else 0.0
        if 1 in classes:
            return float(proba[classes.index(1)])
        return float(proba[1])

    def save(self, filepath='ml/strain_model_general.pkl'):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_medians': self.feature_medians,
            'is_trained': self.is_trained
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"✅ Model saved to {filepath}")

    @staticmethod
    def load(filepath='ml/strain_model_general.pkl'):
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        clf = GeneralStrainClassifier()
        clf.model = data['model']
        clf.feature_names = data['feature_names']
        clf.feature_medians = data.get('feature_medians', {})
        clf.is_trained = data.get('is_trained', True)

        print(f"✅ Model loaded: features={', '.join(clf.feature_names)}")
        return clf


class StrainDetector:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "StrainDetector is deprecated. Use GeneralStrainClassifier and "
            "train via ml/train_general_model.py."
        )
