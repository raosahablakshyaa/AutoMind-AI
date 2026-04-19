"""
Model Training Script

Trains the machine failure prediction model on the AI4I 2020 dataset.
Run this once to generate Models/failure_predictor.pkl and related artifacts.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Paths
RAW_DIR = Path(__file__).parent / "Raw"
MODEL_DIR = Path(__file__).parent / "Models"
MODEL_DIR.mkdir(exist_ok=True)

DATA_PATH = RAW_DIR / "ai4i2020.csv"


def train_model():
    """Train the failure prediction model."""
    
    print("=" * 70)
    print("ML MODEL TRAINING - AI4I 2020 PREDICTIVE MAINTENANCE")
    print("=" * 70)
    
    # Load dataset
    print("\n1. Loading dataset...")
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        return False
    
    df = pd.read_csv(DATA_PATH)
    print(f"   ✓ Loaded {len(df)} samples with {len(df.columns)} features")
    print(f"   ✓ Failure rate: {df['Machine failure'].sum() / len(df) * 100:.2f}%")
    
    # Preprocessing
    print("\n2. Preprocessing data...")
    X = df.drop(['UDI', 'Product ID', 'Machine failure'], axis=1)
    y = df['Machine failure']
    
    # Encode categorical Type column
    le_type = LabelEncoder()
    X['Type'] = le_type.fit_transform(X['Type'])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   ✓ Train set: {len(X_train)} samples ({y_train.sum()} failures)")
    print(f"   ✓ Test set: {len(X_test)} samples ({y_test.sum()} failures)")
    
    # Scaling
    print("\n3. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   ✓ Features scaled using StandardScaler")
    
    # Model training
    print("\n4. Training GradientBoostingClassifier...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbose=0
    )
    model.fit(X_train_scaled, y_train)
    print("   ✓ Model trained successfully")
    
    # Evaluation
    print("\n5. Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = (y_pred == y_test).sum() / len(y_test)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"   ✓ Accuracy: {accuracy:.4f}")
    print(f"   ✓ ROC-AUC: {roc_auc:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure']))
    
    # Save artifacts
    print("\n6. Saving model artifacts...")
    
    with open(MODEL_DIR / "failure_predictor.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"   ✓ Model saved to Models/failure_predictor.pkl")
    
    with open(MODEL_DIR / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"   ✓ Scaler saved to Models/feature_scaler.pkl")
    
    with open(MODEL_DIR / "type_encoder.pkl", "wb") as f:
        pickle.dump(le_type, f)
    print(f"   ✓ Encoder saved to Models/type_encoder.pkl")
    
    # Save feature names
    feature_names = list(X.columns)
    with open(MODEL_DIR / "feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))
    print(f"   ✓ Feature names saved to Models/feature_names.txt")
    
    print("\n" + "=" * 70)
    print("✓ MODEL TRAINING COMPLETE")
    print("=" * 70)
    print("\nModel ready for inference!")
    print(f"Artifacts saved in: {MODEL_DIR}")
    
    return True


if __name__ == "__main__":
    success = train_model()
    exit(0 if success else 1)
