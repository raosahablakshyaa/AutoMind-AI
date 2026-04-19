"""
Machine Failure Predictor

Loads trained ML models and provides failure prediction functions
for the fleet maintenance system.
"""

import pickle
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Optional

MODEL_DIR = Path(__file__).parent / "Models"


class FailurePredictor:
    """Encapsulates ML prediction logic for machine failure."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoder = None
        self.feature_names = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load trained model and preprocessing artifacts."""
        try:
            # Load model
            model_path = MODEL_DIR / "failure_predictor.pkl"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
            
            # Load scaler
            scaler_path = MODEL_DIR / "feature_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
            
            # Load encoder
            encoder_path = MODEL_DIR / "type_encoder.pkl"
            if encoder_path.exists():
                with open(encoder_path, "rb") as f:
                    self.encoder = pickle.load(f)
            
            # Load feature names
            features_path = MODEL_DIR / "feature_names.txt"
            if features_path.exists():
                with open(features_path, "r") as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
        
        except Exception as e:
            print(f"Warning: Could not load ML artifacts: {e}")
    
    def is_available(self) -> bool:
        """Check if model is available for predictions."""
        return self.model is not None and self.scaler is not None
    
    def predict_single(self, machine_data: Dict) -> Optional[Dict]:
        """
        Predict failure probability for a single machine.
        
        Args:
            machine_data: Dictionary with machine operational parameters
            
        Returns:
            Dictionary with prediction results or None if model unavailable
        """
        if not self.is_available():
            return None
        
        try:
            # Map data to model features
            features = self._map_features(machine_data)
            
            # Prepare feature vector
            feature_values = []
            for col in self.feature_names or []:
                if col == "Type":
                    val = self.encoder.transform([features.get(col, "M")])[0]
                else:
                    val = float(features.get(col, 0))
                feature_values.append(val)
            
            # Scale and predict (suppress feature names warning)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_input = np.array([feature_values]).reshape(1, -1)
                X_scaled = self.scaler.transform(X_input)
            
            prob = float(self.model.predict_proba(X_scaled)[0, 1])
            pred = int(self.model.predict(X_scaled)[0])
            conf = float(max(self.model.predict_proba(X_scaled)[0]))
            
            # Map to risk level
            risk_level = self._probability_to_risk(prob)
            
            return {
                "failure_probability": prob,
                "predicted_failure": pred,
                "confidence": conf,
                "risk_level": risk_level
            }
        
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
    def predict_batch(self, machine_data_list: List[Dict]) -> List[Dict]:
        """
        Predict failures for multiple machines.
        
        Args:
            machine_data_list: List of machine data dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for data in machine_data_list:
            result = self.predict_single(data)
            if result:
                results.append(result)
        return results
    
    def _map_features(self, machine_data: Dict) -> Dict:
        """Map raw machine data to model feature space."""
        return {
            "Type": machine_data.get("Type", "M"),
            "Air temperature [K]": float(machine_data.get("Air_temperature", 298)) + 273,
            "Process temperature [K]": float(machine_data.get("Process_temperature", 308)) + 273,
            "Rotational speed [rpm]": float(machine_data.get("Rotational_speed", 0)),
            "Torque [Nm]": float(machine_data.get("Torque", 0)),
            "Tool wear [min]": float(machine_data.get("Tool_wear", 0)),
            "TWF": 0,
            "HDF": 0,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0
        }
    
    def _probability_to_risk(self, probability: float) -> str:
        """Convert failure probability to risk level."""
        if probability < 0.3:
            return "Low"
        elif probability < 0.6:
            return "Medium"
        elif probability < 0.8:
            return "High"
        else:
            return "Critical"


# Global predictor instance
_predictor = None


def get_predictor() -> FailurePredictor:
    """Get or create global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = FailurePredictor()
    return _predictor


def predict_machine_failure(machine_data: Dict) -> Optional[Dict]:
    """
    Convenience function for single machine prediction.
    
    Args:
        machine_data: Machine operational data
        
    Returns:
        Prediction results or None
    """
    predictor = get_predictor()
    return predictor.predict_single(machine_data)
