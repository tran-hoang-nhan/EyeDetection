import pickle
import numpy as np
from .feature_extractor import extract_eye_features, preprocess_eye_image

class MLClassifier:
    def __init__(self, model_path='models/eye_classifier.pkl'):
        self.model = None
        self.model_name = "Unknown"
        self.accuracy = 0.0
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['pipeline']
                self.model_name = model_data['pipeline_name']
                self.accuracy = model_data['accuracy']
                print(f"✓ Loaded ML model: {self.model_name} (accuracy: {self.accuracy:.3f})")
        except Exception as e:
            print(f"❌ Failed to load ML model: {e}")
            print("Run train.py first to train the model.")
    
    def predict_eye_state(self, eye_region):
        """Predict eye state using ML model"""
        if self.model is None:
            return "Unknown", 0.0
        
        try:
            # Preprocess
            processed = preprocess_eye_image(eye_region)
            
            # Extract features
            features = extract_eye_features(processed)
            
            # Predict
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            confidence = probabilities.max()
            
            state = "Open" if prediction == 1 else "Closed"
            return state, confidence
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return "Error", 0.0
    
    def is_available(self):
        """Check if ML model is available"""
        return self.model is not None