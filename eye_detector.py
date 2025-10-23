import cv2
import numpy as np
import os
import pickle
from feature_extractor import extract_eye_features

class EyeStateDetector:
    def __init__(self, model_path='models/eye_state_svm_model.pkl'):
        # Load the SVM model
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = None
            print(f"Model file not found: {model_path}")
            print("Please run train.py first to train the model.")

        # Load Haar Cascade classifiers
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_frontalface_default.xml'))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_eye.xml'))

        if self.face_cascade.empty():
            print("Error: Could not load face cascade classifier")
        if self.eye_cascade.empty():
            print("Error: Could not load eye cascade classifier")

    def detect_faces(self, frame):
        """Detect faces in the frame using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return gray, faces

    def detect_eyes(self, gray_frame, face):
        """Detect eyes within a face region using Haar Cascade"""
        x, y, w, h = face
        roi_gray = gray_frame[y:y+h, x:x+w]

        # Detect eyes in the upper half of the face
        roi_height, roi_width = roi_gray.shape
        upper_half = roi_gray[0:int(roi_height/2), :]

        eyes = self.eye_cascade.detectMultiScale(
            upper_half,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )

        # Adjust eye coordinates to match the full face ROI
        adjusted_eyes = []
        for (ex, ey, ew, eh) in eyes:
            adjusted_eyes.append((ex, ey, ew, eh))

        return adjusted_eyes, roi_gray

    def predict_eye_state(self, eye_img):
        """Predict if an eye is open or closed"""
        if self.model is None:
            return "Unknown"

        # Extract features
        features = extract_eye_features(eye_img)

        # Predict
        prediction = self.model.predict([features])[0]
        probability = self.model.predict_proba([features])[0]

        # Return state and confidence
        if prediction == 1:
            return "Open", probability[1]
        else:
            return "Closed", probability[0]

    def process_frame(self, frame):
        """Process a frame and detect eye states"""
        # Create a copy of the frame for drawing
        result_frame = frame.copy()

        # Detect faces
        gray, faces = self.detect_faces(frame)

        eye_states = []

        # For each face, detect and analyze eyes
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Detect eyes
            eyes, roi_gray = self.detect_eyes(gray, (x, y, w, h))

            # Process each eye
            for (ex, ey, ew, eh) in eyes:
                # Get absolute eye coordinates
                eye_x = x + ex
                eye_y = y + ey

                # Extract eye region
                eye_img = roi_gray[ey:ey+eh, ex:ex+ew]

                # Predict eye state
                state, confidence = self.predict_eye_state(eye_img)
                eye_states.append(state)

                # Draw eye rectangle with state label
                color = (0, 255, 0) if state == "Open" else (0, 0, 255)
                cv2.rectangle(result_frame, (eye_x, eye_y), (eye_x+ew, eye_y+eh), color, 2)
                cv2.putText(result_frame, f"{state} ({confidence:.2f})",
                            (eye_x, eye_y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return result_frame, eye_states
