import cv2
import numpy as np
from .face_detector import FaceDetector
from .ml_classifier import MLClassifier

class EyeStateDetector:
    def __init__(self):
        # Initialize components
        self.face_detector = FaceDetector()
        self.ml_classifier = MLClassifier()
        
        # Eye landmarks indices
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))

    def extract_eye_region(self, image, landmarks, eye_indices):
        """Extract eye region from landmarks"""
        points = np.array([(landmarks[i][0], landmarks[i][1]) for i in eye_indices])
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(points)
        
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Extract region
        eye_region = image[y:y+h, x:x+w]
        return eye_region, (x, y, w, h)

    def process_frame(self, frame):
        """Main processing function - uses ML classifier only"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result_frame = frame.copy()
        
        # Detect faces
        faces = self.face_detector.detect_faces_haar(frame)
        
        results = []
        
        for (x, y, w, h) in faces:
            # Get landmarks
            landmarks = self.face_detector.get_landmarks(frame, (x, y, w, h))
            
            if landmarks is not None:
                # Extract eye regions for ML
                left_eye_region, left_bbox = self.extract_eye_region(gray, landmarks, self.LEFT_EYE)
                right_eye_region, right_bbox = self.extract_eye_region(gray, landmarks, self.RIGHT_EYE)
                
                # ML predictions
                left_ml_state, left_ml_conf = self.ml_classifier.predict_eye_state(left_eye_region)
                right_ml_state, right_ml_conf = self.ml_classifier.predict_eye_state(right_eye_region)
                
                # Final state based on both eyes
                if left_ml_state == "Open" or right_ml_state == "Open":
                    final_state = "Open"
                else:
                    final_state = "Closed"
                
                result = {
                    'face_bbox': (x, y, w, h),
                    'left_eye_bbox': left_bbox,
                    'right_eye_bbox': right_bbox,
                    'landmarks': landmarks,
                    'left_ml_state': left_ml_state,
                    'right_ml_state': right_ml_state,
                    'left_ml_conf': left_ml_conf,
                    'right_ml_conf': right_ml_conf,
                    'final_state': final_state
                }
                
                results.append(result)
        
        return result_frame, results
    
    def draw_results(self, frame, results):
        """Draw detection results on frame"""
        for result in results:
            x, y, w, h = result['face_bbox']
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw eye rectangles
            lx, ly, lw, lh = result['left_eye_bbox']
            rx, ry, rw, rh = result['right_eye_bbox']
            
            # Color based on state
            color = (0, 255, 0) if result['final_state'] == "Open" else (0, 0, 255)
            
            cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), color, 2)
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)
            
            # Draw text
            text = f"Eyes: {result['final_state']}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw states
            state_text = f"L:{result['left_ml_state']} R:{result['right_ml_state']}"
            cv2.putText(frame, state_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw ML confidence
            conf_text = f"ML: L{result['left_ml_conf']:.2f} R{result['right_ml_conf']:.2f}"
            cv2.putText(frame, conf_text, (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
