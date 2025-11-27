import cv2
import numpy as np
from .face_detector import FaceDetector
from .ml_predictor import MLPredictor
from .ear_calculator import calculate_left_ear, calculate_right_ear

class EyeStateDetector:
    def __init__(self):
        # Initialize components
        self.face_detector = FaceDetector()
        self.ml_predictor = MLPredictor()
        
        # Eye landmarks indices
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        
        # Detection thresholds
        self.EAR_THRESHOLD = 0.2 # dlib EAR threshold (< 0.25 = mắt đóng)

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
        """Main processing function - uses EAR threshold + ML prediction combined"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result_frame = frame.copy()
        
        # Detect faces
        faces = self.face_detector.detect_faces_haar(frame)
        
        results = []
        
        for (x, y, w, h) in faces:
            # Get landmarks
            landmarks = self.face_detector.get_landmarks(frame, (x, y, w, h))
            
            if landmarks is not None:
                left_ear = calculate_left_ear(landmarks)
                right_ear = calculate_right_ear(landmarks)
                ear_value = (left_ear + right_ear) / 2.0  # Average for display
                
                left_eye_region, left_bbox = self.extract_eye_region(gray, landmarks, self.LEFT_EYE)
                right_eye_region, right_bbox = self.extract_eye_region(gray, landmarks, self.RIGHT_EYE)
                
                left_ml_conf = self.ml_predictor.predict_eye_state(left_eye_region)
                right_ml_conf = self.ml_predictor.predict_eye_state(right_eye_region)
                
                left_ml_conf = left_ml_conf if left_ml_conf is not None else 0.5
                right_ml_conf = right_ml_conf if right_ml_conf is not None else 0.5
            
                if left_ear < 0.15:
                    left_ml_state = "Closed"  
                else:
                    left_ml_state = "Closed" if left_ml_conf > 0.5 else "Open"
                
                if right_ear < 0.15:
                    right_ml_state = "Closed" 
                else:
                    right_ml_state = "Closed" if right_ml_conf > 0.5 else "Open"
                
                result = {
                    'face_bbox': (x, y, w, h),
                    'left_eye_bbox': left_bbox,
                    'right_eye_bbox': right_bbox,
                    'landmarks': landmarks,
                    'left_ml_state': left_ml_state,
                    'right_ml_state': right_ml_state,
                    'left_ml_conf': left_ml_conf,
                    'right_ml_conf': right_ml_conf,
                    'ear_value': ear_value,
                    'left_ear': left_ear,
                    'right_ear': right_ear
                }
                
                results.append(result)
        
        return result_frame, results
    
    def draw_results(self, frame, results):
        """Draw detection results on frame with 68 landmarks and individual EAR"""
        for result in results:
            x, y, w, h = result['face_bbox']
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            landmarks = result['landmarks']
            for i in range(len(landmarks)):
                px, py = landmarks[i]
                if 36 <= i <= 47:
                    cv2.circle(frame, (int(px), int(py)), 3, (0, 0, 255), -1)
                else:
                    cv2.circle(frame, (int(px), int(py)), 2, (0, 255, 255), -1)
            
            state_text = f"L:{result['left_ml_state']} R:{result['right_ml_state']}"
            cv2.putText(frame, state_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            ear_text = f"L_EAR: {result['left_ear']:.3f} | R_EAR: {result['right_ear']:.3f}"
            cv2.putText(frame, ear_text, (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            conf_text = f"ML: L{result['left_ml_conf']:.2f} R{result['right_ml_conf']:.2f}"
            cv2.putText(frame, conf_text, (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
