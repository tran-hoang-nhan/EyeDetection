import cv2
import dlib
import numpy as np

class FaceDetector:
    def __init__(self):
        # Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Dlib detector
        self.dlib_detector = dlib.get_frontal_face_detector()
        
        # Dlib predictor
        try:
            self.predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        except:
            print("Warning: dlib shape predictor not found")
            self.predictor = None
    
    def detect_faces_haar(self, frame):
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def detect_faces_dlib(self, frame):
        """Detect faces using dlib"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.dlib_detector(gray)
        return [(face.left(), face.top(), face.width(), face.height()) for face in faces]
    
    def get_landmarks(self, frame, face_rect):
        """Get 68 facial landmarks"""
        if self.predictor is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = face_rect
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = self.predictor(gray, dlib_rect)
        
        return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])