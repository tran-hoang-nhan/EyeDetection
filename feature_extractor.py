import cv2
import numpy as np

def extract_eye_features(eye_region):
    """
    Extract features from eye region
    """
    # Resize eye region to a fixed size
    eye_region = cv2.resize(eye_region, (24, 24))

    # Convert to grayscale if it's not already
    if len(eye_region.shape) > 2:
        eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to enhance contrast
    eye_region = cv2.equalizeHist(eye_region)

    # Flatten the image to get features
    features = eye_region.flatten()

    return features
