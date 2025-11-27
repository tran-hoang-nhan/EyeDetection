import numpy as np
from scipy.spatial import distance as dist

def calculate_left_ear(landmarks):
    try:
        left_eye = [landmarks[i] for i in range(36, 42)]
        lv1 = dist.euclidean(left_eye[1], left_eye[5])
        lv2 = dist.euclidean(left_eye[2], left_eye[4])
        lh = dist.euclidean(left_eye[0], left_eye[3])
        left_ear = (lv1 + lv2) / (2.0 * lh) if lh != 0 else 0.0
        return left_ear
    except Exception:
        return 0.0

def calculate_right_ear(landmarks):
    try:
        right_eye = [landmarks[i] for i in range(42, 48)]
        rv1 = dist.euclidean(right_eye[1], right_eye[5])
        rv2 = dist.euclidean(right_eye[2], right_eye[4])
        rh = dist.euclidean(right_eye[0], right_eye[3])
        right_ear = (rv1 + rv2) / (2.0 * rh) if rh != 0 else 0.0
        return right_ear
    except Exception:
        return 0.0