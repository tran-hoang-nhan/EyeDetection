import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from feature_extractor import extract_eye_features

def train():
    """
    Load and prepare dataset, then train SVM model for eye state detection
    """
    print("Loading dataset...")

    # Paths for open and closed eyes dataset
    # Assuming dataset structure similar to the one in D:/XLA
    open_eyes_dir = "dataset/open_eyes"
    closed_eyes_dir = "dataset/closed_eyes"

    # Create placeholders for features and labels
    features = []
    labels = []

    # Check if directories exist, if not create them
    if not os.path.exists(open_eyes_dir):
        os.makedirs(open_eyes_dir, exist_ok=True)
    if not os.path.exists(closed_eyes_dir):
        os.makedirs(closed_eyes_dir, exist_ok=True)

    # Load open eyes images
    if len(os.listdir(open_eyes_dir)) > 0:
        for filename in os.listdir(open_eyes_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(open_eyes_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Extract features
                    img_features = extract_eye_features(img)
                    features.append(img_features)
                    labels.append(1)  # 1 for open eye
    else:
        print(f"Warning: No images found in {open_eyes_dir}")

    # Load closed eyes images
    if len(os.listdir(closed_eyes_dir)) > 0:
        for filename in os.listdir(closed_eyes_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(closed_eyes_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Extract features
                    img_features = extract_eye_features(img)
                    features.append(img_features)
                    labels.append(0)  # 0 for closed eye
    else:
        print(f"Warning: No images found in {closed_eyes_dir}")

    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    if len(features) == 0:
        print("Error: No data loaded. Please ensure dataset directories contain images.")
        return None

    print(f"Dataset loaded. Total samples: {len(features)}")
    print(f"Open eyes: {np.sum(labels == 1)}, Closed eyes: {np.sum(labels == 0)}")

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train SVM model
    print("Training SVM model...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/eye_state_svm_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model saved to models/eye_state_svm_model.pkl")
    return model

if __name__ == "__main__":
    train()
