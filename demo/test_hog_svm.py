import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report

def extract_hog_features(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, (64, 64))
    features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    return features

def main():
    model_path = 'rps_hog_svm_model.pkl'
    test_dir = '../dataset/test'

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run train_hog_svm.py first.")
        return

    print("Loading HOG+SVM model...")
    model = joblib.load(model_path)

    label_map = {'rock': 0, 'paper': 1, 'scissors': 2}
    X_test, y_test = [], []

    print("Processing test features...")
    for category, label_idx in label_map.items():
        category_path = os.path.join(test_dir, category)
        if not os.path.exists(category_path): continue
        
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(category_path, filename))
                if img is not None:
                    X_test.append(extract_hog_features(img))
                    y_test.append(label_idx)

    y_pred = model.predict(X_test)
    print(f"\nHOG+SVM Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Rock', 'Paper', 'Scissors']))

if __name__ == "__main__":
    main()
