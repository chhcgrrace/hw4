import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def extract_hog_features(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, (64, 64))
    return hog(img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

def load_data(folder_path):
    features_list, labels_list = [], []
    label_map = {'rock': 0, 'paper': 1, 'scissors': 2}
    for category, label_idx in label_map.items():
        category_path = os.path.join(folder_path, category)
        if not os.path.exists(category_path):
            subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            if subdirs: category_path = os.path.join(folder_path, subdirs[0], category)
        if not os.path.exists(category_path): continue
        print(f"Processing {category}...")
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(category_path, filename))
                if img is not None:
                    features_list.append(extract_hog_features(img))
                    labels_list.append(label_idx)
    return np.array(features_list), np.array(labels_list)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'dataset', 'train')
    test_dir = os.path.join(base_dir, 'dataset', 'test')
    demo_dir = os.path.join(base_dir, 'demo')

    X_train, y_train = load_data(train_dir)
    X_test, y_test = load_data(test_dir)

    print("\n=== Training Linear SVM (Final Attempt) ===")
    # Linear kernel is often superior for high-dimensional HOG features
    model = SVC(kernel='linear', C=10.0, probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Rock', 'Paper', 'Scissors']))

    os.makedirs(demo_dir, exist_ok=True)
    joblib.dump(model, os.path.join(demo_dir, 'rps_hog_svm_model.pkl'))
    print("Model saved.")

if __name__ == "__main__":
    main()
