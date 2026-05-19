import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def imread_safe(p):
    try:
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None

def extract_hog_features(img):
    if img is None: return None
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use INTER_AREA interpolation for high-quality downsampling
    img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    return hog(img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

def load_data(folder_path):
    features_list, labels_list = [], []
    label_map = {'rock': 0, 'paper': 1, 'scissors': 2}
    
    # Collect all file paths first
    file_list = []
    for category, label_idx in label_map.items():
        category_path = os.path.join(folder_path, category)
        if not os.path.exists(category_path):
            subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            if subdirs: category_path = os.path.join(folder_path, subdirs[0], category)
        if not os.path.exists(category_path): continue
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_list.append((os.path.join(category_path, filename), label_idx))
                
    total = len(file_list)
    print(f"Loading {total} images from {os.path.basename(folder_path)}...")
    for idx, (p, label_idx) in enumerate(file_list):
        img = imread_safe(p)
        if img is not None:
            features_list.append(extract_hog_features(img))
            y_label = label_idx
            labels_list.append(y_label)
        if (idx + 1) % 500 == 0 or idx + 1 == total:
            print(f"  Loaded {idx + 1}/{total} images...")
            
    return np.array(features_list), np.array(labels_list)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'dataset', 'train')
    test_dir = os.path.join(base_dir, 'dataset', 'test')
    demo_dir = os.path.join(base_dir, 'demo')

    X_train, y_train = load_data(train_dir)
    X_test, y_test = load_data(test_dir)

    print("\n=== Training SVM Pipeline ===")
    
    # Directly train with the optimal parameters found in experiments
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced', random_state=42))
    ])

    print("Fitting model...")
    model.fit(X_train, y_train)
    print("Training finished.")

    # Evaluation
    y_pred = model.predict(X_test)
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Rock', 'Paper', 'Scissors']))

    # Save model pipeline
    os.makedirs(demo_dir, exist_ok=True)
    joblib.dump(model, os.path.join(demo_dir, 'rps_hog_svm_model.pkl'))
    print("Model Pipeline saved successfully to demo/rps_hog_svm_model.pkl")

if __name__ == "__main__":
    main()
