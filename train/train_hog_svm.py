import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

def extract_hog_features(img):
    """提取 HOG 特徵"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, (64, 64))
    features = hog(img_resized, 
                   orientations=9, 
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), 
                   visualize=False)
    return features

def load_data(folder_path):
    features_list = []
    labels_list = []
    label_map = {'rock': 0, 'paper': 1, 'scissors': 2}
    
    for category, label_idx in label_map.items():
        category_path = os.path.join(folder_path, category)
        if not os.path.exists(category_path):
            subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            if subdirs:
                category_path = os.path.join(folder_path, subdirs[0], category)
        
        if not os.path.exists(category_path): continue
            
        print(f"Processing HOG features for {category}...")
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(category_path, filename))
                if img is not None:
                    feat = extract_hog_features(img)
                    features_list.append(feat)
                    labels_list.append(label_idx)
                    
    return np.array(features_list), np.array(labels_list)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'dataset', 'train')
    test_dir = os.path.join(base_dir, 'dataset', 'test')
    demo_dir = os.path.join(base_dir, 'demo')

    print("=== [1] Extracting HOG features for SVM ===")
    X_train, y_train = load_data(train_dir)
    X_test, y_test = load_data(test_dir)

    print("\n=== [2] Training SVM model (this may take a moment) ===")
    # probability=True is essential for Error detection via confidence
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    model.fit(X_train, y_train)

    print("\n=== [3] Evaluation Report (HOG+SVM) ===")
    y_pred = model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted')*100:.2f}%")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted')*100:.2f}%")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted')*100:.2f}%")
    
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=['Rock', 'Paper', 'Scissors']))

    os.makedirs(demo_dir, exist_ok=True)
    model_path = os.path.join(demo_dir, 'rps_hog_svm_model.pkl')
    joblib.dump(model, model_path)
    print(f"\nHOG+SVM model saved to: {model_path}")

if __name__ == "__main__":
    main()
