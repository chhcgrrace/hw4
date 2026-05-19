import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

def extract_hog_features(img):
    """膚色偵測專家版 HOG 特徵：利用 YCrCb 過濾背景雜訊"""
    if img is None: return None
    
    # 1. 轉換至 YCrCb 色彩空間 (膚色分離最強大)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # 2. 定義膚色範圍 (Cr 通道與 Cb 通道是關鍵)
    # 這是經典的膚色偵測範圍，能有效過濾掉非人手的顏色
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    
    # 3. 平滑化與降噪
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # 4. 縮放並計算 HOG
    img_resized = cv2.resize(mask, (64, 64))
    features = hog(img_resized, 
                   orientations=9, 
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), 
                   visualize=False)
    return features

def load_data(folder_path):
    """讀取圖片並提取 HOG 特徵"""
    features_list = []
    labels_list = []
    label_map = {'rock': 0, 'paper': 1, 'scissors': 2}
    
    for category, label_idx in label_map.items():
        category_path = os.path.join(folder_path, category)
        
        if not os.path.exists(category_path):
            subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            if subdirs:
                category_path = os.path.join(folder_path, subdirs[0], category)
        
        if not os.path.exists(category_path):
            continue
            
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

    print("=== [1] Extracting HOG features ===")
    X_train, y_train = load_data(train_dir)
    X_test, y_test = load_data(test_dir)

    if len(X_train) == 0:
        print("Error: No training data found!")
        return

    print("\n=== [2] Training Random Forest model ===")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    print("\n=== [3] Evaluation Report (Part 3) ===")
    y_pred = rf_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall: {rec*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")
    
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=['Rock', 'Paper', 'Scissors']))

    os.makedirs(demo_dir, exist_ok=True)
    model_path = os.path.join(demo_dir, 'rps_hog_rf_model.pkl')
    joblib.dump(rf_model, model_path)
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main()
