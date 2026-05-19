import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

def extract_hog_features(img):
    """將圖片轉換為 HOG 特徵"""
    # 1. 轉灰階
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 縮放至固定大小 (64x64)
    img_resized = cv2.resize(img, (64, 64))
    
    # 3. 計算 HOG 特徵
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
            
        print(f"📂 正在處理 {category} 的 HOG 特徵...")
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

    print("=== [1] 提取 HOG 特徵中 ===")
    X_train, y_train = load_data(train_dir)
    X_test, y_test = load_data(test_dir)

    if len(X_train) == 0:
        print("❌ 錯誤：找不到訓練資料！")
        return

    print("\n=== [2] 開始訓練隨機森林模型 ===")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    print("\n=== [3] 模型評估報告 (用於作業 Part 3) ===")
    y_pred = rf_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"🎯 Accuracy (準確率): {acc*100:.2f}%")
    print(f"⚖️ Precision (精確率): {prec*100:.2f}%")
    print(f"🔄 Recall (召回率): {rec*100:.2f}%")
    print(f"🧪 F1-Score: {f1*100:.2f}%")
    
    print("\n📊 詳細分類報告：")
    print(classification_report(y_test, y_pred, target_names=['Rock', 'Paper', 'Scissors']))

    os.makedirs(demo_dir, exist_ok=True)
    model_path = os.path.join(demo_dir, 'rps_hog_rf_model.pkl')
    joblib.dump(rf_model, model_path)
    print(f"\n✅ 模型已儲存至: {model_path}")

if __name__ == "__main__":
    main()
