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
    # 使用剛才訓練好的模型
    model_path = 'rps_hog_rf_model.pkl'
    # 測試集路徑
    test_dir = '../dataset/test'

    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案 {model_path}，請先執行 train_hog_random_forest.py")
        return

    print("⏳ 載入 HOG+RF 模型...")
    model = joblib.load(model_path)

    label_map = {'rock': 0, 'paper': 1, 'scissors': 2}
    X_test, y_test = [], []

    print("📂 正在處理測試集 HOG 特徵...")
    for category, label_idx in label_map.items():
        category_path = os.path.join(test_dir, category)
        if not os.path.exists(category_path):
            continue
        
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(category_path, filename))
                if img is not None:
                    X_test.append(extract_hog_features(img))
                    y_test.append(label_idx)

    if not X_test:
        print("❌ 找不到測試圖片！")
        return

    y_pred = model.predict(X_test)
    print(f"\n🎯 測試準確率: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\n📊 分類報告：")
    print(classification_report(y_test, y_pred, target_names=['Rock', 'Paper', 'Scissors']))

if __name__ == "__main__":
    main()
