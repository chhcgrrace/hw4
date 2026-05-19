import cv2
import joblib
import numpy as np
from skimage.feature import hog

def extract_hog_features(img):
    """提取 HOG 特徵，與訓練時保持一致"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, (64, 64))
    features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    return features

def main():
    model_path = 'rps_hog_rf_model.pkl'
    try:
        model = joblib.load(model_path)
        print("✅ HOG+RF 模型載入成功！")
    except Exception as e:
        print(f"❌ 載入失敗: {e}")
        print("請確保已在 demo 資料夾下執行，或模型檔案存在。")
        return

    cap = cv2.VideoCapture(0)
    # 標籤順序需與訓練時一致: 0=Rock, 1=Paper, 2=Scissors
    labels = ['Rock', 'Paper', 'Scissors']

    print("🎥 啟動相機中，按 'q' 鍵退出...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. 提取 HOG 特徵並調整形狀為 (1, n_features)
        feat = extract_hog_features(frame).reshape(1, -1)

        # 2. 預測類別與機率 (用於識別 Error)
        probs = model.predict_proba(feat)[0]
        max_prob = np.max(probs)
        prediction_idx = np.argmax(probs)
        
        # 3. 判斷是否為 Error (機率門檻法)
        # 如果模型對最高機率的信心低於 0.5 (代表手勢不像石頭、剪刀或布)，則顯示 Error
        if max_prob < 0.5:
            result_text = "Error"
            color = (0, 0, 255) # 紅色
        else:
            result_text = f"{labels[prediction_idx]} ({max_prob*100:.1f}%)"
            color = (0, 255, 0) # 綠色

        # 4. 將預測結果繪製在畫面上
        cv2.putText(frame, f"Result: {result_text}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow("Rock Paper Scissors HOG+RF Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
