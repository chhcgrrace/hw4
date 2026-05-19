import cv2
import joblib
import numpy as np
from skimage.feature import hog

def extract_hog_features(img):
    """膚色偵測版 HOG 特徵"""
    if img is None: return None
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    img_resized = cv2.resize(mask, (64, 64))
    features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    return features, mask

def main():
    model_path = 'rps_hog_rf_model.pkl'
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(0)
    labels = ['Rock', 'Paper', 'Scissors']

    print("Camera ROI mode started. Press 'q' to quit...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- ROI Setup (Center Box) ---
        h, w, _ = frame.shape
        roi_size = 320
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size

        # Crop the ROI
        roi = frame[y1:y2, x1:x2]
        
        # Draw ROI box on the main frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Put hand in box", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 1. 影像預處理與膚色偵測
        feat_raw, mask = extract_hog_features(roi)
        
        # --- 面積檢查 (防止空畫面誤判) ---
        # 計算膚色像素佔比
        white_pixels = np.sum(mask == 255)
        white_per = white_pixels / (roi_size * roi_size)
        
        # 顯示處理後的遮罩 (除錯用)
        cv2.imshow("Skin Mask (Debug)", mask)
        
        # 如果膚色部分太少(沒手)或太多(過曝)，直接判定為 Error
        if white_per < 0.05 or white_per > 0.8:
            result_text = "No Hand Detected"
            color = (0, 0, 255)
        else:
            # 2. 進行正式預測
            feat = feat_raw.reshape(1, -1)
            probs = model.predict_proba(feat)[0]
            max_prob = np.max(probs)
            prediction_idx = np.argmax(probs)
            
            if max_prob < 0.45:
                result_text = "Error"
                color = (0, 0, 255)
            else:
                result_text = f"{labels[prediction_idx]} ({max_prob*100:.1f}%)"
                color = (0, 255, 0)

        # 4. Draw result text
        cv2.putText(frame, f"Result: {result_text}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow("Rock Paper Scissors HOG+RF (ROI)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
