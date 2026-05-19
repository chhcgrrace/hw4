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
            # --- 新增：數指縫判定 (Convexity Defects) ---
            finger_defects = 0
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                hull = cv2.convexHull(c, returnPoints=False)
                if len(hull) > 3:
                    defects = cv2.convexityDefects(c, hull)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(c[s][0])
                            end = tuple(c[e][0])
                            far = tuple(c[f][0])
                            
                            # 計算三角形邊長
                            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                            c_side = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                            
                            # 使用餘弦定理計算夾角 (指縫的角度通常很尖 < 90度)
                            angle = np.arccos((b**2 + c_side**2 - a**2) / (2*b*c_side)) * 57
                            
                            # 降門檻：距離 > 800 (更靈敏抓指縫)
                            if d > 800 and angle <= 90:
                                finger_defects += 1
                                cv2.circle(roi, far, 5, [0, 0, 255], -1) # 畫出偵測到的洞，DEBUG 用
            
            # 2. 進行預測
            feat = feat_raw.reshape(1, -1)
            probs = model.predict_proba(feat)[0]
            
            sorted_probs = np.sort(probs)
            max_prob = sorted_probs[-1]
            margin = sorted_probs[-1] - sorted_probs[-2]
            prediction_idx = np.argmax(probs)
            pred_label = labels[prediction_idx]
            
            # --- 核心邏輯：雙重驗證 ---
            # 石頭: 0 個洞 | 剪刀: 1 個洞 | 布: >= 3 個洞
            # 其餘情況 (例如比 1 或 比 OK) 都容易被判定為 Error
            is_valid_rps = False
            if pred_label == 'Rock' and finger_defects == 0:
                is_valid_rps = True
            elif pred_label == 'Scissors' and finger_defects == 1:
                is_valid_rps = True
            elif pred_label == 'Paper' and finger_defects >= 3:
                is_valid_rps = True
                
            if max_prob < 0.55 or margin < 0.2 or not is_valid_rps:
                result_text = "Error"
                color = (0, 0, 255)
            else:
                result_text = f"{pred_label} ({max_prob*100:.1f}%)"
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
