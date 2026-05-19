import cv2
import joblib
import numpy as np
from skimage.feature import hog

def extract_hog_features(img):
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
    model_path = 'rps_hog_svm_model.pkl'
    try:
        model = joblib.load(model_path)
        print("HOG+Linear-SVM Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(0)
    labels = ['Rock', 'Paper', 'Scissors']

    print("Camera HOG+SVM mode started. Press 'q' to quit...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # ROI Setup
        h, w, _ = frame.shape
        roi_size = 320
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size

        roi = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # 1. 影像預處理與工程優化
        feat_raw, mask = extract_hog_features(roi)
        
        # --- 面積檢查 (防止空畫面誤判) ---
        white_per = np.sum(mask == 255) / (roi_size * roi_size)
        
        # 顯示膚色遮罩供除錯
        cv2.imshow("Skin Mask (Debug)", mask)

        if white_per < 0.05 or white_per > 0.8:
            result_text = "No Hand Detected"
            color = (0, 0, 255)
        else:
            # --- 幾何形狀檢查 ---
            is_geo_error = False
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                x, y, w_h, h_h = cv2.boundingRect(c)
                aspect_ratio = float(w_h) / h_h if h_h > 0 else 0
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = float(cv2.contourArea(c)) / hull_area if hull_area > 0 else 0
                
                # 嚴格認定：長寬比不對或形狀不紮實則為 Error
                if aspect_ratio < 0.4 or aspect_ratio > 2.5 or solidity < 0.55:
                    is_geo_error = True

            feat = feat_raw.reshape(1, -1)
            probs = model.predict_proba(feat)[0]
            sorted_probs = np.sort(probs)
            max_prob = sorted_probs[-1]
            margin = sorted_probs[-1] - sorted_probs[-2]
            prediction_idx = np.argmax(probs)
            
            if max_prob < 0.55 or margin < 0.2 or is_geo_error:
                result_text = "Error"
                color = (0, 0, 255)
            else:
                result_text = f"{labels[prediction_idx]} ({max_prob*100:.1f}%)"
                color = (0, 255, 0)

        cv2.putText(frame, f"Result: {result_text}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Rock Paper Scissors HOG+SVM (Linear)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
