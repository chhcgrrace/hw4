import cv2
import joblib
import numpy as np
from skimage.feature import hog

def extract_hog_features(img):
    """提取 HOG 特徵"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, (64, 64))
    features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    return features

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
        roi_size = 250
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

        # 1. Extract HOG from ROI only
        feat = extract_hog_features(roi).reshape(1, -1)

        # 2. Prediction
        probs = model.predict_proba(feat)[0]
        max_prob = np.max(probs)
        prediction_idx = np.argmax(probs)
        
        # 3. Threshold check for "Error"
        if max_prob < 0.6:
            result_text = "Error"
            color = (0, 0, 255) # Red
        else:
            result_text = f"{labels[prediction_idx]} ({max_prob*100:.1f}%)"
            color = (0, 255, 0) # Green

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
