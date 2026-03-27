import cv2
import numpy as np
import joblib
import os
import sys

# Thêm đường dẫn để import được từ file extract_features
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from extract_features import IMG_SIZE, N_FRAMES, extract_mhi, hog

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'svm_kth.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

def run_predict(video_path):
    if not os.path.exists(MODEL_PATH):
        return "Lỗi: Bạn cần chạy train.py trước để tạo file model!"
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

    # Đọc video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 120: # Lấy khoảng 120 frame để dự đoán
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), IMG_SIZE)
        frames.append(gray)
    cap.release()
    
    if len(frames) < 2: return "Video quá ngắn"
    
    # Trích xuất y hệt quy trình train
    indices = np.linspace(0, len(frames)-1, N_FRAMES, dtype=int)
    selected = [frames[i] for i in indices]
    
    mid_f = selected[len(selected)//2]
    h1 = hog(mid_f, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    mhi = extract_mhi(selected)
    h2 = hog(mhi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    feat = np.concatenate([h1, h2]).reshape(1, -1)

    # Dự đoán
    feat_scaled = scaler.transform(feat)
    probs = model.predict_proba(feat_scaled)[0]
    idx = np.argmax(probs)
    
    return classes[idx], probs[idx]

if __name__ == '__main__':
    # Ví dụ thử nghiệm với 1 file trong tập data
    test_video = r"C:\Users\Admin\Desktop\action_recognition\data\walking\person01_walking_d1_uncomp.avi"
    
    if os.path.exists(test_video):
        label, conf = run_predict(test_video)
        print(f"Video: {os.path.basename(test_video)}")
        print(f"Dự đoán: {label.upper()} (Độ tin cậy: {conf*100:.2f}%)")
    else:
        print("Vui lòng sửa đường dẫn test_video trong file predict.py để chạy thử.")