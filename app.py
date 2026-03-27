import cv2
import numpy as np
import joblib
import os
from skimage.feature import hog

# --- Cấu hình hệ thống ---
MODEL_DIR = 'models'
IMG_SIZE = (64, 64)
N_FRAMES = 12
CLASSES = ['Boxing', 'Handclapping', 'Handwaving', 'Jogging', 'Running', 'Walking']
COLORS = [(0,0,255), (0,255,255), (255,255,0), (0,165,255), (255,0,0), (0,255,0)]

# Load bộ 3 file đã train từ ổ D hoặc Colab
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
pca = joblib.load(os.path.join(MODEL_DIR, 'pca.pkl'))
model = joblib.load(os.path.join(MODEL_DIR, 'svm_kth.pkl'))

def draw_bar_chart(frame, probs):
    """Vẽ biểu đồ xác suất các hành động lên màn hình"""
    for i, p in enumerate(probs):
        width = int(p * 150)
        cv2.rectangle(frame, (10, 60 + i*30), (10 + width, 80 + i*30), COLORS[i], -1)
        cv2.putText(frame, f"{CLASSES[i]}: {p*100:.1f}%", (10, 75 + i*30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def run_demo(video_source=0):
    cap = cv2.VideoCapture(video_source)
    frames_buffer = []

    print("--- Đang khởi động Demo (Nhấn 'q' để thoát) ---")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Tiền xử lý hiển thị
        display_frame = cv2.flip(frame, 1) # Soi gương cho dễ nhìn
        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMG_SIZE)
        frames_buffer.append(resized)

        if len(frames_buffer) == N_FRAMES:
            # 1. Trích xuất đặc trưng (HOG + MHI)
            indices = [0, N_FRAMES//2, -1]
            hog_features = []
            for idx in indices:
                fd = hog(frames_buffer[idx], orientations=12, pixels_per_cell=(8,8), 
                         cells_per_block=(2,2), block_norm='L2-Hys')
                hog_features.append(fd)
            
            # Tính MHI (Motion History Image)
            mhi = np.zeros(IMG_SIZE, dtype=np.float32)
            for i in range(1, len(frames_buffer)):
                diff = cv2.absdiff(frames_buffer[i], frames_buffer[i-1])
                _, thresh = cv2.threshold(diff, 25, 1, cv2.THRESH_BINARY)
                mhi = np.maximum(mhi * 0.7, thresh.astype(np.float32))
            
            mhi_hog = hog(mhi, orientations=12, pixels_per_cell=(8,8), 
                          cells_per_block=(2,2), block_norm='L2-Hys')
            
            # Hợp nhất đặc trưng
            final_features = np.concatenate([*hog_features, mhi_hog]).reshape(1, -1)
            
            # 2. Dự đoán với Scaler và PCA
            feat_scaled = scaler.transform(final_features)
            feat_pca = pca.transform(feat_scaled)
            
            # Lấy xác suất của tất cả các lớp
            probs = model.predict_proba(feat_pca)[0]
            pred_idx = np.argmax(probs)
            
            # 3. Vẽ giao diện
            label = f"PREDICTION: {CLASSES[pred_idx].upper()}"
            cv2.rectangle(display_frame, (0,0), (350, 250), (0,0,0), -1) # Khung đen nền
            cv2.putText(display_frame, label, (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            draw_bar_chart(display_frame, probs)
            
            # Trượt cửa sổ (Sliding window)
            frames_buffer.pop(0)

        cv2.imshow('KTH ACTION RECOGNITION - DUT Senior Project', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Thay 0 bằng link file .avi nếu muốn test video có sẵn
    run_demo(0)