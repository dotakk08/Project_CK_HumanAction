import cv2
import numpy as np
import os
from skimage.feature import hog
from tqdm import tqdm

# --- Cấu hình ---
DATA_DIR = 'data'
FEAT_DIR = 'features'
IMG_SIZE = (64, 64)
N_FRAMES = 12  # Lấy 12 frame để bao quát đủ chu kỳ bước chân
CLASSES = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

if not os.path.exists(FEAT_DIR):
    os.makedirs(FEAT_DIR)

def extract_mhi(frames, decay=0.6):
    mhi = np.zeros(IMG_SIZE, dtype=np.float32)
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i-1])
        _, thresh = cv2.threshold(diff, 20, 1, cv2.THRESH_BINARY)
        mhi = np.maximum(mhi * decay, thresh.astype(np.float32))
    return mhi

def extract_features_from_segment(segment):
    """Trích xuất 3 mốc thời gian + MHI để đạt độ chính xác cao"""
    # 1. Lấy HOG tại 3 thời điểm: Đầu, Giữa, Cuối 
    indices = [0, len(segment)//2, -1]
    hog_parts = []
    for idx in indices:
        f_hog = hog(segment[idx], orientations=12, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), block_norm='L2-Hys')
        hog_parts.append(f_hog)
    
    # 2. Đặc trưng chuyển động MHI (Vết chuyển động)
    mhi_img = extract_mhi(segment)
    hog_mhi = hog(mhi_img, orientations=12, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), block_norm='L2-Hys')
    
    # Kết hợp: (1764*3) + 1764 = 7056 đặc trưng
    return np.concatenate([*hog_parts, hog_mhi])

def main():
    X, y = [], []
    print("--- Đang bắt đầu trích xuất đặc trưng nâng cao ---")
    
    for label_idx, label_name in enumerate(CLASSES):
        class_path = os.path.join(DATA_DIR, label_name)
        if not os.path.exists(class_path): continue
        
        video_files = [f for f in os.listdir(class_path) if f.endswith('.avi')]
        for vid in tqdm(video_files, desc=f"Processing {label_name}"):
            cap = cv2.VideoCapture(os.path.join(class_path, vid))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(cv2.resize(gray, IMG_SIZE))
            cap.release()

            if len(frames) >= N_FRAMES:
                # Chia video thành các đoạn nhỏ (segments) để tăng lượng mẫu
                step = N_FRAMES // 2
                for i in range(0, len(frames) - N_FRAMES + 1, step):
                    segment = frames[i : i + N_FRAMES]
                    feat = extract_features_from_segment(segment)
                    X.append(feat)
                    y.append(label_idx)

    np.save(os.path.join(FEAT_DIR, 'X.npy'), np.array(X))
    np.save(os.path.join(FEAT_DIR, 'y.npy'), np.array(y))
    print(f"✅ Xong! Tổng cộng {len(X)} mẫu với {X[0].shape[0]} đặc trưng.")

if __name__ == '__main__':
    main()