import streamlit as st
import cv2
import numpy as np
import joblib
import os
import tempfile
from skimage.feature import hog

# --- Cấu hình & Load Model ---
MODEL_DIR = 'models'
IMG_SIZE = (64, 64)
N_FRAMES = 12
CLASSES = ['Boxing', 'Handclapping', 'Handwaving', 'Jogging', 'Running', 'Walking']
COLORS = [(255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 165, 0), (0, 0, 255), (0, 255, 0)]

# Load bộ 3 file model (Đảm bảo đã push lên GitHub)
@st.cache_resource
def load_models():
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    pca = joblib.load(os.path.join(MODEL_DIR, 'pca.pkl'))
    model = joblib.load(os.path.join(MODEL_DIR, 'svm_kth.pkl'))
    return scaler, pca, model

try:
    scaler, pca, model = load_models()
except Exception as e:
    st.error(f"Lỗi load model: {e}. Hãy đảm bảo thư mục models/ có đủ 3 file .pkl")

def extract_mhi(frames, decay=0.6):
    mhi = np.zeros(IMG_SIZE, dtype=np.float32)
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i-1])
        _, thresh = cv2.threshold(diff, 20, 1, cv2.THRESH_BINARY)
        mhi = np.maximum(mhi * decay, thresh.astype(np.float32))
    return mhi

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Action Recognition DUT", layout="wide")
st.title("🚀 Hệ thống Nhận diện Hành động Người (KTH Dataset)")
st.sidebar.header("Cấu hình")
confidence_threshold = st.sidebar.slider("Ngưỡng tin cậy (%)", 0, 100, 50)

uploaded_file = st.file_uploader("Chọn video hành động (mp4, avi)...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Lưu file tạm để OpenCV đọc được
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty() # Khung hiển thị video
    st_label = st.sidebar.empty() # Khung hiển thị kết quả
    
    frames_buffer = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Tiền xử lý
        frame_display = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMG_SIZE)
        frames_buffer.append(resized)
        
        if len(frames_buffer) == N_FRAMES:
            # 1. Trích xuất HOG tại 3 mốc
            indices = [0, N_FRAMES//2, -1]
            hog_parts = [hog(frames_buffer[idx], orientations=12, pixels_per_cell=(8,8), 
                             cells_per_block=(2,2), block_norm='L2-Hys') for idx in indices]
            
            # 2. Trích xuất MHI
            mhi_img = extract_mhi(frames_buffer)
            hog_mhi = hog(mhi_img, orientations=12, pixels_per_cell=(8,8), 
                          cells_per_block=(2,2), block_norm='L2-Hys')
            
            # 3. Dự đoán
            features = np.concatenate([*hog_parts, hog_mhi]).reshape(1, -1)
            feat_scaled = scaler.transform(features)
            feat_pca = pca.transform(feat_scaled)
            
            probs = model.predict_proba(feat_pca)[0]
            max_idx = np.argmax(probs)
            confidence = probs[max_idx] * 100
            
            # Hiển thị kết quả nếu vượt ngưỡng
            if confidence >= confidence_threshold:
                label = CLASSES[max_idx]
                color = (0, 255, 0)
                cv2.putText(frame_display, f"{label} ({confidence:.1f}%)", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                # Hiển thị biểu đồ ở sidebar
                st_label.write(f"### Dự đoán: **{label}**")
                st_label.progress(int(confidence))
            
            frames_buffer.pop(0) # Sliding window

        # Render frame lên Streamlit (Thay cho cv2.imshow)
        st_frame.image(frame_display, channels="BGR")
        
    cap.release()
    st.success("Đã xử lý xong video!")
