import streamlit as st
import cv2
import numpy as np
import joblib
import os
from skimage.feature import hog
import time

# --- Cấu hình hệ thống ---
MODEL_DIR = 'models'
IMG_SIZE = (64, 64)
N_FRAMES = 12
CLASSES = ['Boxing', 'Handclapping', 'Handwaving', 'Jogging', 'Running', 'Walking']

# --- Load Models (Sử dụng Cache để mượt hơn) ---
@st.cache_resource
def load_models():
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    pca = joblib.load(os.path.join(MODEL_DIR, 'pca.pkl'))
    model = joblib.load(os.path.join(MODEL_DIR, 'svm_kth.pkl'))
    return scaler, pca, model

try:
    scaler, pca, model = load_models()
except Exception as e:
    st.error(f"❌ Không tìm thấy Model! Admin hãy kiểm tra thư mục {MODEL_DIR}")

def extract_mhi(frames, decay=0.6):
    mhi = np.zeros(IMG_SIZE, dtype=np.float32)
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i-1])
        _, thresh = cv2.threshold(diff, 20, 1, cv2.THRESH_BINARY)
        mhi = np.maximum(mhi * decay, thresh.astype(np.float32))
    return mhi

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="DUT Action AI - Localhost", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stTitle { color: #1E3A8A; font-family: 'Segoe UI'; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Nhận diện Hành động Real-time (Localhost)")
st.caption("Đồ án tốt nghiệp - Khoa Công nghệ Thông tin - DUT")

# Sidebar
st.sidebar.header("⚙️ Cấu hình Webcam")
run_webcam = st.sidebar.checkbox('Bật Camera', value=False)
conf_threshold = st.sidebar.slider("Ngưỡng tin cậy (%)", 0, 100, 40)
show_mhi = st.sidebar.checkbox('Hiển thị MHI (Vết chuyển động)')

# Layout chính
col1, col2 = st.columns([2, 1])

with col1:
    st_frame = st.empty() # Khung video chính

with col2:
    st.write("### 📊 Kết quả phân tích")
    st_label = st.empty()
    st_probs = st.empty()
    st_mhi_view = st.empty()

# --- XỬ LÝ WEBCAM ---
if run_webcam:
    cap = cv2.VideoCapture(0) # Mở Webcam local
    frames_buffer = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Lật ảnh cho giống gương
        frame = cv2.flip(frame, 1)
        frame_out = cv2.resize(frame, (640, 480))
        
        # Tiền xử lý cho Model
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMG_SIZE)
        frames_buffer.append(resized)
        
        if len(frames_buffer) == N_FRAMES:
            # 1. Trích xuất HOG (3 mốc thời gian)
            indices = [0, N_FRAMES//2, -1]
            hog_parts = [hog(frames_buffer[idx], orientations=12, pixels_per_cell=(8,8), 
                             cells_per_block=(2,2), block_norm='L2-Hys') for idx in indices]
            
            # 2. Trích xuất MHI
            mhi_img = extract_mhi(frames_buffer)
            hog_mhi = hog(mhi_img, orientations=12, pixels_per_cell=(8,8), 
                          cells_per_block=(2,2), block_norm='L2-Hys')
            
            # 3. Predict
            features = np.concatenate([*hog_parts, hog_mhi]).reshape(1, -1)
            feat_pca = pca.transform(scaler.transform(features))
            
            probs = model.predict_proba(feat_pca)[0]
            max_idx = np.argmax(probs)
            score = probs[max_idx] * 100
            
            # 4. Hiển thị lên UI
            if score >= conf_threshold:
                label = CLASSES[max_idx]
                cv2.putText(frame_out, f"{label} {score:.1f}%", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                st_label.markdown(f"## Hành động: **{label}**")
                # Vẽ biểu đồ xác suất đơn giản
                st_probs.bar_chart({CLASSES[i]: probs[i] for i in range(len(CLASSES))})
            
            if show_mhi:
                # Chuẩn hóa MHI để hiển thị
                mhi_vis = cv2.normalize(mhi_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                st_mhi_view.image(mhi_vis, caption="Motion History Image (MHI)", width=200)

            frames_buffer.pop(0) # Sliding window

        # Đẩy frame lên trình duyệt
        st_frame.image(frame_out, channels="BGR")
        
        # Tránh quá tải CPU
        time.sleep(0.01)
        
        if not run_webcam:
            break
            
    cap.release()
else:
    st_frame.info("Nhấn 'Bật Camera' ở thanh bên trái để bắt đầu demo.")
