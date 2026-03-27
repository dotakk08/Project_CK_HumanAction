import streamlit as st
import cv2
import numpy as np
import joblib
import os
import tempfile
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from skimage.feature import hog

# --- CẤU HÌNH HỆ THỐNG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
IMG_SIZE = (64, 64)
N_FRAMES = 12
CLASSES = ['Boxing', 'Handclapping', 'Handwaving', 'Jogging', 'Running', 'Walking']

# --- LOAD ASSETS (Sử dụng Cache) ---
@st.cache_resource
def load_assets():
    # Load bộ 3 file model đã push lên GitHub nãy giờ
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    pca = joblib.load(os.path.join(MODEL_DIR, 'pca.pkl'))
    model = joblib.load(os.path.join(MODEL_DIR, 'svm_kth.pkl'))
    return scaler, pca, model

try:
    scaler, pca, model = load_assets()
except Exception as e:
    st.error(f"❌ Lỗi: Không tìm thấy model trong {MODEL_DIR}. Hãy đảm bảo đã Push đủ 3 file .pkl")

# --- HÀM XỬ LÝ ĐẶC TRƯNG ---
def extract_mhi(frames, decay=0.6):
    mhi = np.zeros(IMG_SIZE, dtype=np.float32)
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i-1])
        _, thresh = cv2.threshold(diff, 20, 1, cv2.THRESH_BINARY)
        mhi = np.maximum(mhi * decay, thresh.astype(np.float32))
    return mhi

# --- LỚP XỬ LÝ WEBCAM (Sửa lỗi Connection & Predict) ---
class ActionProcessor(VideoProcessorBase):
    def __init__(self):
        self.buffer = []
        self.result = "Dang cho du frame..."

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Tiền xử lý (Lật ảnh gương cho dễ múa demo)
        display_img = cv2.flip(img, 1)
        gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMG_SIZE)
        
        # 2. Quản lý Buffer
        self.buffer.append(resized)
        if len(self.buffer) > N_FRAMES:
            self.buffer.pop(0)

        # 3. Dự đoán khi đủ 12 frames
        if len(self.buffer) == N_FRAMES:
            try:
                # Trích xuất 3 HOG (Đầu, Giữa, Cuối) + 1 MHI
                indices = [0, N_FRAMES//2, -1]
                hog_parts = [hog(self.buffer[idx], orientations=12, pixels_per_cell=(8,8), 
                                 cells_per_block=(2,2), block_norm='L2-Hys') for idx in indices]
                
                mhi_img = extract_mhi(self.buffer)
                hog_mhi = hog(mhi_img, orientations=12, pixels_per_cell=(8,8), 
                              cells_per_block=(2,2), block_norm='L2-Hys')
                
                # Nối thành vector đặc trưng 7056 chiều
                feat = np.concatenate([*hog_parts, hog_mhi]).reshape(1, -1)
                
                # Transform & Predict
                feat_pca = pca.transform(scaler.transform(feat))
                probs = model.predict_proba(feat_pca)[0]
                idx = np.argmax(probs)
                conf = probs[idx] * 100
                
                if conf > 35: # Ngưỡng tự tin
                    self.result = f"{CLASSES[idx]} ({conf:.1f}%)"
            except:
                pass
        
        # 4. Vẽ kết quả lên màn hình
        cv2.rectangle(display_img, (0, 0), (450, 60), (0, 0, 0), -1)
        cv2.putText(display_img, self.result, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(display_img, format="bgr24")

# --- GIAO DIỆN CHÍNH ---
st.set_page_config(page_title="AI Action Recognition - DUT", layout="centered")
st.title("🏃 Nhận diện Hành động Người Real-time")
st.markdown("---")

tab1, tab2 = st.tabs(["📁 Tải Video File", "⚡ Trực tiếp từ Webcam"])

with tab1:
    uploaded = st.file_uploader("Kéo thả file video (.mp4, .avi)", type=['mp4', 'avi'])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_video_frame = st.empty()
        v_buffer = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            f_display = cv2.resize(frame, (640, 480))
            f_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), IMG_SIZE)
            v_buffer.append(f_gray)
            
            if len(v_buffer) == N_FRAMES:
                # Logic xử lý tương tự Webcam
                indices = [0, N_FRAMES//2, -1]
                h_parts = [hog(v_buffer[i], orientations=12, pixels_per_cell=(8,8), 
                               cells_per_block=(2,2), block_norm='L2-Hys') for i in indices]
                m_img = extract_mhi(v_buffer)
                h_mhi = hog(m_img, orientations=12, pixels_per_cell=(8,8), 
                            cells_per_block=(2,2), block_norm='L2-Hys')
                
                f_vec = np.concatenate([*h_parts, h_mhi]).reshape(1, -1)
                p = model.predict_proba(pca.transform(scaler.transform(f_vec)))[0]
                label = CLASSES[np.argmax(p)]
                cv2.putText(f_display, f"AI: {label}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                v_buffer.pop(0)

            st_video_frame.image(f_display, channels="BGR")
        cap.release()

with tab2:
    st.info("💡 Mẹo: Nếu bị 'Connection error', hãy thử F5 hoặc dùng Google Chrome.")
    
    webrtc_streamer(
        key="action-recognition",
        video_processor_factory=ActionProcessor,
        # Cấu hình STUN mạnh nhất của Google để vượt tường lửa mạng trường
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
            ]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

st.sidebar.markdown("""
### Hướng dẫn:
1. **Local:** `streamlit run app.py`
2. **Web:** Push lên GitHub & Streamlit Cloud.
3. **Lưu ý:** Cần file `requirements.txt` có `streamlit-webrtc` và `av`.
""")