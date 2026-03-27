import numpy as np
import os
import joblib
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- Cấu hình ---
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

def train():
    print("--- 1. Đang tải dữ liệu (float32) ---")
    X = np.load('features/X.npy').astype(np.float32)
    y = np.load('features/y.npy')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    del X 

    print("--- 2. Chuẩn hóa ---")
    scaler = StandardScaler(copy=False) 
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("--- 3. PCA giảm chiều (Giữ 95% thông tin để chạy cực nhanh) ---")
    # Giảm xuống khoảng 500-1000 chiều để LinearSVC chạy mượt
    pca = PCA(n_components=0.95, whiten=True)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    del X_train, X_test
    print(f"Số chiều mới: {X_train_pca.shape[1]}")

    print("--- 4. Huấn luyện LinearSVC (Tối ưu cho dữ liệu lớn) ---")
    # LinearSVC chạy cực nhanh với số lượng mẫu lớn
    base_model = LinearSVC(C=1.0, dual=False, max_iter=2000)
    
    # Bọc trong CalibratedClassifierCV để có thể dùng hàm predict_proba cho Demo/App
    model = CalibratedClassifierCV(base_model, cv=3)
    model.fit(X_train_pca, y_train)

    # --- 5. Lưu kết quả ---
    joblib.dump(model, os.path.join(MODEL_DIR, 'svm_kth.pkl'))
    joblib.dump(pca, os.path.join(MODEL_DIR, 'pca.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    # --- 6. Đánh giá ---
    y_pred = model.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print(f"🔥 ĐỘ CHÍNH XÁC: {acc*100:.2f}%")
    print("="*40)
    print(classification_report(y_test, y_pred, target_names=['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']))

if __name__ == '__main__':
    train()