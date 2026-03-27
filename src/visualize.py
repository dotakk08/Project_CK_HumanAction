import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# --- Cấu hình đường dẫn ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(BASE_DIR, 'features')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CLASSES  = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

def visualize_results():
    print("--- Đang tải dữ liệu và mô hình nâng cao ---")
    
    try:
        # 1. Load dữ liệu mới (7056 đặc trưng)
        X = np.load(os.path.join(FEAT_DIR, 'X.npy'))
        y = np.load(os.path.join(FEAT_DIR, 'y.npy'))
        
        # 2. Load bộ 3 file đã được GridSearch tối ưu
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        pca    = joblib.load(os.path.join(MODEL_DIR, 'pca.pkl'))
        model  = joblib.load(os.path.join(MODEL_DIR, 'svm_kth.pkl'))
    except Exception as e:
        print(f"❌ LỖI: {e}\n(Hãy đảm bảo bạn đã chạy xong extract_features.py và train.py mới)")
        return

    # 3. Chia tập test (Phải dùng random_state=42 để khớp với lúc train)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Đang phân tích {len(X_test)} mẫu test...")

    # 4. QUY TRÌNH CHUẨN: Scaler -> PCA -> Predict
    X_test_scaled = scaler.transform(X_test)
    X_test_pca    = pca.transform(X_test_scaled)
    y_pred        = model.predict(X_test_pca)

    # 5. Vẽ Confusion Matrix (Ma trận nhầm lẫn)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    
    plt.title('Confusion Matrix - KTH Action Recognition\n(Enhanced HOG + MHI + PCA + SVM)')
    plt.xlabel('Hành động dự đoán')
    plt.ylabel('Hành động thực tế')
    
    # Lưu ảnh phục vụ báo cáo
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix_v2.png'))
    print(f"✅ Đã lưu ma trận nhầm lẫn mới tại: confusion_matrix_v2.png")
    
    # 6. In báo cáo chi tiết
    print("\n" + "="*30)
    print("      BÁO CÁO CHI TIẾT")
    print("="*30)
    print(classification_report(y_test, y_pred, target_names=CLASSES))
    
    plt.show()

if __name__ == '__main__':
    visualize_results()