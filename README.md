# Eye State Detection System

## 📋 Mô Tả
Hệ thống phát hiện trạng thái mắt (mở/nhắm) sử dụng:
- **Haar Cascade**: Để phát hiện khuôn mặt
- **Dlib**: Để phát hiện landmarks (điểm đặc trưng) và trích xuất vùng mắt
- **SVM**: Mô hình machine learning phân loại mắt mở/nhắm
- **Tkinter**: Giao diện Windows đơn giản để test

## 📁 Cấu Trúc Project
```
D:\MayHoc/
├── app.py                          # Giao diện Windows
├── eye_detector.py                 # Lớp phát hiện và phân loại mắt
├── train.py                        # Script training SVM model
├── setup_model.py                  # Script tải dataset
├── feature_extractor.py            # Feature extractor cũ (không dùng)
├── requirements.txt                # Các package cần thiết
├── utils/
│   ├── __init__.py
│   └── feature_extractor.py        # Feature extractor mới
├── data/
│   └── eyes/
│       ├── open/                   # Ảnh mắt mở (từ dataset)
│       └── closed/                 # Ảnh mắt nhắm (từ dataset)
└── models/
    ├── eye_state_svm_model.pkl     # Model SVM đã train
    └── training_results.png        # Biểu đồ kết quả training
```

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### 1️⃣ Cài đặt Dependencies
```bash
cd D:\MayHoc
pip install -r requirements.txt
```

### 2️⃣ Setup Dataset
```bash
python setup_model.py
```
- Tạo cấu trúc thư mục
- Tải MRL Eye Dataset từ Kaggle (cần đăng nhập kagglehub)

### 3️⃣ Train Model SVM
```bash
python train.py
```
- Load dataset với preprocessing & feature extraction
- Training SVM với hyperparameter tuning
- Lưu model vào `models/eye_state_svm_model.pkl`
- Hiển thị biểu đồ kết quả training

### 4️⃣ Test Giao Diện Windows
```bash
python app.py
```
- Nhấp "Start Detection" để bắt đầu
- Xem trạng thái mắt trái/phải trên giao diện
- Nhấp "Stop Detection" để dừng
- Nhấp "Exit" để thoát

## 📊 Các Thay Đổi Chính

| Mục | Cũ | Mới |
|-----|-----|-----|
| **Face Detection** | Haar Cascade | ✅ MTCNN |
| **Eye Landmarks** | ❌ Không có | ✅ Dlib 68 landmarks |
| **Model** | Random Forest + SVM | ✅ Chỉ SVM |
| **Feature Extractor** | Root folder | ✅ utils/feature_extractor.py |
| **UI Output** | Tiếng beep | ✅ Hiển thị text: Open/Closed |
| **Dataset** | Y như cũ | ✅ MRL Eye Dataset (Kaggle) |

## ⚙️ Cấu Hình Mô Hình

### SVM Hyperparameters (từ GridSearchCV)
```python
C: [0.1, 1, 10, 100]
gamma: ['scale', 'auto']
kernel: ['rbf', 'linear']
cv: 5
test_size: 0.2
```

## 📝 Chi Tiết File

### eye_detector.py
- `detect_faces_mtcnn()`: Phát hiện khuôn mặt bằng MTCNN
- `get_eye_region_from_landmarks()`: Trích xuất vùng mắt từ dlib landmarks
- `predict_eye_state()`: Phân loại mắt bằng SVM
- `process_frame()`: Xử lý frame từ camera

### train.py
- `load_dataset()`: Load ảnh từ `data/eyes/open` và `data/eyes/closed`
- `train_model()`: Train SVM với tuning hyperparameters
- `plot_results()`: Vẽ biểu đồ accuracy & confusion matrix
- `save_model()`: Lưu model vào `models/eye_state_svm_model.pkl`

### app.py
- Giao diện Tkinter đơn giản
- Hiển thị video từ camera
- Cập nhật trạng thái mắt trái/phải real-time
- Màu xanh lá = Open, Đỏ = Closed

### utils/feature_extractor.py
- `preprocess_eye_image()`: Resize, histogram equalization, normalize
- `extract_eye_features()`: Trích xuất features từ ảnh mắt

## 🔧 Lưu Ý Quan Trọng

1. **Dlib Shape Predictor**: Cần download file `shape_predictor_68_face_landmarks.dat`
   - Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   - Extract và đặt cùng thư mục project

2. **Kaggle Authentication**: 
   - Cần đăng nhập Kaggle để tải dataset
   - Chạy: `kagglehub.login()` nếu chưa xác thực

3. **Dataset**: 
   - MRL Eye Dataset từ Kaggle
   - ~2000 ảnh mắt mở, ~1000 ảnh mắt nhắm

4. **Performance**:
   - CPU: Khoảng 30 FPS
   - GPU (CUDA): Nhanh hơn với MTCNN

## 🐛 Troubleshooting

| Lỗi | Nguyên Nhân | Giải Pháp |
|-----|-----------|---------|
| dlib.error | Thiếu shape_predictor | Download & đặt file vào project |
| MTCNNError | Thiếu TensorFlow | `pip install tensorflow` |
| No data found | Dataset chưa tải | Chạy `python setup_model.py` |
| Camera không hoạt động | Camera bị chiếm | Tắt ứng dụng khác dùng camera |

## 📚 Dependencies
```
opencv-python      # Xử lý video/ảnh
scikit-learn       # SVM, GridSearchCV
numpy              # Xử lý array
mtcnn              # Face detection
dlib               # Landmarks detection
tensorflow         # Backend cho MTCNN
pillow             # Image processing cho Tkinter
kagglehub          # Tải dataset từ Kaggle
matplotlib         # Vẽ biểu đồ
tqdm               # Progress bar
```

---
**Tác giả**: Eye State Detection Team  
**Ngày tạo**: 2025-10-31  
**Version**: 1.0

