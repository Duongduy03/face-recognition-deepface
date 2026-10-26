# Face Recognition with DeepFace & KNN

## 🚀 Cải tiến

### Đã thay đổi:
- ✅ Bỏ Haar Cascade (cũ, kém chính xác)
- ✅ Dùng DeepFace RetinaFace để detect face (chính xác hơn nhiều)
- ✅ Lưu ảnh RGB màu thay vì grayscale
- ✅ Giảm số ảnh cần thiết: 10 ảnh/người (thay vì 20)
- ✅ K=3 cho KNN (robust hơn)
- ✅ UI/UX cải thiện

### Architecture:
```
Ảnh → RetinaFace Detection → Facenet Embedding (128-dim) → KNN → Tên người
```

## 📦 Cài đặt

```bash
pip install deepface opencv-python streamlit pandas numpy pillow
pip install tf-keras  # Nếu cần
```

## 🎯 Sử dụng

### 1. Thu thập dữ liệu
```bash
# Chạy app mới
streamlit run app_deepface.py

# Hoặc nếu đã thay thế
streamlit run app.py
```

- Chọn "📸 Chụp ảnh"
- Nhập tên (không dấu)
- Chụp 5-10 ảnh với góc độ khác nhau

### 2. Train model
```bash
python train_model.py
```

Output:
- `output/model_knn.npz` - Model KNN với embeddings
- `data.csv` - Dữ liệu embeddings

### 3. Nhận diện
- Vào app, chọn "🔍 Nhận diện"
- Upload ảnh
- Xem kết quả

## ⚙️ Config (trong app_deepface.py)

```python
MODEL_NAME = "Facenet"  # Facenet, ArcFace, Facenet512
DETECTOR_BACKEND = "retinaface"  # retinaface, mtcnn, opencv, ssd
TARGET_IMAGES_PER_PERSON = 10
K = 3  # trong train_model.py
```

### Model embedding options:

| Model | Vector size | Speed | Accuracy |
|-------|-------------|-------|----------|
| Facenet | 128 | ⚡⚡⚡ | ✅✅✅ |
| Facenet512 | 512 | ⚡⚡ | ✅✅✅✅ |
| ArcFace | 512 | ⚡⚡ | ✅✅✅✅ |
| VGG-Face | 4096 | 🐌 | ✅✅✅ |

### Detector options:

| Detector | Speed | Accuracy |
|----------|-------|----------|
| retinaface | ⚡⚡ | ✅✅✅✅ (best) |
| mtcnn | ⚡⚡ | ✅✅✅ |
| opencv | ⚡⚡⚡ | ✅✅ |
| ssd | ⚡⚡⚡ | ✅✅ |

## 📊 So sánh trước/sau

### Trước (Haar Cascade + Pixel):
- Haar Cascade detection (kém chính xác)
- 10,000 chiều pixel values
- Cần 15-20 ảnh/người
- Nhạy cảm với ánh sáng
- K=1

### Sau (DeepFace):
- RetinaFace detection (rất chính xác)
- 128 chiều embeddings (Facenet)
- Chỉ cần 5-10 ảnh/người
- Robust với ánh sáng, góc độ
- K=3

## 🔧 Troubleshooting

### Lỗi "No module named 'tf-keras'"
```bash
pip install tf-keras
```

### Detect chậm
- Đổi detector: `DETECTOR_BACKEND = "opencv"`
- Hoặc dùng model nhẹ hơn

### Accuracy thấp
- Chụp thêm ảnh đa dạng
- Thử model khác: `MODEL_NAME = "ArcFace"`
- Tăng K: `K = 5`

## 📁 Files

- `app_deepface.py` - App mới với DeepFace
- `app_old.py` - Backup app cũ
- `train_model.py` - Training script
- `knn_func.py` - KNN implementation
- `dataset/` - Ảnh training
- `output/` - Model files

## 🎓 Học thêm

- [DeepFace GitHub](https://github.com/serengil/deepface)
- [Facenet Paper](https://arxiv.org/abs/1503.03832)
- [RetinaFace Paper](https://arxiv.org/abs/1905.00641)
