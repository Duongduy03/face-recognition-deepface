# Ứng dụng thuật toán KNN vào bài toán nhận diện người trong ảnh

## 1. Giới thiệu dự án

Hệ thống nhận diện khuôn mặt sử dụng **DeepFace** để detect và trích đặc trưng (embedding),
kết hợp thuật toán **K-Nearest Neighbors (KNN) tự cài đặt** (`knn_func.py`) để phân loại.

Dự án gồm hai phần:

- **Ứng dụng demo (Streamlit)**: thu thập ảnh từ camera, nhận diện khuôn mặt trong ảnh upload,
  chọn giữa các mô hình embedding đã train, xem trang so sánh mô hình.
- **Bộ so sánh mô hình (`benchmark.py`)**: so sánh **9 mô hình embedding** (mỗi mô hình ghép với KNN)
  bằng **6 độ đo**, đo **tốc độ FPS** và phân tích **các ca nhận diện sai**. DeepFace hỗ trợ 11 mô hình nhưng
  2 mô hình (`DeepFace`, `Buffalo_L`) không chạy được trong môi trường này, lý do ở mục 2.1 của tài liệu.
  Chi tiết phương pháp và kết quả: [docs/SO_SANH_MO_HINH.md](docs/SO_SANH_MO_HINH.md).

Pipeline: `ảnh → RetinaFace (detect + align) → embedding (ArcFace, Facenet, ...) → L2-normalize → KNN → tên người`

## 2. Mục tiêu

- Xây dựng mô hình nhận diện người dễ hiểu, giúp nắm cách hoạt động của KNN trong thị giác máy tính.
- Minh họa pipeline: thu thập → xử lý → train → nhận diện.
- So sánh hiệu quả của nhiều mô hình embedding khi dùng chung bộ phân loại KNN.

## 3. Cấu trúc project

```
├── app.py               # Giao diện Streamlit: chụp ảnh / nhận diện / so sánh mô hình
├── config.py            # Danh sách mô hình, đường dẫn, K, seed (nơi khai báo duy nhất)
├── face_pipeline.py     # detect, embed, cache khuôn mặt/embedding (dùng chung cho mọi script)
├── knn_func.py          # Thuật toán KNN tự cài đặt
├── prepare_data.py      # Tải dataset LFW về dataset/
├── train_model.py       # Train KNN cho một mô hình -> output/model_knn_<mô hình>.npz
├── benchmark.py         # So sánh 10 mô hình -> results/
├── plot_results.py      # Vẽ biểu đồ từ results/
├── requirements.txt
├── docs/SO_SANH_MO_HINH.md   # Báo cáo so sánh mô hình
├── dataset/<Tên người>/*.jpg # Ảnh dữ liệu (không đưa lên git)
├── output/              # Model đã train, cache embedding/khuôn mặt (không đưa lên git)
└── results/             # Kết quả benchmark: csv, biểu đồ, ảnh các ca lỗi
```

> Thư mục `haar/` là di sản của phiên bản đầu dùng Haar Cascade, hiện không còn được sử dụng.

## 3.1. Cách chạy nhanh

```bash
python prepare_data.py                      # tải LFW về dataset/ (bỏ qua nếu dùng ảnh riêng)
python benchmark.py                         # so sánh 9 mô hình (chạy lâu, xem docs)
python plot_results.py                      # vẽ biểu đồ
python train_model.py                       # train 2 mô hình tốt nhất (đọc results/best_models.json)
streamlit run app.py
```

## 4. Yêu cầu hệ thống

- Python 3.12 hoặc 3.13 (đã chạy thử trên 3.12), Windows/Linux/Mac
- Cần internet ở lần chạy đầu: DeepFace tự tải trọng số mô hình (vài trăm MB mỗi mô hình)

## 5. Cách triển khai

### Bước 1: Tạo môi trường ảo

```bash
py -3.12 -m venv .venv
```

Kích hoạt:
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Windows CMD: `.venv\Scripts\activate.bat`
- Linux/Mac: `source .venv/bin/activate`

### Bước 2: Cài thư viện

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 3: Chuẩn bị dữ liệu

Có hai cách, có thể dùng cả hai (cùng nằm trong `dataset/<Tên người>/`):

- Chạy `python prepare_data.py` để tải bộ LFW (62 người, tối đa 30 ảnh/người).
- Hoặc chạy `streamlit run app.py`, vào mục **📸 Chụp ảnh**, nhập tên và chụp khoảng 10–20 ảnh.

### Bước 4: Train

```bash
python train_model.py --model ArcFace Facenet512
```

Tạo `output/model_knn_<mô hình>.npz`. Không truyền `--model` thì train các mô hình trong
`results/best_models.json` (do `benchmark.py` tạo ra).

### Bước 5: Nhận diện

```bash
streamlit run app.py
```

Vào mục **🔍 Nhận diện**, chọn mô hình ở thanh bên rồi upload ảnh.
Mục **📊 So sánh mô hình** hiển thị kết quả của `benchmark.py`.
