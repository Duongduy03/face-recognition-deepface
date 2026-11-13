# Bài toán nhận diện người trong ảnh

## 1. Giới thiệu dự án

Hệ thống nhận diện khuôn mặt: **DeepFace** detect khuôn mặt và trích đặc trưng (embedding), sau đó một
**thuật toán phân loại** quyết định đây là ai. Thuật toán trọng tâm của đề tài là **K-Nearest Neighbors (KNN) tự cài đặt**
(`knn_func.py`), và đề tài so sánh KNN với **9 thuật toán phân loại khác** chạy trên **cùng một embedding**.

Dự án gồm hai phần:

- **Ứng dụng demo (Streamlit, `app.py`)**: chụp ảnh từ camera, nhận diện khuôn mặt trong ảnh tải lên, xem trang so sánh.
  Demo dùng **2 thuật toán tốt nhất** theo kết quả so sánh.
- **Bộ so sánh 10 thuật toán**: 10 bộ phân loại, **6 độ đo**, **tốc độ (FPS, thời gian dự đoán, thời gian huấn luyện)** và
  **phân tích các ca nhận diện sai**. Phương pháp và kết quả nằm ở
  [docs/SO_SANH_THUAT_TOAN.md](docs/SO_SANH_THUAT_TOAN.md).

Pipeline: `ảnh → RetinaFace (detect + align) → embedding ArcFace → L2-normalize → bộ phân loại (1 trong 10) → tên người / Unknown`

> **Trạng thái (cập nhật 2026-11-05):** so sánh 10 thuật toán, biểu đồ, train và demo trong app **đã có và đã chạy một lần** trên tập con 109 người
> (5.459 ảnh). Báo cáo đầy đủ, kèm các hạn chế: [docs/SO_SANH_THUAT_TOAN.md](docs/SO_SANH_THUAT_TOAN.md); số liệu gốc ở `results/algorithms/`.
> Chưa chạy trên toàn bộ 1.774 người. So sánh **9 bộ embedding** (đều dùng KNN) là [phụ lục](docs/SO_SANH_MO_HINH.md) dùng để chọn ArcFace.

**Kết quả chính** (chi tiết và các hạn chế trong báo cáo):

- Sáu thuật toán đầu (SVM tuyến tính, SVM RBF, KNN, Logistic Regression, MLP, LDA) có F1 macro 0,972–0,989 và **không phân biệt chắc chắn được** theo 5-fold CV.
  SVM tuyến tính cao nhất (0,9886); **KNN hạng 3** (0,9792) và không lọt top 2 theo quy tắc đã đặt. Random Forest, Extra Trees (~0,91–0,92) và Decision Tree (0,35) kém rõ ràng.
- **Demo: SVM tuyến tính + KNN** (SVM RBF của quy tắc được thay bằng KNN, thuật toán của đề tài).
- **Loại "người lạ"** khác xa thứ hạng accuracy: MLP để lọt 1,1% và Logistic Regression 3,3% người lạ, SVM tuyến tính 17,4%, KNN 23,7% (ở cùng mức giữ lại ~94% người quen).
- **Tốc độ end-to-end ≈ 0,35 FPS cho cả 10 thuật toán** vì RetinaFace tốn ~2,7 giây mỗi ảnh trên máy đo (CPU, một máy).
- **7 mẫu test bị cả 10 thuật toán cùng sai**: đều là ảnh khó (mặt nghiêng, hóa trang, thiếu sáng, hai người giống nhau), không phải crop hỏng.

## 2. Yêu cầu của giảng viên và trạng thái

| Yêu cầu | Cách thực hiện | Trạng thái |
|---|---|---|
| Làm nhiều mô hình | 10 thuật toán phân loại trên cùng một embedding (`classifiers.py`) | Đã có |
| So sánh giữa 10 mô hình khác nhau | `python benchmark.py` -> `results/algorithms/` (đã chạy một lần trên 109 người) | Đã có |
| Demo 2 mô hình tốt nhất | `app.py` chọn thuật toán; `python train_model.py` train **SVM tuyến tính + KNN** (xem ghi chú dưới bảng) | Đã có |
| 8 mô hình còn lại chỉ cần code | Có trong `classifiers.py`, chạy trong benchmark, train được bằng `train_model.py --algo` | Đã có |
| 5–6 độ đo | Accuracy, Precision, Recall, F1, Top-3, Cohen's Kappa, kèm 5-fold CV | Đã có |
| Tốc độ phản hồi (FPS) | Thời gian huấn luyện, predict, scores, kích thước mô hình, FPS embed + phân loại và end-to-end (`results/algorithms/timing.csv`) | Đã có |
| Các trường hợp nhận diện sai | `results/algorithms/failures_<thuật toán>.csv`, ảnh minh họa, phân tích chéo `overlap.csv` | Đã có |

**Hai thuật toán demo: SVM tuyến tính và KNN** (`config.DEMO_ALGORITHMS`). Quy tắc chọn đặt trước (F1 macro, rồi Accuracy, rồi tốc độ)
cho ra SVM tuyến tính và SVM RBF, nhưng 6 thuật toán đầu không phân biệt được theo cross-validation, nên SVM RBF được thay bằng KNN
(thuật toán của đề tài) để so sánh trực tiếp trong app. Đây là lựa chọn của người làm đề tài, ghi rõ để báo cáo không nhầm với kết quả của quy tắc.

## 3. Mười thuật toán so sánh

Tất cả chạy trên **cùng một embedding (ArcFace)**, cùng tập train/test, siêu tham số cố định (bảng đầy đủ ở
[docs/SO_SANH_THUAT_TOAN.md](docs/SO_SANH_THUAT_TOAN.md)).

| #   | Thuật toán                          | Cài đặt                        |
| --- | ----------------------------------- | ------------------------------ |
| 1   | **KNN** (K = 3, khoảng cách Euclid) | **tự cài** trong `knn_func.py` |
| 2   | SVM tuyến tính                      | scikit-learn                   |
| 3   | SVM nhân RBF                        | scikit-learn                   |
| 4   | Logistic Regression                 | scikit-learn                   |
| 5   | Random Forest                       | scikit-learn                   |
| 6   | Extra Trees                         | scikit-learn                   |
| 7   | Decision Tree                       | scikit-learn                   |
| 8   | Gaussian Naive Bayes                | scikit-learn                   |
| 9   | Linear Discriminant Analysis (LDA)  | scikit-learn                   |
| 10  | MLP (mạng nơ-ron nhiều lớp)         | scikit-learn                   |

Thêm một bản `KNeighborsClassifier` của scikit-learn chỉ để đối chiếu với KNN tự cài (không tính vào 10).

**Vì sao cố định một embedding.** Embedding quyết định phần lớn độ chính xác. Nếu đổi cả embedding lẫn bộ phân loại
thì không biết chênh lệch đến từ đâu. ArcFace được chọn vì F1 macro cao nhất (ngang Facenet512) trong lần so sánh 9 embedding
ở [phụ lục](docs/SO_SANH_MO_HINH.md). Lần so sánh đó chạy trên 62 người LFW, cần xác nhận lại trên bộ dữ liệu hiện tại.

## 4. Dữ liệu

| Nguồn                           | Nội dung                          | Cách nạp                                                              |
| ------------------------------- | --------------------------------- | --------------------------------------------------------------------- |
| LFW (Labeled Faces in the Wild) | ảnh báo chí, tải qua scikit-learn | `python prepare_data.py`                                              |
| Kaggle Pins Face Recognition    | 105 người nổi tiếng, 17.534 ảnh   | `python prepare_pins.py` (cần tải bộ Pins vào `external/pins/` trước) |
| Ảnh chụp trong app              | ảnh của người dùng                | trang **📸 Chụp ảnh**                                                 |

`dataset/` hiện có **1.774 người, 24.424 ảnh**, nhưng mới detect (và cache ở `output/faces/`) được 6.691 ảnh của 386 người
vì RetinaFace mất khoảng 1,3 giây mỗi ảnh trên CPU.

**Dữ liệu của phần so sánh 10 thuật toán:** tập con cố định gồm **109 người có ít nhất 30 ảnh (5.459 ảnh)**, tất cả đã detect xong.
Sau khi có kết quả sẽ chạy lại trên toàn bộ 1.774 người (kế hoạch ở
[docs/KE_HOACH_DATASET_LON.md](docs/KE_HOACH_DATASET_LON.md)). Mỗi thuật toán dùng cùng một cách chia: 80/20 phân tầng theo người
(`seed = 42`), thêm 5-fold cross-validation để có trung bình ± độ lệch chuẩn.

## 5. Cấu trúc project

```
├── app.py                  # Giao diện Streamlit: chụp ảnh / nhận diện / so sánh
├── config.py               # Danh sách embedding, đường dẫn, K, seed, DB (nơi khai báo duy nhất)
├── face_pipeline.py        # detect, embed, cache khuôn mặt/embedding (dùng chung cho mọi script)
├── knn_func.py             # KNN tự cài đặt
├── classifiers.py          # Bảng 10 thuật toán, cùng giao diện fit/predict/scores
├── benchmark.py            # So sánh 10 thuật toán trên 1 embedding (ArcFace) -> results/algorithms/
├── benchmark_embeddings.py # So sánh 9 embedding + KNN (phụ lục), đổi tên từ benchmark.py cũ
├── train_model.py          # Train KNN -> output/model_knn_<embedding>.npz (sẽ thêm --algo, file .joblib)
├── plot_results.py         # Vẽ biểu đồ từ results/
├── prepare_data.py         # Tải LFW vào dataset/
├── prepare_pins.py         # Chia bộ Pins: ảnh train vào dataset/, ảnh test để demo
├── vector_db.py            # Kho vector PostgreSQL + pgvector (tùy chọn, chỉ phục vụ KNN)
├── build_impostors.py      # Tạo ảnh "người lạ" để đo tỉ lệ nhận nhầm
├── eval_impostors.py       # Đo tỉ lệ nhận nhầm người lạ của KNN (chỉ đọc DB)
├── eval_impostors_algos.py # Tỉ lệ nhận nhầm người lạ cho 10 thuật toán -> results/algorithms/impostor_eval.csv
├── docker-compose.yml      # PostgreSQL + pgvector cục bộ
├── .env.example            # Mẫu FACE_DB_URL cho DB dùng chung
├── requirements.txt
├── docs/
│   ├── SO_SANH_THUAT_TOAN.md   # Báo cáo chính: 10 thuật toán
│   ├── SO_SANH_MO_HINH.md      # Phụ lục: so sánh 9 embedding (62 người LFW)
│   └── KE_HOACH_DATASET_LON.md # Kế hoạch chạy trên toàn bộ dataset
├── dataset/<Tên người>/*.jpg   # Ảnh dữ liệu (không đưa lên git)
├── output/                 # Model đã train, cache embedding/khuôn mặt (không đưa lên git)
└── results/                # Kết quả: csv, biểu đồ, ảnh các ca lỗi
```

> `README_DEEPFACE.md` và thư mục `haar/` là di sản của các phiên bản đầu (Facenet, Haar Cascade), không khớp với mã hiện tại.

## 6. Yêu cầu hệ thống

- Python 3.12 hoặc 3.13 (đã chạy thử trên 3.12), Windows/Linux/Mac
- Cần internet ở lần chạy đầu: DeepFace tự tải trọng số mô hình (vài trăm MB mỗi mô hình)
- Không cần GPU. Số đo tốc độ trong tài liệu là trên CPU
- Docker chỉ cần nếu dùng kho vector PostgreSQL (tùy chọn)

## 7. Cách triển khai

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

Các nguồn cùng nằm trong `dataset/<Tên người>/`:

```bash
python prepare_data.py      # LFW
python prepare_pins.py      # Pins: 30 ảnh/người vào dataset/, 10 ảnh/người ra external/pins_demo_test/
```

Hoặc chạy `streamlit run app.py`, vào mục **📸 Chụp ảnh**, nhập tên và chụp khoảng 10–20 ảnh.

### Bước 4: Chạy so sánh

```bash
python benchmark_embeddings.py --skip-fps      # so sánh 9 embedding + KNN (phụ lục)
python plot_results.py --kind embeddings       # biểu đồ của phụ lục -> results/*.png
```

So sánh 10 thuật toán (báo cáo chính):

```bash
python benchmark.py            # đọc cache khuôn mặt, không detect; ~13 phút trên máy phát triển. --quick: 20 người, ghi vào output/benchmark_quick/
python eval_impostors_algos.py # tỉ lệ người lạ bị nhận nhầm cho 10 thuật toán (~5 phút; cần crop người lạ do build_impostors.py tạo)
python plot_results.py         # đọc results/algorithms/, ghi 8 ảnh PNG vào đó
```

**Lưu ý:** `benchmark_embeddings.py` (và `train_model.py --model`) gọi `ensure_face_cache()`, hàm này detect mọi ảnh trong `dataset/` chưa
có trong cache (hiện khoảng 17.700 ảnh, vài giờ trên CPU). Chi tiết ở [docs/KE_HOACH_DATASET_LON.md](docs/KE_HOACH_DATASET_LON.md).

### Bước 5: Train

```bash
python train_model.py                       # 2 thuật toán demo (SVM-linear + KNN) trên ArcFace -> output/model_ArcFace_*.{joblib,npz}
python train_model.py --algo SVM-RBF MLP    # thuật toán khác trong 10 thuật toán
python train_model.py --model ArcFace       # KNN theo embedding trên cả dataset/ -> output/model_knn_ArcFace.npz (chế độ cũ)
```

Để **cả 10 thuật toán** chọn được trong app, train nốt 8 thuật toán còn lại (một lần, khoảng 10–15 phút CPU, thêm ~2,5 GB vào `output/` do Random Forest ~0,8 GB và Extra Trees ~1,6 GB):

```bash
python train_model.py --algo SVM-RBF LogisticRegression RandomForest ExtraTrees DecisionTree GaussianNB LDA MLP
```

Hai chế độ đầu train trên tập con của benchmark (109 người, đọc cache khuôn mặt, **không detect**), ngưỡng Unknown tính bằng dự đoán
out-of-fold 5-fold (percentile 5 của độ tin cậy). Chỉ chế độ cũ (`--model`) mới gọi `ensure_face_cache()`. Không truyền gì thì train
`config.DEMO_ALGORITHMS`; `--algo` không đi cùng `--model` / `--db`. Người mới chụp trong app chưa nằm trong mô hình `--algo` cho đến khi
có ảnh của họ trong cache khuôn mặt và `benchmark.py`/`train_model.py` được chạy lại với danh sách người mới.

### Bước 6: Nhận diện

```bash
streamlit run app.py
```

Vào mục **🔍 Nhận diện**, chọn nguồn **Thuật toán phân loại (ArcFace)** rồi chọn thuật toán ở thanh bên. Ô chọn luôn liệt kê **đủ 10 thuật toán**, hai thuật toán demo
(⭐ SVM-linear, ⭐ KNN) nằm đầu danh sách; thuật toán chưa train có nhãn "(chưa train)" và app hiện lệnh để train. Sau đó tải ảnh
lên. Hai nguồn còn lại (`File .npz`, PostgreSQL) là KNN theo từng embedding như trước. Mục **📊 So sánh mô hình** có hai thẻ: 10 thuật toán
(`results/algorithms/`) và phụ lục 9 embedding (`results/`).

### Tùy chọn: kho vector PostgreSQL (chỉ cho KNN)

```bash
docker compose up -d                          # PostgreSQL + pgvector cục bộ
python train_model.py --model ArcFace --db    # ghi thêm vector vào DB
```

Cả nhóm dùng chung DB cloud thì đặt `FACE_DB_URL` trong `.env` (mẫu ở `.env.example`). Các thuật toán không phải KNN
không dùng DB mà lưu file trong `output/`.
