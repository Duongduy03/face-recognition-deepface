# Phase 1: Chọn detector nhanh và đo độ chính xác

> **Trạng thái: chưa làm.** Thời lượng: **1 ngày** (nhánh 1B, nếu phải train lại, thêm 0,5 đến 1 ngày).
> Cần xong: [Phase 0](PHASE_0_CHUAN_BI.md). Phase sau: [PHASE_2_LOI_REALTIME.md](PHASE_2_LOI_REALTIME.md). Tổng quan: [README.md](README.md).

## 1. Mục tiêu

RetinaFace mất 1,2 đến 2,7 giây mỗi frame, nên không chạy được video. Phase này trả lời hai câu hỏi bằng số:

1. Detector nào đủ nhanh cho realtime (mục tiêu: detect một frame 640x480 dưới ~60 ms trên CPU)?
2. Đổi detector thì nhận diện có tệ đi không? Model hiện tại được train trên crop của RetinaFace, còn realtime sẽ dùng crop của detector khác.

Kết thúc phase: chọn được detector cho realtime (dự kiến YuNet), có bảng số liệu làm căn cứ.

## 2. Thời gian

| Buổi | Việc | Ước tính |
|---|---|---|
| Ngày 1 sáng | Bước 1 đến 3: thêm tham số `backend`, hằng số, viết `benchmark_realtime_detector.py` | 3 giờ |
| Ngày 1 chiều | Bước 4 đến 7: chạy đo tốc độ, đo độ chính xác, so crop, điền kết quả, hỏi accept | 3 giờ (phần chạy RetinaFace làm mốc mất khoảng 7 đến 20 phút, tuỳ `--per-person`) |
| Nhánh 1B (nếu cần) | Train lại trên crop YuNet | 0,5 đến 1 ngày, chỉ làm khi được accept riêng |

## 3. Điều kiện đầu vào

- Phase 0 đã accept: `cv2.FaceDetectorYN` và trọng số YuNet dùng được (hoặc đã biết phải đổi sang detector khác).
- Có `output/model_ArcFace_SVM-linear.joblib`, `output/model_ArcFace_KNN.npz`.
- Có `external/pins_demo_test/` (105 người x 10 ảnh).

## 4. Vì sao đo trên `external/pins_demo_test/` mà không dùng `dataset/`

`train_model.py:68` (`train_algorithm`) fit mô hình trên **toàn bộ** 5.459 ảnh của tập con 109 người. Ảnh trong `dataset/` vì thế đã nằm trong dữ liệu train, đo trên đó sẽ cho độ chính xác cao giả. `external/pins_demo_test/` gồm 10 ảnh mỗi người của 105 người Pins, tách ra từ lúc chuẩn bị dữ liệu (`prepare_pins.py`) và chưa từng được đưa vào train. Đây là tập kiểm tra đúng, và giống tình huống thật (người đã biết, ảnh mới).

Lưu ý: tên thư mục trong `external/pins_demo_test/` phải khớp tên lớp của model (một số tên viết thường, ví dụ `alycia dabnem carey`). Script bỏ qua người không có trong `model.classes_` và in số người bị bỏ.

Có sẵn một lần đánh giá trước trên tập này: `external/demo_test_result.csv` (cột `person, file, model, pred, dist, thr, result`, dùng KNN theo từng embedding). Chỉ để tham khảo, không phải mốc của phase này.

## 5. Các bước

### Bước 1. Thêm tham số `backend` cho `detect_faces`

Sửa `face_pipeline.py:45` (hiện: `detect_faces(bgr_image, drop_undetected=True)`):

```python
def detect_faces(bgr_image, drop_undetected=True, backend=None):
    ...
    face_objs = DeepFace.extract_faces(
        img_path=bgr_image,
        detector_backend=backend or DETECTOR_BACKEND,   # None -> RetinaFace như cũ
        enforce_detection=False,
        align=True,
    )
```

Chỉ đổi một dòng và chữ ký. Mọi nơi đang gọi `detect_faces(img)` (app.py, train, benchmark) không đổi hành vi. Cập nhật docstring nói rõ `backend=None` nghĩa là `DETECTOR_BACKEND`.

### Bước 2. Thêm hằng số vào `config.py`

Cạnh `DETECTOR_BACKEND` (`config.py:77`):

```python
# Detector dùng cho realtime (nhanh hơn RetinaFace nhiều). Chọn sau khi đo ở docs/realtime/PHASE_1_DETECTOR.md.
REALTIME_DETECTOR = "yunet"
```

**Không** đổi `DETECTOR_BACKEND` và **không** đổi `PIPELINE_FINGERPRINT` ở phase này: chúng mô tả pipeline đã train model.

### Bước 3. Viết `benchmark_realtime_detector.py`

Giao diện dòng lệnh:

```bash
python benchmark_realtime_detector.py \
    --detectors yunet opencv ssd fastmtcnn centerface retinaface \
    --algos SVM-linear KNN \
    --per-person 3 \
    --out results/realtime
# tuỳ chọn: --skip-speed  --skip-accuracy  --no-crops
```

Quy ước: `import config` trước `deepface` (như các script khác). Dùng `imread_unicode`. Đọc model bằng `classifiers.load(classifiers.model_file(algo, BENCH_EMBEDDING))`, embed bằng `embed_face(face_bgr, BENCH_EMBEDDING)`, tính độ tin cậy bằng `classifiers.scores` và ngưỡng `meta["threshold"]` giống `predict_name_with_algo` (`app.py:201`). Detector không build được (ví dụ `centerface` thiếu trọng số) thì ghi `unavailable` vào bảng và đi tiếp, không làm script dừng.

**Phần A: tốc độ** (`detector_speed.csv`)
- Tạo khung hình giả lập webcam 640x480 từ ảnh Pins: khung 1 mặt (một ảnh giữ tỉ lệ, đặt giữa nền xám), khung 2 mặt và khung 3 mặt (2 hoặc 3 ảnh xếp ngang, cao 360 px).
- Mỗi detector: chạy thử 1 lần để nạp model, rồi đo 30 lần mỗi loại khung. Ghi trung vị, p95, số mặt tìm thấy trung bình.
- Cột: `detector, frame_kind, n_runs, median_ms, p95_ms, mean_faces, status`.

**Phần B: độ chính xác** (`detector_compare.csv`)
- Với mỗi người trong `external/pins_demo_test/` (chỉ người có trong `model.classes_`), lấy `--per-person` ảnh đầu theo thứ tự tên file (cố định, không ngẫu nhiên để lặp lại được; mặc định 3, tức khoảng 315 ảnh).
- Với mỗi (detector, ảnh): detect, lấy mặt lớn nhất bằng `largest_face`, embed, dự đoán bằng từng thuật toán, áp ngưỡng Unknown.
- Tính cho mỗi (detector, thuật toán): `n_images`, `detect_rate` (tỉ lệ ảnh tìm thấy ít nhất một mặt), `top1_acc_detected` (đúng trong số ảnh có mặt), `top1_acc_all` (đúng trên toàn bộ ảnh, ảnh không tìm thấy mặt tính là sai), `unknown_rate` (tỉ lệ bị gán Unknown), `median_detect_ms`.
- `retinaface` là mốc so sánh. Ghi thêm cột `delta_vs_retinaface` (điểm phần trăm của `top1_acc_all`).

**Phần C: so sánh crop** (`crop_similarity.csv`, `crops_compare.jpg`)
- Với 100 ảnh đầu của phần B: embed crop RetinaFace và crop của từng detector, tính độ tương đồng cosine giữa hai embedding (embedding đã L2-normalize nên là tích vô hướng). Ghi trung vị, phân vị 5%.
- Ghép 12 ảnh cạnh nhau (crop RetinaFace, crop từng detector) vào `results/realtime/crops_compare.jpg` để nhìn bằng mắt: căn chỉnh, độ phủ trán cằm, viền đen.
- Ý nghĩa: cosine cao (gần 1) nghĩa là crop gần như cho cùng embedding; thấp nghĩa là model đã train có thể bị lệch.

Đồng thời in ra `shape` của `face_bgr` của từng detector cho vài ảnh, vì DeepFace có thể trả về kích thước crop khác nhau giữa các detector (cần kiểm tra thực tế, chưa xác nhận).

Script **không** sửa model, không ghi vào `output/` hay `dataset/`. Chỉ ghi vào `results/realtime/`.

### Bước 4. Chạy đo tốc độ

```bash
python benchmark_realtime_detector.py --skip-accuracy
```

Mong đợi: YuNet và `opencv` (Haar) nhanh nhất; RetinaFace chậm nhất (khoảng 1,2 đến 2,7 giây). Điền bảng ở mục 7.1.

### Bước 5. Chạy đo độ chính xác

```bash
python benchmark_realtime_detector.py --skip-speed --per-person 3
```

Nếu muốn số chắc hơn thì `--per-person 10` (toàn bộ ~1.000 ảnh; với RetinaFace mất khoảng 20 đến 45 phút). Điền bảng ở mục 7.2 và 7.3.

### Bước 6. Quyết định

Áp dụng bảng quyết định (ngưỡng đề xuất, người dùng có thể đổi khi accept):

| Điều kiện (với thuật toán demo SVM-linear; KNN cũng không tệ hơn nhiều) | Quyết định |
|---|---|
| `detect_rate` của detector nhanh ≥ 97% **và** `top1_acc_all` tụt không quá 3 điểm so với RetinaFace **và** detect ≤ 60 ms/khung 640x480 | **Nhánh A:** dùng detector đó với model hiện tại, không train lại. Đi tiếp Phase 2 |
| Tụt quá 3 điểm nhưng một detector khác đạt điều kiện | **Nhánh B:** đổi `REALTIME_DETECTOR` sang detector đó |
| Không detector nhanh nào đạt | **Nhánh C (1B):** train lại model trên crop của detector nhanh, hoặc giữ RetinaFace + tracking thưa (mỗi 1 đến 2 giây một lần detect) và chấp nhận giật hơn. Hỏi người dùng |

### Bước 7. Cập nhật `config.py` theo quyết định

Nhánh A: không cần sửa gì thêm ngoài hằng `REALTIME_DETECTOR` đã có. Nhánh B: sửa giá trị hằng. Ghi quyết định và lý do vào mục 7.4.

### Nhánh 1B (chỉ làm khi người dùng accept riêng): train lại trên crop của detector nhanh

Mục đích: model quen với crop YuNet nên không bị lệch. **Giữ nguyên model cũ** để trang upload không bị ảnh hưởng.

1. Detect lại toàn bộ ảnh của tập con 109 người bằng detector nhanh (nhanh: vài phút), lưu cache vào thư mục riêng `output/faces_yunet/` (không đụng `output/faces/`).
2. Embed ArcFace các crop mới (khoảng 5.459 ảnh x 0,1 đến 0,18 giây, tương đương 10 đến 16 phút).
3. Thêm cờ `--detector` cho `train_model.py`: khi có cờ thì đọc cache mới, lưu model với **hậu tố tên file** (ví dụ `output/model_ArcFace_SVM-linear_yunet.joblib`) và ghi `pipeline_fingerprint` đúng (`detector=yunet;...`). Mặc định (không cờ) y hệt hiện tại.
4. Đo lại theo bước 5 với model mới; trang upload tiếp tục dùng model cũ (RetinaFace).
5. Cập nhật `classifiers.model_file` hoặc tham số để trang realtime chọn được model hậu tố (Phase 3).

Nhánh 1B là việc lớn hơn dự kiến; phải được accept trước khi bắt đầu, và ước tính thêm 0,5 đến 1 ngày.

## 6. Kiểm tra trang upload không đổi

Sau bước 1 (đã sửa `detect_faces`): chạy `streamlit run app.py`, upload lại 3 ảnh tham chiếu của Phase 0 với SVM-linear và KNN; kết quả phải khớp bảng tham chiếu (tên, số mặt, độ tin cậy giống đến 3 chữ số thập phân). Nếu khác, hoàn tác bước 1 và điều tra trước khi đi tiếp.

## 7. Kết quả (điền sau khi làm)

### 7.1 Tốc độ detect (ms, khung 640x480, CPU)

| Detector | 1 mặt: trung vị | 1 mặt: p95 | 2 mặt: trung vị | 3 mặt: trung vị | Ghi chú |
|---|---|---|---|---|---|
| retinaface | | | | | mốc |
| yunet | | | | | |
| opencv | | | | | |
| ssd | | | | | |
| fastmtcnn | | | | | |
| centerface | | | | | |

### 7.2 Độ chính xác trên `external/pins_demo_test/` (SVM-linear)

| Detector | n ảnh | detect_rate | top1_acc_detected | top1_acc_all | unknown_rate | Chênh so với RetinaFace (điểm %) |
|---|---|---|---|---|---|---|
| retinaface | | | | | | 0 |
| yunet | | | | | | |
| ... | | | | | | |

### 7.3 Độ chính xác (KNN) và tương đồng crop

| Detector | top1_acc_all (KNN) | Cosine crop so với RetinaFace: trung vị | Phân vị 5% |
|---|---|---|---|
| yunet | | | |
| ... | | | |

### 7.4 Quyết định

| Mục | Giá trị |
|---|---|
| Detector chọn cho realtime | |
| Nhánh (A, B hay C/1B) | |
| Lý do (dẫn số từ bảng trên) | |
| Người accept, ngày | |

## 8. Tiêu chí hoàn thành

- [ ] `detect_faces(..., backend=None)` hoạt động; trang upload cho kết quả y như bảng tham chiếu Phase 0.
- [ ] `results/realtime/detector_speed.csv`, `detector_compare.csv`, `crop_similarity.csv`, `crops_compare.jpg` đã tạo.
- [ ] Các bảng 7.1 đến 7.4 đã điền.
- [ ] `config.REALTIME_DETECTOR` đúng với quyết định.
- [ ] Không file nào trong `output/model_ArcFace_*` bị ghi đè.

## 9. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Detector nhanh trả `facial_area` hoặc landmark thiếu nên align khác RetinaFace | Phần C đo tương đồng crop; nếu thấp thì xét nhánh 1B |
| Một số detector không dùng được (thiếu trọng số, `mediapipe` chưa cài) | Ghi `unavailable`, bỏ qua; không cài thêm gói mới nếu chưa hỏi người dùng |
| Mẫu 3 ảnh mỗi người quá ít nên chênh lệch nằm trong sai số | Chạy `--per-person 10`; báo khoảng tin cậy thô (khoảng 1/√n) cạnh số liệu |
| Tên người thư mục không khớp lớp của model | Script in số người bị bỏ; báo trong kết quả |

## 10. Điểm dừng

Báo người dùng: bảng 7.1 đến 7.3, quyết định đề xuất theo bước 6 và lý do. Hỏi: **"Chọn detector nào cho realtime, và có đi tiếp Phase 2 không (hay phải làm nhánh 1B trước)?"** Cập nhật bảng trạng thái ở [README.md](README.md) mục 7.
