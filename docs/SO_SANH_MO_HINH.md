# So sánh các mô hình embedding kết hợp KNN

Tài liệu này mô tả phần mở rộng theo yêu cầu của giảng viên: so sánh nhiều mô hình, đo bằng nhiều độ đo,
đo tốc độ (FPS), phân tích các ca nhận diện sai, và demo 2 mô hình tốt nhất.
Mọi con số dưới đây do `benchmark.py` tạo ra và nằm trong thư mục [`results/`](../results/).

## 1. Tóm tắt

- **So sánh được 9 mô hình, không phải 10.** `DeepFace` và `Buffalo_L` không chạy được trong môi trường này
  (lý do ở mục 2.1). Mã vẫn khai báo hai mô hình này trong `config.py` (`UNAVAILABLE_MODELS`).
- **Hai mô hình demo: ArcFace và Facenet512.** Cả hai cùng đúng 330/335 mẫu test (accuracy 0,9851).
- **5 mô hình đầu (ArcFace, Facenet512, Facenet, VGG-Face, SFace) chỉ cách nhau 1–4 mẫu test** (accuracy 0,973–0,985),
  nên không thể kết luận chắc chắn mô hình nào nhỉnh hơn. Việc chọn ArcFace và Facenet512 dựa trên F1 macro
  chênh nhau 0,0008 (xem mục 4.3).
- **OpenFace (0,30) và DeepID (0,27) kém hẳn** các mô hình còn lại. Đã kiểm tra đây không phải lỗi tiền xử lý
  của dự án (mục 4.3).
- **Tốc độ end-to-end chỉ khoảng 0,64–0,79 FPS** trên CPU, vì RetinaFace tốn khoảng 1,2 giây mỗi ảnh. Riêng bước
  embed + KNN thì từ 2,5 FPS (Dlib) đến 76 FPS (DeepID).
- **Ca lỗi:** phần lớn lỗi của các mô hình tốt không phải do bộ phân loại mà do crop khuôn mặt bị hỏng sau bước
  detect/căn chỉnh (mục 6.2).

## 2. Yêu cầu của giảng viên và nơi thực hiện

| Yêu cầu | Thực hiện | Ghi chú |
|---|---|---|
| Làm nhiều mô hình | 9 mô hình DeepFace + KNN, khai báo trong `config.py` | Muốn 10 thì xem mục 2.1 |
| So sánh giữa 10 mô hình | `benchmark.py` → `results/metrics.csv` | **9 mô hình** |
| Demo 2 mô hình tốt nhất | `app.py` chọn giữa các mô hình đã train | ArcFace, Facenet512 |
| 7 mô hình còn lại chỉ cần code | Có sẵn trong `config.py`; chạy `python train_model.py --model <tên>` | Không bị khóa cứng 2 mô hình |
| 5–6 độ đo | 6 độ đo (mục 4) | Accuracy, Precision, Recall, F1, Top-3, Kappa |
| Tốc độ FPS | `results/fps.csv` (mục 5) | CPU, 50 ảnh/mô hình |
| Các trường hợp nhận diện sai | `results/failures_<mô hình>.csv` và `results/failures/<mô hình>/` (mục 6) | |

### 2.1. Vì sao chỉ có 9 mô hình

DeepFace 0.0.101 hỗ trợ 11 mô hình nhận diện. Hai mô hình không chạy được trên máy này:

- **DeepFace:** báo lỗi `DeepFace model requires LocallyConnected2D but it is no longer supported after tf 2.12`.
  Máy dùng TensorFlow 2.21 (Python 3.12), không có cách hạ xuống TF ≤ 2.12.
- **Buffalo_L:** cần `insightface` và `onnxruntime` (đã cài được), nhưng trọng số được tải từ Google Drive; Drive trả về
  trang cảnh báo virus (HTML) thay vì file nên DeepFace không tải được. Đã thử tải thẳng bằng `curl` và cũng nhận
  được trang HTML. Tôi đã xóa file HTML rác này.

Nếu bạn tải được `webface_r50.onnx` về thủ công và đặt vào `C:\Users\<tên>\.deepface\weights\buffalo_l\`,
có thể thêm `"Buffalo_L"` vào `MODEL_NAMES` trong `config.py` rồi chạy `python benchmark.py --models Buffalo_L`.
Cách này **chưa được thử**.

## 3. Phương pháp

**Dữ liệu.** Bộ LFW (Labeled Faces in the Wild) tải qua `scikit-learn` (`prepare_data.py`): các người có ít nhất 20 ảnh,
lấy tối đa 30 ảnh mỗi người. Kết quả: **62 người, 1673 ảnh**. Chia train/test theo tỉ lệ 80/20, phân tầng theo người,
`seed = 42`: **1338 ảnh train, 335 ảnh test**, dùng chung cho mọi mô hình.

**Pipeline** (`face_pipeline.py`, dùng chung cho app, train và benchmark):

1. RetinaFace detect + căn chỉnh khuôn mặt, lấy khuôn mặt lớn nhất. Làm một lần và cache ở `output/faces/`.
2. Mô hình embedding của DeepFace biến khuôn mặt thành vector. Cache ở `output/embeddings/<mô hình>.npz`.
3. **Chuẩn hóa L2** vector. Các mô hình có thang đo khác nhau (vector 128 đến 4096 chiều), chuẩn hóa để khoảng cách
   Euclid so sánh được.
4. KNN tự cài đặt (`knn_func.py`): K = 3, khoảng cách Euclid, bỏ phiếu theo đa số. Khi hòa phiếu chọn nhãn của láng giềng
   gần nhất.

**Ngưỡng "Unknown".** Mỗi mô hình có ngưỡng riêng: percentile 95 của khoảng cách tới láng giềng gần nhất, tính trên các mẫu
được dự đoán đúng. Ngưỡng này **không** ảnh hưởng 6 độ đo (tính trước khi áp ngưỡng), chỉ dùng cho demo và phân tích lỗi.

## 4. Sáu độ đo

### 4.1. Định nghĩa

Tính trên 335 mẫu test bằng `scikit-learn`.

| Độ đo | Ý nghĩa |
|---|---|
| Accuracy | Tỉ lệ mẫu dự đoán đúng |
| Precision (macro) | Trung bình theo từng người của: trong các ảnh bị dự đoán là người đó, bao nhiêu phần đúng |
| Recall (macro) | Trung bình theo từng người của: trong các ảnh thật của người đó, bao nhiêu phần được nhận ra |
| F1 (macro) | Trung bình điều hòa của Precision và Recall, rồi lấy trung bình theo người |
| Top-3 accuracy | Tỉ lệ mẫu mà nhãn đúng nằm trong 3 người (khác nhau) gần nhất |
| Cohen's Kappa | Mức đồng thuận giữa dự đoán và nhãn thật sau khi trừ phần trùng hợp ngẫu nhiên |

### 4.2. Kết quả

| Mô hình | Số chiều | Accuracy | Precision | Recall | F1 | Top-3 | Kappa |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ArcFace** | 512 | 0,9851 | 0,9876 | 0,9855 | 0,9853 | 0,9851 | 0,9848 |
| **Facenet512** | 512 | 0,9851 | 0,9868 | 0,9855 | 0,9845 | 0,9881 | 0,9848 |
| Facenet | 128 | 0,9821 | 0,9854 | 0,9828 | 0,9824 | 0,9851 | 0,9818 |
| VGG-Face | 4096 | 0,9761 | 0,9786 | 0,9774 | 0,9764 | 0,9821 | 0,9757 |
| SFace | 128 | 0,9731 | 0,9770 | 0,9734 | 0,9728 | 0,9761 | 0,9727 |
| GhostFaceNet | 512 | 0,9284 | 0,9406 | 0,9280 | 0,9281 | 0,9552 | 0,9272 |
| Dlib | 128 | 0,9045 | 0,9209 | 0,9032 | 0,9042 | 0,9672 | 0,9029 |
| OpenFace | 128 | 0,3015 | 0,2899 | 0,2941 | 0,2776 | 0,4716 | 0,2895 |
| DeepID | 160 | 0,2687 | 0,3034 | 0,2728 | 0,2737 | 0,4149 | 0,2563 |

Biểu đồ: [`f1_ranking.png`](../results/f1_ranking.png), [`metrics_heatmap.png`](../results/metrics_heatmap.png),
[`confusion_ArcFace.png`](../results/confusion_ArcFace.png), [`confusion_Facenet512.png`](../results/confusion_Facenet512.png).

### 4.3. Cách đọc kết quả

- **Độ tin cậy của thứ hạng.** Một mẫu test tương đương 0,30 điểm phần trăm accuracy (1/335). ArcFace và Facenet512
  đúng cùng 330 mẫu. Năm mô hình đầu cách nhau tối đa 4 mẫu. Chỉ có một lần chia dữ liệu (không chạy lặp hay cross-validation)
  nên **không đủ cơ sở để nói ArcFace "tốt hơn" Facenet, VGG-Face hay SFace**. Hai mô hình demo được chọn theo F1 macro
  (ArcFace 0,9853, Facenet512 0,9845, Facenet 0,9824).
- **Nhóm giữa và nhóm cuối thì khác biệt rõ.** GhostFaceNet và Dlib thấp hơn nhóm đầu 5–8 điểm. OpenFace và DeepID chỉ
  đúng khoảng 3 trên 10 mẫu.
- **OpenFace và DeepID thấp không phải do lỗi tiền xử lý của dự án.** Tôi so embedding do `face_pipeline.embed_face` tạo ra
  với embedding do chính pipeline gốc của DeepFace (`DeepFace.represent` với detector RetinaFace) trên cùng một ảnh:
  độ tương đồng cosine là 1,0000 với OpenFace và DeepID. Vì vậy điểm thấp là đặc tính của mô hình trên dữ liệu này.
  Nguyên nhân sâu hơn (ví dụ đầu vào độ phân giải thấp của DeepID) là **suy đoán, chưa được kiểm chứng**.
- **Top-3 của Dlib (0,967) cao hơn nhiều so với accuracy (0,905):** mô hình thường xếp đúng người ở hạng 2–3 nhưng không
  đứng đầu.

## 5. Tốc độ (FPS)

Đo trên **CPU Intel Core i5-1135G7 (4 nhân, 8 luồng), Windows 11, TensorFlow 2.21, không dùng GPU**.
Mỗi mô hình xử lý 50 ảnh test được chọn ngẫu nhiên (cùng seed), bỏ lượt chạy đầu để tránh tính thời gian nạp mô hình,
mỗi ảnh xử lý riêng lẻ (không gom batch). Thời gian đọc file không được tính.

| Mô hình | Detect (ms) | Embed (ms) | KNN (ms) | FPS embed + KNN | FPS end-to-end |
|---|---:|---:|---:|---:|---:|
| DeepID | 1256 | 12,5 | 0,65 | **76,3** | 0,79 |
| SFace | 1249 | 14,2 | 0,55 | 67,8 | 0,79 |
| OpenFace | 1208 | 112,8 | 0,44 | 8,8 | 0,76 |
| ArcFace | 1214 | 178,3 | 2,56 | 5,5 | 0,72 |
| VGG-Face | 1237 | 211,2 | 20,30 | 4,3 | 0,68 |
| GhostFaceNet | 1197 | 244,7 | 2,41 | 4,1 | 0,69 |
| Facenet | 1165 | 260,9 | 0,38 | 3,8 | 0,70 |
| Facenet512 | 1178 | 280,3 | 2,24 | 3,5 | 0,68 |
| Dlib | 1159 | 397,9 | 0,36 | 2,5 | 0,64 |

Biểu đồ: [`time_breakdown.png`](../results/time_breakdown.png), [`speed_vs_f1.png`](../results/speed_vs_f1.png).

- **RetinaFace là nút thắt.** Bước detect chiếm khoảng 1,2 giây mỗi ảnh nên FPS end-to-end gần như bằng nhau
  (0,64–0,79) giữa các mô hình. Muốn nhanh hơn cần đổi detector (`DETECTOR_BACKEND` trong `config.py`, ví dụ `opencv`
  hoặc `ssd`); **tôi chưa đo các detector này** nên không biết độ chính xác và tốc độ thực tế.
- **KNN không phải vấn đề** với 1338 mẫu (dưới 1 ms, trừ VGG-Face 20 ms do vector 4096 chiều).
- **SFace và DeepID nhanh gấp khoảng 8–30 lần** các mô hình còn lại ở bước embed. Có thể do hai mô hình này chạy qua OpenCV/ONNX
  còn các mô hình khác chạy qua `predict` của Keras cho từng ảnh, nhưng tôi **chưa kiểm chứng** nguyên nhân.
- **Độ lặp lại.** Hai lần đo SFace cho 75,5 FPS (lần chạy thử, chỉ 5 ảnh) và 67,8 FPS (lần chạy đầy đủ, 50 ảnh),
  tức lệch khoảng 10% (hai lần dùng số ảnh khác nhau nên đây chỉ là chỉ báo thô).
  Nên coi các số trên là cỡ độ lớn, chênh vài chục phần trăm giữa các mô hình gần nhau chưa chắc có ý nghĩa.
  Thứ tự embed của Facenet (128 chiều, 261 ms) chậm hơn ArcFace (178 ms) cũng có thể do sai số đo.

## 6. Các trường hợp nhận diện sai

Một mẫu test được coi là **lỗi** nếu dự đoán sai, hoặc dự đoán đúng nhưng khoảng cách vượt ngưỡng "Unknown" (bị loại).
Mỗi lỗi được xếp vào một nguyên nhân (theo thứ tự ưu tiên):

| Nguyên nhân | Định nghĩa |
|---|---|
| `no_face_detected` | RetinaFace không tìm thấy mặt (`confidence = 0`) |
| `bad_crop` | Crop khuôn mặt có hơn 15% pixel đen: bước detect/căn chỉnh tạo ra crop hỏng |
| `misidentified` | Crop bình thường nhưng KNN nhận nhầm người |
| `rejected_unknown` | Nhận đúng nhưng khoảng cách vượt ngưỡng nên bị loại thành Unknown |

### 6.1. Số liệu (335 mẫu test)

| Mô hình | Nhận nhầm | Crop hỏng | Không thấy mặt | Bị loại Unknown |
|---|---:|---:|---:|---:|
| ArcFace | 2 | 3 | 0 | 17 |
| Facenet512 | 2 | 3 | 0 | 17 |
| Facenet | 3 | 3 | 0 | 17 |
| VGG-Face | 5 | 3 | 0 | 17 |
| SFace | 6 | 3 | 0 | 17 |
| GhostFaceNet | 21 | 3 | 0 | 16 |
| Dlib | 29 | 3 | 0 | 16 |
| OpenFace | 231 | 3 | 0 | 5 |
| DeepID | 242 | 3 | 0 | 5 |

Biểu đồ: [`failure_breakdown.png`](../results/failure_breakdown.png). Chi tiết từng ca (ảnh thật, nhãn đúng, nhãn dự đoán,
khoảng cách): `results/failures_<mô hình>.csv`. Ảnh minh họa (trái: ảnh test, phải: mẫu train gần nhất của người bị nhận nhầm)
trong `results/failures/<mô hình>/`.

**Cách đọc bảng:**

- **"Bị loại Unknown" là hệ quả của quy tắc đặt ngưỡng, không phải lỗi của mô hình.** Ngưỡng được đặt để giữ lại 95% mẫu đúng,
  nên luôn khoảng 5% mẫu đúng bị loại (330 × 5% ≈ 17). Trong benchmark không có mẫu "người lạ" thật nên không đo được
  tỉ lệ nhận nhầm người lạ thành người quen.
- **"Không thấy mặt" bằng 0** vì RetinaFace tìm được mặt trong cả 1673 ảnh LFW (ảnh báo chí, mặt nằm gần giữa khung hình).
  Loại lỗi này có thể xuất hiện với ảnh webcam thật; **chưa được đo ở đây**.
- Trong lần chạy này, cả 3 mẫu crop hỏng đều bị dự đoán sai ở mọi mô hình, nên nhận nhầm + crop hỏng đúng bằng số mẫu dự đoán sai.
  Ví dụ ArcFace dự đoán sai 5 mẫu = 2 nhận nhầm + 3 crop hỏng.

### 6.2. Phát hiện chính: crop hỏng gây phần lớn lỗi của các mô hình tốt

Chỉ có **16 trong 1673 crop (dưới 1%) bị hỏng**, nhưng chúng gây **3 trong 5 lỗi của ArcFace** và 3 trong 5 lỗi của Facenet512.
Ba mẫu test này bị sai ở **cả 7 mô hình tốt**, và với ArcFace khoảng cách tới người bị nhầm rất nhỏ (0,07–0,10):
hai crop hỏng đều chủ yếu là màu đen nên embedding gần giống nhau, dù là hai người khác nhau.

Ví dụ ArcFace:

- [Jeremy Greenstock → Naomi Watts](../results/failures/ArcFace/01_Jeremy_Greenstock_as_Naomi_Watts.jpg): cả hai crop đều là mảnh mặt bị xoay, nền đen
- [Roh Moo-hyun → Megawati Sukarnoputri](../results/failures/ArcFace/02_Roh_Moo-hyun_as_Megawati_Sukarnoputri.jpg)
- [Gray Davis → John Ashcroft](../results/failures/ArcFace/03_Gray_Davis_as_John_Ashcroft.jpg)

Hai ca nhận nhầm "thật" còn lại của ArcFace có khoảng cách lớn (0,96 và 1,13), tức là mô hình gần như không chắc chắn:

- [Laura Bush → George W Bush](../results/failures/ArcFace/14_Laura_Bush_as_George_W_Bush.jpg), khoảng cách 0,955
- [Recep Tayyip Erdogan → Tom Ridge](../results/failures/ArcFace/22_Recep_Tayyip_Erdogan_as_Tom_Ridge.jpg), khoảng cách 1,129

**Nguyên nhân của crop hỏng:**

- Đã loại trừ giả thuyết "viền đen có sẵn của dataset": bản LFW mà `scikit-learn` tải mặc định là bản *funneled*
  (đã xoay căn chỉnh) nên một nửa số ảnh nguồn (835/1673) có góc đen. Tuy nhiên tỉ lệ pixel đen của ảnh nguồn **không tương quan**
  với tỉ lệ pixel đen của crop (hệ số tương quan −0,03), và ảnh nguồn của 16 crop hỏng không nhiều viền đen hơn mức bình thường.
- Vì vậy lỗi nằm ở bước detect/căn chỉnh của DeepFace (ví dụ mặt gần mép ảnh hoặc bị xoay nhiều khi `align=True`).
  **Cơ chế chính xác chưa được tìm hiểu sâu.**
- Cách khắc phục khả dĩ (chưa làm): loại các crop có tỉ lệ đen cao khỏi tập train và yêu cầu chụp lại ở app,
  hoặc thử `align=False`/detector khác. Mỗi cách cần chạy lại benchmark.

### 6.3. Kiểm tra thủ công với người lạ

Một lần kiểm tra thủ công (script chạy một lần, **không có trong repo nên không tái lập được bằng lệnh có sẵn**), dùng chính hàm
`predict_name_from_face` của `app.py`: 20 ảnh LFW của những người **không có trong dataset** (có 5–19 ảnh trong LFW).

| Mô hình | Số ảnh người lạ bị loại thành "Unknown" | Ảnh bị nhận nhầm thành người quen (khoảng cách) |
|---|---:|---|
| ArcFace | 19/20 | Condoleezza Rice → David Beckham (0,39) |
| Facenet512 | 18/20 | Al Sharpton → Kofi Annan (0,59); Taha Yassin Ramadan → Hamid Karzai (0,66) |

Đây là mẫu rất nhỏ (20 ảnh), chỉ cho thấy cơ chế ngưỡng hoạt động, không dùng để kết luận độ chính xác với người lạ.
Cùng lần kiểm tra có 20 ảnh người đã biết (đã đổi độ sáng) cho 20/20 đúng, nhưng các ảnh này nằm trong tập train
nên **kết quả đó không có giá trị đánh giá**.

## 7. Hạn chế

- **Một lần chia dữ liệu, tập test nhỏ (335 mẫu).** Xem mục 4.3.
- **LFW là ảnh báo chí**, khác với ảnh webcam của người dùng (ánh sáng, góc mặt, độ phân giải). Kết quả ở đây
  không nên được diễn giải là độ chính xác của ứng dụng với dữ liệu của bạn. Muốn có số liệu trên dữ liệu riêng thì chụp ảnh
  trong app rồi chạy lại `benchmark.py` (ảnh của bạn sẽ nằm cùng thư mục `dataset/`).
- **Ngưỡng Unknown** đặt theo quy tắc percentile 95, chưa được tối ưu và chưa đánh giá với người lạ ở quy mô lớn.
- **FPS chỉ đo trên CPU**, ảnh 250×250, mỗi ảnh một lần gọi, không batch; không có số liệu GPU.
- **Không có thời gian huấn luyện** để so sánh: KNN không có bước huấn luyện, chi phí nằm ở bước tạo embedding cho tập train
  (xem `embed_s` trong `output/embeddings/`).
- **10 mô hình:** mới có 9, xem mục 2.1.

## 8. Các lỗi của mã gốc đã sửa trong quá trình làm

| Vấn đề trong mã gốc | Cách xử lý |
|---|---|
| `app.py` đổi ảnh sang RGB trước khi gọi `DeepFace.represent`, trong khi DeepFace coi mảng numpy là **BGR**. Lúc train đưa đường dẫn file (BGR). Embedding lúc nhận diện bị đảo kênh màu so với lúc train | Mọi nơi dùng chung `face_pipeline.embed_face` và truyền BGR. **Chưa đo mức ảnh hưởng đến độ chính xác** |
| Lúc train dùng detector mặc định của DeepFace, lúc nhận diện dùng RetinaFace | Cùng dùng RetinaFace (`config.DETECTOR_BACKEND`) |
| Tên mô hình `ArcFace` viết cứng ở hai file | Khai báo một nơi trong `config.py` |
| Ngưỡng Unknown cố định 3,7 (chỉ hợp lý với một mô hình, một thang đo) | Ngưỡng riêng cho từng mô hình, lưu trong file `.npz` |
| `knn_func.py`: khi hòa phiếu chọn theo thứ tự chữ cái | Chọn nhãn của láng giềng gần nhất |
| `app.py`: `detect_and_extract_faces` gọi DeepFace với `enforce_detection=False`, khi không thấy mặt DeepFace trả về nguyên cả ảnh như một "khuôn mặt", nên thông báo "không tìm thấy khuôn mặt" không bao giờ hiện | `detect_faces` bỏ kết quả có `confidence = 0` |
| `app.py`: font `arial.ttf` chỉ có trên Windows | Thử thêm DejaVu rồi font mặc định của PIL |
| In log có emoji/tiếng Việt lỗi `UnicodeEncodeError` khi stdout bị pipe trên Windows | `config.py` đặt stdout sang UTF-8 |

## 9. Tái lập kết quả

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python prepare_data.py        # tải LFW (~200MB) vào dataset/
python benchmark.py           # so sánh các mô hình, ghi vào results/
python plot_results.py        # vẽ biểu đồ
python train_model.py         # train 2 mô hình tốt nhất cho demo (đọc results/best_models.json)
streamlit run app.py
```

Thời gian trên máy đã dùng (i5-1135G7, CPU): detect 1673 ảnh khoảng 35 phút; embed, đánh giá và đo FPS của cả 9 mô hình
khoảng 55 phút (VGG-Face chậm nhất, khoảng 8 phút). Detect và embed được cache trong `output/` nên các lần chạy lại chỉ mất vài phút. Nếu thay ảnh trong `dataset/`
mà giữ nguyên tên file thì phải xóa `output/faces/` và `output/embeddings/` để tính lại.

Các tham số (K, tỉ lệ test, seed, detector, danh sách mô hình) nằm trong [`config.py`](../config.py).
Ngưỡng Unknown (percentile 95) và ngưỡng crop hỏng (15% pixel đen) nằm đầu [`benchmark.py`](../benchmark.py).

## 10. Các file liên quan

| File | Vai trò |
|---|---|
| `config.py` | Danh sách mô hình, đường dẫn, K, seed |
| `face_pipeline.py` | detect, embed, cache khuôn mặt và embedding |
| `knn_func.py` | KNN tự cài đặt |
| `prepare_data.py` | Tải LFW |
| `benchmark.py` | Đánh giá, đo FPS, phân tích lỗi |
| `plot_results.py` | Vẽ biểu đồ |
| `train_model.py` | Train KNN cho từng mô hình (dùng cho demo) |
| `app.py` | Giao diện, có trang "So sánh mô hình" đọc `results/` |
| `results/` | `metrics.csv`, `fps.csv`, `failure_summary.csv`, `failures_*.csv`, `predictions.csv`, `run_info.json`, biểu đồ |
