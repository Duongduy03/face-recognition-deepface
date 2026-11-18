# Phase 0: Chuẩn bị môi trường và đo nền

> **Trạng thái: chưa làm.** Thời lượng: **0,5 ngày**. Phase này chỉ cài thư viện, kiểm tra và đo. **Không sửa logic nhận diện.**
> Tổng quan và lịch: [README.md](README.md). Phase sau: [PHASE_1_DETECTOR.md](PHASE_1_DETECTOR.md).

## 1. Mục tiêu

1. Cài `streamlit-webrtc` và `av` mà không làm hỏng TensorFlow, DeepFace, OpenCV, numpy.
2. Xác nhận các thành phần phase sau sẽ cần chạy được: `cv2.FaceDetectorYN` (YuNet), trọng số YuNet của DeepFace, webcam trong trình duyệt.
3. Đo nền tốc độ `detect_faces` (RetinaFace) và `embed_face` trên máy của người dùng, với ảnh cỡ webcam (640x480).
4. **Lưu kết quả tham chiếu của trang upload** (cùng ảnh, cùng model) để mọi phase sau so sánh, bảo đảm upload ảnh tĩnh không đổi.
5. Chốt máy nào chạy lệnh (xem README mục 8).

## 2. Thời gian

| Buổi | Việc | Ước tính |
|---|---|---|
| Sáng (khoảng 2 giờ) | Bước 1 đến 5: máy chạy lệnh, lưu trạng thái, cài, kiểm tra import, kiểm tra YuNet | 2 giờ |
| Chiều (khoảng 2 giờ) | Bước 6 đến 8: đo nền, kết quả tham chiếu trang upload, kiểm tra webcam, ghi kết quả, hỏi accept | 2 giờ |

## 3. Điều kiện đầu vào

- Đã có model: `output/model_ArcFace_SVM-linear.joblib` và `output/model_ArcFace_KNN.npz` (đang có sẵn trong `output/`).
- Có `external/pins_demo_test/` (105 thư mục người, mỗi thư mục 10 ảnh).
- Máy có kết nối Internet (tải gói pip và trọng số YuNet).

## 4. Các bước

### Bước 1. Chốt máy chạy lệnh

Hỏi người dùng và ghi vào bảng kết quả (mục 7):

- Cách 1: người dùng chạy lệnh trên Windows (`.venv\Scripts\python.exe`) và gửi lại kết quả.
- Cách 2: dựng venv Linux riêng để Claude tự chạy. Chỉ chọn nếu người dùng đồng ý; tạo ở thư mục riêng, không đụng `.venv`:
  ```bash
  python3.12 -m venv .venv-linux
  .venv-linux/bin/pip install -r requirements.txt
  ```
  (`.venv-linux` phải được thêm vào `.gitignore` nếu chưa có; kiểm tra file `.gitignore` trước, không tự sửa nếu người dùng chưa đồng ý.)

### Bước 2. Lưu trạng thái trước khi cài

Cây làm việc hiện có rất nhiều file đã sửa chưa commit (xem `git status`). Đề nghị người dùng tự quyết định: commit hoặc tạo nhánh `realtime` từ `duy` trước khi phase 1 bắt đầu sửa code. Phase 0 không commit gì.

Lưu danh sách gói hiện tại để so sánh:

```bash
python -m pip freeze > output/freeze_before.txt
```

Ghi lại phiên bản của: `streamlit` (1,66,1 lúc khảo sát), `tensorflow` (2,21,0), `deepface` (0,0,103), `numpy` (2,5,4), `opencv-python` (5,0,0,93), `protobuf`, `scikit-learn`.

### Bước 3. Cài thư viện

```bash
python -m pip install streamlit-webrtc av
python -m pip freeze > output/freeze_after.txt
```

So hai file freeze. Quy tắc:

- Được phép: thêm `streamlit-webrtc`, `av`, `aiortc`, `aioice`, `pylibsrtp`, `pyee`, `cryptography`, `dnspython`, `ifaddr`, `google-crc32c` và các phụ thuộc nhỏ của chúng.
- **Không được đổi** (nếu đổi thì rollback và báo người dùng): `numpy`, `opencv-python`, `opencv-contrib-python-headless`, `protobuf`, `tensorflow`, `keras`, `tf-keras`, `deepface`, `streamlit`, `scikit-learn`.
- Rollback: `python -m pip install -r output/freeze_before.txt` hoặc cài lại đúng phiên bản bị đổi (`pip install numpy==<bản cũ>`).

Kiểm tra:

```bash
python -c "import streamlit_webrtc, av, aiortc; print('streamlit-webrtc', streamlit_webrtc.__version__); print('av', av.__version__); print('aiortc', aiortc.__version__)"
python -m pip check
python -c "import config; import numpy, cv2, tensorflow as tf, deepface; print(numpy.__version__, cv2.__version__, tf.__version__)"
```

(`import config` trước, theo quy ước của dự án: `config.py` phải được import trước `deepface`.)

### Bước 4. Cập nhật `requirements.txt`

Thêm hai dòng (file hiện không ghim phiên bản trừ `deepface==0.0.103`, giữ cùng kiểu):

```
streamlit-webrtc
av
```

Ghi phiên bản đã cài vào bảng kết quả (mục 7) để phase 5 ghi vào báo cáo.

### Bước 5. Kiểm tra YuNet

```bash
python -c "import cv2; print(cv2.__version__); print('FaceDetectorYN:', hasattr(cv2, 'FaceDetectorYN'))"
python -c "import config; from deepface import DeepFace; m = DeepFace.build_model(task='face_detector', model_name='yunet'); print('YuNet OK', type(m))"
```

Lệnh thứ hai lần đầu sẽ tải trọng số `face_detection_yunet_2023mar.onnx` vào thư mục `.deepface/weights` của người dùng. Ghi lại: có tải được không, dung lượng, thời gian.

Nếu `FaceDetectorYN` không có hoặc DeepFace không build được YuNet: **không dừng phase**, ghi vào mục 7 và báo người dùng ở điểm dừng. Phase 1 sẽ chuyển sang detector khác (`opencv`, `ssd`, `fastmtcnn`, `centerface`).

### Bước 6. Đo nền tốc độ (RetinaFace)

Chạy đoạn sau (một lần, không lưu file). Mục đích: biết trên máy này detect và embed mất bao lâu, để đặt mục tiêu cho phase 2.

```python
import time, glob, cv2, numpy as np
import config
from face_pipeline import detect_faces, embed_face, imread_unicode

paths = sorted(glob.glob("external/pins_demo_test/*/*.jpg"))[::100][:8]   # 8 ảnh rải đều
def to_vga(img):                       # giả lập frame webcam 640x480
    return cv2.resize(img, (640, 480))

detect_faces(to_vga(imread_unicode(paths[0])))   # chạy thử để nạp model, không tính giờ
det, emb = [], []
for p in paths:
    img = to_vga(imread_unicode(p))
    t = time.perf_counter(); faces = detect_faces(img); det.append(time.perf_counter() - t)
    if faces:
        t = time.perf_counter(); embed_face(faces[0]["face_bgr"], "ArcFace"); emb.append(time.perf_counter() - t)
print(f"detect RetinaFace 640x480: trung vị {np.median(det)*1000:.0f} ms, p95 {np.percentile(det,95)*1000:.0f} ms (n={len(det)})")
print(f"embed ArcFace: trung vị {np.median(emb)*1000:.0f} ms (n={len(emb)})")
```

Ghi số vào mục 7. Ảnh Pins có kích thước khác nhau; ép về 640x480 làm méo tỉ lệ nhưng chỉ để đo thời gian, không đo độ chính xác.

### Bước 7. Lưu kết quả tham chiếu của trang upload

Mục đích: sau mỗi phase có thể chứng minh trang upload không đổi.

1. Chọn 3 ảnh trong `external/pins_demo_test/` (cùng 3 ảnh dùng suốt các phase; ghi đường dẫn vào mục 7).
2. Chạy `streamlit run app.py`, vào "🔍 Nhận diện", nguồn "Thuật toán phân loại (ArcFace)", lần lượt chọn **SVM-linear** rồi **KNN**, upload từng ảnh.
3. Với mỗi (thuật toán, ảnh) ghi: tên nhận diện, độ tin cậy hoặc khoảng cách trong bảng chi tiết, số khuôn mặt tìm thấy, và chụp màn hình lưu vào `results/realtime/upload_reference/` (thư mục mới, tạo khi lưu).
4. Ghi vào bảng ở mục 7.

Script tự động `check_upload_regression.py` sẽ được viết ở Phase 2 Ngày 1; lúc đó chạy nó **trước** khi sửa code để chốt bản tham chiếu bằng số, đối chiếu với bảng thủ công này.

### Bước 8. Kiểm tra webcam của trình duyệt

- Mở `http://localhost:8501`, vào "📸 Chụp ảnh", xác nhận `st.camera_input` hiện được hình webcam (nghĩa là trình duyệt đã được cấp quyền camera và camera hoạt động).
- Ghi trình duyệt và hệ điều hành người dùng sẽ dùng để thử webcam ở Phase 3 và 4.

### Bước 9. Kiểm tra app cũ vẫn chạy

`streamlit run app.py`: mở cả 3 trang ("📸 Chụp ảnh", "🔍 Nhận diện", "📊 So sánh mô hình"), không có lỗi trong console. Việc này chứng minh việc cài thư viện không làm hỏng app.

## 5. Tiêu chí hoàn thành

- [ ] `import streamlit_webrtc, av, aiortc` chạy được.
- [ ] `pip check` không báo xung đột mới; các gói trong danh sách "không được đổi" không đổi phiên bản.
- [ ] `requirements.txt` có `streamlit-webrtc` và `av`.
- [ ] Biết `cv2.FaceDetectorYN` có hay không, và YuNet có build được hay không.
- [ ] Có số đo nền (detect, embed) trên máy người dùng.
- [ ] Có bảng kết quả tham chiếu của trang upload (3 ảnh x 2 thuật toán).
- [ ] App cũ chạy đủ 3 trang không lỗi.
- [ ] Chốt máy chạy lệnh (cách 1 hay cách 2).

## 6. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| `pip install` nâng numpy hoặc đổi opencv | Rollback bằng `freeze_before.txt`; nếu `av` không tương thích numpy 2,5 thì thử bản `av` khác hoặc báo người dùng |
| `aiortc` không có wheel cho Python 3.12 trên máy | Báo người dùng; phương án: Python 3.13 (CLAUDE.md nói 3.13 cũng dùng được) hoặc cài trình biên dịch |
| Không tải được trọng số YuNet (mạng, tường lửa) | Người dùng tải tay file vào thư mục `.deepface/weights`, hoặc chuyển detector ở Phase 1 |
| Hai bản OpenCV cùng cài (`opencv-python` và `opencv-contrib-python-headless`) dùng chung thư mục `cv2` | Không đụng vào ở phase này; chỉ ghi nhận nếu `FaceDetectorYN` bị ảnh hưởng |

## 7. Kết quả (điền sau khi làm)

| Mục | Giá trị |
|---|---|
| Ngày làm | |
| Máy chạy lệnh (cách 1 hay 2) | |
| Phiên bản: streamlit / streamlit-webrtc / av / aiortc | |
| Gói nào bị đổi khi cài (nếu có) | |
| `cv2.FaceDetectorYN` có không | |
| Tải trọng số YuNet | |
| Đo nền detect RetinaFace 640x480 (trung vị, p95) | |
| Đo nền embed ArcFace (trung vị) | |
| Trình duyệt, hệ điều hành dùng để thử webcam | |
| 3 ảnh tham chiếu (đường dẫn) | |

Bảng tham chiếu trang upload:

| Ảnh | Thuật toán | Số mặt | Tên | Độ tin cậy / khoảng cách |
|---|---|---|---|---|
| | SVM-linear | | | |
| | KNN | | | |
| | SVM-linear | | | |
| | KNN | | | |
| | SVM-linear | | | |
| | KNN | | | |

## 8. Điểm dừng

Báo người dùng: đã cài được chưa, YuNet có dùng được không, số đo nền, bảng tham chiếu. Hỏi: **"Accept Phase 0 để sang Phase 1 (chọn detector)?"** Cập nhật bảng trạng thái ở [README.md](README.md) mục 7.
