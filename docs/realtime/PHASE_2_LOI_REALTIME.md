# Phase 2: Lõi realtime (chưa có giao diện)

> **Trạng thái: chưa làm.** Thời lượng: **2 ngày** (Ngày 1: tách hàm thuần và khoá hồi quy; Ngày 2: worker, tracker, bỏ phiếu, chạy offline).
> Cần xong: [Phase 1](PHASE_1_DETECTOR.md) (đã chọn detector, `config.REALTIME_DETECTOR` đúng). Phase sau: [PHASE_3_TRANG_STREAMLIT.md](PHASE_3_TRANG_STREAMLIT.md). Tổng quan: [README.md](README.md).

## 1. Mục tiêu

Xây phần lõi nhận diện cho video, **kiểm được mà không cần Streamlit và không cần webcam**:

- Ngày 1: đưa các hàm thuần (dự đoán, vẽ khung) ra khỏi `app.py` vào `recognition.py` để dùng chung, **không đổi hành vi trang upload**; có script chứng minh điều đó bằng số.
- Ngày 2: `realtime.py` gồm worker thread, tracker khung theo IoU, bỏ phiếu để nhãn ổn định, giới hạn số lần embed; `realtime_demo.py` chạy offline trên video hoặc chuỗi ảnh, in FPS, độ trễ và tỉ lệ nhận đúng.

## 2. Thời gian

| Ngày | Buổi | Việc | Ước tính |
|---|---|---|---|
| Ngày 1 | Sáng | Bước 1.1 đến 1.3: viết `check_upload_regression.py`, chạy **trước** khi sửa để chốt bản tham chiếu | 2,5 giờ |
| Ngày 1 | Chiều | Bước 1.4 đến 1.6: tạo `recognition.py`, chuyển hàm, `app.py` import lại, chạy lại script, kiểm tra tay trang upload | 3,5 giờ |
| Ngày 2 | Sáng | Bước 2.1 đến 2.4: tracker, `RealtimeRecognizer`, worker | 4 giờ |
| Ngày 2 | Chiều | Bước 2.5 đến 2.7: `realtime_demo.py`, đo, chỉnh, điền kết quả, hỏi accept | 3 giờ |

## 3. Điều kiện đầu vào

- Phase 1 accept; `config.REALTIME_DETECTOR` đã chốt; `detect_faces(..., backend=...)` hoạt động.
- Bảng tham chiếu trang upload ở Phase 0 đã có.
- Model `SVM-linear` và `KNN` đã train trong `output/`.

---

# NGÀY 1: tách hàm thuần, khoá hồi quy trang upload

Nguyên tắc: **thay đổi tối thiểu trong `app.py`**. Chỉ chuyển các hàm thuần sang `recognition.py` và để `app.py` import lại đúng tên cũ. Thân `page_nhan_dien` (`app.py:361-501`) **không sửa**, nên trang upload không thể đổi hành vi vì lý do refactor.

## 4. Các bước Ngày 1

### Bước 1.1. Viết `check_upload_regression.py`

Script này chạy mô phỏng đúng đường đi của trang upload, bằng hàm thật, không cần mở trình duyệt:

```bash
python check_upload_regression.py --save      # ghi bản tham chiếu: results/realtime/upload_golden.json
python check_upload_regression.py --compare   # chạy lại và so với bản tham chiếu
```

Nội dung:

- Dùng 3 ảnh tham chiếu của Phase 0 và thêm 1 ảnh **nhiều mặt** (ghép 2 ảnh Pins cạnh nhau bằng `np.hstack` sau khi cùng chiều cao; ghi cách ghép vào docstring).
- Với mỗi (ảnh, thuật toán trong `DEMO_ALGORITHMS`): `faces = detect_faces(img)`; với từng mặt gọi `app.predict_name_with_algo(face_bgr, model, meta)` (nạp model bằng `classifiers.load`, không dùng `st.cache_resource`) và ghi `{box (x,y,w,h), name, confidence làm tròn 4 chữ số}`.
- Thêm kiểm tra vẽ: gọi `app.draw_rounded_rectangle` và `app.draw_vietnamese_text` trên một ảnh cố định, ghi **băm SHA1 của mảng ảnh kết quả**. Nếu hàm vẽ bị đổi vô tình, băm sẽ khác.
- `--compare`: so tên và khung phải **giống hệt**, độ tin cậy lệch không quá `1e-4`, băm vẽ giống hệt. In `OK` hoặc danh sách khác biệt và thoát mã khác 0.
- Script gọi hàm qua `import app` (`app.predict_name_with_algo`, `app.draw_*`). Vì `app.py` sẽ import lại các hàm đã chuyển, tên `app.<hàm>` vẫn tồn tại ở cả hai trạng thái: trước và sau refactor, nên **cùng một script chạy được ở cả hai bên**. Lưu ý: `import app` ngoài Streamlit sẽ in vài cảnh báo "ScriptRunContext"; bỏ qua được. Nếu `import app` gây lỗi vì `st.set_page_config` (`app.py:32`) thì bọc import trong `try` và chuyển sang chạy qua `streamlit.testing.v1.AppTest` hoặc đặt cờ môi trường; ghi cách xử lý vào docstring.

### Bước 1.2. Chốt bản tham chiếu **trước khi sửa `app.py`**

```bash
python check_upload_regression.py --save
```

Đối chiếu bằng mắt với bảng tham chiếu thủ công của Phase 0 (tên, số mặt, độ tin cậy). Hai bên phải khớp; nếu không, dừng và điều tra trước khi sửa gì.

### Bước 1.3. Chuẩn bị danh sách hàm sẽ chuyển

| Hàm | Vị trí hiện tại | Phụ thuộc | Ghi chú |
|---|---|---|---|
| `draw_rounded_rectangle` | `app.py:161` | `cv2` | Thuần |
| `predict_name_from_face` | `app.py:181` | `embed_face`, đối tượng KNN | Thuần; nguồn `.npz` và PG |
| `predict_name_with_algo` | `app.py:201` | `embed_face`, `classifiers`, `BENCH_EMBEDDING` | Thuần |
| `_load_font`, `draw_vietnamese_text` | `app.py:217-234` | `PIL`, `cv2` | Giữ **nguyên văn bản** ở `app.py` (không chuyển): quirk màu BGR/RGB của hàm này được giữ để trang upload không đổi |

Quyết định: `recognition.py` nhận 3 hàm đầu (nguyên văn, chỉ đổi chỗ). `draw_vietnamese_text` ở lại `app.py`; bản vẽ nhanh cho video là hàm **mới** trong `recognition.py` (bước 1.5), không thay bản cũ.

### Bước 1.4. Tạo `recognition.py`

Quy ước: không import `streamlit`. `import config` trước `deepface`. Nội dung:

```python
"""Hàm nhận diện thuần (không phụ thuộc Streamlit), dùng chung cho app.py và realtime.py."""
import config                                    # phải đứng trước deepface
from dataclasses import dataclass
from functools import lru_cache
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
import classifiers
from config import BENCH_EMBEDDING
from face_pipeline import detect_faces, embed_face, largest_face

# --- chuyển nguyên văn từ app.py: draw_rounded_rectangle, predict_name_from_face, predict_name_with_algo ---

@dataclass
class FaceResult:
    box: tuple            # (x, y, w, h) trên frame gốc
    name: str             # "Unknown", tên người, hoặc "" khi chưa nhận diện xong
    confidence: float | None
    detect_conf: float    # độ tin cậy của detector
    track_id: int | None = None

def make_algo_predictor(model, meta):
    """Trả về hàm face_bgr -> (name, confidence), bọc predict_name_with_algo."""
    return lambda face_bgr: predict_name_with_algo(face_bgr, model, meta)

def draw_results(frame_bgr, results, font_size=26):
    """Vẽ khung bo góc và nhãn cho TẤT CẢ mặt với một lần chuyển PIL cho cả frame. Sửa trực tiếp và trả về frame."""
```

Yêu cầu của `draw_results`:

- Vẽ khung bằng `draw_rounded_rectangle` (cv2) cho mọi mặt trước, rồi **một** lần `Image.fromarray` để vẽ tất cả nhãn tiếng Việt, rồi một lần chuyển ngược. Chi phí không tăng theo số mặt (bản cũ `draw_vietnamese_text` copy cả frame cho mỗi nhãn).
- Font cache bằng `@lru_cache` theo `font_size` (thử `arial.ttf`, `DejaVuSans.ttf`, rồi mặc định, giống `_load_font` ở `app.py:217`).
- Màu: xanh lá `(0,255,0)` cho người đã biết, đỏ `(0,0,255)` cho Unknown (BGR, dùng cho cv2). **Với PIL phải truyền RGB** (đảo thứ tự) để nhãn Unknown ra đỏ đúng như khung. Đây là chỗ khác bản cũ: `CLAUDE.md` ghi nhận lỗi cũ của `draw_vietnamese_text` (nhãn Unknown màu xanh dương). Giữ lỗi đó ở trang upload, sửa ở hàm mới; ghi chú rõ trong docstring.
- Nhãn rỗng (`name == ""`, chưa nhận diện xong) hiện khung xám và chữ "..." .
- Đặt nhãn phía trên khung; nếu sát mép trên thì đặt bên trong khung (giống `max(y - 35, 0)` ở bản cũ).

### Bước 1.5. `app.py` import lại

Trong `app.py`, xoá định nghĩa của 3 hàm đã chuyển và thêm ngay dưới các import hiện có:

```python
from recognition import draw_rounded_rectangle, predict_name_from_face, predict_name_with_algo
```

Không động vào `page_nhan_dien`, `main`, `draw_vietnamese_text`. Kiểm tra không còn ai khác trong repo gọi các hàm này từ `app` (tìm bằng `grep -n "predict_name_\|draw_rounded_rectangle" *.py`).

### Bước 1.6. Kiểm tra hồi quy

```bash
python check_upload_regression.py --compare
```

Phải ra `OK`. Sau đó kiểm tra tay: `streamlit run app.py` → "🔍 Nhận diện" → upload cùng 3 ảnh với SVM-linear và KNN, so với bảng tham chiếu Phase 0 và chụp màn hình so với ảnh tham chiếu (khung, nhãn, màu giống hệt).

**Cổng cuối Ngày 1:** `--compare` OK **và** kiểm tra tay khớp. Nếu không, hoàn tác (`git checkout -- app.py`, xoá `recognition.py`) và điều tra; không sang Ngày 2.

---

# NGÀY 2: worker, tracker, bỏ phiếu, chạy offline

## 5. Thiết kế

### 5.1 Các tham số (đặt trong `config.py`, gom dưới nhóm REALTIME)

| Tham số | Mặc định | Ý nghĩa | Khoảng thử ở Phase 4 |
|---|---|---|---|
| `REALTIME_DETECTOR` | `"yunet"` (hoặc theo Phase 1) | Detector dùng cho realtime | |
| `RT_DETECT_WIDTH` | 640 | Thu nhỏ frame về độ rộng này trước khi detect (giữ tỉ lệ); toạ độ quy đổi về frame gốc | 480 đến 960 |
| `RT_MIN_FACE` | 60 | Cạnh nhỏ nhất của khung (px, trên frame gốc); nhỏ hơn thì bỏ qua (mặt quá bé, embed không đáng tin) | 40 đến 100 |
| `RT_EMBED_EVERY_S` | 1.5 | Làm mới nhãn của một track sau ngần này giây | 0,5 đến 3 |
| `RT_MAX_EMBEDS_PER_CYCLE` | 1 | Số mặt tối đa được embed trong mỗi vòng worker (chặn độ trễ khi có nhiều mặt) | 1 đến 3 |
| `RT_VOTE_LEN` | 5 | Số kết quả gần nhất dùng bỏ phiếu | 3 đến 9 |
| `RT_MIN_VOTES` | 2 | Cần ít nhất ngần này phiếu cùng tên mới hiển thị nhãn | 1 đến 3 |
| `RT_IOU_MATCH` | 0.3 | Ngưỡng IoU để coi hai khung là cùng một mặt | 0,2 đến 0,5 |
| `RT_MAX_AGE_S` | 1.0 | Track không khớp quá ngần này giây thì xoá | 0,5 đến 2 |
| `RT_BOX_SMOOTH` | 0.6 | Hệ số làm mượt khung (EMA) để khung đỡ rung | 0 đến 0,8 |

Giá trị mặc định chỉ là điểm xuất phát; Phase 4 chỉnh bằng số liệu thật.

### 5.2 Tracker (trong `realtime.py`, không thêm thư viện)

- `Track`: `id`, `box` (đã làm mượt), `name`, `confidence`, `votes` (`deque(maxlen=RT_VOTE_LEN)` các tên), `last_embed_t`, `last_seen_t`, `created_t`, `n_embeds`.
- Mỗi vòng: tính IoU giữa các khung detect mới và các track hiện có; ghép **tham lam** theo IoU giảm dần, chỉ ghép khi IoU ≥ `RT_IOU_MATCH`; khung chưa ghép tạo track mới; track không được ghép giữ lại đến `RT_MAX_AGE_S` rồi xoá.
- Khi ghép: cập nhật khung bằng EMA `box = a*box_cu + (1-a)*box_moi` với `a = RT_BOX_SMOOTH`.
- Quy tắc **cần embed lại** một track (theo thứ tự ưu tiên): track mới (chưa có phiếu nào); đã quá `RT_EMBED_EVERY_S`; nhãn hiện tại là Unknown và đã quá 0,5 giây (cho cơ hội nhận diện lại khi mặt quay thẳng); diện tích khung đổi hơn 50% so với lần embed gần nhất.
- Mỗi vòng chỉ embed tối đa `RT_MAX_EMBEDS_PER_CYCLE` track, ưu tiên track chưa có nhãn rồi track lâu chưa được làm mới nhất. Track khác giữ nhãn cũ.
- Bỏ phiếu: sau mỗi lần embed thêm tên (hoặc `"Unknown"`) vào `votes`; nhãn hiển thị là tên được nhiều phiếu nhất nếu có ≥ `RT_MIN_VOTES` phiếu, ngược lại là `""` (chưa chắc, vẽ "...").
- Crop để embed: lấy `face_bgr` đã căn chỉnh do chính lần `detect_faces(frame_nho, backend=REALTIME_DETECTOR)` trả về, khớp với khung của track. Không chạy detect lần hai.

### 5.3 `RealtimeRecognizer` (trong `realtime.py`)

```python
class RealtimeRecognizer:
    def __init__(self, predictor, detector=None, **params): ...
    def start(self): ...                 # tạo và chạy worker thread (daemon)
    def stop(self): ...                  # dừng worker, đợi thread kết thúc (có timeout)
    def submit(self, frame_bgr): ...     # gọi từ luồng video; chỉ giữ frame MỚI NHẤT (ghi đè frame chờ), không chặn
    def latest(self) -> "Snapshot": ...  # kết quả mới nhất: danh sách FaceResult (khung đã quy về frame gốc) + thống kê
    def update_params(self, **params): ...   # đổi tham số khi đang chạy (Phase 3 dùng cho thanh trượt)
```

- Khoá (`threading.Lock`) chỉ bảo vệ việc đổi chỗ tham chiếu: `submit` ghi một biến, `latest` đọc một biến `Snapshot` bất biến (đã tạo xong rồi mới gán). Luồng video không bao giờ chờ worker.
- `Snapshot`: `results: list[FaceResult]`, `frame_shape`, `stats` (`worker_fps`, `detect_ms`, `embed_ms`, `n_faces`, `frames_dropped`, `timestamp`).
- Worker lặp: lấy frame mới nhất (nếu không có thì ngủ ~5 ms) → thu nhỏ → detect → tracker → embed (có giới hạn) → tạo `Snapshot` → gán. Bắt mọi ngoại lệ trong vòng lặp: ghi log, tăng bộ đếm lỗi, **không để thread chết**; sau 20 lỗi liên tiếp thì dừng và báo qua `stats["error"]`.
- TensorFlow/Keras chạy trong thread nền: **chỉ một worker duy nhất**; nạp model (gọi thử `embed_face` một ảnh giả) ngay trong `start()` trước khi vòng lặp bắt đầu, để lần đầu chậm không rơi vào lúc đang xem video.
- Không dùng `st.*` hay `st.cache_resource` trong module này; model và predictor được truyền từ ngoài vào.

### 5.4 Tối ưu `scores` của SVM-linear (tuỳ chọn, có kiểm chứng)

`predict_name_with_algo` gọi `model.predict` rồi `classifiers.scores` (`classifiers.py:111`); với SVM-linear `scores` mất ~63 ms mỗi mặt. Có thể tính `scores` một lần rồi lấy nhãn bằng `model.classes_[argmax]`. **Chỉ áp dụng nếu chứng minh được kết quả giống 100%** trên toàn bộ ảnh `external/pins_demo_test/` đã qua pipeline; nếu có dù một mẫu khác thì giữ gọi cả hai. Nếu áp dụng, làm trong `make_algo_predictor` của `recognition.py`, **không** sửa `predict_name_with_algo` (đang được trang upload dùng).

## 6. Các bước Ngày 2

### Bước 2.1. Thêm tham số vào `config.py`

Thêm nhóm REALTIME (bảng 5.1), có chú thích tiếng Việt như phong cách hiện có.

### Bước 2.2. Viết tracker và hàm IoU trong `realtime.py`

Tách thành hàm/class thuần (`iou(boxA, boxB)`, `class Tracker` có `update(detections, now) -> (tracks, to_embed)`) để kiểm thử riêng bằng khung giả không cần model.

### Bước 2.3. Viết `RealtimeRecognizer`

Theo mục 5.3.

### Bước 2.4. Kiểm thử tracker bằng khung giả

Một đoạn kiểm thử nhỏ (có thể là `python -m realtime --selftest`) cho các tình huống: một khung đứng yên (giữ cùng id), khung di chuyển chậm (giữ id), hai khung gần nhau (không đổi id chéo), khung biến mất rồi xuất hiện lại sau `RT_MAX_AGE_S` (id mới). Cho kết quả `PASS/FAIL` từng tình huống.

### Bước 2.5. Viết `realtime_demo.py`

```bash
python realtime_demo.py --images external/pins_demo_test --faces 3 --seconds 30 --algo SVM-linear --out results/realtime/demo_3faces.mp4
python realtime_demo.py --video duong_dan/video.mp4 --algo KNN
```

Hai chế độ nguồn:

- `--video FILE`: đọc bằng `cv2.VideoCapture(FILE)` (đọc file trên đĩa, không cần webcam).
- `--images DIR --faces N`: **video tổng hợp** có ground truth. Mỗi giây chọn ngẫu nhiên (hạt giống cố định) N người khác nhau từ `DIR`, lấy 1 ảnh mỗi người, đặt lên khung 640x480 và cho trượt chậm ngang qua khung; mỗi mặt có nhãn đúng là tên thư mục. Cho phép N = 1, 2, 3, và có chế độ "người ra vào" (ảnh xuất hiện rồi biến mất).

Vòng lặp: phát khung với tốc độ ~25 khung/giây thật (giả lập webcam: `time.sleep` theo thời gian); mỗi khung gọi `recognizer.submit(frame)`, rồi `draw_results(frame, recognizer.latest().results)`; nếu có `--out` thì ghi video, nếu có `--show` thì hiện cửa sổ (`cv2.imshow`; bỏ qua khi máy không có màn hình). Cuối cùng in báo cáo.

Số liệu in ra và ghi `results/realtime/demo_<tên>.csv`:

| Số liệu | Cách tính |
|---|---|
| `display_fps` | Số khung vẽ được mỗi giây (luồng video; phải gần bằng tốc độ nguồn) |
| `worker_fps` | Số vòng worker mỗi giây |
| `detect_ms` trung vị / p95 | Từ `stats` |
| `embed_ms` trung vị | Từ `stats` |
| `time_to_label_s` | Từ lúc một mặt xuất hiện đến lúc có nhãn đầu tiên khác `""` (trung vị, p95) |
| `label_accuracy` | (chỉ chế độ `--images`) trong số các thời điểm một mặt đang hiển thị nhãn, tỉ lệ nhãn đúng tên thư mục; Unknown của người đã biết tính là sai |
| `flicker_per_min` | Số lần nhãn của cùng một track đổi sau khi đã ổn định (kỳ vọng gần 0) |
| `id_switches` | Số lần một mặt thật bị đổi id track (chỉ chế độ tổng hợp) |

### Bước 2.6. Chạy đo

Chạy cho N = 1, 2, 3 mặt và cho mỗi thuật toán demo (SVM-linear, KNN), mỗi lần 30 giây. Ghi vào bảng 8.

Mục tiêu (đặt trước, điều chỉnh theo số đo nền ở Phase 0 nếu máy yếu hơn dự kiến):

| Chỉ tiêu | Mục tiêu |
|---|---|
| `display_fps` | ≥ 20 (luồng video không phụ thuộc nhận diện) |
| `worker_fps` với 3 mặt | ≥ 4 vòng/giây |
| `time_to_label_s` trung vị | ≤ 1,5 giây |
| `label_accuracy` | không thấp hơn `top1_acc_all` của Phase 1 quá 2 điểm |
| `flicker_per_min` | ≤ 2 mỗi track |

### Bước 2.7. Kiểm tra lại trang upload

`python check_upload_regression.py --compare` phải `OK` (Ngày 2 không được đổi gì liên quan đến upload) và upload tay 1 ảnh để chắc chắn.

## 7. Tiêu chí hoàn thành

- [ ] Ngày 1: `check_upload_regression.py --compare` OK sau refactor; kiểm tra tay trang upload khớp.
- [ ] `recognition.py`, `realtime.py`, `realtime_demo.py` đã có; `realtime.py` không import `streamlit`.
- [ ] Tự kiểm tra tracker PASS cả 4 tình huống.
- [ ] Bảng 8 đã điền cho N = 1, 2, 3 và 2 thuật toán; đạt các chỉ tiêu hoặc có giải thích vì sao không đạt.
- [ ] Không có thread rò: `recognizer.stop()` xong thì `threading.enumerate()` không còn worker.
- [ ] Trang upload không đổi.

## 8. Kết quả (điền sau khi làm)

| N mặt | Thuật toán | display_fps | worker_fps | detect_ms (trung vị / p95) | embed_ms | time_to_label_s | label_accuracy | flicker/phút | id_switches |
|---|---|---|---|---|---|---|---|---|---|
| 1 | SVM-linear | | | | | | | | |
| 1 | KNN | | | | | | | | |
| 2 | SVM-linear | | | | | | | | |
| 2 | KNN | | | | | | | | |
| 3 | SVM-linear | | | | | | | | |
| 3 | KNN | | | | | | | | |

| Mục | Giá trị |
|---|---|
| Kết quả `check_upload_regression.py --compare` | |
| Tham số đã chỉnh so với mặc định (và vì sao) | |
| Có áp dụng tối ưu mục 5.4 không (và bằng chứng giống 100%) | |
| Ngày làm, người accept | |

## 9. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| `import app` ngoài Streamlit lỗi hoặc ồn | Cách xử lý ở bước 1.1; phương án cuối: kiểm tra hồi quy trực tiếp trên `recognition.py` cộng kiểm tra tay trang upload |
| TensorFlow trong thread nền chậm hoặc lỗi | Một worker; nạp model trước khi chạy vòng lặp; nếu vẫn lỗi, chuyển worker thành process riêng (`multiprocessing`) và ghi vào phase |
| 3 mặt vẫn quá chậm (embed 95 đến 180 ms và `scores` 63 ms mỗi mặt) | `RT_MAX_EMBEDS_PER_CYCLE=1`, tăng `RT_EMBED_EVERY_S`, giảm `RT_DETECT_WIDTH`; xét tối ưu 5.4 |
| Khung giả (ảnh Pins dán lên nền) khác webcam thật | Chấp nhận ở phase này; đo webcam thật ở Phase 4 |
| Gán nhầm track khi hai người đi ngang nhau | Đo `id_switches`; nếu cao thì thêm kiểm tra khoảng cách embedding khi ghép |

## 10. Điểm dừng

Báo người dùng: `--compare` OK, bảng 8, các chỉ tiêu đạt hay chưa, chỗ nào cần chỉnh. Hỏi: **"Accept Phase 2 để sang Phase 3 (trang Streamlit)?"** Cập nhật bảng trạng thái ở [README.md](README.md) mục 7.
