# Kế hoạch: nhận diện khuôn mặt realtime từ webcam

> **Trạng thái: mới có kế hoạch (viết 2026-11-18). Chưa phase nào được làm, chưa sửa dòng code nào.**
> Mỗi phase là một file riêng trong thư mục này. Chỉ làm phase mà người phụ trách đề tài chỉ định; xong một phase thì dừng, kiểm tra, rồi hỏi accept mới sang phase kế.

## 1. Mục tiêu và phạm vi

Hiện app chỉ nhận diện trên ảnh tĩnh: upload ảnh (`page_nhan_dien`, `app.py:361`) hoặc chụp một ảnh (`st.camera_input`, `app.py:271`). Mục tiêu mới:

- Mở webcam trong app; hễ có mặt người trong khung hình thì **tự nhận diện ngay**, không cần upload.
- **Nhiều người trong một khung hình** đều được nhận diện, mỗi người một khung và một nhãn.
- Người lạ (không có trong dữ liệu) hiện "Unknown" như ở trang upload.

Phạm vi bản đầu:

| Trong phạm vi | Ngoài phạm vi (để sau) |
|---|---|
| Nguồn "Thuật toán phân loại (ArcFace)": SVM-linear, KNN và các thuật toán khác đã train | Nguồn `.npz` (KNN theo từng embedding) và PostgreSQL |
| Webcam của trình duyệt qua `streamlit-webrtc` | Camera IP/RTSP, `cv2.VideoCapture` phía server |
| Chạy trên CPU | GPU (venv Windows, TF 2.21 không có GPU) |
| Nhận diện, vẽ khung và nhãn tiếng Việt, hiện FPS | Chống giả mạo (ảnh/video của người khác đưa lên camera) |

**Ràng buộc bắt buộc: chức năng upload ảnh tĩnh phải được giữ nguyên.** Trang "🔍 Nhận diện" (upload) và trang "📸 Chụp ảnh" vẫn chạy như cũ. Realtime là một trang mới thêm vào, không thay thế. Mọi thay đổi ở các phase sau phải giữ nguyên kết quả của trang upload (từng phase có bước kiểm tra riêng, phase 2 có script so sánh trước và sau).

## 2. Quyết định đã chốt (2026-11-18)

| Chủ đề | Quyết định | Lý do |
|---|---|---|
| Lấy hình webcam | `streamlit-webrtc` (webcam của trình duyệt, xử lý từng frame ở server) | Chạy được cả khi app ở máy khác; môi trường Linux của Claude không có `/dev/video*`. `cv2.VideoCapture` chỉ mở camera của máy chạy server |
| Tăng tốc | Detector nhanh **YuNet** cho realtime, giữ ArcFace và các model đã train; có phase đo lại độ chính xác | RetinaFace mất 1,2 đến 2,7 giây mỗi frame trên CPU, không thể chạy video |
| Đơn vị thời gian | Ngày làm việc, mỗi phase 0,5 đến 1 ngày, phase dài tách "Ngày 1 / Ngày 2" | Theo yêu cầu |
| Nguồn model | Chỉ nguồn "Thuật toán phân loại (ArcFace)" | Hai thuật toán demo đã train sẵn: SVM-linear và KNN |

## 3. Hiện trạng đã khảo sát

Số liệu lấy từ code, `results/fps.csv`, `results/algorithms/timing.csv` và hai báo cáo trong `docs/`.

| Phát hiện | Vị trí | Hệ quả cho realtime |
|---|---|---|
| `detect_faces` dùng RetinaFace, 1,2 s/ảnh (máy i5-1135G7) đến 2,7 s/ảnh (máy 16 luồng) trên CPU | `face_pipeline.py:45`, `config.py` (`DETECTOR_BACKEND`) | Không thể chạy từng frame. Cần detector nhanh và đo lại độ chính xác |
| `embed_face` (ArcFace) 95 đến 180 ms/mặt, chỉ nhận một ảnh mỗi lần | `face_pipeline.py:86` | Không embed mọi frame: chỉ embed khi mặt mới xuất hiện hoặc đến hạn làm mới (tracking) |
| SVM-linear: `scores` ~63 ms/mặt (`predict` chỉ ~0,8 ms); KNN ~6 ms cho `predict` và ~6 ms cho `scores` (đo với 200 vector, tập train thật lớn hơn nên KNN sẽ chậm hơn) | `classifiers.py`, `results/algorithms/timing.csv` | Chấp nhận được nếu có tracking |
| Logic nhận diện nằm lẫn với giao diện trong `page_nhan_dien` | `app.py:409-465` | Hàm thuần dùng lại được: `predict_name_with_algo` (`app.py:201`), `predict_name_from_face` (`app.py:181`), `draw_rounded_rectangle` (`app.py:161`) |
| `draw_vietnamese_text` copy cả frame cho mỗi nhãn và dựng lại font mỗi lần | `app.py:217-234` | Cần hàm vẽ riêng cho video (một lần chuyển PIL cho cả frame) |
| Chỉ có chụp một ảnh, chưa có luồng video; `streamlit-webrtc` và `av` chưa cài | `app.py:271` | Phase 0 cài thêm |
| Model đã train dùng crop của RetinaFace; model lưu `pipeline_fingerprint` có `detector=retinaface` | `config.py` (`PIPELINE_FINGERPRINT`), `train_model.py:68` | Đổi detector có thể làm lệch. Phase 1 đo, có điểm quyết định |
| Model hiện tại **được train trên toàn bộ** 5.459 ảnh của tập con 109 người | `train_model.py:68` (`train_algorithm` fit trên toàn bộ X, y) | Không được đo độ chính xác bằng ảnh trong `dataset/` (lộ đáp án). Dùng `external/pins_demo_test/` (105 người, 10 ảnh mỗi người, chưa từng đưa vào train) |
| `.venv` là venv của Windows (`.venv/Scripts`, `.venv/Lib`); shell Linux của Claude không chạy được app và không có `/dev/video*` | `.venv/pyvenv.cfg`, `.claude/launch.json` | Lệnh chạy DeepFace do người dùng chạy trên Windows, hoặc dựng venv Linux riêng (xem mục 8) |

## 4. Kiến trúc đích

```
Trình duyệt (webcam) ──WebRTC──► streamlit-webrtc
                                   │  mỗi frame: VideoProcessor.recv(frame)
                                   ├─► recognizer.submit(frame)             chỉ giữ frame mới nhất, bỏ frame cũ
                                   └─► draw_results(frame, recognizer.latest())   vẽ kết quả mới nhất, trả frame ngay

Worker thread (RealtimeRecognizer, realtime.py), lặp:
   frame mới nhất → thu nhỏ về detect_width → YuNet detect + align
                  → tracker IoU: khớp khung với track cũ, tạo track mới
                  → track mới / đến hạn làm mới (tối đa N mặt mỗi vòng)
                       → ArcFace embed → classifier (SVM-linear, KNN, ...) → bỏ phiếu nhiều lần → nhãn
                  → công bố ảnh chụp (snapshot) kết quả cho luồng video vẽ
```

Nguyên tắc: **luồng video không bao giờ chờ nhận diện.** Hình luôn mượt, khung và nhãn đi sau hình vài trăm mili giây. Việc nặng (detect, embed, classify) chạy ở một worker duy nhất.

File mới dự kiến: `recognition.py` (hàm thuần dùng chung), `realtime.py` (worker, tracker), `realtime_demo.py` (chạy offline trên video/ảnh), `benchmark_realtime_detector.py`, `check_upload_regression.py`. File sửa: `face_pipeline.py` (thêm tham số `backend`), `config.py` (hằng realtime), `app.py` (re-export hàm đã chuyển, thêm trang realtime), `requirements.txt`.

## 5. Các phase và lịch

| Phase | File | Nội dung | Thời lượng | Cộng dồn | Điểm dừng cần accept |
|---|---|---|---|---|---|
| 0 | [PHASE_0_CHUAN_BI.md](PHASE_0_CHUAN_BI.md) | Cài thư viện, kiểm tra môi trường, đo nền, ghi kết quả tham chiếu của trang upload | 0,5 ngày | 0,5 | Cài được, app cũ vẫn chạy, có số đo nền |
| 1 | [PHASE_1_DETECTOR.md](PHASE_1_DETECTOR.md) | Thêm tham số `backend`, đo tốc độ và độ chính xác của detector nhanh, quyết định dùng YuNet hay không | 1 ngày (+0,5 đến 1 ngày nếu phải train lại, nhánh 1B) | 1,5 | Chọn detector dựa trên bảng so sánh |
| 2 | [PHASE_2_LOI_REALTIME.md](PHASE_2_LOI_REALTIME.md) | Ngày 1: tách hàm thuần, vẽ nhanh, script kiểm tra hồi quy. Ngày 2: worker, tracker, bỏ phiếu, chạy offline trên video | 2 ngày | 3,5 | Chạy offline đúng và đủ nhanh; upload không đổi |
| 3 | [PHASE_3_TRANG_STREAMLIT.md](PHASE_3_TRANG_STREAMLIT.md) | Trang "🎥 Nhận diện realtime" dùng `streamlit-webrtc` | 1 ngày | 4,5 | Người dùng thử webcam thật theo checklist |
| 4 | [PHASE_4_TINH_CHINH.md](PHASE_4_TINH_CHINH.md) | Thử nhiều người, chỉnh ngưỡng và tham số, đo FPS thật, ghi lỗi đã biết | 0,5 đến 1 ngày | 5 đến 5,5 | Bảng FPS và danh sách lỗi |
| 5 | [PHASE_5_TAI_LIEU.md](PHASE_5_TAI_LIEU.md) | Báo cáo hoàn chỉnh, cập nhật README, CLAUDE.md, báo cáo 10 thuật toán | 0,5 ngày | 5,5 đến 6 | Người dùng đọc docs |

Tổng: **5,5 đến 6 ngày làm việc** (6 đến 7 nếu rơi vào nhánh 1B). Ước tính cho một người làm cùng Claude, gồm kiểm tra sau mỗi phase, **chưa tính thời gian chờ** người dùng thử webcam và accept.

Thứ tự bắt buộc: 0, 1, 2, 3, 4, 5. Phase 1 quyết định detector cho các phase sau. Phase 2 cần kết quả của phase 1. Phase 3 cần phase 2. Phase 4 cần webcam thật.

## 6. Quy tắc làm việc

1. **Chỉ làm phase được chỉ định.** Dùng câu như "làm Phase 1".
2. **Xong một phase thì dừng.** Chạy các bước kiểm tra trong file phase, điền bảng kết quả, cập nhật bảng trạng thái ở mục 7, rồi hỏi accept. Không tự sang phase sau.
3. **Không phá trang upload ảnh.** Mỗi phase có bước kiểm tra lại: mở app, upload cùng ảnh tham chiếu (lưu ở Phase 0) và so kết quả.
4. Hàm có sẵn chỉ được thêm tham số có giá trị mặc định giữ hành vi cũ (ví dụ `detect_faces(..., backend=None)`).
5. Không ghi đè model đã train (`output/model_ArcFace_*`). Nếu phải train lại thì lưu tên khác.
6. Không commit, không push nếu người dùng chưa yêu cầu.
7. Ghi quyết định và số đo vào chính file phase đó (mục "Kết quả"), để đọc lại hiểu được vì sao chọn như vậy.

## 7. Theo dõi trạng thái

| Phase | Trạng thái | Ngày làm | Người accept | Ghi chú |
|---|---|---|---|---|
| 0 | Chưa làm | | | |
| 1 | Chưa làm | | | |
| 2 | Chưa làm | | | |
| 3 | Chưa làm | | | |
| 4 | Chưa làm | | | |
| 5 | Chưa làm | | | |

## 8. Môi trường chạy

- Dự án phát triển trên **Windows, Python 3.12, CPU**. Lệnh trong các file phase viết dạng `python ...`; trên Windows dùng `.venv\Scripts\python.exe ...`.
- Shell Linux của Claude **không** có TensorFlow và không dùng được `.venv` (đó là venv Windows). Với bước cần chạy DeepFace (đo tốc độ, đo độ chính xác, chạy app), có hai cách, quyết định ở Phase 0:
  - **Cách 1 (mặc định):** người dùng chạy lệnh trên Windows và gửi lại kết quả (dán output hoặc đường dẫn file CSV).
  - **Cách 2:** dựng venv Linux riêng (`python3.12 -m venv .venv-linux`, cài `requirements.txt`) để Claude tự chạy được. Có GPU GTX 1660 trên máy Linux nhưng TensorFlow 2.21 trong `requirements.txt` bản CPU; GPU chưa dùng trong kế hoạch này.
- Webcam thật chỉ thử được trên máy có webcam và trình duyệt: do người dùng thử (Phase 3, 4).
- `getUserMedia` của trình duyệt chỉ cho phép webcam trên `localhost` hoặc HTTPS. Truy cập app từ máy khác qua `http://<ip>:8501` sẽ bị chặn webcam; cần HTTPS hoặc dùng chung máy.

## 9. Rủi ro chung

| Rủi ro | Mức | Cách xử lý | Phase |
|---|---|---|---|
| YuNet làm giảm độ chính xác so với RetinaFace (crop và căn chỉnh khác) | Cao | Đo trên `external/pins_demo_test/`, có điểm quyết định, nhánh train lại trên crop YuNet | 1 |
| `cv2.FaceDetectorYN` không có trong bản OpenCV 5.0.0.93 đang cài, hoặc DeepFace không tải được trọng số YuNet | Trung bình | Kiểm tra ở Phase 0; phương án: detector khác (`opencv`, `ssd`, `fastmtcnn`) | 0, 1 |
| Cài `streamlit-webrtc`/`av` làm đổi phiên bản numpy, opencv, protobuf và hỏng TF/DeepFace | Trung bình | Lưu `pip freeze` trước và sau, so sánh, rollback nếu đổi | 0 |
| TensorFlow chạy trong thread nền lỗi hoặc rò bộ nhớ | Trung bình | Một worker duy nhất, tải model một lần ngoài vòng lặp, không gọi `st.*` trong thread | 2, 3 |
| Vẫn chậm trên CPU yếu | Trung bình | Tracking, giới hạn số lần embed mỗi vòng, thu nhỏ frame | 2, 4 |
| Ngưỡng Unknown hiệu chỉnh trên ảnh dataset, ảnh webcam khác ánh sáng và góc | Trung bình | Chỉnh ở Phase 4, có thể hiệu chỉnh lại trên ảnh qua YuNet | 4 |
| Người dùng chưa có model đã train | Thấp | Trang realtime báo lệnh `python train_model.py` giống trang upload | 3 |

## 10. Câu hỏi còn mở (chốt ở phase tương ứng)

- Máy nào chạy lệnh: Windows của người dùng hay dựng venv Linux? (Phase 0)
- Ngưỡng chấp nhận khi đổi detector: đề xuất tụt không quá 3 điểm phần trăm độ chính xác và tỉ lệ tìm thấy mặt từ 97% trở lên. (Phase 1)
- Có muốn thêm nguồn `.npz` và PostgreSQL vào trang realtime sau này? (sau Phase 5)

## 11. File liên quan hiện có

- `app.py`, `face_pipeline.py`, `classifiers.py`, `knn_func.py`, `config.py`, `train_model.py`
- `results/fps.csv`, `results/algorithms/timing.csv`, `results/algorithms/subset_people.txt`
- `external/pins_demo_test/` (105 người, mỗi người 10 ảnh chưa từng train), `external/demo_test_result.csv` (lần đánh giá trước trên tập này)
- `docs/KE_HOACH_DATASET_LON.md` (kiểu trình bày kế hoạch), `docs/SO_SANH_THUAT_TOAN.md` (báo cáo 10 thuật toán)
