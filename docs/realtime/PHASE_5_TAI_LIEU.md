# Phase 5: Hoàn thiện tài liệu

> **Trạng thái: chưa làm.** Thời lượng: **0,5 ngày**. Chỉ viết và cập nhật tài liệu, không đổi logic.
> Cần xong: [Phase 4](PHASE_4_TINH_CHINH.md) (có số đo và giá trị mặc định cuối). Tổng quan: [README.md](README.md).

## 1. Mục tiêu

Để người khác (giảng viên, thành viên nhóm, chính bạn sau vài tháng) đọc là hiểu tính năng realtime làm gì, hoạt động ra sao, chạy thế nào, đo được gì, giới hạn ở đâu. Gồm:

1. Báo cáo hoàn chỉnh `docs/realtime/BAO_CAO_REALTIME.md`.
2. Cập nhật `README.md` (cách dùng).
3. Cập nhật `CLAUDE.md` (kiến trúc cho các phiên làm việc sau).
4. Bổ sung mục Demo của `docs/SO_SANH_THUAT_TOAN.md`.
5. Chốt các file phase: điền hết các bảng kết quả, đổi trạng thái.

## 2. Thời gian

| Buổi | Việc | Ước tính |
|---|---|---|
| Sáng | Bước 1 và 2: báo cáo `BAO_CAO_REALTIME.md` | 2,5 giờ |
| Chiều | Bước 3 đến 6: README, CLAUDE.md, báo cáo 10 thuật toán, rà soát, hỏi accept | 1,5 giờ |

## 3. Điều kiện đầu vào

- Mọi bảng kết quả ở Phase 0 đến 4 đã có số (hoặc ghi lý do trống).
- Người dùng đã duyệt giá trị mặc định cuối.

## 4. Các bước

### Bước 1. Dàn ý `BAO_CAO_REALTIME.md`

Viết theo phong cách `docs/SO_SANH_THUAT_TOAN.md`: dòng trạng thái ở đầu dạng blockquote (`> **Trạng thái: ... (ngày) ...**`), phần "Tóm tắt" gạch đầu dòng có chữ đậm dẫn ý, bảng, lệnh tái lập trong khối mã, số theo cách viết tiếng Việt (0,9886; 1.092), nêu rõ hạn chế và phần chưa đo. Kết thúc bằng "Tái lập" và "File liên quan".

| Mục | Nội dung |
|---|---|
| Tóm tắt | 5 đến 7 gạch đầu dòng: làm được gì, nhanh bao nhiêu, đúng bao nhiêu, hạn chế lớn nhất |
| 1. Yêu cầu | Webcam, tự nhận diện, nhiều người cùng lúc, giữ chức năng upload |
| 2. Hiện trạng trước khi làm | Bảng khảo sát (từ README của thư mục này mục 3), vì sao RetinaFace không đủ nhanh |
| 3. Thiết kế | Sơ đồ kiến trúc (mục 4 của README thư mục này), vai trò từng file, luồng frame, tracker, bỏ phiếu, giới hạn embed |
| 4. Chọn detector | Bảng tốc độ và độ chính xác (Phase 1 mục 7), quyết định và lý do |
| 5. Tham số | Bảng tham số mặc định cuối và ý nghĩa (Phase 2 mục 5.1, giá trị theo Phase 4) |
| 6. Đo hiệu năng | FPS theo số mặt, thuật toán, có và không tracking (Phase 4 mục 7.1) |
| 7. Kết quả theo kịch bản | Bảng S1 đến S10 (Phase 4 mục 7.2) |
| 8. Ngưỡng Unknown | Ngưỡng cũ và ngưỡng realtime, vì sao khác (Phase 4 bước 2) |
| 9. Các trường hợp sai và hạn chế | Phase 4 mục 7.3; không chống giả mạo; chỉ nguồn "Thuật toán phân loại"; CPU; HTTPS cho truy cập từ xa; cỡ mẫu thử |
| 10. Hồi quy trang upload | Cách `check_upload_regression.py` chứng minh upload không đổi, kết quả cuối |
| 11. Cách dùng | Các bước chạy và dùng trang (giống nội dung thêm vào `README.md` ở bước 3 bên dưới) |
| 12. Tái lập | Các lệnh: cài, benchmark detector, `realtime_demo.py`, `check_upload_regression.py`, `calibrate_live_threshold.py` |
| 13. File liên quan | Danh sách file mới và file đã sửa |

Quy tắc: mọi con số trong báo cáo phải dẫn được về một file CSV hoặc bảng trong file phase; không viết số nào chưa đo. Chỗ chưa đo ghi "chưa đo".

### Bước 2. Hình minh hoạ

- Chụp màn hình trang realtime có 1 mặt và nhiều mặt (người dùng chụp khi thử ở Phase 4; che mặt nếu cần). Lưu vào `results/realtime/` và nhúng vào báo cáo.
- Biểu đồ FPS theo số mặt (nếu muốn): có thể vẽ bằng matplotlib vào `results/realtime/fps_vs_faces.png`; giữ cùng phong cách với `plot_results.py` (màu nhấn xanh dương cho 2 thuật toán demo, xám cho phần còn lại).

### Bước 3. Cập nhật `README.md`

Thêm mục "Nhận diện realtime bằng webcam" gồm:

- Yêu cầu: đã train model (`python train_model.py`), đã cài `streamlit-webrtc` và `av` (`pip install -r requirements.txt`).
- Cách dùng: `streamlit run app.py` → "🎥 Nhận diện realtime" → chọn thuật toán → START → cho phép camera.
- Lưu ý: webcam chỉ chạy trên `localhost` hoặc HTTPS; nhãn lên sau khoảng 1 giây; chạy CPU.
- Liên kết tới `docs/realtime/BAO_CAO_REALTIME.md`.
- Giữ nguyên mô tả trang upload và chụp ảnh.

### Bước 4. Cập nhật `CLAUDE.md`

CLAUDE.md viết bằng tiếng Anh và dành cho các phiên làm việc sau; thêm ngắn gọn:

- Mục Overview/Current state: realtime webcam page exists (nêu ngày), 6 phase trong `docs/realtime/`.
- Mục Commands: `python realtime_demo.py ...`, `python check_upload_regression.py --compare`, `python benchmark_realtime_detector.py ...`, `python calibrate_live_threshold.py`.
- Mục Architecture: `recognition.py` (pure predict/draw helpers shared by app and realtime; `predict_name_*` and `draw_rounded_rectangle` now live here and are re-imported by `app.py`), `realtime.py` (worker thread + IoU tracker + vote smoothing), `config.REALTIME_DETECTOR` (differs from `DETECTOR_BACKEND`), page "🎥 Nhận diện realtime" (algorithm source only).
- Mục Gotchas: realtime crops come from `REALTIME_DETECTOR`, the saved models were trained on RetinaFace crops (measured gap in `results/realtime/detector_compare.csv`); `draw_results` fixes the Unknown colour quirk, the upload page's `draw_vietnamese_text` keeps it on purpose; do not call `st.*` or `st.cache_*` from the worker thread; run `check_upload_regression.py --compare` after touching `app.py`, `recognition.py`, `detect_faces`.
- Cập nhật câu "Still not done" nếu có phần chưa làm (nguồn `.npz`/PostgreSQL cho realtime).

Chỉ sửa các đoạn liên quan; không viết lại những phần khác của CLAUDE.md.

### Bước 5. Bổ sung `docs/SO_SANH_THUAT_TOAN.md`

Ở mục "Demo" (mục 8 của báo cáo) thêm một đoạn ngắn: app có thêm chế độ realtime, trỏ sang `docs/realtime/BAO_CAO_REALTIME.md`, nhắc rằng số FPS trong báo cáo 10 thuật toán là FPS xử lý ảnh tĩnh (RetinaFace) còn FPS realtime là cấu hình khác (detector nhanh + tracking), không so trực tiếp. Không đổi số liệu cũ.

### Bước 6. Chốt các file phase và rà soát

- Điền nốt mọi bảng kết quả trong `PHASE_0` đến `PHASE_4`; đổi dòng trạng thái đầu file thành "Đã xong (ngày), accept bởi ...".
- Cập nhật bảng trạng thái ở [README.md](README.md) mục 7 và dòng trạng thái đầu README.
- Rà soát: các đường dẫn file và số dòng trong docs còn đúng sau khi code thay đổi (số dòng của `app.py` đã đổi sau Phase 2 và 3; cập nhật hoặc bỏ số dòng, chỉ giữ tên hàm).
- Kiểm tra liên kết giữa các file `.md` trong `docs/realtime/` mở được.
- Chạy lại một lần `python check_upload_regression.py --compare` và `streamlit run app.py` để chắc chắn bản cuối vẫn sạch.

## 5. Tiêu chí hoàn thành

- [ ] `docs/realtime/BAO_CAO_REALTIME.md` đủ 13 mục, không có ô số liệu trống không giải thích.
- [ ] `README.md`, `CLAUDE.md`, `docs/SO_SANH_THUAT_TOAN.md` đã cập nhật đúng phần.
- [ ] Các file phase đã điền kết quả và đổi trạng thái; bảng ở README thư mục này đầy đủ.
- [ ] Mọi lệnh trong mục "Tái lập" chạy đúng như viết (Claude chạy thử hoặc người dùng xác nhận).
- [ ] Trang upload vẫn đúng.

## 6. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Số liệu trong báo cáo lệch với CSV sau lần chỉnh cuối | Viết báo cáo sau khi Phase 4 chốt; lấy số trực tiếp từ CSV, không gõ từ trí nhớ |
| Docs nhanh chóng lỗi thời khi code đổi | Ghi ngày ở dòng trạng thái; ghi rõ "số liệu tạo từ file CSV nào, lệnh nào" như báo cáo 10 thuật toán |
| CLAUDE.md quá dài | Chỉ thêm các điều thật sự giúp phiên sau (kiến trúc, lệnh, bẫy), không chép số đo |

## 7. Kết quả (điền sau khi làm)

| Mục | Giá trị |
|---|---|
| Ngày làm | |
| File đã tạo | |
| File đã sửa | |
| Lệnh tái lập đã chạy thử | |
| Người đọc và accept | |

## 8. Điểm dừng

Báo người dùng danh sách file đã tạo và sửa, nhờ đọc `docs/realtime/BAO_CAO_REALTIME.md`. Hỏi: **"Accept để kết thúc dự án realtime?"** Chỉ commit khi người dùng yêu cầu. Cập nhật bảng trạng thái ở [README.md](README.md) mục 7.
