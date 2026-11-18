# Phase 4: Tinh chỉnh và đo trên webcam thật

> **Trạng thái: chưa làm.** Thời lượng: **0,5 đến 1 ngày** (Ngày 1 là đủ nếu ít lỗi; Ngày 2 chỉ khi phải chỉnh lại nhiều).
> Cần xong: [Phase 3](PHASE_3_TRANG_STREAMLIT.md) (trang realtime chạy được trên webcam thật). Phase sau: [PHASE_5_TAI_LIEU.md](PHASE_5_TAI_LIEU.md). Tổng quan: [README.md](README.md).

## 1. Mục tiêu

Phase 2 đo trên khung giả (ảnh Pins dán lên nền), Phase 3 mới chỉ thử sơ. Phase này đo **trên webcam thật, nhiều người, nhiều điều kiện** và chỉnh tham số theo số liệu:

1. Chốt ngưỡng Unknown cho khung hình live (ngưỡng hiện tại hiệu chỉnh trên ảnh dataset, ảnh webcam khác về ánh sáng và góc).
2. Chốt tham số mặc định: kích thước mặt tối thiểu, độ rộng detect, tần suất embed, số phiếu.
3. Có bảng FPS và bảng lỗi đã biết, làm dữ liệu cho báo cáo (Phase 5).

## 2. Thời gian

| Ngày | Việc | Ước tính |
|---|---|---|
| Ngày 1 sáng | Bước 1 và 2: thêm ghi log vào trang, hiệu chỉnh lại ngưỡng Unknown trên ảnh qua pipeline realtime | 3 giờ |
| Ngày 1 chiều | Bước 3 và 4: người dùng thử theo kịch bản, Claude tổng hợp log và chỉnh tham số | 3 giờ |
| Ngày 2 (chỉ khi cần) | Bước 5 và 6: chỉnh lại, đo lại, điền kết quả, hỏi accept | 0,5 ngày |

## 3. Điều kiện đầu vào

- Phase 3 accept; người dùng đã thử webcam thật.
- Có ít nhất 2 người (hoặc ảnh người trên điện thoại) để thử đa người. Cần cả người **có trong dữ liệu** (một trong 105 người Pins hoặc thành viên đã chụp ảnh qua trang "📸 Chụp ảnh") và người **lạ**.

## 4. Các bước

### Bước 1. Ghi log từ trang realtime

Thêm tuỳ chọn "Ghi log phiên này" vào expander "Cài đặt nâng cao". Khi bật, mỗi 0,5 giây ghi một dòng vào `results/realtime/live_log_<giờ>.csv`:

`timestamp, scenario, track_id, name, confidence, box_w, box_h, display_fps, worker_fps, detect_ms, embed_ms, n_faces`

`scenario` là ô nhập chữ ngắn (ví dụ "2 người, sáng") để tách kịch bản khi phân tích. Ghi log nằm ở worker/processor, không ở luồng `recv`, và không làm chậm video.

### Bước 2. Hiệu chỉnh lại ngưỡng Unknown cho pipeline realtime

Vấn đề: `meta["threshold"]` được tính từ dự đoán out-of-fold trên crop RetinaFace của dataset (`train_model.calibrate_confidence_threshold`, `train_model.py:53-65`: phân vị 5% của độ tin cậy các mẫu đoán đúng). Crop từ detector realtime và ảnh webcam có phân bố khác, nên ngưỡng có thể quá chặt (nhiều người quen bị Unknown) hoặc quá lỏng.

Làm hai việc, **không ghi đè `meta` của model cũ**:

1. Script `calibrate_live_threshold.py` (mới): chạy ảnh `external/pins_demo_test/` qua đúng pipeline realtime (detector realtime, embed, classifier), tính phân bố độ tin cậy của các dự đoán **đúng** và của dự đoán **sai**, đề xuất ngưỡng (phân vị 5% của độ tin cậy đúng, giống quy ước hiện tại) cho từng thuật toán demo, in so với ngưỡng cũ. Ghi `results/realtime/live_threshold.csv`.
2. Trang realtime có thanh "Độ nhạy Unknown" cộng thêm một độ lệch vào ngưỡng của model (mặc định 0 = dùng nguyên `meta["threshold"]`). Nếu quyết định dùng ngưỡng mới, lưu vào `config.py` dưới dạng hằng riêng cho realtime (`RT_THRESHOLD_OFFSET = {"SVM-linear": ..., "KNN": ...}`), **không đụng ngưỡng của trang upload**.

Lưu ý khi giải thích kết quả: ngưỡng của mỗi thuật toán ở thang điểm riêng (KNN: âm của khoảng cách; SVM: `decision_function`; các thuật toán khác: xác suất), không so sánh giữa các thuật toán. DecisionTree, GaussianNB, LDA có xác suất bão hoà nên ngưỡng gần như vô dụng (xem `SATURATED_CONFIDENCE`, `app.py:27`).

### Bước 3. Kịch bản thử (người dùng làm, bật ghi log)

Mỗi kịch bản 60 giây, mỗi thuật toán demo (SVM-linear, KNN) một lượt. Ghi tên kịch bản vào ô `scenario`.

| # | Kịch bản | Điều cần quan sát |
|---|---|---|
| S1 | 1 người quen, nhìn thẳng, đủ sáng | Mốc chuẩn: đúng tên, thời gian lên nhãn |
| S2 | 1 người lạ, nhìn thẳng | Phải ra Unknown; ghi tỉ lệ bị nhận nhầm thành người quen |
| S3 | 2 người quen cùng khung | Hai khung, hai nhãn đúng, không đổi nhãn chéo |
| S4 | 3 người (một lạ) | Số mặt đúng, FPS worker, độ trễ |
| S5 | Người ra vào khung liên tục | Track mới có nhãn sau bao lâu; không còn khung ma |
| S6 | Mặt nghiêng 30 đến 45 độ, cúi, ngẩng | Nhãn đứng yên hay nhảy; Unknown tăng bao nhiêu |
| S7 | Thiếu sáng, ngược sáng (cửa sổ phía sau) | Có còn tìm thấy mặt không (detect_rate), đúng tên không |
| S8 | Đeo kính, đội mũ, đeo khẩu trang | Độ chính xác giảm bao nhiêu |
| S9 | Người ở xa (mặt nhỏ, gần ngưỡng `RT_MIN_FACE`) | Từ khoảng cách nào trở đi không nhận được |
| S10 | Ảnh người khác hiện trên điện thoại giơ lên | Ghi nhận đúng/sai (chỉ để nêu hạn chế: hệ thống **không** chống giả mạo) |

### Bước 4. Phân tích log và chỉnh tham số

Claude đọc các file log và tính cho từng kịch bản: tỉ lệ thời gian có nhãn đúng, tỉ lệ Unknown, thời gian lên nhãn đầu tiên, số lần đổi nhãn (nhấp nháy), `display_fps`, `worker_fps`. Vì log không có ground truth tự động, **người dùng ghi tên người thật** cho mỗi kịch bản (ô `scenario` hoặc bảng riêng) để so khớp.

Chỉnh theo bảng ưu tiên (mỗi lần chỉ đổi một tham số, đo lại):

| Triệu chứng | Tham số cần xem |
|---|---|
| Nhãn nhấp nháy giữa hai tên | Tăng `RT_VOTE_LEN`, `RT_MIN_VOTES`; kiểm tra ngưỡng Unknown |
| Lên nhãn chậm | Giảm `RT_EMBED_EVERY_S` cho track mới, tăng `RT_MAX_EMBEDS_PER_CYCLE` |
| FPS worker thấp khi nhiều mặt | Giảm `RT_DETECT_WIDTH`, tăng `RT_EMBED_EVERY_S`, xét tối ưu `scores` (Phase 2 mục 5.4) |
| Khung rung | Tăng `RT_BOX_SMOOTH` |
| Đổi nhãn chéo khi hai người đi ngang nhau | Giảm `RT_IOU_MATCH` hoặc thêm kiểm tra embedding khi ghép track |
| Người quen thường bị Unknown | Hạ ngưỡng (bước 2), kiểm tra ánh sáng và `RT_MIN_FACE` |
| Người lạ bị nhận nhầm | Tăng ngưỡng; nếu thuật toán bão hoà thì ghi vào hạn chế, đổi sang thuật toán khác |
| Mặt nhỏ không nhận được | Hạ `RT_MIN_FACE` hoặc tăng `RT_DETECT_WIDTH` (đánh đổi tốc độ) |

### Bước 5. Chốt giá trị mặc định

Ghi các giá trị cuối vào `config.py` (nhóm REALTIME) kèm chú thích lý do (một dòng, trỏ về file này).

### Bước 6. Đo FPS cuối cùng và kiểm tra lại trang upload

- Đo lại cho SVM-linear và KNN, với 1, 2, 3 mặt, **có tracking** (cấu hình cuối) và **không tracking** (`RT_EMBED_EVERY_S` rất nhỏ và `RT_MAX_EMBEDS_PER_CYCLE` lớn, để minh hoạ lợi ích của tracking). Điền bảng 7.1.
- `python check_upload_regression.py --compare` phải OK; upload tay một ảnh.

## 5. Tiêu chí hoàn thành

- [ ] Đã thử đủ kịch bản S1 đến S10 cho cả 2 thuật toán demo (hoặc ghi rõ kịch bản nào bỏ và vì sao).
- [ ] Bảng 7.1 (FPS), 7.2 (kết quả theo kịch bản), 7.3 (lỗi đã biết) đã điền.
- [ ] Ngưỡng Unknown realtime đã quyết định (giữ hay đổi) và có số liệu `live_threshold.csv`.
- [ ] Giá trị mặc định cuối đã ghi vào `config.py`.
- [ ] Trang upload không đổi.

## 6. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Không đủ người có trong dữ liệu để thử | Dùng ảnh người trong `external/pins_demo_test/` hiện trên điện thoại (ghi rõ đây là ảnh, không phải mặt thật; chỉ dùng để đo, không dùng để kết luận về giả mạo) hoặc thêm vài người bằng trang "📸 Chụp ảnh" rồi train lại (tăng phạm vi, hỏi người dùng trước) |
| Số mẫu thử ít nên kết luận không chắc | Nêu rõ cỡ mẫu trong báo cáo, không dùng từ khẳng định |
| Ngưỡng mới tốt cho người quen nhưng làm người lạ lọt vào | Đo cả hai; ưu tiên không nhận nhầm người lạ nếu mục đích là điểm danh, và ghi quyết định |
| Chỉnh tham số theo một người thử nên không tổng quát | Thử ít nhất 2 người, ghi điều kiện ánh sáng |

## 7. Kết quả (điền sau khi làm)

### 7.1 FPS cuối cùng

| N mặt | Thuật toán | Tracking | display_fps | worker_fps | detect_ms | embed_ms | Thời gian lên nhãn |
|---|---|---|---|---|---|---|---|
| 1 | SVM-linear | có | | | | | |
| 1 | SVM-linear | không | | | | | |
| 2 | SVM-linear | có | | | | | |
| 3 | SVM-linear | có | | | | | |
| 3 | SVM-linear | không | | | | | |
| 1 | KNN | có | | | | | |
| 3 | KNN | có | | | | | |

### 7.2 Kết quả theo kịch bản

| Kịch bản | Thuật toán | Đúng tên (% thời gian có nhãn) | Unknown (%) | Nhấp nháy/phút | Ghi chú |
|---|---|---|---|---|---|
| S1 | SVM-linear | | | | |
| S1 | KNN | | | | |
| S2 | ... | | | | |

### 7.3 Lỗi đã biết

| Tình huống | Hiện tượng | Nguyên nhân dự đoán | Có khắc phục được không |
|---|---|---|---|
| | | | |

### 7.4 Quyết định

| Mục | Giá trị |
|---|---|
| Ngưỡng Unknown realtime (SVM-linear, KNN) và độ lệch so với `meta` | |
| Tham số mặc định cuối (bảng 5.1 của Phase 2) | |
| Ngày làm, người accept | |

## 8. Điểm dừng

Báo người dùng: bảng 7.1 đến 7.3, các giá trị mặc định đề xuất, hạn chế còn lại. Hỏi: **"Accept Phase 4 để sang Phase 5 (hoàn thiện tài liệu)?"** Cập nhật bảng trạng thái ở [README.md](README.md) mục 7.
