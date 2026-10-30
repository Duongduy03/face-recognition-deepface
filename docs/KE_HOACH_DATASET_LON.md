# Kế hoạch: chạy pipeline trên dataset lớn (Pins + LFW ≥ 2 ảnh)

Trạng thái dataset hiện tại: **1.774 người, 24.424 ảnh**.

| Nguồn | Người | Ảnh | Ghi chú |
|---|---|---|---|
| Kaggle Pins | 105 | 16.484 | toàn bộ trừ 1.050 ảnh test demo (`external/pins_demo_test/`, không lọt vào dataset) |
| LFW (≥ 2 ảnh) | 1.669 | 7.940 | bỏ người chỉ có 1 ảnh; 7 người LFW cũ vẫn chỉ 20–22 ảnh; người trùng tên với Pins giữ ảnh Pins |

`prepare_pins.py --all` bổ sung ảnh Pins còn lại; chạy lại an toàn (không ghi đè, không đụng ảnh test).

## Phần 3: Thời gian và cách chạy

**Ước tính** (CPU, ~1,3 giây/ảnh detect):

- Detect 24.424 ảnh ≈ **9 giờ** (một lần, cache ở `output/faces/`). Hiện chỉ ~3.150 ảnh Pins + 142 ảnh LFW đã có cache, nên còn ~21.000 ảnh mới ≈ 7,5 giờ.
- Embed mỗi model trên ~24.000 mặt: tùy model, từ vài chục phút đến vài giờ. 9 model cộng lại có thể tới nhiều giờ nữa. Cần đo thật bằng 200 ảnh trước khi chốt.

**Rủi ro cần xử lý trước khi chạy dài:**

1. `ensure_face_cache` lưu index mỗi 100 ảnh nên ngắt rồi chạy lại được.
2. `embed_dataset` chỉ ghi `output/embeddings/<model>.npz` ở **cuối** mỗi model. Nếu bị ngắt giữa chừng thì mất hết phần đã embed của model đó. Nên thêm lưu định kỳ trước khi chạy dài.
3. Máy không được sleep khi chạy qua đêm (dùng keep-awake của app hoặc chạy trong Terminal panel).
4. Cache cũ `output/embeddings/*.npz` và `output/model_knn_*.npz` hiện thuộc dataset cũ. Train lại sẽ ghi đè, nên sao lưu nếu muốn giữ kết quả cũ.

**Các bước đề xuất:**

1. Thử nhỏ: `python benchmark.py --models ArcFace --skip-fps` trên một tập con để đo thời gian detect/embed/model thật.
2. Chạy detect toàn bộ một lần (qua đêm), có log ra `output/`.
3. Benchmark từng nhóm model, chạy trước ArcFace, Facenet512, SFace (3 model đang có trong DB). Rồi mới đến 6 model còn lại.
4. `python plot_results.py`, cập nhật `docs/SO_SANH_MO_HINH.md` (số liệu cũ ứng với 7 người LFW nên không so sánh được).
5. `python train_model.py` cho các model tốt nhất.

**Điểm cần lưu ý khi đánh giá:**

- Người chỉ có 2 ảnh: chia 80/20 stratified cho ra 1 train + 1 test, KNN rất khó. Có thể báo cáo thêm kết quả riêng cho nhóm ≥ 10 ảnh.
- Mất cân bằng lớp lớn: Pins ~157 ảnh/người so với LFW 2–530 ảnh/người (7 người đầu cắt ở 20).

## Phần 4: DB Neon dùng chung

DB đang giữ Pins (30 ảnh/người) cho ArcFace, Facenet512, SFace và 7 người LFW cho cả 9 model. Không ghi bộ mới lên đó cho đến khi cả nhóm đồng ý.

1. Thống nhất với nhóm: có nạp 24.000 vector/model lên Neon không (dung lượng gói miễn phí, thời gian ingest).
2. Nếu có: sao lưu DB trước (`pg_dump`), hoặc dùng DB/schema riêng để thử.
3. Chỉ nạp 3 model ArcFace, Facenet512, SFace trước (`train_model.py --model X --db`). Không dùng `--db-prune` khi dataset local khác dataset của người khác.
4. Sau khi nạp, `--db-recalibrate` tự chạy (ngưỡng Unknown tính lại trên toàn bộ vector trong DB).
5. Kiểm tra `eval_impostors.py`: FAR phụ thuộc số người đã quen trong DB nên sẽ đổi mạnh, cần chạy lại và ghi chú trong báo cáo.
6. Tên thư mục LFW là tên thật người nổi tiếng, trùng quy ước `normalize_name`. Hai thành viên cùng chạy `--db` thì `image_key` là `<người>/<file>`: ảnh LFW tên `N.jpg` có thể trùng nhau giữa các máy nếu dataset khác nhau, nên cả nhóm phải dùng cùng một dataset (chia sẻ qua Drive).
