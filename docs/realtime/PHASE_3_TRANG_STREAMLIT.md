# Phase 3: Trang Streamlit "🎥 Nhận diện realtime"

> **Trạng thái: chưa làm.** Thời lượng: **1 ngày**.
> Cần xong: [Phase 2](PHASE_2_LOI_REALTIME.md) (`recognition.py`, `realtime.py` chạy đúng offline). Phase sau: [PHASE_4_TINH_CHINH.md](PHASE_4_TINH_CHINH.md). Tổng quan: [README.md](README.md).

## 1. Mục tiêu

Thêm một trang mới vào app: bật webcam của trình duyệt, nhận diện nhiều mặt theo thời gian thực bằng `RealtimeRecognizer` của Phase 2, hiện khung và nhãn tiếng Việt, hiện FPS và số mặt, có thanh chỉnh tham số.

**Không thay đổi trang upload và trang chụp ảnh.** Trang mới là mục menu thứ tư. Phần chọn thuật toán của trang upload (`app.py:656-668`) **không bị sửa**; trang realtime có đoạn chọn thuật toán riêng, ngắn, để không đụng vào nhánh đang chạy.

## 2. Thời gian

| Buổi | Việc | Ước tính |
|---|---|---|
| Sáng | Bước 1 đến 4: menu, chọn thuật toán, `webrtc_streamer`, bộ xử lý video, vòng đời recognizer | 4 giờ |
| Chiều | Bước 5 đến 7: thanh trượt, số liệu, xử lý lỗi, kiểm tra không lỗi (Claude), đưa checklist cho người dùng thử webcam thật, hỏi accept | 3 giờ (chưa tính thời gian người dùng thử) |

## 3. Điều kiện đầu vào

- Phase 2 accept; `realtime_demo.py` chạy đúng.
- `streamlit-webrtc` và `av` đã cài (Phase 0).
- Người dùng có webcam và trình duyệt cấp được quyền camera (đã kiểm tra ở Phase 0 bước 8).

## 4. Thiết kế

### 4.1 Luồng dữ liệu

```
webrtc_streamer(video_processor_factory=RealtimeProcessor, ...)
   RealtimeProcessor.recv(frame):                 # chạy ở luồng của thư viện, mỗi frame
       img = frame.to_ndarray(format="bgr24")     # BGR: đúng quy ước DeepFace, KHÔNG đổi sang RGB
       self.recognizer.submit(img)                # không chặn
       draw_results(img, self.recognizer.latest().results)
       return av.VideoFrame.from_ndarray(img, format="bgr24")
```

Chọn kiểu **class `VideoProcessorBase`** (không dùng `video_frame_callback`) vì cần đẩy tham số vào recognizer khi người dùng kéo thanh trượt: `ctx.video_processor.recognizer.update_params(...)`.

### 4.2 Vòng đời recognizer

- Model và predictor được nạp **ở luồng script Streamlit**, trước khi mở webcam: dùng `get_algo(model_name)` (đã có, `app.py:131`, có `st.cache_resource(max_entries=3)`), rồi `recognition.make_algo_predictor(model, meta)`.
- `RealtimeProcessor` **không** gọi `st.*` và không gọi hàm có `st.cache_*` trong `recv` hay trong worker.
- Tạo recognizer khi nhận được `ctx.video_processor` đầu tiên (hoặc trong `__init__` của processor qua tham số); gọi `recognizer.start()` khi stream bắt đầu và `recognizer.stop()` khi stream dừng (`on_ended` của processor, hoặc `ctx.state.playing == False`). Không để thread chạy mồ côi khi người dùng chuyển trang, đổi thuật toán hay đóng tab.
- Đổi thuật toán trong sidebar khi đang chạy: dừng stream cũ (key của `webrtc_streamer` đổi theo tên thuật toán, ví dụ `key=f"rt-{algo}"`), tạo stream mới, để mỗi stream có đúng một recognizer.

### 4.3 Cấu hình WebRTC

```python
webrtc_streamer(
    key=f"rt-{model_name}",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
    video_processor_factory=...,
    async_processing=True,
)
```

- Độ phân giải đề nghị 640x480 để khớp `RT_DETECT_WIDTH`. Người dùng không cần chọn.
- STUN công khai chỉ để thương lượng kết nối; hình ảnh vẫn chạy giữa trình duyệt và app. Trên cùng máy (`localhost`) thường không cần STUN; vẫn giữ để chạy qua mạng nội bộ. Ghi chú trong docs: HTTPS bắt buộc nếu mở app từ máy khác (README mục 8).

### 4.4 Giao diện

| Thành phần | Chi tiết |
|---|---|
| Menu | Thêm "🎥 Nhận diện realtime" vào `st.sidebar.radio` ở `app.py:652` (chỉ thêm một mục vào danh sách) |
| Sidebar | Chỉ nguồn "Thuật toán phân loại (ArcFace)": selectbox thuật toán dùng `classifiers.display_order()` và `algo_label` (đã có); hộp thông tin giống `app.py:663` nhưng ghi **Detector realtime** = `REALTIME_DETECTOR` |
| Nếu chưa train thuật toán | Cảnh báo kèm lệnh `python train_model.py --algo <tên>`, giống `app.py:379-383`, không mở webcam |
| Cảnh báo bão hoà | Với DecisionTree, GaussianNB, LDA (`SATURATED_CONFIDENCE`, `app.py:27`) hiện cảnh báo giống trang upload |
| Cảnh báo detector | Nếu `meta["pipeline_fingerprint"]` ghi `detector=retinaface` còn realtime dùng detector khác: hiện **thông tin** (không phải lỗi): "Model được train trên crop RetinaFace, realtime dùng <detector>; đã đo chênh X điểm ở Phase 1" (X lấy từ Phase 1 mục 7). Không dùng cảnh báo fingerprint của trang upload (`app.py:391-395`) vì nó sẽ báo sai trong trường hợp này |
| Vùng video | Một cột rộng; dưới video: số mặt, FPS hiển thị, FPS worker, thời gian detect/embed |
| Thanh chỉnh (expander "Cài đặt nâng cao") | `RT_DETECT_WIDTH`, `RT_MIN_FACE`, `RT_EMBED_EVERY_S`, `RT_MAX_EMBEDS_PER_CYCLE`, `RT_VOTE_LEN`; thay đổi gọi `update_params` ngay, không khởi động lại stream |
| Hướng dẫn ngắn | Một dòng: "Cho phép trình duyệt dùng camera. Nhãn xuất hiện sau khoảng 1 giây" |

### 4.5 Hiển thị số liệu

Dưới video, cập nhật định kỳ khi stream đang chạy (mẫu của `streamlit-webrtc`):

```python
placeholder = st.empty()
while ctx.state.playing:
    snap = ctx.video_processor.recognizer.latest()
    placeholder.metric_row(...)      # hoặc st.columns + st.metric
    time.sleep(0.5)
```

Vòng lặp này giữ script Streamlit đang chạy. Khi người dùng đổi một widget, Streamlit chạy lại script (ngắt vòng lặp cũ), nên thanh trượt vẫn hoạt động. Dùng `time.sleep(0.5)` để không đốt CPU.

## 5. Các bước

### Bước 1. Thêm mục menu và hàm trang

- Thêm `"🎥 Nhận diện realtime"` vào danh sách ở `app.py:652`.
- Thêm nhánh `elif choice == "🎥 Nhận diện realtime": page_realtime(model_name)` trong `main()` (`app.py:670-675`), **giữ nguyên** các nhánh cũ.
- Trong `main()` thêm một khối riêng (không sửa khối `if choice == "🔍 Nhận diện"` ở `app.py:656-668`) cho `choice == "🎥 Nhận diện realtime"`: selectbox thuật toán và hộp thông tin.

### Bước 2. Viết `RealtimeProcessor`

Đặt trong `realtime.py` hoặc một file riêng `realtime_page.py` (đề xuất: `realtime.py` giữ phần thuần không import `streamlit_webrtc`; processor dùng `streamlit_webrtc` đặt trong `app.py` hoặc `realtime_page.py`, để `realtime_demo.py` vẫn chạy mà không cần cài `streamlit-webrtc`).

### Bước 3. Viết `page_realtime(model_name)`

Theo mục 4: kiểm tra `algo_trained`, nạp model (có spinner), tạo predictor, dựng `webrtc_streamer`, hiện số liệu và thanh chỉnh.

### Bước 4. Vòng đời

Dừng recognizer khi stream dừng; thử: bấm Stop, chuyển sang trang khác, đổi thuật toán, đóng tab. Sau mỗi thao tác kiểm tra số thread (ghi `threading.active_count()` ở sidebar khi bật chế độ gỡ lỗi) không tăng dần.

### Bước 5. Xử lý lỗi

| Tình huống | Hành vi |
|---|---|
| Chưa có model | Thông báo lệnh train, không mở webcam |
| Người dùng từ chối quyền camera | `streamlit-webrtc` tự hiện trạng thái; thêm một dòng hướng dẫn cấp quyền |
| Worker lỗi liên tiếp | `stats["error"]` hiện thành `st.error` dưới video; video vẫn chạy, không có khung |
| Không có mặt trong khung | Không vẽ gì; số mặt = 0 |
| Model nặng (RandomForest 0,8 GB, ExtraTrees 1,6 GB) | Spinner khi nạp; ghi chú cảnh báo chậm khi chọn hai thuật toán này |

### Bước 6. Kiểm tra bởi Claude (không cần webcam)

- Cú pháp và import: `python -c "import app"` không lỗi.
- Khởi động `streamlit run app.py`, mở trang mới: không exception trong log, thấy đủ thành phần (selectbox, cảnh báo, nút START của webrtc). Dùng `preview_start`, `preview_snapshot`, `preview_console_logs`, `preview_logs`.
- Chuyển qua lại cả 4 trang: không lỗi.
- **Trang upload:** chạy `python check_upload_regression.py --compare` (phải OK) và upload tay 1 ảnh.

Claude **không** thử được webcam thật (môi trường Linux không có camera; nếu dùng trình duyệt xem trước thì cũng không có camera). Bước 7 do người dùng làm.

### Bước 7. Checklist cho người dùng thử webcam thật

1. Chạy `streamlit run app.py`, mở `http://localhost:8501`, chọn "🎥 Nhận diện realtime", chọn SVM-linear.
2. Bấm START, cho phép camera. Hình webcam hiện, mượt (không giật theo thời gian nhận diện).
3. Đưa mặt vào: trong khoảng 1 đến 2 giây có khung và nhãn. Người có trong dữ liệu ra đúng tên; người lạ ra Unknown.
4. Hai người cùng vào khung hình: mỗi người một khung, nhãn riêng.
5. Một người đi ra rồi vào lại: khung biến mất rồi hiện lại, nhãn đúng.
6. Đổi sang KNN: stream đổi mà không lỗi.
7. Kéo thanh "Nhận diện mỗi ... giây": FPS worker và độ trễ nhãn đổi theo.
8. Bấm STOP, chuyển sang "🔍 Nhận diện", upload một ảnh: vẫn nhận diện bình thường như trước.
9. Ghi lại: FPS hiển thị, FPS worker, thời gian lên nhãn, và mọi lỗi (chụp màn hình).

## 6. Tiêu chí hoàn thành

- [ ] Có mục "🎥 Nhận diện realtime"; 3 trang cũ không đổi hành vi.
- [ ] `check_upload_regression.py --compare` OK; upload tay vẫn đúng.
- [ ] Trang dựng không lỗi (log sạch); xử lý đủ các tình huống lỗi ở bước 5.
- [ ] Không rò thread sau Stop, đổi trang, đổi thuật toán.
- [ ] Người dùng đã thử checklist bước 7 trên webcam thật và gửi kết quả.

## 7. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| `streamlit-webrtc` không tương thích streamlit 1.66 hoặc numpy 2,5 | Phát hiện ở Phase 0; nếu chỉ lộ ra ở đây thì ghi và đề xuất bản `streamlit-webrtc` khác hoặc ghim phiên bản |
| Video giật vì `recv` làm việc nặng | `recv` chỉ vẽ và gọi `submit`; mọi việc nặng ở worker. Đo `display_fps` |
| Nhiều tab trình duyệt cùng mở realtime làm tải gấp đôi | Mỗi stream một recognizer; ghi chú giới hạn vào docs (khuyến nghị một tab) |
| TF trong thread nền xung đột với TF trong luồng script (khi trang upload và realtime cùng dùng) | Hai trang không chạy đồng thời trong một phiên; nếu gặp lỗi thì ghi lại và xét chạy worker bằng process riêng |
| Không mở được webcam từ máy khác (HTTP) | Ghi trong docs; dùng `localhost` hoặc HTTPS |

## 8. Kết quả (điền sau khi làm)

| Mục | Giá trị |
|---|---|
| Ngày làm | |
| Phiên bản `streamlit-webrtc` | |
| Log khởi động sạch? | |
| `check_upload_regression.py --compare` | |
| Trình duyệt người dùng thử | |
| FPS hiển thị / FPS worker trên webcam thật | |
| Thời gian lên nhãn (cảm nhận hoặc đo) | |
| Lỗi gặp khi thử (và đã xử lý chưa) | |

Kết quả checklist bước 7:

| Mục | Đạt? | Ghi chú |
|---|---|---|
| 2. Hình mượt | | |
| 3. Một người | | |
| 4. Hai người | | |
| 5. Ra vào khung | | |
| 6. Đổi thuật toán | | |
| 7. Thanh chỉnh | | |
| 8. Upload vẫn đúng | | |

## 9. Điểm dừng

Báo người dùng: kết quả kiểm tra của Claude và đưa checklist bước 7. Sau khi người dùng thử và gửi kết quả, hỏi: **"Accept Phase 3 để sang Phase 4 (tinh chỉnh và đo trên webcam thật)?"** Cập nhật bảng trạng thái ở [README.md](README.md) mục 7.
