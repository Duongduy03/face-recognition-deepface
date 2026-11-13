import hashlib
import os
import json
import time
import uuid

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from PIL import ImageFont, ImageDraw, Image
from config import (
    ALGO_RESULTS_DIR, BENCH_EMBEDDING, DATASET_DIR, DEMO_ALGORITHMS, RESULTS_DIR, MODEL_NAMES, DETECTOR_BACKEND,
    PIPELINE_FINGERPRINT, model_path, normalize_name,
)
import classifiers
import train_model
import vector_db
from face_pipeline import detect_faces, embed_face, ensure_face_cache
from knn_func import load_knn_from_npz

TARGET_IMAGES_PER_PERSON = 10
SOURCE_NPZ = "File .npz"
SOURCE_PG = "PostgreSQL (pgvector)"
SOURCE_ALGO = f"Thuật toán phân loại ({BENCH_EMBEDDING})"
SATURATED_CONFIDENCE = {"DecisionTree", "GaussianNB", "LDA"}   # độ tin cậy bão hòa: ngưỡng Unknown vô hiệu (xem báo cáo mục 7.1)

# =========================
# CẤU HÌNH GIAO DIỆN
# =========================
st.set_page_config(
    page_title="AI HAUI - Face Recognition with DeepFace",
    layout="wide",
    page_icon="📷",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 30px;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.25rem;
        }
        .sub-title {
            text-align: center;
            font-size: 14px;
            color: #666666;
            margin-bottom: 1.5rem;
        }
        .step-box {
            padding: 1rem 1.2rem;
            border-radius: 0.6rem;
            border: 1px solid #e0e0e0;
            margin-bottom: 1rem;
            background-color: #fafafa;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# LOAD MODEL KNN
# =========================
@st.cache_data(ttl=30)
def pg_models():
    """Các mô hình đã nạp vào PostgreSQL (rỗng nếu DB không chạy)."""
    return vector_db.available_models()


def available_models(source=SOURCE_NPZ):
    """Các mô hình đã được train: có file output/model_knn_<tên>.npz, hoặc đã nạp vào PostgreSQL."""
    if source == SOURCE_PG:
        in_db = pg_models()
        return [m for m in MODEL_NAMES if m in in_db]
    return [m for m in MODEL_NAMES if os.path.exists(model_path(m))]


@st.cache_resource
def load_model(model_name, mtime):
    # mtime nằm trong khóa cache: train lại thì file đổi và model được nạp lại
    return load_knn_from_npz(model_path(model_name))


@st.cache_resource
def pg_connection():
    return vector_db.connect()


def get_pg_conn():
    conn = pg_connection()
    if conn.closed or conn.broken:   # DB khởi động lại: mở kết nối mới
        pg_connection.clear()
        conn = pg_connection()
    return conn


def get_knn(model_name, source=SOURCE_NPZ):
    if source == SOURCE_PG:
        return vector_db.load_pg_knn(model_name, get_pg_conn())
    return load_model(model_name, os.path.getmtime(model_path(model_name)))


def algo_trained(algo):
    """Đã train bằng `python train_model.py [--algo ...]`: có file output/model_<embedding>_<thuật toán>.*"""
    return os.path.exists(classifiers.model_file(algo, BENCH_EMBEDDING))


def algo_models():
    """Các thuật toán đã train, theo thứ tự hiển thị."""
    return [a for a in classifiers.display_order() if algo_trained(a)]


def algo_label(algo):
    """Nhãn trong ô chọn: 2 thuật toán demo có ⭐ (và nằm đầu danh sách, xem classifiers.display_order); chưa train thì ghi rõ."""
    label = f"⭐ {algo} (demo)" if algo in DEMO_ALGORITHMS else algo
    return label if algo_trained(algo) else f"{label} (chưa train)"


# Random Forest (~0,8 GB) và Extra Trees (~1,6 GB) rất nặng: chỉ giữ tối đa 3 mô hình trong RAM, mô hình ít dùng nhất bị bỏ
@st.cache_resource(max_entries=3)
def load_algo(algo, mtime):
    # mtime nằm trong khóa cache: train lại thì file đổi và mô hình được nạp lại. File .joblib là pickle, chỉ nạp file tự train
    return classifiers.load(classifiers.model_file(algo, BENCH_EMBEDDING))


def get_algo(algo):
    return load_algo(algo, os.path.getmtime(classifiers.model_file(algo, BENCH_EMBEDDING)))


# =========================
# HÀM TIỆN ÍCH
# =========================
def _save_image(frame, person_folder_path: str):
    """Lưu ảnh BGR vào dataset (hỗ trợ Unicode path)"""
    if not os.path.exists(person_folder_path):
        os.makedirs(person_folder_path, exist_ok=True)

    try:
        is_success, img_encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not is_success:
            return False, "Lỗi mã hóa ảnh"

        # tên file không dựa vào số ảnh đang có: image_key (<Tên>/<file>) là khóa chung trong DB của nhóm,
        # hai máy cùng đặt "1.jpg" sẽ ghi đè ảnh của nhau
        file_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
        file_path = os.path.join(person_folder_path, file_name)

        with open(file_path, "wb") as f:
            f.write(img_encoded.tobytes())

        return True, file_path
    except Exception as e:
        return False, str(e)


def draw_rounded_rectangle(img, top_left, bottom_right, color, thickness=2, radius=15):
    """
    Vẽ khung bo góc bằng line + ellipse.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    radius = int(min(radius, (x2 - x1) / 2, (y2 - y1) / 2))

    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def predict_name_from_face(face_bgr, model_name, knn_model):
    """Trả về (tên, khoảng cách tới mẫu gần nhất). Vượt ngưỡng của mô hình -> "Unknown"."""
    try:
        embedding = embed_face(face_bgr, model_name)
        pred = knn_model.predict(embedding)[0]

        dists, _ = knn_model.kneighbors(embedding)
        min_distance = float(dists[0])

        # Ngưỡng được hiệu chỉnh riêng cho từng mô hình khi train (train_model.py)
        if knn_model.threshold is not None and min_distance > knn_model.threshold:
            return "Unknown", min_distance

        return str(pred), min_distance

    except Exception as e:
        print(f"Error predicting: {e}")
        return "Unknown", None


def predict_name_with_algo(face_bgr, model, meta):
    """Trả về (tên, độ tin cậy) với một thuật toán trong classifiers.py. Độ tin cậy = điểm cao nhất của classifiers.scores
    (lớn hơn = chắc hơn; KNN là -khoảng cách tới láng giềng gần nhất). Dưới ngưỡng trong meta thì là "Unknown"."""
    try:
        embedding = embed_face(face_bgr, BENCH_EMBEDDING)[None, :]
        pred = str(model.predict(embedding)[0])
        confidence = float(classifiers.scores(model, embedding)[0].max())
        threshold = meta.get("threshold")
        if threshold is not None and confidence < threshold:
            return "Unknown", confidence
        return pred, confidence
    except Exception as e:
        print(f"Error predicting: {e}")
        return "Unknown", None


def _load_font(font_size):
    # arial.ttf có sẵn trên Windows; các hệ khác thử DejaVu rồi font mặc định của PIL
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_vietnamese_text(img_bgr, text, pos, font_size=24, color=(0, 255, 0)):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)

    draw.text(pos, text, font=_load_font(font_size), fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# =========================
# PAGE 1: CHỤP ẢNH
# =========================
def page_chup_anh():
    st.markdown('<div class="main-title">HỆ THỐNG NHẬN DIỆN KHUÔN MẶT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Thu thập dữ liệu với DeepFace RetinaFace Detection</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("📸 Chụp ảnh thu thập dữ liệu")

    person_name = normalize_name(st.text_input(
        "Nhập tên (có thể có dấu, ví dụ: Mạnh, Khang, Nguyễn Văn A):",
        "",
    ))

    if person_name.strip():
        person_folder_path = os.path.join(DATASET_DIR, person_name.strip())
        current_count = 0
        if os.path.exists(person_folder_path):
            current_count = len(
                [
                    f
                    for f in os.listdir(person_folder_path)
                    if os.path.isfile(os.path.join(person_folder_path, f))
                ]
            )
        st.info(f"Hiện có {current_count} ảnh của `{person_name}` trong dataset.")
        st.progress(min(current_count / TARGET_IMAGES_PER_PERSON, 1.0))
    else:
        st.warning("Hãy nhập tên trước khi chụp ảnh.")

    picture = st.camera_input("Chụp ảnh khuôn mặt", key="camera_capture")

    if picture is not None:
        _process_capture(picture.getvalue(), person_name)

    st.markdown("</div>", unsafe_allow_html=True)
    _sync_section()


def _process_capture(bytes_data, person_name):
    if not person_name.strip():
        st.error("Bạn chưa nhập tên/mã định danh.")
        return

    # Streamlit chạy lại cả trang mỗi khi bấm nút (kể cả nút đồng bộ) và camera_input vẫn giữ ảnh cũ:
    # không nhớ ảnh đã lưu thì mỗi lần chạy lại sẽ ghi thêm một bản trùng vào dataset/
    photo_id = hashlib.md5(bytes_data).hexdigest()
    if st.session_state.get("saved_photo_id") == photo_id:
        st.caption("Ảnh này đã được lưu. Bấm 'Clear photo' rồi chụp ảnh mới.")
        return

    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    if cv2_img is None or cv2_img.size == 0:
        st.error("Không đọc được ảnh từ camera.")
        return

    with st.spinner("Đang detect khuôn mặt..."):
        faces = detect_faces(cv2_img)

    if not faces:
        st.error("Không tìm thấy khuôn mặt. Hãy chụp lại gần hơn / sáng hơn.")
        return

    # Lấy face lớn nhất
    best = max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])
    face_bgr, facial_area = best["face_bgr"], best["facial_area"]

    person_folder_path = os.path.join(DATASET_DIR, person_name.strip())
    success, path = _save_image(face_bgr, person_folder_path)

    if success:
        st.session_state["saved_photo_id"] = photo_id
        new_count = len([f for f in os.listdir(person_folder_path)
                       if os.path.isfile(os.path.join(person_folder_path, f))])
        st.success(f"✅ Đã lưu ảnh: {os.path.basename(path)}")
        st.info(f"📊 Tổng số ảnh hiện có: {new_count}/{TARGET_IMAGES_PER_PERSON}")

        st.image(
            cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB),
            caption=f"Khuôn mặt đã lưu ({facial_area['w']}x{facial_area['h']}px)",
            width="stretch",
        )
    else:
        st.error(f"Lỗi khi lưu ảnh: {path}")


def _sync_section():
    """Ảnh chụp chỉ nằm trong dataset/. Bước này giống `train_model.py --db`: embed dataset/, ghi vector
    vào PostgreSQL và tính lại ngưỡng Unknown một lần cho cả lô, không làm sau từng ảnh."""
    st.subheader("☁️ Đồng bộ vector lên PostgreSQL")
    st.caption("Bấm sau khi chụp đủ ảnh. Ảnh mới cần detect + embed nên có thể mất vài phút; "
               "ảnh đã xử lý trước đó được lấy từ cache.")

    models = st.multiselect("Mô hình cần đồng bộ", MODEL_NAMES, default=available_models(SOURCE_NPZ),
                            key="sync_models")
    if not st.button("Đồng bộ lên DB", disabled=not models):
        return

    try:
        with st.status("Đang đồng bộ...", expanded=True) as status:
            conn = get_pg_conn()
            faces_df = ensure_face_cache()
            if faces_df.empty:
                raise ValueError("Không có ảnh hợp lệ trong dataset/")
            for model_name in models:
                st.write(f"{model_name}: embed, ghi vector, tính lại ngưỡng Unknown...")
                train_model.train(model_name, faces_df, conn)
            status.update(label="Đồng bộ xong", state="complete")
    except Exception as e:
        st.error(f"Đồng bộ thất bại: {e}")
    else:
        pg_models.clear()   # để trang Nhận diện thấy ngay các mô hình mới trong DB
        st.success(f"Đã đồng bộ {len(faces_df)} ảnh ({faces_df['person'].nunique()} người) "
                   f"cho {', '.join(models)}.")


# =========================
# PAGE 2: NHẬN DIỆN ẢNH UPLOAD
# =========================
def page_nhan_dien(model_name, source=SOURCE_NPZ):
    st.markdown('<div class="main-title">NHẬN DIỆN KHUÔN MẶT</div>', unsafe_allow_html=True)
    use_algo = source == SOURCE_ALGO
    st.markdown(
        f'<div class="sub-title">DeepFace RetinaFace + '
        f'{BENCH_EMBEDDING + " Embedding + " + str(model_name) if use_algo else str(model_name) + " Embedding + KNN"}</div>',
        unsafe_allow_html=True,
    )

    if model_name is None:
        st.error(
            "⚠️ Chưa có model nào.\n\n"
            "Hãy chạy `python train_model.py` (SVM tuyến tính + KNN trên ArcFace) hoặc "
            "`python train_model.py --model ArcFace` (KNN trên cả dataset/)."
        )
        return
    if use_algo:
        if not algo_trained(model_name):
            st.warning(
                f"Thuật toán **{model_name}** chưa được train. Chạy `python train_model.py --algo {model_name}` "
                "(xong thì tải lại trang) rồi chọn lại."
            )
            return
        with st.spinner(f"Đang nạp mô hình {model_name}..."):
            algo_model, algo_meta = get_algo(model_name)
        knn_model = None
        db_pipeline = algo_meta.get("pipeline_fingerprint")
    else:
        knn_model = get_knn(model_name, source)
        db_pipeline = getattr(knn_model, "pipeline_fingerprint", None)
    if db_pipeline and db_pipeline != PIPELINE_FINGERPRINT:
        st.warning(
            f"Vector trong DB được tạo bằng pipeline `{db_pipeline}`, máy này dùng `{PIPELINE_FINGERPRINT}`. "
            "Kết quả nhận diện có thể sai."
        )

    uploaded_img = st.file_uploader("Chọn ảnh để nhận diện", type=["jpg", "jpeg", "png"])

    if uploaded_img is None:
        return

    bytes_data = uploaded_img.read()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    if cv2_img is None or cv2_img.size == 0:
        st.error("Không đọc được ảnh.")
        return

    # Detect faces
    with st.spinner("Đang phát hiện khuôn mặt..."):
        faces = detect_faces(cv2_img)

    if not faces:
        st.warning("Không phát hiện khuôn mặt nào trong ảnh.")
        st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), width="stretch")
        return

    # Predict cho từng face
    recognized_count = 0
    unknown_count = 0
    details = []

    with st.spinner(f"Đang nhận diện {len(faces)} khuôn mặt..."):
        for face in faces:
            face_bgr, facial_area = face["face_bgr"], face["facial_area"]
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

            if use_algo:
                name, confidence = predict_name_with_algo(face_bgr, algo_model, algo_meta)
                # KNN: hiển thị khoảng cách (= -độ tin cậy) cho giống các nguồn KNN khác; thuật toán khác: điểm tin cậy
                if model_name == "KNN":
                    score_col, score = "Khoảng cách", None if confidence is None else -confidence
                else:
                    score_col, score = "Độ tin cậy", confidence
            else:
                name, score = predict_name_from_face(face_bgr, model_name, knn_model)
                score_col = "Khoảng cách"
            details.append({"Tên": name, score_col: None if score is None else round(score, 3)})

            # Chọn màu
            if name == "Unknown":
                box_color = (0, 0, 255)  # Đỏ
                text_color = (0, 0, 255)
                unknown_count += 1
            else:
                box_color = (0, 255, 0)  # Xanh lá
                text_color = (0, 255, 0)
                recognized_count += 1

            draw_rounded_rectangle(
                cv2_img,
                (x, y),
                (x + w, y + h),
                color=box_color,
                thickness=3,
                radius=20,
            )

            cv2_img = draw_vietnamese_text(
                cv2_img,
                name,
                (x, max(y - 35, 0)),
                font_size=28,
                color=text_color
            )

    # Hiển thị kết quả
    if unknown_count == 0:
        st.success(f"✅ Nhận diện thành công {recognized_count} khuôn mặt!")
    else:
        st.info(f"📊 Kết quả: {recognized_count} nhận diện được, {unknown_count} không xác định")

    st.image(
        cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB),
        caption="Kết quả nhận diện",
        width="stretch",
    )
    if use_algo:
        thr = algo_meta.get("threshold")
        if thr is None:
            thr_text = "không dùng"
        elif model_name == "KNN":
            thr_text = f"khoảng cách > {-thr:.3f} thì là Unknown (Euclid trên embedding đã L2-normalize)"
        else:
            thr_text = f"độ tin cậy < {thr:.3f} thì là Unknown (thang điểm riêng của {model_name}, không so sánh với thuật toán khác)"
        st.caption(
            f"Thuật toán {model_name} trên {BENCH_EMBEDDING}, huấn luyện {algo_meta.get('n_train', '?')} ảnh / "
            f"{algo_meta.get('n_classes', '?')} người ({algo_meta.get('trained_at', '?')}). Ngưỡng Unknown: {thr_text}."
        )
        if model_name in SATURATED_CONFIDENCE:
            st.caption(
                f"⚠️ Xác suất đầu ra của {model_name} gần như luôn bằng 1 (bão hòa) nên ngưỡng Unknown gần như không loại được ai: "
                "người lạ hầu hết vẫn bị gán thành một người quen (báo cáo mục 7.1 và 7.3)."
            )
    else:
        thr = knn_model.threshold
        st.caption(
            f"Mô hình {model_name} ({source}), K={knn_model.K}, ngưỡng Unknown = "
            f"{'không dùng' if thr is None else round(thr, 3)} (khoảng cách Euclid trên embedding đã L2-normalize)."
        )
    st.dataframe(pd.DataFrame(details), hide_index=True)


# =========================
# PAGE 3: SO SÁNH MÔ HÌNH
# =========================
def _results_path(name):
    return os.path.join(RESULTS_DIR, name)


def page_so_sanh_embedding():
    """Phụ lục: so sánh 9 embedding (cùng dùng KNN), số liệu từ benchmark_embeddings.py trên 62 người LFW."""
    st.caption("Phụ lục: so sánh 9 bộ embedding của DeepFace, cả 9 đều dùng KNN. Số liệu đo trên 62 người LFW (benchmark_embeddings.py).")

    if not os.path.exists(_results_path("metrics.csv")):
        st.error("Chưa có kết quả. Hãy chạy `python benchmark_embeddings.py` rồi `python plot_results.py --kind embeddings`.")
        return

    metrics = pd.read_csv(_results_path("metrics.csv"))
    best = []
    if os.path.exists(_results_path("best_models.json")):
        with open(_results_path("best_models.json"), encoding="utf-8") as f:
            best = json.load(f)
    if os.path.exists(_results_path("run_info.json")):
        with open(_results_path("run_info.json"), encoding="utf-8") as f:
            info = json.load(f)
        st.caption(
            f"Dataset: {info['n_people']} người, {info['n_images']} ảnh "
            f"(train {info['n_train']} / test {info['n_test']}), K={info['K']}, detector {info['detector']}. "
            f"CPU: {info['processor']} ({info['cpu_count']} luồng)."
        )

    st.subheader("6 độ đo trên tập test")
    st.dataframe(metrics.round(4), hide_index=True, width="stretch")
    if best:
        st.success(f"2 mô hình tốt nhất theo F1: **{best[0]}**, **{best[1]}**")

    if os.path.exists(_results_path("fps.csv")):
        st.subheader("Tốc độ (FPS)")
        st.dataframe(pd.read_csv(_results_path("fps.csv")).round(2), hide_index=True, width="stretch")
        st.caption("fps_embed_knn: chỉ embed + KNN. fps_end_to_end: gồm cả detect bằng RetinaFace.")

    st.subheader("Biểu đồ")
    for fname, caption in [
        ("f1_ranking.png", "Xếp hạng theo F1"),
        ("metrics_heatmap.png", "6 độ đo của 9 embedding"),
        ("speed_vs_f1.png", "Độ chính xác so với tốc độ"),
        ("time_breakdown.png", "Thời gian từng bước"),
        ("failure_breakdown.png", "Các loại ca lỗi"),
    ]:
        if os.path.exists(_results_path(fname)):
            st.image(_results_path(fname), caption=caption, width="stretch")

    st.subheader("Các trường hợp nhận diện sai")
    st.caption("Mỗi ảnh: bên trái là ảnh test, bên phải là mẫu train gần nhất thuộc lớp bị dự đoán nhầm.")
    models = list(metrics["model"])
    model = st.selectbox("Chọn mô hình", models, index=models.index(best[0]) if best else 0)
    fail_dir = os.path.join(RESULTS_DIR, "failures", model)
    if os.path.isdir(fail_dir):
        files = sorted(os.listdir(fail_dir))[:12]
        cols = st.columns(3)
        for i, fn in enumerate(files):
            cols[i % 3].image(os.path.join(fail_dir, fn), width="stretch")
    else:
        st.info("Chưa có ảnh lỗi cho mô hình này.")


def page_so_sanh_algo():
    """So sánh 10 thuật toán phân loại trên cùng embedding, số liệu và biểu đồ từ benchmark.py + plot_results.py."""
    path = lambda name: os.path.join(ALGO_RESULTS_DIR, name)
    needed = ["metrics.csv", "cv.csv", "timing.csv", "best_algorithms.json"]
    if not all(os.path.exists(path(f)) for f in needed):
        st.error("Chưa có kết quả. Hãy chạy `python benchmark.py` rồi `python plot_results.py`.")
        return

    metrics, cv, timing = (pd.read_csv(path(f)) for f in ("metrics.csv", "cv.csv", "timing.csv"))
    with open(path("best_algorithms.json"), encoding="utf-8") as f:
        best = json.load(f)
    info = {}
    if os.path.exists(path("run_info.json")):
        with open(path("run_info.json"), encoding="utf-8") as f:
            info = json.load(f)
        st.caption(
            f"Embedding {info['embedding']} cố định cho cả 10 thuật toán. Dữ liệu: {info['n_people']} người, {info['n_images']} ảnh "
            f"(train {info['n_train']} / test {info['n_test']}, chia 80/20 phân tầng, seed {info['seed']}). "
            f"CPU: {info['processor']} ({info['cpu_count']} luồng)."
        )

    st.subheader("6 độ đo trên tập test (một lần chia)")
    cols = ["algorithm", "accuracy", "precision_macro", "recall_macro", "f1_macro", "top3_accuracy", "cohen_kappa"]
    st.dataframe(metrics[cols].round(4), hide_index=True, width="stretch")
    msg = f"2 thuật toán đứng đầu theo quy tắc đặt trước: **{best['best'][0]}**, **{best['best'][1]}**."
    if best.get("indistinguishable"):
        msg += " Chênh lệch giữa hai thuật toán này nhỏ hơn độ lệch chuẩn của 5-fold CV, nên chưa thể nói thuật toán nào tốt hơn."
    st.success(msg)

    st.subheader("5-fold cross-validation")
    st.dataframe(cv.round(4), hide_index=True, width="stretch")
    st.caption("Trung bình và độ lệch chuẩn (ddof = 1) trên 5 fold. Khi các thuật toán chênh nhau ít hơn độ lệch chuẩn thì thứ hạng không chắc chắn.")

    st.subheader("Tốc độ và kích thước")
    tcols = ["algorithm", "fit_s", "predict_ms", "scores_ms", "model_mb", "fps_embed_classify", "fps_end_to_end"]
    st.dataframe(timing[[c for c in tcols if c in timing.columns]].round(3), hide_index=True, width="stretch")
    shared = info.get("timing", {}).get("shared")
    st.caption(
        "predict_ms: một vector chỉ lấy nhãn; scores_ms: lấy thêm độ tin cậy (cần để loại Unknown). "
        + (f"Detect {shared['detect_ms']:.0f} ms và embed {shared['embed_ms']:.0f} ms mỗi ảnh là chi phí chung cho cả 10 thuật toán, "
           "nên FPS end-to-end gần như bằng nhau." if shared else "")
    )

    st.subheader("Biểu đồ")
    charts = [("f1_ranking.png", "Xếp hạng F1"), ("metrics_heatmap.png", "6 độ đo"),
              ("speed_vs_f1.png", "Độ chính xác so với tốc độ"), ("time_and_size.png", "Chi phí"),
              ("failure_breakdown.png", "Các loại ca lỗi"), ("overlap_hist.png", "Một mẫu bị sai bởi bao nhiêu thuật toán")]
    charts += [(f"confusion_{a}.png", f"Ma trận nhầm lẫn: {a}") for a in best["best"]]
    for fname, caption in charts:
        if os.path.exists(path(fname)):
            st.image(path(fname), caption=caption, width="stretch")

    st.subheader("Các trường hợp nhận diện sai")
    st.caption("Mỗi ảnh: bên trái là ảnh test, bên phải là mẫu train gần nhất thuộc người bị dự đoán nhầm. "
               "'_all_wrong' là các mẫu mà cả 10 thuật toán cùng sai.")
    options = list(metrics["algorithm"]) + (["_all_wrong"] if os.path.isdir(path(os.path.join("failures", "_all_wrong"))) else [])
    choice = st.selectbox("Chọn thuật toán", options, index=options.index(best["best"][0]) if best["best"][0] in options else 0)
    fail_dir = path(os.path.join("failures", choice))
    if os.path.isdir(fail_dir):
        cols_ui = st.columns(3)
        for i, fn in enumerate(sorted(os.listdir(fail_dir))[:12]):
            cols_ui[i % 3].image(os.path.join(fail_dir, fn), width="stretch")
    else:
        st.info("Chưa có ảnh lỗi cho lựa chọn này.")


def page_so_sanh():
    st.markdown('<div class="main-title">SO SÁNH CÁC MÔ HÌNH</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">So sánh 10 thuật toán phân loại (benchmark.py) và phụ lục so sánh 9 embedding</div>',
        unsafe_allow_html=True,
    )
    tab_algo, tab_embed = st.tabs(["10 thuật toán phân loại", "Phụ lục: 9 embedding"])
    with tab_algo:
        page_so_sanh_algo()
    with tab_embed:
        page_so_sanh_embedding()


# =========================
# MAIN
# =========================
def main():
    st.sidebar.title("🎯 Menu")
    choice = st.sidebar.radio("Chọn chức năng", ["📸 Chụp ảnh", "🔍 Nhận diện", "📊 So sánh mô hình"])

    model_name = None
    source = SOURCE_NPZ
    if choice == "🔍 Nhận diện":
        sources = ([SOURCE_ALGO] if algo_models() else []) + [SOURCE_NPZ] + ([SOURCE_PG] if pg_models() else [])
        if len(sources) > 1:   # SOURCE_PG chỉ hiện khi PostgreSQL đang chạy và đã có dữ liệu
            source = st.sidebar.radio("Nguồn mô hình", sources)
        if source == SOURCE_ALGO:
            # đủ 10 thuật toán, 2 thuật toán demo ở đầu; thuật toán chưa train vẫn hiện nhưng có nhãn "(chưa train)"
            model_name = st.sidebar.selectbox("Thuật toán", classifiers.display_order(), format_func=algo_label)
            st.sidebar.info(f"**Embedding:** {BENCH_EMBEDDING}\n\n**Thuật toán:** {model_name}\n\n**Detector:** {DETECTOR_BACKEND}")
        else:
            models = available_models(source)
            if models:
                model_name = st.sidebar.selectbox("Mô hình embedding", models)
                st.sidebar.info(f"**Model:** {model_name}\n\n**Detector:** {DETECTOR_BACKEND}")

    if choice == "📸 Chụp ảnh":
        page_chup_anh()
    elif choice == "🔍 Nhận diện":
        page_nhan_dien(model_name, source)
    else:
        page_so_sanh()


if __name__ == "__main__":
    main()
