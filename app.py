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
    DATASET_DIR, RESULTS_DIR, MODEL_NAMES, DETECTOR_BACKEND, PIPELINE_FINGERPRINT, model_path, normalize_name,
)
import train_model
import vector_db
from face_pipeline import detect_faces, embed_face, ensure_face_cache
from knn_func import load_knn_from_npz

TARGET_IMAGES_PER_PERSON = 10
SOURCE_NPZ = "File .npz"
SOURCE_PG = "PostgreSQL (pgvector)"

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
    st.markdown(
        f'<div class="sub-title">DeepFace RetinaFace + {model_name} Embedding + KNN</div>',
        unsafe_allow_html=True,
    )

    if model_name is None:
        st.error(
            "⚠️ Chưa có model KNN nào.\n\n"
            "Hãy chạy `python train_model.py --model ArcFace` (hoặc `python train_model.py` "
            "sau khi chạy `python benchmark.py`)."
        )
        return
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

            name, distance = predict_name_from_face(face_bgr, model_name, knn_model)
            details.append({"Tên": name, "Khoảng cách": None if distance is None else round(distance, 3)})

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


def page_so_sanh():
    st.markdown('<div class="main-title">SO SÁNH CÁC MÔ HÌNH</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Embedding (DeepFace) + KNN — số liệu từ benchmark.py</div>',
        unsafe_allow_html=True,
    )

    if not os.path.exists(_results_path("metrics.csv")):
        st.error("Chưa có kết quả. Hãy chạy `python benchmark.py` rồi `python plot_results.py`.")
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
        ("metrics_heatmap.png", "6 độ đo của 10 mô hình"),
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


# =========================
# MAIN
# =========================
def main():
    st.sidebar.title("🎯 Menu")
    choice = st.sidebar.radio("Chọn chức năng", ["📸 Chụp ảnh", "🔍 Nhận diện", "📊 So sánh mô hình"])

    model_name = None
    source = SOURCE_NPZ
    if choice == "🔍 Nhận diện":
        if pg_models():   # chỉ hiện lựa chọn khi PostgreSQL đang chạy và đã có dữ liệu
            source = st.sidebar.radio("Nguồn vector", [SOURCE_NPZ, SOURCE_PG])
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
