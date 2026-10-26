import os
import sys

# Windows: khi stdout bị pipe/redirect, mã hóa mặc định (cp1252) không in được tiếng Việt và emoji
# trong log của DeepFace -> UnicodeEncodeError. config.py luôn được import trước DeepFace.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_ROOT = os.getcwd()

# =========================
# ĐƯỜNG DẪN
# =========================
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
EMBED_DIR = os.path.join(OUTPUT_DIR, "embeddings")      # cache embedding của benchmark
FACES_CACHE_DIR = os.path.join(OUTPUT_DIR, "faces")     # cache khuôn mặt đã detect
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

IMAGE_EXTS = (".png", ".jpg", ".jpeg")

# =========================
# CẤU HÌNH MÔ HÌNH
# =========================
# Các mô hình embedding của DeepFace, mỗi mô hình ghép với KNN.
# Đây là nơi duy nhất khai báo tên mô hình: app.py, train_model.py, benchmark.py đều đọc từ đây.
MODEL_NAMES = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepID",
    "ArcFace",
    "SFace",
    "GhostFaceNet",
    "Dlib",
]

# DeepFace hỗ trợ thêm các mô hình dưới đây nhưng không chạy được trong môi trường hiện tại
# (đã thử, xem docs/SO_SANH_MO_HINH.md). Không đưa vào MODEL_NAMES để benchmark không báo lỗi.
UNAVAILABLE_MODELS = {
    "DeepFace": "cần LocallyConnected2D, Keras/TensorFlow > 2.12 không còn hỗ trợ (máy dùng TF 2.21)",
    "Buffalo_L": "cần insightface + onnxruntime và tải trọng số từ Google Drive; Drive trả về trang cảnh báo "
                 "thay vì file nên DeepFace không tải được",
}

DETECTOR_BACKEND = "retinaface"

# =========================
# KNN & ĐÁNH GIÁ
# =========================
K = 3
TEST_SIZE = 0.2
SEED = 42


def model_path(model_name):
    """Đường dẫn file model KNN đã train cho một mô hình embedding."""
    return os.path.join(OUTPUT_DIR, f"model_knn_{model_name}.npz")
