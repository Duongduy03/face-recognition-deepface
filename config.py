import os
import sys
import unicodedata
from importlib import metadata

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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")                # so sánh 9 embedding (phụ lục)
ALGO_RESULTS_DIR = os.path.join(RESULTS_DIR, "algorithms")         # so sánh 10 thuật toán (báo cáo chính)

# Ảnh "người lạ" không có tên (Kaggle face-detection-dataset, git bỏ qua) dùng để đo tỉ lệ nhận nhầm.
# Để ngoài dataset/: ở đó mỗi thư mục con là một người và sẽ bị đưa vào train.
IMPOSTOR_SRC_DIR = os.path.join(PROJECT_ROOT, "external", "face-detection")
IMPOSTOR_CACHE_DIR = os.path.join(OUTPUT_DIR, "impostor_faces")   # khuôn mặt đã detect + index.csv

IMAGE_EXTS = (".png", ".jpg", ".jpeg")

def _load_dotenv(path):
    """Đọc các dòng KEY=VALUE trong file .env (nếu có) vào os.environ, không ghi đè biến đã đặt sẵn."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Kho embedding trên PostgreSQL + pgvector. Mặc định là container cục bộ (docker-compose.yml);
# cả nhóm dùng chung một DB cloud thì đặt FACE_DB_URL trong file .env (file này không đưa lên git).
DB_URL = os.environ.get("FACE_DB_URL") or "postgresql://face:face@localhost:5432/face"

# =========================
# CẤU HÌNH MÔ HÌNH
# =========================
# Các mô hình embedding của DeepFace, mỗi mô hình ghép với KNN.
# Đây là nơi duy nhất khai báo tên mô hình: app.py, train_model.py, benchmark_embeddings.py đều đọc từ đây.
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
# SO SÁNH 10 THUẬT TOÁN PHÂN LOẠI (xem docs/SO_SANH_THUAT_TOAN.md)
# =========================
# Cả 10 thuật toán chạy trên cùng một embedding, để chênh lệch chỉ đến từ bộ phân loại.
BENCH_EMBEDDING = "ArcFace"
# Tập con để so sánh: những người có ít nhất ngần này ảnh trong cache khuôn mặt. Danh sách người được
# ghi cố định vào results/algorithms/subset_people.txt (cache lớn dần nên số người ≥ ngưỡng sẽ đổi theo thời gian).
BENCH_MIN_IMAGES = 30
# Thứ tự cố định; classifiers.build(name) dựng từng thuật toán. KNN là bản tự cài (knn_func.py), 9 cái còn lại là scikit-learn.
CLASSIFIER_NAMES = [
    "KNN",
    "SVM-linear",
    "SVM-RBF",
    "LogisticRegression",
    "RandomForest",
    "ExtraTrees",
    "DecisionTree",
    "GaussianNB",
    "LDA",
    "MLP",
]
# 2 thuật toán demo trong app (do người làm đề tài chọn 2026-11-05, xem docs). Không trùng hẳn với
# results/algorithms/best_algorithms.json: quy tắc đặt trước chọn SVM-linear + SVM-RBF, nhưng 6 thuật toán đầu
# không phân biệt được theo CV, nên thay SVM-RBF bằng KNN (thuật toán của đề tài) để so sánh trực tiếp trong app.
# `python train_model.py` không đối số train đúng các thuật toán này.
DEMO_ALGORITHMS = ["SVM-linear", "KNN"]


def _deepface_version():
    # đọc metadata của gói, không import deepface (config.py phải được import trước deepface)
    try:
        return metadata.version("deepface")
    except metadata.PackageNotFoundError:
        return "unknown"


# Dấu vết của pipeline tạo embedding. Vector chỉ so sánh được khi cùng detector, cùng thứ tự màu BGR, cùng L2
# và cùng phiên bản DeepFace; vector_db.py lưu chuỗi này và từ chối ghi chung DB khi nó khác.
PIPELINE_FINGERPRINT = f"detector={DETECTOR_BACKEND};align=1;color=bgr;norm=l2;deepface={_deepface_version()}"


def normalize_name(name):
    """Tên người: Unicode NFC và bỏ khoảng trắng thừa, để "Nguyễn" gõ bằng ký tự dựng sẵn hay tổ hợp dấu
    đều ra cùng một người. Không đổi chữ hoa/thường."""
    return " ".join(unicodedata.normalize("NFC", name).split())

# =========================
# KNN & ĐÁNH GIÁ
# =========================
K = 3
TEST_SIZE = 0.2
SEED = 42


def model_path(model_name):
    """Đường dẫn file model KNN đã train cho một mô hình embedding."""
    return os.path.join(OUTPUT_DIR, f"model_knn_{model_name}.npz")
