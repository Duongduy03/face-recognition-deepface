"""10 thuật toán phân loại dùng chung một giao diện, để benchmark.py / train_model.py / app.py xử lý như nhau.

Cả 10 chạy trên CÙNG một ma trận embedding (config.BENCH_EMBEDDING, đã L2-normalize), nên chênh lệch giữa chúng
là do bộ phân loại. Xem docs/SO_SANH_THUAT_TOAN.md mục 3.

Giao diện (giống scikit-learn):
    model = build("SVM-RBF")
    model.fit(X_train, y_train)
    model.predict(X)         # nhãn dự đoán (tên người)
    scores(model, X)         # ma trận (n_mẫu x n_người), cột theo model.classes_; LỚN HƠN = CHẮC HƠN

`scores` dùng cho Top-3 và độ tin cậy (ngưỡng Unknown):
    KNN tự cài  : -(khoảng cách nhỏ nhất tới từng người). Top-3 = 3 người KHÁC NHAU gần nhất.
    SVM         : decision_function (không bật probability=True vì huấn luyện chậm hơn nhiều lần).
    còn lại     : predict_proba.
Ba loại này có thang đo khác nhau, nên độ tin cậy chỉ so sánh được trong cùng một thuật toán.

Lưu ý: file .joblib là pickle, chỉ nạp file do chính mình train.
"""
import os
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import CLASSIFIER_NAMES, DEMO_ALGORITHMS, K, OUTPUT_DIR, SEED
from knn_func import KNN, load_knn_from_npz


class KNNClassifier:
    """KNN tự cài (knn_func.KNN) bọc lại cho có giao diện fit / predict / scores / classes_ như scikit-learn."""

    name = "KNN"

    def __init__(self, k=K):
        self.k = k
        self.knn = KNN(K=k)
        self.classes_ = None

    def fit(self, X, y):
        self.knn.fit(X, y)
        self.classes_ = np.unique(self.knn.y_train)
        class_idx = np.searchsorted(self.classes_, self.knn.y_train)
        # sắp mẫu train theo lớp một lần để lấy "khoảng cách nhỏ nhất tới từng lớp" bằng reduceat
        self._order = np.argsort(class_idx, kind="stable")
        self._starts = np.searchsorted(class_idx[self._order], np.arange(len(self.classes_)))
        return self

    def predict(self, X):
        return self.knn.predict(X)    # K láng giềng gần nhất, bỏ phiếu đa số, hòa phiếu chọn láng giềng gần nhất

    def scores(self, X):
        X = np.asarray(X, dtype="float32")
        if X.ndim == 1:
            X = X[None, :]
        out = np.empty((len(X), len(self.classes_)), dtype="float64")
        for i, x in enumerate(X):
            dists = np.linalg.norm(self.knn.X_train - x, axis=1)
            out[i] = -np.minimum.reduceat(dists[self._order], self._starts)
        return out


# Siêu tham số cố định trước khi chạy, không chỉnh theo kết quả test (bảng ở docs/SO_SANH_THUAT_TOAN.md mục 3).
# n_jobs=1 cho mọi thuật toán: thời gian đo được lặp lại được và giống cách ứng dụng chạy (một ảnh mỗi lần).
_BUILDERS = {
    "KNN":                lambda: KNNClassifier(K),
    "SVM-linear":         lambda: SVC(kernel="linear", C=1),
    "SVM-RBF":            lambda: SVC(kernel="rbf", C=10, gamma="scale"),
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
    "RandomForest":       lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
    "ExtraTrees":         lambda: ExtraTreesClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
    "DecisionTree":       lambda: DecisionTreeClassifier(random_state=SEED),
    "GaussianNB":         lambda: GaussianNB(),
    "LDA":                lambda: LinearDiscriminantAnalysis(solver="svd"),
    "MLP":                lambda: MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=SEED),
}
assert list(_BUILDERS) == CLASSIFIER_NAMES, "classifiers._BUILDERS phải khớp config.CLASSIFIER_NAMES (cả thứ tự)"


def build(name):
    """Thuật toán mới (chưa fit) theo tên trong config.CLASSIFIER_NAMES."""
    if name not in _BUILDERS:
        raise ValueError(f"Thuật toán không có: {name!r}. Chọn một trong {CLASSIFIER_NAMES}")
    return _BUILDERS[name]()


def display_order():
    """Cả 10 thuật toán để hiển thị (app.py): các thuật toán demo (config.DEMO_ALGORITHMS) ở đầu, rồi các thuật toán còn lại
    theo thứ tự config.CLASSIFIER_NAMES."""
    return list(DEMO_ALGORITHMS) + [a for a in CLASSIFIER_NAMES if a not in DEMO_ALGORITHMS]


def build_reference_knn():
    """KNN của scikit-learn (K giống nhau, Euclid), chỉ để đối chiếu với KNN tự cài. Không nằm trong 10 thuật toán.

    Khác biệt duy nhất đáng kể: khi hòa phiếu (K = 3 mà 3 láng giềng thuộc 3 người khác nhau) scikit-learn chọn
    nhãn đứng đầu theo thứ tự chữ cái, còn KNN tự cài chọn nhãn của láng giềng gần nhất.
    """
    return KNeighborsClassifier(n_neighbors=K, metric="euclidean", algorithm="brute")


def scores(model, X):
    """Ma trận điểm (n_mẫu x n_người), cột theo model.classes_, lớn hơn = chắc hơn. Xem docstring đầu file."""
    if isinstance(model, KNNClassifier):
        s = model.scores(X)
    elif isinstance(model, SVC):
        s = model.decision_function(X)
    else:
        s = model.predict_proba(X)
    s = np.asarray(s, dtype="float64")
    if s.ndim != 2 or s.shape[1] != len(model.classes_):
        raise ValueError(f"scores có hình dạng {s.shape}, cần (n_mẫu, {len(model.classes_)})")
    return s


# =========================
# LƯU / NẠP
# =========================
_NPZ_TEXT_META = ("algo", "embedding", "pipeline_fingerprint", "trained_at")    # meta dạng chữ ghi kèm trong .npz của KNN

def model_file(algo, embedding):
    """output/model_<embedding>_<thuật toán>.joblib; riêng KNN là .npz (cùng định dạng với model_knn_<embedding>.npz).

    KNN ở đây KHÔNG dùng config.model_path: file model_knn_<embedding>.npz là KNN do `train_model.py --model` train
    trên cả dataset/ (app.py dùng cho nguồn "File .npz"), ghi đè nó sẽ đổi tập người mà app nhận ra.
    """
    ext = "npz" if algo == "KNN" else "joblib"
    return os.path.join(OUTPUT_DIR, f"model_{embedding}_{algo}.{ext}")


def save(model, path, meta=None):
    """Lưu mô hình. meta (dict): algo, embedding, threshold, pipeline_fingerprint, ...

    meta["threshold"] tính theo ĐỘ TIN CẬY (lớn hơn = chắc hơn; dưới ngưỡng thì là Unknown). Riêng file .npz của KNN
    vẫn lưu theo khoảng cách như định dạng cũ (ngưỡng khoảng cách = -ngưỡng độ tin cậy); load() đổi lại.
    Trả về path đã ghi (np.savez tự thêm đuôi .npz nếu thiếu).
    """
    meta = dict(meta or {})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if isinstance(model, KNNClassifier):
        thr = meta.get("threshold")
        extra = {} if thr is None else {"threshold": np.array([-float(thr)])}
        for key in _NPZ_TEXT_META:    # knn_func.load_knn_from_npz bỏ qua các khóa thừa, nên file vẫn đọc được bằng hàm cũ
            if meta.get(key) is not None:
                extra[key] = np.array([str(meta[key])])
        np.savez(path, X=model.knn.X_train, y=model.knn.y_train, k=np.array([model.k]), **extra)
        return path if path.endswith(".npz") else path + ".npz"
    meta.setdefault("algo", type(model).__name__)
    meta["sklearn"] = sklearn.__version__
    joblib.dump({"model": model, "meta": meta}, path)
    return path


def load(path):
    """Nạp (model, meta) từ file do save() ghi. File .npz (KNN) cũng đọc được, kể cả file do train_model.py cũ tạo."""
    if path.endswith(".npz"):
        knn = load_knn_from_npz(path)
        if knn is None:
            raise FileNotFoundError(path)
        model = KNNClassifier(knn.K).fit(knn.X_train, knn.y_train)
        thr = None if knn.threshold is None else -float(knn.threshold)
        meta = {"algo": "KNN", "K": knn.K, "threshold": thr,
                "n_train": int(len(model.knn.y_train)), "n_classes": int(len(model.classes_))}
        with np.load(path, allow_pickle=True) as data:
            for key in _NPZ_TEXT_META:
                if key in data.files:
                    meta[key] = str(data[key][0])
        return model, meta
    data = joblib.load(path)
    meta = data["meta"]
    if meta.get("sklearn") and meta["sklearn"] != sklearn.__version__:
        warnings.warn(f"{os.path.basename(path)} được lưu bằng scikit-learn {meta['sklearn']}, "
                      f"máy này dùng {sklearn.__version__}; kết quả có thể khác, nên train lại.")
    return data["model"], meta
