
import os
import numpy as np


def l2_normalize(x):
    """Chuẩn hóa L2 theo hàng để embedding của các mô hình khác thang đo vẫn so sánh được."""
    x = np.array(x, dtype="float32")
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


class KNN:
    def __init__(self, K=3):
        self.K = K
        self.X_train = None
        self.y_train = None
        self.threshold = None   # ngưỡng khoảng cách để coi là "Unknown" (None = không dùng)

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train, dtype="float32")
        self.y_train = np.array(y_train)

    def kneighbors(self, x, exclude=None):
        """Khoảng cách Euclid tới toàn bộ tập train, sắp xếp tăng dần -> (dists, idx).

        exclude: chỉ số mẫu train bị bỏ qua (dùng cho leave-one-out khi x chính là mẫu train).
        """
        x = np.array(x, dtype="float32").reshape(1, -1)

        dists = np.linalg.norm(self.X_train - x, axis=1)
        idx = np.argsort(dists)
        if exclude is not None:
            idx = idx[idx != exclude]
        return dists[idx], idx

    def _predict_one(self, x, exclude=None):
        _, idx = self.kneighbors(x, exclude)
        k_labels = self.y_train[idx[: self.K]]

        values, counts = np.unique(k_labels, return_counts=True)
        top = values[counts == counts.max()]
        if len(top) == 1:
            return top[0]

        # Hòa phiếu: chọn nhãn có láng giềng gần nhất (k_labels đã sắp theo khoảng cách tăng dần)
        for label in k_labels:
            if label in top:
                return label

    def predict(self, X_test):
        X_test = np.array(X_test, dtype="float32")
        if X_test.ndim == 1:
            return np.array([self._predict_one(X_test)])

        preds = [self._predict_one(x) for x in X_test]
        return np.array(preds)


def load_knn_from_npz(model_path):

    if not os.path.exists(model_path):
        return None

    data = np.load(model_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    k = int(data["k"][0]) if "k" in data.files else 5

    model = KNN(K=k)
    model.fit(X, y)
    if "threshold" in data.files:
        model.threshold = float(data["threshold"][0])
    return model
