"""Train KNN cho một hoặc nhiều mô hình embedding và lưu ra output/model_knn_<mô hình>.npz.

    python train_model.py --model ArcFace
    python train_model.py --model ArcFace Facenet512
    python train_model.py                       # mô hình trong results/best_models.json (nếu đã chạy benchmark)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from config import MODEL_NAMES, OUTPUT_DIR, RESULTS_DIR, K, model_path
from face_pipeline import ensure_face_cache, embed_dataset
from knn_func import KNN

THRESHOLD_PERCENTILE = 95   # ngưỡng "Unknown": giữ lại 95% mẫu được nhận đúng


def calibrate_threshold(X, y, k):
    """Leave-one-out: với mỗi mẫu, KNN (bỏ chính nó) dự đoán đúng thì lấy khoảng cách tới
    láng giềng gần nhất; ngưỡng là percentile 95 của các khoảng cách đó."""
    knn = KNN(K=k)
    knn.fit(X, y)

    dists = []
    for i in range(len(X)):
        d, _ = knn.kneighbors(X[i], exclude=i)
        if knn._predict_one(X[i], exclude=i) == y[i]:
            dists.append(d[0])
    if not dists:
        return None
    return float(np.percentile(dists, THRESHOLD_PERCENTILE))


def save_model(model_name, X, y, k, threshold):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    extra = {} if threshold is None else {"threshold": np.array([threshold])}
    np.savez(model_path(model_name), X=X, y=y, k=np.array([k]), **extra)


def train(model_name, faces_df):
    print(f"\n===== {model_name} =====")
    X, _ = embed_dataset(model_name, faces_df)
    y = faces_df["person"].to_numpy()

    threshold = calibrate_threshold(X, y, K)
    save_model(model_name, X, y, K, threshold)

    df = pd.DataFrame(X, columns=map(str, range(X.shape[1])))
    df["name"] = y
    csv_path = os.path.join(OUTPUT_DIR, f"data_{model_name}.csv")
    df.to_csv(csv_path)
    print(f"Đã tạo {model_path(model_name)} ({len(df)} mẫu, ngưỡng Unknown = {threshold})")
    print(f"Đã tạo {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", choices=MODEL_NAMES, default=None)
    args = parser.parse_args()

    models = args.model
    if models is None:
        best_path = os.path.join(RESULTS_DIR, "best_models.json")
        if not os.path.exists(best_path):
            parser.error("Chưa có results/best_models.json, hãy chạy benchmark.py hoặc dùng --model <tên>")
        with open(best_path, encoding="utf-8") as f:
            models = json.load(f)

    faces_df = ensure_face_cache()
    if faces_df.empty:
        raise ValueError("Không có ảnh hợp lệ trong dataset/")
    n_undetected = int((faces_df["detected"] == 0).sum())
    print(f"{len(faces_df)} ảnh, {faces_df['person'].nunique()} người "
          f"({n_undetected} ảnh không detect được mặt, dùng nguyên ảnh)")

    for model_name in models:
        train(model_name, faces_df)


if __name__ == "__main__":
    main()
