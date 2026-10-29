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

import vector_db
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


def recalibrate_db(db_conn, model_name):
    """Ngưỡng Unknown dùng chung: tính trên mọi vector trong DB (gồm dữ liệu của thành viên khác),
    không phải trên dataset/ cục bộ vốn có thể chỉ là một phần."""
    X, y = vector_db.load_all(db_conn, model_name)
    threshold = calibrate_threshold(X, y, K)
    vector_db.set_threshold(db_conn, model_name, threshold)
    print(f"Ngưỡng Unknown trong DB = {threshold} ({len(X)} vector, {len(set(y))} người)")


def train(model_name, faces_df, use_db=False, prune=False):
    print(f"\n===== {model_name} =====")
    # Mở kết nối mới ngay trước mỗi lần dùng: embed mất hàng chục phút, Postgres cloud tự ngắt kết nối nhàn rỗi
    # (Neon free: AdminShutdown) nên kết nối mở từ đầu sẽ chết trước khi kịp ghi.
    if use_db:
        with vector_db.connect() as db_conn:
            vector_db.check_pipeline(db_conn, model_name)   # báo lỗi trước khi tốn thời gian embed
    X, _ = embed_dataset(model_name, faces_df)
    y = faces_df["person"].to_numpy()

    threshold = calibrate_threshold(X, y, K)
    save_model(model_name, X, y, K, threshold)
    if use_db:
        with vector_db.connect() as db_conn:
            vector_db.ingest(db_conn, model_name, faces_df, X, K, prune=prune)
            print(f"Đã ghi {len(X)} vector vào PostgreSQL (bảng {vector_db._table(model_name)})")
            recalibrate_db(db_conn, model_name)

    df = pd.DataFrame(X, columns=map(str, range(X.shape[1])))
    df["name"] = y
    csv_path = os.path.join(OUTPUT_DIR, f"data_{model_name}.csv")
    df.to_csv(csv_path)
    print(f"Đã tạo {model_path(model_name)} ({len(df)} mẫu, ngưỡng Unknown = {threshold})")
    print(f"Đã tạo {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", choices=MODEL_NAMES, default=None)
    parser.add_argument("--db", action="store_true",
                        help="Ghi thêm vector vào PostgreSQL + pgvector (docker compose up -d)")
    parser.add_argument("--db-prune", action="store_true",
                        help="Với --db: xóa khỏi DB các khuôn mặt không có trong dataset/ trên máy này. "
                             "Chỉ dùng khi dataset/ của bạn là toàn bộ dữ liệu chung, vì DB dùng chung sẽ mất dữ liệu của người khác")
    parser.add_argument("--db-recalibrate", action="store_true",
                        help="Chỉ tính lại ngưỡng Unknown trong DB từ mọi vector đang có ở đó "
                             "(không cần dataset/, không embed, không ghi vector)")
    args = parser.parse_args()
    if args.db_prune and not args.db:
        parser.error("--db-prune cần đi kèm --db")
    if args.db_recalibrate and args.db:
        parser.error("--db-recalibrate không đi cùng --db (--db đã tính lại ngưỡng sau khi ghi)")

    models = args.model
    if models is None:
        best_path = os.path.join(RESULTS_DIR, "best_models.json")
        if not os.path.exists(best_path):
            parser.error("Chưa có results/best_models.json, hãy chạy benchmark.py hoặc dùng --model <tên>")
        with open(best_path, encoding="utf-8") as f:
            models = json.load(f)

    if args.db_recalibrate:
        db_conn = vector_db.connect()
        in_db = vector_db.models_in_db(db_conn)
        for model_name in models:
            print(f"\n===== {model_name} =====")
            if model_name in in_db:
                recalibrate_db(db_conn, model_name)
            else:
                print("Chưa có trong DB, bỏ qua")
        return

    faces_df = ensure_face_cache()
    if faces_df.empty:
        raise ValueError("Không có ảnh hợp lệ trong dataset/")
    n_undetected = int((faces_df["detected"] == 0).sum())
    print(f"{len(faces_df)} ảnh, {faces_df['person'].nunique()} người "
          f"({n_undetected} ảnh không detect được mặt, dùng nguyên ảnh)")

    for model_name in models:
        train(model_name, faces_df, args.db, prune=args.db_prune)


if __name__ == "__main__":
    main()
