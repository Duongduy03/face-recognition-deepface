"""Train mô hình cho app.py. Hai chế độ:

(1) Thuật toán phân loại (mặc định): train trên embedding config.BENCH_EMBEDDING, dữ liệu là tập con của benchmark
    (face_pipeline.load_bench_subset, đọc cache khuôn mặt, KHÔNG detect thêm), lưu ra output/model_<embedding>_<thuật toán>.*

    python train_model.py                       # config.DEMO_ALGORITHMS (SVM-linear + KNN)
    python train_model.py --algo SVM-RBF MLP    # thuật toán tùy chọn trong config.CLASSIFIER_NAMES

(2) KNN theo từng embedding trên cả dataset/ (chế độ cũ, nguồn "File .npz" và --db của app), lưu output/model_knn_<mô hình>.npz.
    Chú ý: chế độ này gọi ensure_face_cache(), detect MỌI ảnh trong dataset/ chưa có trong cache.

    python train_model.py --model ArcFace
    python train_model.py --model ArcFace Facenet512 --db
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import classifiers
import vector_db
from config import (
    BENCH_EMBEDDING, CLASSIFIER_NAMES, DEMO_ALGORITHMS, MODEL_NAMES, OUTPUT_DIR, PIPELINE_FINGERPRINT,
    RESULTS_DIR, SEED, K, model_path,
)
from face_pipeline import ensure_face_cache, embed_dataset, load_bench_subset
from knn_func import KNN

THRESHOLD_PERCENTILE = 95   # ngưỡng "Unknown": giữ lại 95% mẫu được nhận đúng
CONF_PERCENTILE = 100 - THRESHOLD_PERCENTILE   # cùng ý nghĩa nhưng tính trên độ tin cậy (lớn hơn = chắc hơn)


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


def calibrate_confidence_threshold(algo, X, y, folds=5):
    """Ngưỡng Unknown cho một thuật toán trong classifiers.py: dự đoán out-of-fold (5-fold, cùng seed benchmark), lấy độ tin cậy
    của các mẫu được nhận đúng, ngưỡng là percentile 5 (giữ lại 95%). Dưới ngưỡng thì app coi là Unknown.

    Leave-one-out (như calibrate_threshold) không khả thi với thuật toán phải huấn luyện lại mỗi lần (SVM, Random Forest, MLP).
    """
    kept = []
    for tr, te in StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED).split(X, y):
        model = classifiers.build(algo).fit(X[tr], y[tr])
        pred = model.predict(X[te])
        conf = classifiers.scores(model, X[te]).max(axis=1)
        kept.append(conf[pred == y[te]])
    return float(np.percentile(np.concatenate(kept), CONF_PERCENTILE))


def train_algorithm(algo, X, y):
    """Train một thuật toán trên toàn bộ (X, y), ngưỡng Unknown out-of-fold, lưu bằng classifiers.save. Trả về đường dẫn file."""
    print(f"\n===== {algo} ({BENCH_EMBEDDING}) =====")
    t0 = time.perf_counter()
    threshold = calibrate_confidence_threshold(algo, X, y)
    print(f"Ngưỡng Unknown (độ tin cậy) = {threshold:.4f}  [{time.perf_counter() - t0:.1f}s cho 5 lượt out-of-fold]")

    t0 = time.perf_counter()
    model = classifiers.build(algo).fit(X, y)
    fit_s = time.perf_counter() - t0
    meta = {
        "algo": algo, "embedding": BENCH_EMBEDDING, "threshold": threshold, "unknown_percentile": CONF_PERCENTILE,
        "pipeline_fingerprint": PIPELINE_FINGERPRINT, "n_train": int(len(y)), "n_classes": int(len(set(y))),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = classifiers.save(model, classifiers.model_file(algo, BENCH_EMBEDDING), meta)
    print(f"Đã train {len(y)} mẫu, {meta['n_classes']} người, fit {fit_s:.2f}s -> {path} ({os.path.getsize(path) / 1e6:.2f} MB)")
    return path


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
    parser.add_argument("--algo", nargs="+", choices=CLASSIFIER_NAMES, default=None,
                        help=f"Thuật toán phân loại cần train trên {BENCH_EMBEDDING} (mặc định khi không truyền gì: {DEMO_ALGORITHMS}). "
                             "Không dùng chung với --model / --db*")
    args = parser.parse_args()
    if args.db_prune and not args.db:
        parser.error("--db-prune cần đi kèm --db")
    if args.db_recalibrate and args.db:
        parser.error("--db-recalibrate không đi cùng --db (--db đã tính lại ngưỡng sau khi ghi)")
    legacy = args.model is not None or args.db or args.db_recalibrate
    if args.algo and legacy:
        parser.error("--algo không đi cùng --model / --db / --db-recalibrate (pgvector và --model chỉ dành cho KNN theo embedding)")

    if not legacy:   # chế độ thuật toán: không đối số = DEMO_ALGORITHMS
        algos = args.algo or DEMO_ALGORITHMS
        faces_df = load_bench_subset()     # đọc cache, không detect
        X, _ = embed_dataset(BENCH_EMBEDDING, faces_df)
        y = faces_df["person"].to_numpy()
        print(f"{BENCH_EMBEDDING}: {len(y)} ảnh, {len(set(y))} người (tập con của benchmark)")
        for algo in algos:
            train_algorithm(algo, X, y)
        return

    models = args.model
    if models is None:
        best_path = os.path.join(RESULTS_DIR, "best_models.json")
        if not os.path.exists(best_path):
            parser.error("Chưa có results/best_models.json, hãy chạy benchmark_embeddings.py hoặc dùng --model <tên>")
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
