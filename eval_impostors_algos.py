"""Tỉ lệ "người lạ" bị nhận nhầm thành người quen, cho cả 10 thuật toán của benchmark.py.

Khác eval_impostors.py (chỉ KNN, đọc DB): script này huấn luyện lại từng thuật toán trên tập train của benchmark (cùng chia 80/20,
seed 42) rồi chấm các khuôn mặt "người lạ" ở output/impostor_faces/ (do build_impostors.py tạo: crop lấy từ bộ Kaggle
face-detection, không có danh tính, không thuộc người nào trong mô hình).

Với mỗi thuật toán, ngưỡng Unknown là ngưỡng benchmark.py đã dùng (results/algorithms/failure_summary.csv, percentile 5 của độ tin cậy
trên mẫu test được nhận đúng). Kết quả:
    far                     : tỉ lệ người lạ có độ tin cậy >= ngưỡng, tức bị nhận nhầm thành người quen (thấp hơn = tốt hơn)
    known_accept            : tỉ lệ mẫu test của người quen có độ tin cậy >= ngưỡng (khoảng 95% theo cách đặt ngưỡng)
    known_accept_at_far5    : tỉ lệ mẫu test của người quen được giữ lại khi chọn ngưỡng sao cho 5% người lạ bị nhận nhầm
                              (không phụ thuộc cách đặt ngưỡng: cao hơn = độ tin cậy phân biệt quen/lạ tốt hơn)

    python eval_impostors_algos.py              # -> results/algorithms/impostor_eval.csv (khoảng 4-5 phút)
    python eval_impostors_algos.py --algos KNN SVM-linear --limit 300
Các crop người lạ khá nhiễu (tối, mờ, xoay), giống người lạ thật; số đo phụ thuộc vào bộ crop này và vào số người quen trong mô hình.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import classifiers
from benchmark import Data, fit_model
from config import ALGO_RESULTS_DIR, BENCH_EMBEDDING, CLASSIFIER_NAMES, IMPOSTOR_CACHE_DIR, SEED
from face_pipeline import embed_dataset, embed_face, imread_unicode, load_bench_subset

CACHE = os.path.join(IMPOSTOR_CACHE_DIR, f"{BENCH_EMBEDDING}.npz")


def impostor_embeddings(limit):
    """Embedding (cache ở output/impostor_faces/<embedding>.npz) của tối đa `limit` crop người lạ đã được build_impostors.py chấp nhận."""
    index = pd.read_csv(os.path.join(IMPOSTOR_CACHE_DIR, "index.csv"))
    keys = index[index["accepted"] == 1]["key"].tolist()
    keys = [k for k in keys if os.path.exists(os.path.join(IMPOSTOR_CACHE_DIR, k.replace("/", "__")))]
    if not keys:
        raise SystemExit(f"Chưa có crop người lạ trong {IMPOSTOR_CACHE_DIR}. Chạy `python build_impostors.py --limit 1000` trước.")
    cached = {}
    if os.path.exists(CACHE):
        with np.load(CACHE, allow_pickle=True) as d:
            cached = dict(zip(map(str, d["keys"]), d["X"]))
    todo = [k for k in keys if k not in cached]
    if todo:
        print(f"Embed {len(todo)} crop người lạ ({len(cached)} đã có cache)...")
        for n, k in enumerate(todo, start=1):
            cached[k] = embed_face(imread_unicode(os.path.join(IMPOSTOR_CACHE_DIR, k.replace("/", "__"))), BENCH_EMBEDDING)
            if n % 200 == 0:
                print(f"  {n}/{len(todo)}")
        np.savez(CACHE, keys=np.array(list(cached)), X=np.array(list(cached.values()), dtype="float32"))
    rng = np.random.default_rng(SEED)
    chosen = keys if len(keys) <= limit else [keys[i] for i in sorted(rng.choice(len(keys), size=limit, replace=False))]
    return np.array([cached[k] for k in chosen], dtype="float32")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algos", nargs="+", default=CLASSIFIER_NAMES, choices=CLASSIFIER_NAMES)
    parser.add_argument("--limit", type=int, default=1000, help="số crop người lạ tối đa")
    args = parser.parse_args()

    summary_path = os.path.join(ALGO_RESULTS_DIR, "failure_summary.csv")
    if not os.path.exists(summary_path):
        raise SystemExit("Chưa có results/algorithms/failure_summary.csv. Chạy `python benchmark.py` trước.")
    thresholds = pd.read_csv(summary_path).set_index("algorithm")["unknown_threshold"]

    faces_df = load_bench_subset()
    X, _ = embed_dataset(BENCH_EMBEDDING, faces_df)
    data = Data(faces_df, X)
    imp = impostor_embeddings(args.limit)
    print(f"{len(imp)} người lạ | {len(data.train_idx)} train / {len(data.test_idx)} test của người quen\n")

    rows = []
    for name in sorted(args.algos, key=CLASSIFIER_NAMES.index):
        t0 = time.perf_counter()
        model, _, _ = fit_model(name, data.X_train, data.y_train)
        conf_imp = classifiers.scores(model, imp).max(axis=1)
        conf_known = classifiers.scores(model, data.X_test).max(axis=1)
        thr = float(thresholds[name])
        thr5 = float(np.percentile(conf_imp, 95))     # 5% người lạ có độ tin cậy >= thr5
        rows.append({
            "algorithm": name, "n_impostors": len(imp), "n_known_test": len(conf_known), "threshold": thr,
            "far": float(np.mean(conf_imp >= thr)), "known_accept": float(np.mean(conf_known >= thr)),
            "known_accept_at_far5": float(np.mean(conf_known >= thr5)),
            "impostor_conf_median": float(np.median(conf_imp)), "known_conf_median": float(np.median(conf_known)),
        })
        print(f"{name:20s} FAR {rows[-1]['far']:.3f} | giữ người quen {rows[-1]['known_accept']:.3f} | "
              f"giữ người quen khi FAR=5%: {rows[-1]['known_accept_at_far5']:.3f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        del model

    out = os.path.join(ALGO_RESULTS_DIR, "impostor_eval.csv")
    new = pd.DataFrame(rows)
    if os.path.exists(out):    # chạy một phần thì giữ dòng của thuật toán không chạy lần này (nếu cùng số người lạ)
        old = pd.read_csv(out)
        old = old[~old["algorithm"].isin(new["algorithm"]) & (old["n_impostors"] == len(imp))]
        new = pd.concat([old, new], ignore_index=True)
    new["_o"] = new["algorithm"].map(CLASSIFIER_NAMES.index)
    new.sort_values("_o").drop(columns="_o").to_csv(out, index=False)
    print("Đã ghi", out)


if __name__ == "__main__":
    main()
