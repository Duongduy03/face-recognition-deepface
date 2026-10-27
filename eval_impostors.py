"""Đo tỉ lệ "người lạ" bị nhận nhầm thành người quen ở ngưỡng Unknown hiện tại của từng mô hình.

Chỉ đọc PostgreSQL (vector người quen: faces/emb_*, người lạ: impostors/imp_*), không cần ảnh. Người lạ do
build_impostors.py nạp. Quyết định Unknown của app là "khoảng cách tới láng giềng gần nhất > ngưỡng", nên ở đây:

    far          = tỉ lệ người lạ có khoảng cách gần nhất <= ngưỡng  (bị nhận nhầm, càng thấp càng tốt)
    known_accept = tỉ lệ người quen (leave-one-out) có khoảng cách gần nhất <= ngưỡng (được nhận, càng cao càng tốt)
    thr_far1     = ngưỡng cho far ~ 1%, kèm known_accept ở ngưỡng đó (đánh đổi nếu siết ngưỡng)

    python eval_impostors.py [--model ArcFace ...]       # -> results/impostor_eval.csv

far phụ thuộc số người trong DB: càng nhiều người quen thì càng dễ có ai đó giống người lạ.
"""
import argparse
import os

import numpy as np
import pandas as pd

import vector_db
from config import MODEL_NAMES, RESULTS_DIR


def nn_distances(A, B, exclude_self=False):
    """Khoảng cách Euclid từ mỗi hàng của A tới hàng gần nhất của B. Vector đã L2-normalize nên d^2 = 2 - 2 a.b."""
    d2 = 2.0 - 2.0 * (A @ B.T)
    if exclude_self:
        np.fill_diagonal(d2, np.inf)
    return np.sqrt(np.maximum(d2.min(axis=1), 0.0))


def evaluate(conn, model_name):
    row = conn.execute("SELECT threshold FROM model_meta WHERE model_name = %s", (model_name,)).fetchone()
    if row is None or row[0] is None:
        return None, "chưa có ngưỡng trong DB (chạy train_model.py --db hoặc --db-recalibrate)"
    threshold = float(row[0])
    X_imp = vector_db.load_impostors(conn, model_name)
    if len(X_imp) == 0:
        return None, "chưa có người lạ (chạy build_impostors.py)"
    X_known, _ = vector_db.load_all(conn, model_name)

    d_imp = nn_distances(X_imp, X_known)
    d_known = nn_distances(X_known, X_known, exclude_self=True)
    thr_far1 = float(np.percentile(d_imp, 1))
    return {
        "model": model_name,
        "n_known": len(X_known),
        "n_impostors": len(X_imp),
        "threshold": round(threshold, 4),
        "far": round(float((d_imp <= threshold).mean()), 4),
        "known_accept": round(float((d_known <= threshold).mean()), 4),
        "imp_nn_p5": round(float(np.percentile(d_imp, 5)), 4),
        "imp_nn_median": round(float(np.median(d_imp)), 4),
        "thr_far1": round(thr_far1, 4),
        "known_accept_at_far1": round(float((d_known <= thr_far1).mean()), 4),
    }, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", choices=MODEL_NAMES, default=None)
    args = parser.parse_args()

    conn = vector_db.connect()
    in_db = vector_db.models_in_db(conn)
    rows = []
    for model_name in args.model or MODEL_NAMES:
        if model_name not in in_db:
            print(f"{model_name}: chưa có trong DB, bỏ qua")
            continue
        result, why = evaluate(conn, model_name)
        if result is None:
            print(f"{model_name}: {why}, bỏ qua")
        else:
            rows.append(result)

    if not rows:
        raise SystemExit("Không có mô hình nào đủ dữ liệu để đo.")
    df = pd.DataFrame(rows).sort_values("far").reset_index(drop=True)
    print("\n" + df.to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "impostor_eval.csv")
    df.to_csv(out, index=False)
    print(f"\nĐã ghi {out}")


if __name__ == "__main__":
    main()
