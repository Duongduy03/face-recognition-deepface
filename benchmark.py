"""So sánh 10 thuật toán phân loại trên CÙNG một embedding (config.BENCH_EMBEDDING). Xem docs/SO_SANH_THUAT_TOAN.md.

Mỗi thuật toán (classifiers.py): huấn luyện -> 6 độ đo -> ngưỡng Unknown và phân loại ca lỗi -> thời gian phân loại,
kích thước mô hình -> 5-fold cross-validation. Cuối cùng phân tích chéo (mẫu nào bị sai bởi bao nhiêu thuật toán)
và chọn 2 thuật toán tốt nhất. Kết quả ghi vào results/algorithms/.

    python benchmark.py --quick                 # 20 người, ghi vào output/benchmark_quick/ (thử mã, không đụng results/)
    python benchmark.py                         # đủ 10 thuật toán trên tập con cố định
    python benchmark.py --algos KNN SVM-RBF --skip-cv --skip-timing

Dữ liệu: face_pipeline.load_bench_subset() đọc cache khuôn mặt, KHÔNG detect thêm ảnh nào (khác benchmark_embeddings.py).
Benchmark so sánh embedding cũ nằm ở benchmark_embeddings.py.
"""
import argparse
import json
import os
import platform
import tempfile
import time
import warnings
from importlib import metadata

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

import classifiers
from benchmark_embeddings import save_failure_images
from config import (
    ALGO_RESULTS_DIR, BENCH_EMBEDDING, CLASSIFIER_NAMES, DETECTOR_BACKEND, K, OUTPUT_DIR, SEED, TEST_SIZE,
)
from face_pipeline import (
    black_ratio, detect_faces, embed_dataset, embed_face, imread_unicode, largest_face, load_bench_subset,
)

TOPK = 3                      # Top-3 accuracy
UNKNOWN_PERCENTILE = 5        # ngưỡng Unknown: giữ lại 95% mẫu test được nhận đúng (độ tin cậy dưới ngưỡng thì bị loại)
BAD_CROP_BLACK_RATIO = 0.15   # crop có hơn 15% pixel đen coi là crop hỏng (lỗi detect/căn chỉnh), cùng benchmark_embeddings.py
QUICK_PEOPLE = 20
QUICK_DIR = os.path.join(OUTPUT_DIR, "benchmark_quick")
ORDER = {name: i for i, name in enumerate(CLASSIFIER_NAMES)}


# =========================
# DỮ LIỆU VÀ CÁCH CHIA
# =========================
class Data:
    """Ma trận embedding, nhãn và chỉ số train/test chung cho mọi thuật toán."""

    def __init__(self, faces_df, X):
        self.faces = faces_df
        self.X = X
        self.y = faces_df["person"].to_numpy()
        self.train_idx, self.test_idx = train_test_split(
            np.arange(len(self.y)), test_size=TEST_SIZE, stratify=self.y, random_state=SEED
        )
        tr, te = self.train_idx, self.test_idx
        self.X_train, self.y_train = X[tr], self.y[tr]
        self.X_test, self.y_test = X[te], self.y[te]
        self.train_paths = faces_df["face_path"].to_numpy()[tr]
        self.test_face_paths = faces_df["face_path"].to_numpy()[te]
        self.test_src_paths = faces_df["src_path"].to_numpy()[te]
        self.detected = faces_df["detected"].to_numpy()[te].astype(bool)
        self.black = np.array([black_ratio(imread_unicode(p)) for p in self.test_face_paths])

    def nearest_of_class(self, x, label):
        """(đường dẫn crop, khoảng cách) của mẫu train gần x nhất thuộc người `label`. Dùng để minh họa ca lỗi."""
        mask = self.y_train == label
        d = np.linalg.norm(self.X_train[mask] - x, axis=1)
        j = int(np.argmin(d))
        return self.train_paths[mask][j], float(d[j])


# =========================
# TỪNG BƯỚC ĐÁNH GIÁ
# =========================
def fit_model(name, X, y):
    """(model, thời gian fit giây, số ConvergenceWarning)."""
    model = classifiers.build(name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        model.fit(X, y)
        fit_s = time.perf_counter() - t0
    n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, fit_s, n_conv


def macro_prf(y_true, pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    return float(precision), float(recall), float(f1)


def evaluate(name, model, data):
    """6 độ đo, ngưỡng Unknown, ca lỗi và dự đoán của một thuật toán trên tập test."""
    X_test, y_test = data.X_test, data.y_test
    pred = model.predict(X_test)
    S = classifiers.scores(model, X_test)
    classes = np.asarray(model.classes_)

    ranked = classes[np.argsort(-S, axis=1)[:, :TOPK]]
    top3 = float(np.mean([t in r for t, r in zip(y_test, ranked)]))
    precision, recall, f1 = macro_prf(y_test, pred)

    conf = S.max(axis=1)     # độ tin cậy: điểm cao nhất (KNN: -khoảng cách tới láng giềng gần nhất)
    correct = pred == y_test
    threshold = float(np.percentile(conf[correct], UNKNOWN_PERCENTILE)) if correct.any() else float("nan")
    rejected = conf < threshold
    wrong = ~correct
    failing = wrong | rejected

    # lỗi ở bước detect/căn chỉnh được xếp trước lỗi của bộ phân loại (cùng quy tắc benchmark_embeddings.py)
    bad_crop = data.black > BAD_CROP_BLACK_RATIO
    reason = np.where(
        ~data.detected, "no_face_detected",
        np.where(bad_crop, "bad_crop", np.where(wrong, "misidentified", "rejected_unknown")),
    )

    fail_rows = []
    for i in np.where(failing)[0]:
        neighbor, dist = data.nearest_of_class(X_test[i], pred[i])
        fail_rows.append({
            "algorithm": name, "face_path": data.test_face_paths[i], "src_path": data.test_src_paths[i],
            "true": y_test[i], "pred": pred[i], "confidence": conf[i], "nn_dist": dist,
            "reason": reason[i], "black_ratio": data.black[i], "neighbor_face_path": neighbor,
        })
    failures = pd.DataFrame(fail_rows, columns=[
        "algorithm", "face_path", "src_path", "true", "pred", "confidence", "nn_dist", "reason",
        "black_ratio", "neighbor_face_path",
    ])
    if len(failures):
        failures = failures.sort_values("nn_dist").reset_index(drop=True)

    metrics = {
        "algorithm": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_macro": precision, "recall_macro": recall, "f1_macro": f1,
        "top3_accuracy": top3,
        "cohen_kappa": float(cohen_kappa_score(y_test, pred)),
        "n_train": len(data.train_idx), "n_test": len(y_test), "n_classes": int(len(classes)),
    }
    fail_reason = reason[failing]
    summary = {
        "algorithm": name, "n_test": len(y_test), "n_correct": int(correct.sum()),
        "n_misidentified": int((fail_reason == "misidentified").sum()),
        "n_no_face": int((fail_reason == "no_face_detected").sum()),
        "n_bad_crop": int((fail_reason == "bad_crop").sum()),
        "n_rejected_unknown": int((fail_reason == "rejected_unknown").sum()),
        "n_no_face_in_test": int((~data.detected).sum()), "n_bad_crop_in_test": int(bad_crop.sum()),
        "unknown_threshold": threshold,
    }
    predictions = pd.DataFrame({
        "algorithm": name, "face_path": data.test_face_paths, "true": y_test, "pred": pred,
        "confidence": conf, "rejected": rejected, "correct": correct, "detected": data.detected,
    })
    return metrics, summary, failures, predictions


def model_size_mb(name, model):
    """Kích thước file khi lưu bằng chính classifiers.save (file tạm, xóa ngay)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.npz" if name == "KNN" else "m.joblib")
        path = classifiers.save(model, path, {"algo": name})
        return os.path.getsize(path) / 1e6


def time_classifier(model, X_test, n_vectors, repeats):
    """Thời gian phân loại MỘT vector (như ứng dụng): predict, và scores (để có thêm độ tin cậy). Đơn vị ms."""
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(X_test), size=min(n_vectors, len(X_test)), replace=False)
    vecs = [X_test[i:i + 1] for i in sel]
    model.predict(vecs[0])                    # khởi động, không tính giờ
    classifiers.scores(model, vecs[0])

    pred_runs, score_runs = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for v in vecs:
            model.predict(v)
        pred_runs.append((time.perf_counter() - t0) / len(vecs))
        t0 = time.perf_counter()
        for v in vecs:
            classifiers.scores(model, v)
        score_runs.append((time.perf_counter() - t0) / len(vecs))
    ddof = 1 if repeats > 1 else 0
    return {
        "predict_ms": float(np.mean(pred_runs) * 1000),
        "predict_ms_std": float(np.std(pred_runs, ddof=ddof) * 1000),
        "scores_ms": float(np.mean(score_runs) * 1000),
        "n_vectors": len(vecs), "repeats": repeats,
    }


def measure_shared(data, n_images):
    """Thời gian detect (RetinaFace, ảnh gốc) và embed một lần cho cả 10 thuật toán: hai bước này chung nên không cần đo lặp."""
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(data.test_idx), size=min(n_images, len(data.test_idx)), replace=False)
    imgs = [imread_unicode(p) for p in data.test_src_paths[chosen]]     # đọc file không tính giờ

    face0 = detect_faces(imgs[0], drop_undetected=False)                 # khởi động: nạp trọng số
    embed_face(largest_face(face0)["face_bgr"], BENCH_EMBEDDING)

    detect_t, embed_t = [], []
    for img in imgs:
        t0 = time.perf_counter()
        faces = detect_faces(img, drop_undetected=False)
        t1 = time.perf_counter()
        if not faces:
            continue
        face = largest_face(faces)["face_bgr"]
        t2 = time.perf_counter()
        embed_face(face, BENCH_EMBEDDING)
        t3 = time.perf_counter()
        detect_t.append(t1 - t0)
        embed_t.append(t3 - t2)
    # thời gian detect phụ thuộc kích thước ảnh gốc (Pins lớn hơn LFW 250x250), nên ghi kèm để báo cáo nêu rõ điều kiện đo
    heights = [im.shape[0] for im in imgs]
    widths = [im.shape[1] for im in imgs]
    return {"detect_ms": float(np.mean(detect_t) * 1000), "embed_ms": float(np.mean(embed_t) * 1000),
            "n_images": len(detect_t),
            "image_height_mean": float(np.mean(heights)), "image_width_mean": float(np.mean(widths)),
            "image_height_max": int(np.max(heights)), "image_width_max": int(np.max(widths))}


def cross_validate(name, X, y, folds):
    """5-fold phân tầng (cùng seed): accuracy, F1 macro và thời gian fit của từng fold."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        model, fit_s, _ = fit_model(name, X[tr], y[tr])
        pred = model.predict(X[te])
        rows.append({"algorithm": name, "fold": fold, "accuracy": float(accuracy_score(y[te], pred)),
                     "f1_macro": macro_prf(y[te], pred)[2], "fit_s": fit_s})
        del model
    return rows


def check_knn_reference(model, data):
    """KNN tự cài so với KNeighborsClassifier của scikit-learn: số mẫu lệch và bao nhiêu trong đó là hòa phiếu 1-1-1."""
    ref = classifiers.build_reference_knn().fit(data.X_train, data.y_train)
    ours, theirs = model.predict(data.X_test), ref.predict(data.X_test)
    diff = np.where(ours != theirs)[0]
    ties = 0
    for i in diff:
        _, idx = model.knn.kneighbors(data.X_test[i])
        labels = model.knn.y_train[idx[:model.k]]
        ties += int(len(set(labels)) == len(labels))
    return {"n_test": len(ours), "n_mismatch": int(len(diff)), "n_mismatch_that_are_ties": ties}


# =========================
# GHI KẾT QUẢ
# =========================
def write_merged(path, new_df, ran, merge):
    """Ghi CSV; merge=True thì giữ dòng của các thuật toán không chạy lần này (chạy một phần bằng --algos)."""
    if merge and os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~old["algorithm"].isin(ran)]
        if len(old):
            new_df = pd.concat([old, new_df], ignore_index=True)
    new_df["_o"] = new_df["algorithm"].map(ORDER)
    sort_cols = ["_o"] + [c for c in ("fold", "face_path") if c in new_df.columns]
    new_df = new_df.sort_values(sort_cols, kind="stable").drop(columns="_o").reset_index(drop=True)
    new_df.to_csv(path, index=False)
    return new_df


def pick_best(metrics, cv, timing):
    """2 thuật toán tốt nhất: F1 macro, hòa thì Accuracy, rồi thời gian predict (ms) ngắn hơn."""
    df = metrics.copy()
    df["predict_ms"] = df["algorithm"].map(timing.set_index("algorithm")["predict_ms"]) if timing is not None else 0.0
    df = df.sort_values(["f1_macro", "accuracy", "predict_ms"], ascending=[False, False, True]).reset_index(drop=True)
    best = list(df["algorithm"].head(2))
    out = {"best": best, "rule": "F1 macro, hòa thì Accuracy, rồi predict_ms nhỏ hơn",
           "f1_macro": [float(v) for v in df["f1_macro"].head(2)], "indistinguishable": None}
    if cv is not None and all(a in set(cv["algorithm"]) for a in best):
        c = cv.set_index("algorithm")
        gap = abs(float(c.loc[best[0], "cv_f1_mean"] - c.loc[best[1], "cv_f1_mean"]))
        std = float(max(c.loc[best[0], "cv_f1_std"], c.loc[best[1], "cv_f1_std"]))
        out.update({"cv_f1_mean": [float(c.loc[a, "cv_f1_mean"]) for a in best],
                    "cv_f1_std": [float(c.loc[a, "cv_f1_std"]) for a in best],
                    "cv_gap": gap, "indistinguishable": bool(gap < std),
                    "note": "indistinguishable = chênh lệch F1 trung bình CV giữa hai thuật toán nhỏ hơn độ lệch chuẩn CV lớn hơn"})
    return out


def write_overlap(preds, data, out_dir):
    """Mỗi mẫu test bị sai bởi bao nhiêu thuật toán. Trả về dict tóm tắt cho run_info."""
    present = [a for a in CLASSIFIER_NAMES if a in set(preds["algorithm"])]
    correct = preds.pivot(index="face_path", columns="algorithm", values="correct")[present].reindex(data.test_face_paths)
    pred = preds.pivot(index="face_path", columns="algorithm", values="pred")[present].reindex(data.test_face_paths)
    wrong = ~correct.to_numpy(dtype=bool)
    n_wrong = wrong.sum(axis=1)

    overlap = pd.DataFrame({
        "face_path": data.test_face_paths, "src_path": data.test_src_paths, "true": data.y_test,
        "n_algos": len(present), "n_wrong": n_wrong,
        "algos_wrong": [";".join(a for a, w in zip(present, row) if w) for row in wrong],
        "black_ratio": data.black,
    })
    overlap.to_csv(os.path.join(out_dir, "overlap.csv"), index=False)
    summary = (overlap["n_wrong"].value_counts().reindex(range(len(present) + 1), fill_value=0)
               .rename_axis("n_algos_wrong").reset_index(name="n_samples"))
    summary.to_csv(os.path.join(out_dir, "overlap_summary.csv"), index=False)

    info = {"n_algos": len(present), "n_all_wrong": 0, "n_only_knn_wrong": None, "n_knn_wrong": None}
    if "KNN" in present:
        k = present.index("KNN")
        info["n_knn_wrong"] = int(wrong[:, k].sum())
        info["n_only_knn_wrong"] = int((wrong[:, k] & (n_wrong == 1)).sum())

    # ảnh minh họa các mẫu mà TẤT CẢ thuật toán cùng sai (nhiều khả năng là lỗi dữ liệu)
    if len(present) > 1:
        all_wrong = np.where(n_wrong == len(present))[0]
        info["n_all_wrong"] = int(len(all_wrong))
        rows = []
        for i in all_wrong:
            label = pd.Series(pred.iloc[i]).mode().iloc[0]     # nhãn sai được nhiều thuật toán chọn nhất
            neighbor, dist = data.nearest_of_class(data.X_test[i], label)
            rows.append({"face_path": data.test_face_paths[i], "neighbor_face_path": neighbor, "true": data.y_test[i],
                         "pred": label, "nn_dist": dist,
                         "reason": "bad_crop" if data.black[i] > BAD_CROP_BLACK_RATIO else "all_wrong"})
        df = pd.DataFrame(rows, columns=["face_path", "neighbor_face_path", "true", "pred", "nn_dist", "reason"])
        save_failure_images("_all_wrong", df.sort_values("nn_dist"), results_dir=out_dir)
    return info


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "unknown"


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algos", nargs="+", default=CLASSIFIER_NAMES, choices=CLASSIFIER_NAMES)
    parser.add_argument("--quick", action="store_true",
                        help=f"chỉ {QUICK_PEOPLE} người đầu của tập con, ghi vào output/benchmark_quick/ (để thử mã)")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--skip-timing", action="store_true", help="bỏ đo detect/embed/predict (vẫn tính kích thước mô hình)")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--fps-images", type=int, default=None, help="số ảnh gốc đo detect + embed (mặc định 50, --quick: 5)")
    parser.add_argument("--classify-vectors", type=int, default=200, help="số vector đo thời gian predict mỗi lượt")
    parser.add_argument("--classify-repeats", type=int, default=3)
    args = parser.parse_args()
    algos = sorted(args.algos, key=ORDER.get)
    fps_images = args.fps_images or (5 if args.quick else 50)

    out_dir = QUICK_DIR if args.quick else ALGO_RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    faces_df = load_bench_subset()
    if args.quick:
        people = sorted(faces_df["person"].unique())[:QUICK_PEOPLE]
        faces_df = faces_df[faces_df["person"].isin(people)].reset_index(drop=True)
    X, _ = embed_dataset(BENCH_EMBEDDING, faces_df)      # cache đầy đủ nên không embed thêm
    data = Data(faces_df, X)
    counts = faces_df["person"].value_counts()
    print(f"{BENCH_EMBEDDING}: {len(faces_df)} khuôn mặt, {counts.size} người ({counts.min()}-{counts.max()} ảnh/người) "
          f"| train {len(data.train_idx)} / test {len(data.test_idx)} | thư mục kết quả: {out_dir}")

    # chạy một phần chỉ gộp với kết quả cũ nếu cùng một tập test (nếu không, trộn hai lần chia là sai)
    pred_path = os.path.join(out_dir, "predictions.csv")
    merge = True
    if os.path.exists(pred_path):
        old_paths = set(pd.read_csv(pred_path, usecols=["face_path"])["face_path"])
        if old_paths != set(data.test_face_paths):
            print("Cảnh báo: tập test khác lần chạy trước, bỏ kết quả cũ trong thư mục này (không gộp).")
            merge = False

    shared = None
    if not args.skip_timing:
        print(f"\nĐo detect + embed trên {fps_images} ảnh gốc (một lần, chung cho mọi thuật toán)...")
        shared = measure_shared(data, fps_images)
        print(f"  detect {shared['detect_ms']:.0f} ms | embed {shared['embed_ms']:.1f} ms ({shared['n_images']} ảnh)")

    metrics, summaries, all_failures, all_preds, timing_rows, cv_rows, hyper, conv, errors = [], [], [], [], [], [], {}, {}, {}
    knn_reference = None
    for name in algos:
        print(f"\n===== {name} =====", flush=True)
        t_start = time.perf_counter()
        try:
            model, fit_s, n_conv = fit_model(name, data.X_train, data.y_train)
            m, summary, failures, predictions = evaluate(name, model, data)
            m["fit_s"] = fit_s
            print(f"  fit {fit_s:.1f}s | acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} top3={m['top3_accuracy']:.4f}"
                  f"{' | CẢNH BÁO: chưa hội tụ' if n_conv else ''}", flush=True)
            metrics.append(m)
            summaries.append(summary)
            all_failures.append(failures)
            all_preds.append(predictions)
            conv[name] = n_conv
            hyper[name] = {"K": model.k} if name == "KNN" else model.get_params()

            row = {"algorithm": name, "fit_s": fit_s, "model_mb": model_size_mb(name, model)}
            if not args.skip_timing:
                row.update(time_classifier(model, data.X_test, args.classify_vectors, args.classify_repeats))
                p = row["predict_ms"] / 1000
                row.update(shared)
                row["fps_classify"] = 1 / p
                row["fps_embed_classify"] = 1 / (shared["embed_ms"] / 1000 + p)
                row["fps_end_to_end"] = 1 / ((shared["detect_ms"] + shared["embed_ms"]) / 1000 + p)
                print(f"  predict {row['predict_ms']:.2f} ms (±{row['predict_ms_std']:.2f}) | model {row['model_mb']:.2f} MB", flush=True)
            else:
                print(f"  model {row['model_mb']:.2f} MB", flush=True)
            timing_rows.append(row)
            if name == "KNN":
                knn_reference = check_knn_reference(model, data)
                print(f"  so với KNN scikit-learn: lệch {knn_reference['n_mismatch']}/{knn_reference['n_test']} mẫu "
                      f"(trong đó hòa phiếu 1-1-1: {knn_reference['n_mismatch_that_are_ties']})")
            del model

            if not args.skip_cv:
                folds = cross_validate(name, X, data.y, args.cv_folds)
                cv_rows.extend(folds)
                f = pd.DataFrame(folds)
                print(f"  CV {args.cv_folds}-fold: acc {f['accuracy'].mean():.4f} ± {f['accuracy'].std():.4f} | "
                      f"f1 {f['f1_macro'].mean():.4f} ± {f['f1_macro'].std():.4f}", flush=True)
            print(f"  ({time.perf_counter() - t_start:.0f}s)", flush=True)
        except Exception as e:
            print(f"  LỖI {name}: {e!r}")
            errors[name] = repr(e)

    ran = [m["algorithm"] for m in metrics]
    if not ran:
        print("Không thuật toán nào chạy được.")
        return

    merged = write_merged(os.path.join(out_dir, "metrics.csv"), pd.DataFrame(metrics), ran, merge)
    merged = merged.sort_values(["f1_macro", "accuracy"], ascending=False).reset_index(drop=True)
    merged.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    write_merged(os.path.join(out_dir, "failure_summary.csv"), pd.DataFrame(summaries), ran, merge)
    preds = write_merged(os.path.join(out_dir, "predictions.csv"), pd.concat(all_preds, ignore_index=True), ran, merge)
    for failures in all_failures:
        if len(failures):
            name = failures["algorithm"].iloc[0]
            failures.to_csv(os.path.join(out_dir, f"failures_{name}.csv"), index=False)
            save_failure_images(name, failures.head(30), results_dir=out_dir)
    timing = write_merged(os.path.join(out_dir, "timing.csv"), pd.DataFrame(timing_rows), [r["algorithm"] for r in timing_rows], merge)

    cv_summary = None
    if cv_rows:
        folds_df = pd.DataFrame(cv_rows)
        write_merged(os.path.join(out_dir, "cv_folds.csv"), folds_df, list(folds_df["algorithm"].unique()), merge)
        g = folds_df.groupby("algorithm")
        cv_new = pd.DataFrame({
            "algorithm": g.size().index, "n_folds": g.size().to_numpy(),
            "cv_acc_mean": g["accuracy"].mean().to_numpy(), "cv_acc_std": g["accuracy"].std(ddof=1).to_numpy(),
            "cv_f1_mean": g["f1_macro"].mean().to_numpy(), "cv_f1_std": g["f1_macro"].std(ddof=1).to_numpy(),
            "cv_fit_s_mean": g["fit_s"].mean().to_numpy(),
        })
        cv_summary = write_merged(os.path.join(out_dir, "cv.csv"), cv_new, list(cv_new["algorithm"]), merge)
    elif os.path.exists(os.path.join(out_dir, "cv.csv")) and merge:
        cv_summary = pd.read_csv(os.path.join(out_dir, "cv.csv"))

    best = pick_best(merged, cv_summary, timing if "predict_ms" in timing.columns and timing["predict_ms"].notna().all() else None)
    with open(os.path.join(out_dir, "best_algorithms.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"\nTop 2 theo F1 macro: {best['best']}"
          f"{' (không phân biệt được theo CV)' if best['indistinguishable'] else ''}")

    overlap_info = write_overlap(preds, data, out_dir) if preds["algorithm"].nunique() >= 1 else {}

    # run_info: chạy một phần thì giữ siêu tham số/lỗi của các thuật toán không chạy lần này
    info_path = os.path.join(out_dir, "run_info.json")
    old = json.load(open(info_path, encoding="utf-8")) if (merge and os.path.exists(info_path)) else {}
    info = {
        "embedding": BENCH_EMBEDDING, "quick": args.quick,
        "n_images": int(len(faces_df)), "n_people": int(counts.size),
        "images_per_person_min": int(counts.min()), "images_per_person_max": int(counts.max()),
        "n_train": int(len(data.train_idx)), "n_test": int(len(data.test_idx)),
        "n_undetected": int((faces_df["detected"] == 0).sum()),
        "K": K, "test_size": TEST_SIZE, "seed": SEED, "detector": DETECTOR_BACKEND,
        "topk": TOPK, "unknown_percentile": UNKNOWN_PERCENTILE, "bad_crop_black_ratio": BAD_CROP_BLACK_RATIO,
        "cv_folds": args.cv_folds, "cv_std_ddof": 1,
        "timing": {"fps_images": fps_images, "classify_vectors": args.classify_vectors,
                   "classify_repeats": args.classify_repeats, "shared": shared},
        "algorithms_in_last_run": ran,
        "hyperparameters": {**old.get("hyperparameters", {}), **hyper},
        "convergence_warnings": {**old.get("convergence_warnings", {}), **conv},
        "errors": errors,
        "knn_vs_sklearn": knn_reference or old.get("knn_vs_sklearn"),
        "overlap": overlap_info,
        "platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
        "python": platform.python_version(), "tensorflow": _version("tensorflow"), "deepface": _version("deepface"),
        "scikit_learn": _version("scikit-learn"), "numpy": _version("numpy"),
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"Đã ghi kết quả vào {out_dir}")


if __name__ == "__main__":
    main()
