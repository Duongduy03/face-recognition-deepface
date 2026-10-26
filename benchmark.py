"""So sánh các mô hình embedding (DeepFace) + KNN trên cùng một dataset.

Với mỗi mô hình: embed -> KNN tự cài (knn_func.KNN) -> 6 độ đo, tốc độ (FPS), các ca nhận diện sai.
Kết quả ghi vào results/. Xem docs/SO_SANH_MO_HINH.md để biết cách đọc kết quả.

    python benchmark.py                      # chạy đủ MODEL_NAMES trong config.py
    python benchmark.py --models ArcFace SFace
    python benchmark.py --skip-fps
"""
import argparse
import json
import os
import platform
import re
import time

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from config import (
    MODEL_NAMES, RESULTS_DIR, K, TEST_SIZE, SEED, DETECTOR_BACKEND,
)
from face_pipeline import (
    ensure_face_cache, embed_dataset, embed_face, detect_faces, largest_face,
    imread_unicode, imwrite_unicode, black_ratio,
)
from knn_func import KNN

TOPK = 3                    # Top-3 accuracy
THRESHOLD_PERCENTILE = 95   # ngưỡng "Unknown": giữ lại 95% mẫu đúng
MAX_FAILURE_IMAGES = 30     # số ảnh lỗi lưu ra cho mỗi mô hình (CSV vẫn đủ)
BAD_CROP_BLACK_RATIO = 0.15  # crop có hơn 15% pixel đen coi là crop hỏng (lỗi detect/căn chỉnh)


# =========================
# ĐÁNH GIÁ MỘT MÔ HÌNH
# =========================
def evaluate_model(model_name, faces_df, train_idx, test_idx):
    y = faces_df["person"].to_numpy()
    X, embed_s = embed_dataset(model_name, faces_df)

    knn = KNN(K=K)
    knn.fit(X[train_idx], y[train_idx])
    y_train = knn.y_train

    preds, nn_dist, topk_hit, nn_pred_idx = [], [], [], []
    knn_times = []
    for i in test_idx:
        t0 = time.perf_counter()
        dists, idx = knn.kneighbors(X[i])
        pred = knn._predict_one(X[i])
        knn_times.append(time.perf_counter() - t0)

        # top-k lớp gần nhất (mỗi lớp tính một lần, theo thứ tự láng giềng gần nhất)
        ranked = list(dict.fromkeys(y_train[idx]))[:TOPK]
        topk_hit.append(y[i] in ranked)
        preds.append(pred)
        nn_dist.append(dists[0])
        # mẫu train gần nhất thuộc lớp được dự đoán (dùng để minh họa ca nhầm)
        nn_pred_idx.append(idx[np.argmax(y_train[idx] == pred)])

    y_test = y[test_idx]
    preds = np.array(preds)
    nn_dist = np.array(nn_dist)
    correct = preds == y_test

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )

    # Ngưỡng Unknown: percentile 95 của khoảng cách ở các mẫu được nhận đúng
    threshold = float(np.percentile(nn_dist[correct], THRESHOLD_PERCENTILE)) if correct.any() else float("nan")
    rejected = nn_dist > threshold
    knn.threshold = threshold

    detected = faces_df["detected"].to_numpy()[test_idx].astype(bool)
    face_paths = faces_df["face_path"].to_numpy()[test_idx]
    black = np.array([black_ratio(imread_unicode(p)) for p in face_paths])
    bad_crop = black > BAD_CROP_BLACK_RATIO
    wrong = ~correct
    failing = wrong | rejected

    # Nguyên nhân của ca lỗi: lỗi ở bước detect/căn chỉnh được xếp trước lỗi của bộ phân loại
    reason = np.where(
        ~detected, "no_face_detected",
        np.where(bad_crop, "bad_crop", np.where(wrong, "misidentified", "rejected_unknown")),
    )
    failures = pd.DataFrame({
        "model": model_name,
        "face_path": face_paths,
        "src_path": faces_df["src_path"].to_numpy()[test_idx],
        "true": y_test,
        "pred": preds,
        "nn_dist": nn_dist,
        "reason": reason,
        "black_ratio": black,
        "neighbor_face_path": faces_df["face_path"].to_numpy()[train_idx][nn_pred_idx],
    })[failing].sort_values("nn_dist").reset_index(drop=True)

    predictions = pd.DataFrame({
        "model": model_name,
        "face_path": faces_df["face_path"].to_numpy()[test_idx],
        "true": y_test, "pred": preds, "nn_dist": nn_dist,
        "rejected": rejected, "detected": detected,
    })

    row = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, preds),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "top3_accuracy": float(np.mean(topk_hit)),
        "cohen_kappa": cohen_kappa_score(y_test, preds),
        "embedding_dim": X.shape[1],
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }
    fail_reason = reason[failing]
    summary = {
        "model": model_name,
        "n_test": len(test_idx),
        "n_correct": int(correct.sum()),
        "n_misidentified": int((fail_reason == "misidentified").sum()),
        "n_no_face": int((fail_reason == "no_face_detected").sum()),
        "n_bad_crop": int((fail_reason == "bad_crop").sum()),
        "n_rejected_unknown": int((fail_reason == "rejected_unknown").sum()),
        "n_no_face_in_test": int((~detected).sum()),
        "n_bad_crop_in_test": int(bad_crop.sum()),
        "unknown_threshold": threshold,
    }
    return row, summary, failures, predictions, knn


# =========================
# FPS
# =========================
def measure_fps(model_name, knn, faces_df, test_idx, n_images):
    """Đo thời gian từng bước trên ảnh gốc: detect -> embed -> KNN (đã bỏ lượt khởi động)."""
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(test_idx, size=min(n_images, len(test_idx)), replace=False)
    imgs = [imread_unicode(faces_df["src_path"].iloc[i]) for i in chosen]   # đọc file không tính giờ

    # khởi động
    f = largest_face(detect_faces(imgs[0], drop_undetected=False))
    knn._predict_one(embed_face(f["face_bgr"], model_name))

    detect_t, embed_t, knn_t = [], [], []
    for img in imgs:
        t0 = time.perf_counter()
        faces = detect_faces(img, drop_undetected=False)
        t1 = time.perf_counter()
        face = largest_face(faces)["face_bgr"]
        t2 = time.perf_counter()
        emb = embed_face(face, model_name)
        t3 = time.perf_counter()
        knn._predict_one(emb)
        t4 = time.perf_counter()
        detect_t.append(t1 - t0)
        embed_t.append(t3 - t2)
        knn_t.append(t4 - t3)

    d, e, k = np.mean(detect_t), np.mean(embed_t), np.mean(knn_t)
    return {
        "model": model_name,
        "n_images": len(imgs),
        "detect_ms": d * 1000,
        "embed_ms": e * 1000,
        "knn_ms": k * 1000,
        "fps_embed_knn": 1.0 / (e + k),
        "fps_end_to_end": 1.0 / (d + e + k),
    }


# =========================
# ẢNH CÁC CA LỖI
# =========================
def _safe(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:40]


def save_failure_images(model_name, failures):
    out_dir = os.path.join(RESULTS_DIR, "failures", model_name)
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, old))

    size = 160
    for n, r in enumerate(failures.head(MAX_FAILURE_IMAGES).itertuples(), start=1):
        left = cv2.resize(imread_unicode(r.face_path), (size, size))
        right = cv2.resize(imread_unicode(r.neighbor_face_path), (size, size))
        band = np.full((58, size * 2, 3), 255, dtype=np.uint8)
        lines = [
            (f"true: {r.true}"[:44], (0, 0, 0)),
            (f"pred: {r.pred}"[:44], (0, 0, 180)),
            (f"d={r.nn_dist:.2f} | {r.reason}", (90, 90, 90)),
        ]
        for row, (text, color) in enumerate(lines):
            text = text.encode("ascii", "replace").decode()
            cv2.putText(band, text, (4, 15 + row * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        montage = np.vstack([band, np.hstack([left, right])])
        imwrite_unicode(os.path.join(out_dir, f"{n:02d}_{_safe(r.true)}_as_{_safe(r.pred)}.jpg"), montage)


# =========================
# GHI KẾT QUẢ (gộp với lần chạy trước khi chỉ chạy một số mô hình)
# =========================
def merge_csv(path, new_df, ran_models):
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~old["model"].isin(ran_models)]
        if len(old):   # bảng cũ rỗng thì bỏ qua để không giữ lại các cột đã lỗi thời
            new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)
    return new_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--fps-images", type=int, default=50, help="Số ảnh dùng để đo FPS")
    parser.add_argument("--skip-fps", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    faces_df = ensure_face_cache()
    y = faces_df["person"].to_numpy()
    idx_all = np.arange(len(faces_df))
    train_idx, test_idx = train_test_split(
        idx_all, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    print(f"{len(faces_df)} khuôn mặt, {len(set(y))} người | train {len(train_idx)} / test {len(test_idx)}")

    metrics, fps_rows, summaries, all_failures, all_preds, errors = [], [], [], [], [], {}
    for model_name in args.models:
        print(f"\n===== {model_name} =====")
        try:
            row, summary, failures, predictions, knn = evaluate_model(
                model_name, faces_df, train_idx, test_idx
            )
            print(f"  acc={row['accuracy']:.3f} f1={row['f1_macro']:.3f} top3={row['top3_accuracy']:.3f}")
            metrics.append(row)
            summaries.append(summary)
            all_failures.append(failures)
            all_preds.append(predictions)
            failures.to_csv(os.path.join(RESULTS_DIR, f"failures_{model_name}.csv"), index=False)
            save_failure_images(model_name, failures)

            if not args.skip_fps:
                fps = measure_fps(model_name, knn, faces_df, test_idx, args.fps_images)
                print(f"  FPS embed+knn={fps['fps_embed_knn']:.1f} end-to-end={fps['fps_end_to_end']:.2f}")
                fps_rows.append(fps)
        except Exception as e:
            print(f"  LỖI {model_name}: {e!r}")
            errors[model_name] = repr(e)

    ran = [m["model"] for m in metrics]
    if ran:
        merged = merge_csv(os.path.join(RESULTS_DIR, "metrics.csv"), pd.DataFrame(metrics), ran)
        merged = merged.sort_values(["f1_macro", "accuracy"], ascending=False).reset_index(drop=True)
        merged.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
        merge_csv(os.path.join(RESULTS_DIR, "failure_summary.csv"), pd.DataFrame(summaries), ran)
        merge_csv(os.path.join(RESULTS_DIR, "predictions.csv"), pd.concat(all_preds, ignore_index=True), ran)
        if fps_rows:
            merge_csv(os.path.join(RESULTS_DIR, "fps.csv"), pd.DataFrame(fps_rows), [r["model"] for r in fps_rows])

        best = list(merged["model"].head(2))
        with open(os.path.join(RESULTS_DIR, "best_models.json"), "w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False)
        print(f"\nTop 2 theo F1 macro: {best}")

    import tensorflow as tf, deepface
    info = {
        "n_images": int(len(faces_df)), "n_people": int(len(set(y))),
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "n_undetected": int((faces_df["detected"] == 0).sum()),
        "K": K, "test_size": TEST_SIZE, "seed": SEED, "detector": DETECTOR_BACKEND,
        "topk": TOPK, "threshold_percentile": THRESHOLD_PERCENTILE,
        "bad_crop_black_ratio": BAD_CROP_BLACK_RATIO,
        "platform": platform.platform(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "python": platform.python_version(),
        "tensorflow": tf.__version__, "deepface": getattr(deepface, "__version__", ""),
        "errors": errors,
    }
    with open(os.path.join(RESULTS_DIR, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
