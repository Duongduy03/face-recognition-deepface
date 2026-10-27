"""Tạo embedding "người lạ" từ ảnh không có tên và nạp vào PostgreSQL (bảng impostors, imp_<mô hình>).

Chỉ một người trong nhóm cần chạy (cần ảnh trong external/face-detection/ và DB dùng chung). Cả nhóm sau đó đo
bằng eval_impostors.py mà không cần ảnh.

    python build_impostors.py --limit 1000                  # detect 1000 khuôn mặt rồi embed bằng cả 9 mô hình
    python build_impostors.py --model ArcFace --limit 300
    python build_impostors.py --detect-only                 # chỉ detect + cache, chưa động tới DB

Chạy lại an toàn: ảnh đã detect nằm trong output/impostor_faces/ (index.csv), embedding đã có trong DB thì bỏ qua.
"""
import argparse
import os
import random

import numpy as np
import pandas as pd

import vector_db
from config import IMPOSTOR_CACHE_DIR, IMPOSTOR_SRC_DIR, IMAGE_EXTS, MODEL_NAMES, SEED
from face_pipeline import black_ratio, detect_faces, embed_face, imread_unicode, imwrite_unicode, largest_face

MIN_CONFIDENCE = 0.9    # RetinaFace chắc chắn đây là khuôn mặt
MAX_BLACK_RATIO = 0.15  # cùng ngưỡng "bad_crop" của benchmark.py
BATCH = 100             # số vector ghi DB mỗi lần, để bị ngắt giữa chừng không mất hết
INDEX_COLS = ["key", "accepted", "confidence", "reason"]


def list_images(src_dir):
    """image_key = đường dẫn tương đối của ảnh, dấu "/" (không chứa đường dẫn của máy nào)."""
    keys = []
    for root, _, files in os.walk(src_dir):
        for name in files:
            if name.lower().endswith(IMAGE_EXTS):
                rel = os.path.relpath(os.path.join(root, name), src_dir)
                keys.append(rel.replace(os.sep, "/"))
    return sorted(keys)


def face_path(key):
    return os.path.join(IMPOSTOR_CACHE_DIR, key.replace("/", "__"))


def detect_stage(src_dir, limit, min_face):
    """Detect ngẫu nhiên (seed cố định) cho tới khi đủ `limit` khuôn mặt đạt chất lượng. Trả về index đã nhận."""
    index_path = os.path.join(IMPOSTOR_CACHE_DIR, "index.csv")
    done = pd.read_csv(index_path) if os.path.exists(index_path) else pd.DataFrame(columns=INDEX_COLS)
    seen = set(done["key"])
    accepted = int(done["accepted"].sum())

    keys = list_images(src_dir)
    if not keys:
        raise FileNotFoundError(f"Không có ảnh trong {src_dir}. Hãy tải dataset Kaggle vào đó trước.")
    random.Random(SEED).shuffle(keys)
    todo = [k for k in keys if k not in seen]
    print(f"{len(keys)} ảnh nguồn, đã xử lý {len(seen)} (nhận {accepted}), mục tiêu {limit}")

    rows = []
    try:
        for n, key in enumerate(todo, start=1):
            if accepted >= limit:
                break
            img = imread_unicode(os.path.join(src_dir, key))
            reason, conf = "", 0.0
            if img is None:
                reason = "unreadable"
            else:
                faces = detect_faces(img)
                if not faces:
                    reason = "no_face"
                else:
                    best = largest_face(faces)
                    conf = best["confidence"]
                    area = best["facial_area"]
                    if conf < MIN_CONFIDENCE:
                        reason = "low_confidence"
                    elif min(area["w"], area["h"]) < min_face:
                        reason = "too_small"
                    elif black_ratio(best["face_bgr"]) > MAX_BLACK_RATIO:
                        reason = "bad_crop"
                    else:
                        imwrite_unicode(face_path(key), best["face_bgr"])
                        accepted += 1
            rows.append({"key": key, "accepted": int(reason == ""), "confidence": conf, "reason": reason})
            if n % 25 == 0:
                print(f"  detect {n} ảnh, nhận {accepted}/{limit}")
    finally:
        # ghi index cả khi bị ngắt (Ctrl+C) để lần sau không phải detect lại
        if rows:
            os.makedirs(IMPOSTOR_CACHE_DIR, exist_ok=True)
            pd.concat([done, pd.DataFrame(rows, columns=INDEX_COLS)], ignore_index=True).to_csv(index_path, index=False)

    done = pd.read_csv(index_path)
    print("Lý do loại:", done.loc[done["accepted"] == 0, "reason"].value_counts().to_dict())
    return done[done["accepted"] == 1].head(limit)


def embed_stage(accepted, models):
    conn = vector_db.connect()
    keys_all = accepted["key"].tolist()
    conf = dict(zip(accepted["key"], accepted["confidence"]))
    for model_name in models:
        print(f"\n===== {model_name} =====")
        vector_db.check_pipeline(conn, model_name)   # báo lỗi trước khi tốn thời gian embed
        have = vector_db.impostor_keys(conn, model_name)
        todo = [k for k in keys_all if k not in have]
        print(f"{len(have)} đã có trong DB, cần embed {len(todo)}")
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            X = np.array([embed_face(imread_unicode(face_path(k)), model_name) for k in chunk])
            vector_db.ingest_impostors(conn, model_name, chunk, [conf[k] for k in chunk], X)
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=IMPOSTOR_SRC_DIR, help="thư mục ảnh nguồn (mặc định external/face-detection)")
    parser.add_argument("--limit", type=int, default=1000, help="số khuôn mặt người lạ cần có")
    parser.add_argument("--min-face", type=int, default=60, help="cạnh nhỏ nhất (pixel) của khung mặt trong ảnh gốc")
    parser.add_argument("--model", nargs="+", choices=MODEL_NAMES, default=None)
    parser.add_argument("--detect-only", action="store_true")
    args = parser.parse_args()

    accepted = detect_stage(args.src, args.limit, args.min_face)
    print(f"\nCó {len(accepted)} khuôn mặt người lạ trong {IMPOSTOR_CACHE_DIR}")
    if not args.detect_only:
        embed_stage(accepted, args.model or MODEL_NAMES)


if __name__ == "__main__":
    main()
