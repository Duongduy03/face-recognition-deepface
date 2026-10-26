"""Các bước xử lý dùng chung cho app.py, train_model.py và benchmark.py.

Quy ước: ảnh numpy đưa vào DeepFace phải là BGR (cùng quy ước với cv2.imread).
Pipeline: ảnh -> detect + align (retinaface) -> embedding -> L2-normalize -> KNN.
"""
import os
import time

import cv2
import numpy as np
import pandas as pd

# config phải được import trước deepface: nó đặt stdout sang UTF-8 để log của DeepFace không gây lỗi
from config import (
    DATASET_DIR, FACES_CACHE_DIR, EMBED_DIR, DETECTOR_BACKEND, IMAGE_EXTS,
)
from deepface import DeepFace
from knn_func import l2_normalize


# =========================
# I/O ẢNH (hỗ trợ đường dẫn Unicode trên Windows)
# =========================
def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img, quality=95):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


# =========================
# DETECT & EMBED
# =========================
def detect_faces(bgr_image, drop_undetected=True):
    """Detect + align khuôn mặt bằng DeepFace.

    Trả về list dict {"face_bgr", "facial_area", "confidence"}.
    enforce_detection=False nên khi không thấy mặt DeepFace trả về nguyên ảnh với
    confidence = 0; drop_undetected=True sẽ bỏ các kết quả đó.
    """
    try:
        face_objs = DeepFace.extract_faces(
            img_path=bgr_image,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
        )
    except Exception as e:
        print(f"Lỗi detect face: {e}")
        return []

    results = []
    for face_obj in face_objs:
        confidence = float(face_obj.get("confidence", 0) or 0)
        if drop_undetected and confidence <= 0:
            continue
        face_rgb = np.clip(face_obj["face"] * 255.0, 0, 255).astype("uint8")
        results.append({
            "face_bgr": cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR),
            "facial_area": face_obj["facial_area"],
            "confidence": confidence,
        })
    return results


def black_ratio(face_bgr):
    """Tỉ lệ pixel (gần) đen của khuôn mặt. Crop bị hỏng sau khi căn chỉnh/xoay thường có phần lớn là đen."""
    return float((face_bgr.max(axis=2) < 8).mean())


def largest_face(faces):
    return max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])


def embed_face(face_bgr, model_name):
    """Embedding (đã L2-normalize) của một khuôn mặt đã được cắt sẵn."""
    result = DeepFace.represent(
        img_path=face_bgr,
        model_name=model_name,
        detector_backend="skip",
        enforce_detection=False,
        align=False,
    )
    return l2_normalize(result[0]["embedding"])


# =========================
# CACHE KHUÔN MẶT CỦA DATASET
# =========================
def _list_dataset_images(dataset_dir):
    rows = []
    for person in sorted(os.listdir(dataset_dir)):
        folder = os.path.join(dataset_dir, person)
        if not os.path.isdir(folder):
            continue
        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith(IMAGE_EXTS):
                rows.append((person, filename))
    return rows


def ensure_face_cache(dataset_dir=DATASET_DIR, verbose=True):
    """Detect mỗi ảnh trong dataset đúng một lần.

    Khuôn mặt lớn nhất được lưu ở output/faces/<người>/<file> và ghi vào index.csv.
    Chạy lại chỉ xử lý ảnh mới. Nếu detect không ra mặt, dùng nguyên ảnh (detected = 0)
    để mẫu vẫn nằm trong tập đánh giá.

    Trả về DataFrame: person, filename, src_path, face_path, detected, confidence, detect_s.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Không tìm thấy dataset : {dataset_dir}")

    index_path = os.path.join(FACES_CACHE_DIR, "index.csv")
    cols = ["person", "filename", "src_path", "face_path", "detected", "confidence", "detect_s"]
    done = pd.read_csv(index_path) if os.path.exists(index_path) else pd.DataFrame(columns=cols)
    done_keys = set(zip(done["person"], done["filename"]))

    todo = [r for r in _list_dataset_images(dataset_dir) if r not in done_keys]
    if verbose and todo:
        print(f"Detect {len(todo)} ảnh mới ({len(done)} ảnh đã có trong cache)...")

    new_rows = []
    for n, (person, filename) in enumerate(todo, start=1):
        src_path = os.path.join(dataset_dir, person, filename)
        img = imread_unicode(src_path)
        if img is None or img.size == 0:
            print(f"  Không đọc được {src_path}")
            continue

        t0 = time.perf_counter()
        faces = detect_faces(img, drop_undetected=False)
        detect_s = time.perf_counter() - t0
        if not faces:
            continue

        face = largest_face(faces)
        face_path = os.path.join(FACES_CACHE_DIR, person, filename)
        imwrite_unicode(face_path, face["face_bgr"])
        new_rows.append({
            "person": person, "filename": filename, "src_path": src_path,
            "face_path": face_path, "detected": int(face["confidence"] > 0),
            "confidence": face["confidence"], "detect_s": detect_s,
        })

        if verbose and n % 100 == 0:
            print(f"  {n}/{len(todo)}")
        if n % 100 == 0:   # lưu định kỳ để chạy lại được nếu bị ngắt
            done = _write_index(index_path, done, new_rows, cols)

    done = _write_index(index_path, done, new_rows, cols)

    # chỉ giữ ảnh còn tồn tại trong dataset hiện tại
    current = set(_list_dataset_images(dataset_dir))
    keep = [(p, f) in current for p, f in zip(done["person"], done["filename"])]
    return done[keep].sort_values(["person", "filename"]).reset_index(drop=True)


def _write_index(index_path, done, new_rows, cols):
    if new_rows:
        done = pd.concat([done, pd.DataFrame(new_rows, columns=cols)], ignore_index=True)
        new_rows.clear()
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    done.to_csv(index_path, index=False)
    return done


# =========================
# CACHE EMBEDDING THEO MÔ HÌNH
# =========================
def embed_dataset(model_name, faces_df, verbose=True):
    """Embedding (L2-normalize) của mọi khuôn mặt trong faces_df, cache ở output/embeddings/<model>.npz.

    Trả về (X, embed_s): ma trận embedding và thời gian embed (giây) của từng ảnh.
    """
    os.makedirs(EMBED_DIR, exist_ok=True)
    cache_path = os.path.join(EMBED_DIR, f"{model_name}.npz")

    cache = {}
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        for p, x, t in zip(data["paths"], data["X"], data["embed_s"]):
            cache[str(p)] = (x, float(t))

    paths = list(faces_df["face_path"])
    missing = [p for p in paths if p not in cache]
    if missing:
        if verbose:
            print(f"[{model_name}] embed {len(missing)} ảnh ({len(cache)} ảnh đã có cache)...")
        # khởi động: nạp trọng số / build model, không tính vào thời gian
        embed_face(imread_unicode(missing[0]), model_name)
        for n, p in enumerate(missing, start=1):
            img = imread_unicode(p)
            t0 = time.perf_counter()
            emb = embed_face(img, model_name)
            cache[p] = (emb, time.perf_counter() - t0)
            if verbose and n % 300 == 0:
                print(f"  {n}/{len(missing)}")

        all_paths = list(cache.keys())
        np.savez(
            cache_path,
            paths=np.array(all_paths),
            X=np.array([cache[p][0] for p in all_paths], dtype="float32"),
            embed_s=np.array([cache[p][1] for p in all_paths], dtype="float64"),
        )

    X = np.array([cache[p][0] for p in paths], dtype="float32")
    embed_s = np.array([cache[p][1] for p in paths], dtype="float64")
    return X, embed_s
