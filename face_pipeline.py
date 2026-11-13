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
    ALGO_RESULTS_DIR, BENCH_MIN_IMAGES,
)
from deepface import DeepFace
from knn_func import l2_normalize

EMBED_SAVE_EVERY = 1000   # lưu cache embedding sau mỗi ngần này ảnh


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
    # đường dẫn trong index có thể do máy khác ghi: tính lại theo thư mục hiện tại từ (person, filename)
    done["src_path"] = [os.path.join(dataset_dir, p, f) for p, f in zip(done["person"], done["filename"])]
    done["face_path"] = [os.path.join(FACES_CACHE_DIR, p, f) for p, f in zip(done["person"], done["filename"])]
    done_keys = set(zip(done["person"], done["filename"]))

    todo = [r for r in _list_dataset_images(dataset_dir) if r not in done_keys]
    if verbose and todo:
        print(f"Detect {len(todo)} ảnh mới ({len(done)} ảnh đã có trong cache)...")

    new_rows = []
    try:
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
    finally:
        done = _write_index(index_path, done, new_rows, cols)   # Ctrl-C cũng lưu nốt phần đã detect

    # chỉ giữ ảnh còn tồn tại trong dataset hiện tại
    current = set(_list_dataset_images(dataset_dir))
    keep = [(p, f) in current for p, f in zip(done["person"], done["filename"])]
    return done[keep].sort_values(["person", "filename"]).reset_index(drop=True)


def _write_index(index_path, done, new_rows, cols):
    if new_rows:
        done = pd.concat([done, pd.DataFrame(new_rows, columns=cols)], ignore_index=True)
        new_rows.clear()
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    tmp_path = index_path + ".tmp"   # ghi file tạm rồi đổi tên, ngắt giữa chừng không làm hỏng index cũ
    done.to_csv(tmp_path, index=False)
    os.replace(tmp_path, index_path)
    return done


def load_cached_faces(people=None, min_images=None, dataset_dir=DATASET_DIR):
    """Đọc cache khuôn mặt (output/faces/index.csv) mà KHÔNG detect thêm ảnh nào.

    Khác ensure_face_cache: hàm kia detect mọi ảnh trong dataset/ chưa có trong cache (hàng chục nghìn ảnh nếu cache
    đang tụt lại, ~1,3 giây/ảnh). Hàm này chỉ trả những gì đã có, nên dùng cho benchmark trên một tập con.

    people: chỉ giữ những người trong danh sách này. min_images: chỉ giữ người có ít nhất ngần này ảnh hợp lệ
    (cả hai cùng đặt thì áp dụng cả hai). Ảnh hợp lệ là ảnh gốc còn trong dataset/ và file crop còn trong cache.
    Cột và thứ tự giống ensure_face_cache: person, filename, src_path, face_path, detected, confidence, detect_s.
    """
    index_path = os.path.join(FACES_CACHE_DIR, "index.csv")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Chưa có cache khuôn mặt {index_path}. Chạy ensure_face_cache() (detect) một lần trước.")

    df = pd.read_csv(index_path)
    # đường dẫn trong index có thể do máy khác ghi: tính lại theo thư mục hiện tại từ (person, filename)
    df["src_path"] = [os.path.join(dataset_dir, p, f) for p, f in zip(df["person"], df["filename"])]
    df["face_path"] = [os.path.join(FACES_CACHE_DIR, p, f) for p, f in zip(df["person"], df["filename"])]

    if people is not None:
        df = df[df["person"].isin(set(people))]
    df = df[[os.path.exists(s) and os.path.exists(c) for s, c in zip(df["src_path"], df["face_path"])]]
    if min_images is not None:
        counts = df["person"].value_counts()
        df = df[df["person"].isin(counts[counts >= min_images].index)]
    return df.sort_values(["person", "filename"]).reset_index(drop=True)


def load_bench_subset(min_images=BENCH_MIN_IMAGES, subset_file=None):
    """Tập con cố định để so sánh 10 thuật toán (docs/SO_SANH_THUAT_TOAN.md mục 2.1), đọc từ cache, không detect.

    Danh sách người nằm trong results/algorithms/subset_people.txt (mỗi dòng một tên). Chưa có file thì tạo từ cache
    hiện tại (người có ít nhất min_images ảnh). Từ đó trở đi luôn dùng đúng danh sách trong file, dù cache có thêm
    người mới, để các lần chạy so sánh được với nhau. Số ảnh của mỗi người thì vẫn là mọi ảnh đang có trong cache.
    """
    subset_file = subset_file or os.path.join(ALGO_RESULTS_DIR, "subset_people.txt")
    if os.path.exists(subset_file):
        with open(subset_file, encoding="utf-8") as f:
            people = [line.strip() for line in f if line.strip()]
    else:
        people = sorted(load_cached_faces(min_images=min_images)["person"].unique())
        os.makedirs(os.path.dirname(subset_file), exist_ok=True)
        with open(subset_file, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(people) + "\n")
        print(f"Đã tạo {subset_file} ({len(people)} người có ≥ {min_images} ảnh trong cache)")

    df = load_cached_faces(people=people)
    missing = sorted(set(people) - set(df["person"]))
    if missing:
        print(f"Cảnh báo: {len(missing)} người trong {subset_file} không còn ảnh hợp lệ trong cache: {missing[:5]}...")
    return df


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
        # phải đóng file (with): trên Windows file .npz còn mở thì os.replace ở _write_embed_cache báo PermissionError
        with np.load(cache_path, allow_pickle=True) as data:
            for p, x, t in zip(data["paths"], data["X"], data["embed_s"]):
                # khóa là đường dẫn tuyệt đối, có thể do máy khác ghi: lấy <người>/<file> rồi gắn vào thư mục hiện tại
                person, filename = str(p).replace("\\", "/").split("/")[-2:]
                cache[os.path.join(FACES_CACHE_DIR, person, filename)] = (x, float(t))

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
            if n % EMBED_SAVE_EVERY == 0:   # lưu định kỳ để chạy lại được nếu bị ngắt
                _write_embed_cache(cache_path, cache)

        _write_embed_cache(cache_path, cache)

    X = np.array([cache[p][0] for p in paths], dtype="float32")
    embed_s = np.array([cache[p][1] for p in paths], dtype="float64")
    return X, embed_s


def _write_embed_cache(cache_path, cache):
    all_paths = list(cache.keys())
    tmp_path = cache_path[:-len(".npz")] + ".tmp.npz"   # ghi file tạm rồi đổi tên, ngắt giữa chừng không làm hỏng cache cũ
    np.savez(
        tmp_path,
        paths=np.array(all_paths),
        X=np.array([cache[p][0] for p in all_paths], dtype="float32"),
        embed_s=np.array([cache[p][1] for p in all_paths], dtype="float64"),
    )
    os.replace(tmp_path, cache_path)
