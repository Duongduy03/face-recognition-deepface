"""Tải dataset LFW (Labeled Faces in the Wild) và ghi ra dataset/<Tên>/N.jpg.

Cùng cấu trúc thư mục mà app.py (trang chụp ảnh) đang dùng, nên train_model.py và
benchmark.py đọc được ngay. Không ghi đè / xóa thư mục người đã tồn tại.
"""
import argparse
import os

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

from config import DATASET_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-faces", type=int, default=20,
                        help="Chỉ lấy người có ít nhất ngần này ảnh trong LFW")
    parser.add_argument("--max-per-person", type=int, default=30,
                        help="Tối đa số ảnh lấy cho mỗi người (giảm mất cân bằng lớp)")
    args = parser.parse_args()

    # slice_=None giữ nguyên ảnh 250x250 (mặc định sklearn cắt còn 125x94, quá nhỏ cho detector)
    lfw = fetch_lfw_people(
        min_faces_per_person=args.min_faces,
        color=True,
        resize=1.0,
        slice_=None,
    )
    print(f"LFW: {len(lfw.images)} ảnh, {len(lfw.target_names)} người (min_faces={args.min_faces})")

    saved_total = 0
    skipped_people = 0
    for label, name in enumerate(lfw.target_names):
        person_dir = os.path.join(DATASET_DIR, name)
        if os.path.isdir(person_dir) and os.listdir(person_dir):
            skipped_people += 1
            continue
        os.makedirs(person_dir, exist_ok=True)

        idx = np.where(lfw.target == label)[0][: args.max_per_person]
        for n, i in enumerate(idx, start=1):
            img = lfw.images[i]
            if img.max() <= 1.0:  # sklearn mới trả về float 0..1
                img = img * 255.0
            img_bgr = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                continue
            with open(os.path.join(person_dir, f"{n}.jpg"), "wb") as f:
                f.write(buf.tobytes())
            saved_total += 1

    print(f"Đã lưu {saved_total} ảnh vào {DATASET_DIR}")
    if skipped_people:
        print(f"Bỏ qua {skipped_people} người vì thư mục đã tồn tại.")


if __name__ == "__main__":
    main()
