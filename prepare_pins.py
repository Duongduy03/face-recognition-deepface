"""Chia bộ Kaggle "Pins Face Recognition" (105 người) thành ảnh nạp vào dataset/ và ảnh test để demo.

    python prepare_pins.py                          # 30 ảnh/người vào dataset/, 10 ảnh/người ra external/pins_demo_test/
    python prepare_pins.py --train 30 --test 10

Mỗi người lấy ngẫu nhiên (seed cố định) 30 + 10 ảnh không trùng nhau. Ảnh test KHÔNG nằm trong dataset/ nên không
bao giờ được nạp vào DB: dùng chúng để tải lên trang "Nhận diện" khi demo (người đã quen, ảnh chưa từng thấy).
Tên thư mục bỏ tiền tố "pins_" và qua config.normalize_name. Không ghi đè thư mục người đã tồn tại.
"""
import argparse
import os
import random
import shutil

from config import DATASET_DIR, IMAGE_EXTS, PROJECT_ROOT, SEED, normalize_name

PINS_DIR = os.path.join(PROJECT_ROOT, "external", "pins", "105_classes_pins_dataset")
TEST_DIR = os.path.join(PROJECT_ROOT, "external", "pins_demo_test")


def copy_files(files, src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for f in files:
        shutil.copyfile(os.path.join(src_dir, f), os.path.join(dest_dir, f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=PINS_DIR)
    parser.add_argument("--train", type=int, default=30, help="số ảnh mỗi người chép vào dataset/")
    parser.add_argument("--test", type=int, default=10, help="số ảnh mỗi người để riêng làm ảnh test demo")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--test-dir", default=TEST_DIR)
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        raise SystemExit(f"Chưa có {args.src}. Tải bằng: kaggle datasets download -d hereisburak/pins-face-recognition "
                         "-p external/pins --unzip")

    n_people = n_train = n_test = n_skipped = 0
    for folder in sorted(os.listdir(args.src)):
        src_dir = os.path.join(args.src, folder)
        if not os.path.isdir(src_dir):
            continue
        name = normalize_name(folder.removeprefix("pins_"))
        files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(IMAGE_EXTS))
        if len(files) < args.train + args.test:
            print(f"Bỏ qua {name}: chỉ có {len(files)} ảnh")
            continue

        person_dir = os.path.join(args.dataset_dir, name)
        if os.path.isdir(person_dir) and os.listdir(person_dir):
            n_skipped += 1
            continue

        random.Random(f"{SEED}-{name}").shuffle(files)   # mỗi người một thứ tự cố định, chạy lại ra cùng kết quả
        train_files, test_files = files[:args.train], files[args.train:args.train + args.test]
        copy_files(train_files, src_dir, person_dir)
        copy_files(test_files, src_dir, os.path.join(args.test_dir, name))
        n_people += 1
        n_train += len(train_files)
        n_test += len(test_files)

    print(f"Đã chép {n_train} ảnh của {n_people} người vào {args.dataset_dir}")
    print(f"Đã để riêng {n_test} ảnh test vào {args.test_dir}")
    if n_skipped:
        print(f"Bỏ qua {n_skipped} người vì thư mục đã tồn tại trong dataset/.")


if __name__ == "__main__":
    main()
