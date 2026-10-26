# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Face-recognition student project (HAUI), documented in Vietnamese. DeepFace detects faces and produces embeddings; a hand-written KNN (`knn_func.py`, no scikit-learn classifier) classifies them. Two parts: a Streamlit demo (`app.py`) and a benchmark that compares 9 DeepFace embedding models + KNN with 6 metrics, FPS and failure analysis (`benchmark.py`, report in `docs/SO_SANH_MO_HINH.md`). DeepFace's `DeepFace` and `Buffalo_L` models cannot run here (TF 2.21 / Google Drive weights) and sit in `config.UNAVAILABLE_MODELS`.

## Commands

No test suite or linter. Python 3.12 venv in `.venv` (3.13 also fine).

```bash
pip install -r requirements.txt
python prepare_data.py                  # download LFW into dataset/<Name>/N.jpg
python benchmark.py [--models A B] [--skip-fps]   # -> results/*.csv, failures/, best_models.json
python plot_results.py                  # charts from results/*.csv
python train_model.py --model ArcFace   # -> output/model_knn_<model>.npz (no flag = models in results/best_models.json)
streamlit run app.py
```

`config.py` reconfigures stdout/stderr to UTF-8 (needed on Windows when output is piped, because DeepFace logs emoji). It must be imported before `deepface`; keep that order in new modules.

## Architecture

Pipeline: image -> RetinaFace detect + align -> embedding -> L2-normalize -> KNN -> name.

- `config.py` — the single place for `MODEL_NAMES` (9 runnable models), `DETECTOR_BACKEND`, `K`, `SEED`, paths, `model_path(name)`. Add or rename models here only.
- `face_pipeline.py` — shared by app, train and benchmark: `detect_faces`, `embed_face`, `ensure_face_cache` (detects each dataset image once into `output/faces/` + `index.csv`), `embed_dataset` (per-model embedding cache in `output/embeddings/<model>.npz`). Detection is slow (~1.3 s/image on CPU), so these caches matter; delete them if dataset images change in place.
- `knn_func.py` — `KNN` (Euclidean, majority vote, ties broken by nearest neighbor), `kneighbors`, `l2_normalize`, `load_knn_from_npz` (returns `None` if the file is missing; also reads optional `threshold`).
- `benchmark.py` — fixed stratified 80/20 split (seed in config), evaluates every model with the same split, writes `results/`. Re-running a subset with `--models` merges into existing CSVs.
- `train_model.py` — trains on all of `dataset/` per model and calibrates a per-model "Unknown" distance threshold by leave-one-out (95th percentile), stored in the npz.
- `app.py` — pages: capture photos, recognize (model chosen from the sidebar among trained `output/model_knn_*.npz`), compare models (reads `results/`).
- Git-ignored/regenerable: `dataset/`, `output/`. Charts and failure images in `results/` are kept in git on purpose (`.gitignore` exceptions).

## Gotchas

- DeepFace treats numpy image inputs as **BGR** (OpenCV convention). Never convert to RGB before `DeepFace.represent`/`extract_faces`; `extract_faces` returns the face in RGB, which `detect_faces` converts back to BGR. The original app got this wrong (embeddings at inference had swapped channels versus training).
- Embeddings are L2-normalized everywhere (benchmark, train, app); distances and thresholds are only meaningful in that space. A threshold from one model does not transfer to another.
- `benchmark.py` and `train_model.py` embed with `detector_backend="skip"` on already-cropped faces; app crops come from `detect_faces`. Keep that consistent if changing the pipeline.
- `ensure_face_cache` falls back to the whole image when no face is found (`detected = 0`) so the sample stays in the evaluation set.
- `haar/` and `README_DEEPFACE.md` are stale leftovers (Haar Cascade, `app_deepface.py`, Facenet) and do not match the code. `tutor.md` is an unrelated AI-tutor persona prompt.
