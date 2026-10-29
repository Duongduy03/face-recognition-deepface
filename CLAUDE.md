# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Face-recognition student project (HAUI), documented in Vietnamese. DeepFace detects faces and produces embeddings; a hand-written KNN (`knn_func.py`, no scikit-learn classifier) classifies them. Two parts: a Streamlit demo (`app.py`) and a benchmark that compares 9 DeepFace embedding models + KNN with 6 metrics, FPS and failure analysis (`benchmark.py`, report in `docs/SO_SANH_MO_HINH.md`). DeepFace's `DeepFace` and `Buffalo_L` models cannot run here (TF 2.21 / Google Drive weights) and sit in `config.UNAVAILABLE_MODELS`.

## Commands

No test suite or linter. Python 3.12 venv in `.venv` (3.13 also fine).

```bash
pip install -r requirements.txt
python prepare_data.py                  # download LFW into dataset/<Name>/N.jpg
python prepare_pins.py                  # Kaggle Pins (external/pins): 30 imgs/person -> dataset/, 10 imgs/person -> external/pins_demo_test/ (never ingested; upload them in the app to demo)
python benchmark.py [--models A B] [--skip-fps] [--fps-images N]   # -> results/*.csv, failures/, best_models.json
python plot_results.py                  # charts from results/*.csv
python train_model.py --model ArcFace   # -> output/model_knn_<model>.npz (no flag = models in results/best_models.json)
streamlit run app.py
docker compose up -d                    # PostgreSQL + pgvector (optional vector store)
python train_model.py --model ArcFace --db   # also writes the vectors into PostgreSQL
python train_model.py --model ArcFace --db-recalibrate   # only recompute the DB's Unknown threshold from all its vectors
python build_impostors.py --limit 1000  # one person only: unnamed faces from external/face-detection -> impostors tables in the DB
python eval_impostors.py                # whole team, DB only: false-accept rate of "strangers" at each model's threshold -> results/impostor_eval.csv
```

`config.py` reconfigures stdout/stderr to UTF-8 (needed on Windows when output is piped, because DeepFace logs emoji). It must be imported before `deepface`; keep that order in new modules.

## Architecture

Pipeline: image -> RetinaFace detect + align -> embedding -> L2-normalize -> KNN -> name.

- `config.py` — the single place for `MODEL_NAMES` (9 runnable models), `DETECTOR_BACKEND`, `K`, `SEED`, paths, `model_path(name)`. Add or rename models here only.
- `face_pipeline.py` — shared by app, train and benchmark: `detect_faces`, `embed_face`, `ensure_face_cache` (detects each dataset image once into `output/faces/` + `index.csv`), `embed_dataset` (per-model embedding cache in `output/embeddings/<model>.npz`). Detection is slow (~1.3 s/image on CPU), so these caches matter; delete them if dataset images change in place.
- `knn_func.py` — `KNN` (Euclidean, majority vote, ties broken by nearest neighbor), `kneighbors`, `l2_normalize`, `load_knn_from_npz` (returns `None` if the file is missing; also reads optional `threshold`).
- `vector_db.py` — optional PostgreSQL + pgvector store, meant to be shared by the team on a cloud Postgres (Supabase/Neon): put `FACE_DB_URL` in `.env` (git-ignored, see `.env.example`; `config.py` loads it, default is the local `docker-compose.yml` container). Tables `persons`, `faces` (keyed by machine-independent `image_key` = `<person>/<file>`), one `emb_<model>` per model (dimension differs), `model_meta` (K, threshold). `train_model.py --db` only upserts; `--db-prune` also deletes faces missing from the local `dataset/`, so never use it against a shared DB from a partial dataset. Team rules for the shared DB: images are not stored in the DB or git (share the `dataset/<Name>/<file>` folder via Drive or similar; the DB only keeps `image_key`); capture filenames are timestamp+uuid (not `N.jpg`) so two members never write the same `image_key`, and old `N.jpg` files from before this change can still collide; names go through `config.normalize_name` (NFC, collapsed spaces); `model_meta.pipeline_fingerprint` (`config.PIPELINE_FINGERPRINT`: detector, align, BGR, L2, deepface version) makes `check_pipeline` refuse to ingest or mix vectors from a different pipeline, and `app.py` warns on mismatch; `ingest` never sets `threshold`, `train_model.py --db` recalibrates it on all vectors in the DB (`vector_db.load_all`) after ingesting, and `--db-recalibrate` does only that. `PgVectorKNN` has the same interface as `KNN` (`predict`, `kneighbors`, `K`, `threshold`) but the DB finds the K nearest (`<->` = Euclidean) and `knn_func.majority_vote` still votes. The `.npz` flow is unchanged; `train_model.py --db` writes both and `app.py` shows a "Nguồn vector" switch when the DB has data. HNSW index only for models up to 2000 dims (VGG-Face, 4096, is searched without an index). `benchmark.py` does not use the DB.
- `build_impostors.py` / `eval_impostors.py` — measure how often a stranger is accepted as someone known. Source images are the Kaggle `fareselmenshawii/face-detection-dataset` (face detection only: no identities, so it cannot train recognition), downloaded once by one member into `external/face-detection/` (git-ignored; the kaggle zip had a corrupt stretch from a resumed download, about 2% of files, which is harmless because only ~1000 faces are used). `build_impostors.py` detects with the same `detect_faces`, keeps the largest face (confidence >= 0.9, small side >= 60 px, black ratio <= 0.15), caches crops in `output/impostor_faces/`, embeds with each model and writes `impostors` + `imp_<model>` (separate from `faces`, so KNN never votes for them; `check_pipeline` applies). `eval_impostors.py` reads only the DB, so teammates need neither the images nor Kaggle. FAR depends on how many known people are in the DB; the crops are noisy (some dark, blurred or rotated), like real strangers.
- `prepare_pins.py` — splits the Kaggle `hereisburak/pins-face-recognition` set (105 celebrities, 17,534 images, scraped from Pinterest so a few images carry the wrong label; names are partly lowercase) into 30 train + 10 test images per person with a fixed seed; train goes to `dataset/` (then `train_model.py --db`), test stays in `external/pins_demo_test/` (git-ignored) for demos. The shared Neon DB holds these people on purpose (demo scale), but only for ArcFace, Facenet512 and SFace; the other 6 models cover just the 7 LFW people plus impostors, so `eval_impostors.py` numbers for them are not comparable.
- `benchmark.py` — fixed stratified 80/20 split (seed in config), evaluates every model with the same split, writes `results/`. Re-running a subset with `--models` merges into existing CSVs.
- `train_model.py` — trains on all of `dataset/` per model and calibrates a per-model "Unknown" distance threshold by leave-one-out (95th percentile), stored in the npz.
- `app.py` — pages: capture photos, recognize (model chosen from the sidebar among trained `output/model_knn_*.npz`), compare models (reads `results/`).
- Git-ignored/regenerable: `dataset/`, `output/`. Charts and failure images in `results/` are kept in git on purpose (`.gitignore` exceptions).

## Gotchas

- DeepFace treats numpy image inputs as **BGR** (OpenCV convention). Never convert to RGB before `DeepFace.represent`/`extract_faces`; `extract_faces` returns the face in RGB, which `detect_faces` converts back to BGR. The original app got this wrong (embeddings at inference had swapped channels versus training).
- Embeddings are L2-normalized everywhere (benchmark, train, app); distances and thresholds are only meaningful in that space. A threshold from one model does not transfer to another.
- `benchmark.py` and `train_model.py` embed with `detector_backend="skip"` on already-cropped faces; app crops come from `detect_faces`. Keep that consistent if changing the pipeline.
- Read/write images through `imread_unicode` / `imwrite_unicode` in `face_pipeline.py`, not `cv2.imread`/`imwrite`; plain OpenCV fails on non-ASCII Windows paths (dataset folders are person names).
- The Unknown threshold is calibrated differently in the two places: `benchmark.py` uses the 95th percentile of nearest-neighbor distance over correctly classified test samples, `train_model.py` uses leave-one-out over the full dataset. Do not compare the values.
- `ensure_face_cache` falls back to the whole image when no face is found (`detected = 0`) so the sample stays in the evaluation set.
- `haar/` and `README_DEEPFACE.md` are stale leftovers (Haar Cascade, `app_deepface.py`, Facenet) and do not match the code. `tutor.md` is an unrelated AI-tutor persona prompt.
