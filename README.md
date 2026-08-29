# FruitVision

FruitVision is a Flask web application that classifies fruit ripeness (ripe / unripe / rotten) from photos or live video, using eight independently trained models that can be compared side by side. It also tracks harvest history, fruit stock, and market-readiness, behind role-based accounts (admin / farmer).

Originally a university group project (each "member" folder corresponds to one team member's own feature-engineering approach); this README documents the system as it stands today.

---

## Table of Contents

- [Features](#features)
- [Supported Fruits](#supported-fruits)
- [Model Registry](#model-registry)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training](#training)
- [Testing](#testing)
- [Security Notes](#security-notes)
- [Database](#database)

---

## Features

### Fruit Classification (`/classify`)
- **Single Fruit Analysis** — upload one photo, pick a fruit and a model, get an instant ripeness call with confidence, blemish/surface-quality analysis, and a marketability recommendation.
- **Batch Analysis** — upload many photos of the same fruit at once; get a distribution chart and a PDF report. Optional toggles: treat a photo as containing *multiple* fruits (apple/banana/orange only — needs YOLO object detection, not just classification), skip the fruit-match validation for a batch (useful when testing with mismatched or unusual photos), and auto-log results into Fruit Stock.
- **Mixed-Fruit Analysis** — one photo containing a mix of apples, bananas, and oranges together; YOLOv8n localises each fruit, then the Merged 1+4 model classifies every crop individually.
- Every prediction is validated first: whole-image CLIP comparison checks the selected identity against all ten supported fruits and common non-fruit categories. YOLO-World and classical shape/contour checks provide fallback validation when CLIP is inconclusive, before the image is run through the per-model ripeness prediction.

### Real-Time Tracking (`/realtime`)
Live webcam or uploaded-video tracking, with its own engine per model (temporal smoothing / rolling-vote across frames, FPS logging, stock-eligible detections counted once per physical fruit). Session results export to PDF.

### Dashboards
- **Dashboard** (`/`) — live overview: recent predictions, confidence/label breakdown, urgent marketability alerts.
- **Analytics** (`/analytics`) — global, all-time, all-model charts.
- **Marketability Dashboard** (`/marketability`) — every prediction re-interpreted as a farmer-facing handling recommendation (ready to sell / hold to ripen / remove), with an operator review workflow to confirm or correct a model's call without ever overwriting the original prediction.

### Harvest Records (`/history`)
Every prediction ever logged, filterable by fruit/model/date, with detail view, edit, delete, single-record PDF export, and CSV export of the filtered set.

### Fruit Stock (`/stock`)
An append-only ledger of stock movements (auto-logged from predictions, or added/adjusted manually), with running on-hand totals by fruit and ripeness, plus CSV and PDF export.

### Model Lab (`/model-lab`)
Side-by-side comparison of all 8 models: accuracy, macro-F1, balanced accuracy, weakest per-class recall, inference latency, and model size — reads live from each model's saved training report. Includes an interactive per-fruit, per-model confusion matrix, per-fruit recall bars, and (for the YOLO pipeline) an epoch-by-epoch loss/accuracy chart. `/training-report/<model_key>` shows one model's full per-fruit confusion matrices and class-distribution plots from its actual training run.

### Admin Panel (`/admin`, admin role only)
- Create accounts directly with a name, email, and admin-chosen password (min. 8 characters), and assign a role.
- Change a user's role, **deactivate/reactivate** an account (blocks login and drops any live session immediately, without deleting their history), or permanently remove one — with guardrails so you can't lock yourself out or strip the last admin.
- Model registry usage stats and a rolling activity log (logins, role changes, exports, reviews, ...).

### Accounts & Roles
Two roles: **admin** (sees and manages everything, system-wide) and **farmer** (sees only their own harvest records and stock — a farmer cannot view, edit, delete, or export another farmer's data, including by guessing a record's URL). Session cookies are `HttpOnly` + `SameSite=Lax`, signed with a randomly generated key persisted outside git (see [Security Notes](#security-notes)).

### Settings & Profile
Per-account defaults (default model, confidence-flagging threshold), password change, dark mode, and a personal activity/stats view.

---

## Supported Fruits

`apple`, `banana`, `orange`, `mango`, `pear`, `peach`, `strawberry`, `tomato`, `lemon`, `guava`

Upload validation uses whole-image CLIP identity comparison, with YOLO-World configured with all ten fruit names as a secondary detector. Mixed-fruit analysis and real-time tracking still use the existing COCO detector and therefore support bounding-box detection only for `apple`, `banana`, and `orange`; extending those paths to every fruit would require a custom annotated object-detection dataset.

---

## Model Registry

Every "model" is really one ripeness classifier per fruit (so e.g. "AB" is really 10 separate small classifiers, one per fruit above), unless noted otherwise.

| Key | Label | Approach |
|---|---|---|
| `ab` | Ensemble AB | SVM on hand-crafted **colour + shape** features |
| `bc` | Ensemble BC | SVM on hand-crafted **shape + texture** features |
| `cd` | Ensemble CD | SVM on hand-crafted **texture + gabor** features |
| `da` | Ensemble DA | SVM on hand-crafted **gabor + colour** features |
| `all_four` | Ensemble (All 4, soft-voted) | Averages AB/BC/CD/DA's predicted probabilities |
| `merged_1_4` | Merged 1+4 | Single SVM, features from AB + DA fused: colour + shape + gabor |
| `m14v2` | Merged 1+4 v2 | Merged 1+4 plus texture: colour + shape + gabor + texture |
| `m14v3` | Merged 1+4 v3 | Same features as v2, but detection unions Member 1's Otsu box with Member 4's HSV-saturation box, and calibration uses Member 4's deskew |
| `yolo_pure` | YOLOv8 Classification | Fine-tuned YOLOv8-cls, no hand-crafted features — a fully independent 5th predictor, deliberately **not** folded into the `all_four` soft vote (its CNN features aren't the complementary hand-crafted pair the ensemble design assumes) |

Every one of these 9 selectable options (all 8 models plus `all_four`) has its own real-time tracking engine under `realtime/`.

---

## Tech Stack

- **Backend**: Flask, SQLite (three tables' worth of app data: auth/settings, harvest results, stock ledger — no ORM, hand-written parameterised SQL)
- **Classical CV**: OpenCV, scikit-image (colour space, shape/contour, GLCM texture, Gabor filter feature extraction)
- **ML**: scikit-learn (SVMs), CLIP + Ultralytics YOLO-World (open-vocabulary upload validation), YOLOv8 (COCO multi-fruit detection and YOLOv8-cls ripeness classification), PyTorch
- **Reporting**: fpdf2 (branded PDF exports), matplotlib (training/confusion-matrix plots, dashboard charts)
- **Frontend**: server-rendered Jinja templates, vanilla JS, one hand-written CSS design system (no frontend build step, no framework)

---

## Project Structure

```
FruitVision/
├── app.py                     # The Flask app: every route, the model registry, auth gating
├── train_all.py                # Runs every model's training script from one command
├── analyze_member_performance.py  # Consolidates every model's per-fruit accuracy/recall into one comparison
├── requirements.txt
│
├── database/
│   ├── auth_db.py              # Users, roles, activity log, app-wide settings
│   ├── history_db.py           # Every logged prediction ("harvest record")
│   └── stock_db.py             # Stock movement ledger
│
├── core_modules/                # Shared logic used across every route/model
│   ├── fruit_validation.py      # "Is this photo actually the selected fruit?" gate
│   ├── multi_fruit_detect.py    # Same-fruit multi-box detection for one photo
│   ├── mixed_fruit_m14.py       # Mixed apple/banana/orange detection + M14 classification
│   ├── blemish_analysis.py      # Surface-quality / blemish-percentage analysis
│   ├── marketability.py         # Prediction → farmer-facing handling recommendation
│   ├── filter_photos.py         # Saves each model's intermediate filtered images
│   ├── model_lab.py             # Reads outputs/training/ + trained_models/ for the Model Lab page
│   ├── dashboard_charts.py      # Chart generation for every dashboard
│   ├── pdf_report.py            # Branded PDF report generation (predictions + stock)
│   └── ma_/mb_/mc_/md_*.py      # The four hand-crafted feature extractors (colour, shape, texture, gabor)
│
├── member_apps/                 # One folder per model's own preprocessing/detection/calibration/predict/train
│   ├── member_1_ab .. member_4_da/
│   ├── merged_member_1_4/, merged_member_1_4_v2/, merged_member_1_4_v3/
│   └── predict_ensemble.py      # The all_four soft vote
│
├── pipeline/pure_yolo/          # The independent YOLOv8-cls pipeline (dataset prep, train, predict)
├── pipeline/fruit_detector/     # Auto-labeled dataset builder + trainer for the single-class "fruit" detector used by real-time tracking on non-COCO fruits
│
├── realtime/                    # Webcam/video tracking: one tracker per model + shared routes
│
├── templates/                   # Jinja templates (one per page)
├── static/design.css            # The whole design system, one file
│
├── datasets/fruit_ripeness/     # Source images, {fruit}/{ripe,unripe,rotten}/ (not in git)
├── trained_models/              # Trained weights per model (not in git)
├── outputs/training/            # Confusion matrices, classification reports, training metadata (not in git)
│
└── tests/                       # pytest suite
```

---

## Getting Started

### Prerequisites
- Python **3.11** (`py --list` to check what's installed)

### Installation
```bash
pip install -r requirements.txt
```

The YOLO-World validation vocabulary uses Ultralytics' CLIP text encoder,
which `requirements.txt` installs directly from the official Ultralytics
GitHub repository. Git and network access are therefore required during the
first dependency installation.

### Dataset Setup
Download the base dataset from Kaggle: https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten

Arrange it (and any additional fruit you add) under `datasets/fruit_ripeness/` like this:

```
datasets/fruit_ripeness/
├── apple/
│   ├── ripe/
│   ├── unripe/
│   └── rotten/
├── banana/
│   ├── ripe/
│   ├── unripe/
│   └── rotten/
└── ... one folder per fruit in the Supported Fruits list, same 3 subfolders each
```

### Running the App
```bash
python app.py
```
Opens on `http://127.0.0.1:5001`. On first run, the database is created automatically and seeded with two default accounts:

| Email | Password | Role |
|---|---|---|
| `admin@fruitvision.local` | `admin123` | Admin |
| `farmer@fruitvision.local` | `farmer123` | Farmer |

**Change both passwords from Settings after your first login.** The login page shows this same hint until a 3rd account exists.

---

## Training

Every model trains independently and reads from `datasets/fruit_ripeness/`. From the project root:

```bash
python train_all.py              # trains all 8 models sequentially
python train_all.py --parallel   # first model sequential, the rest in parallel
```

This writes trained weights to `trained_models/<model>/`, per-fruit confusion matrices / class-distribution plots / classification reports to `outputs/training/<model>/`, and a log per model to `trained_logs/`.

The YOLOv8-cls pipeline (`yolo_pure`) needs an extra one-time step first, since it needs a `train`/`val` split rather than raw class folders:
```bash
python pipeline/pure_yolo/dataset_prep.py
```
If this hasn't been run for a fruit, `yolo_cls_train.py` silently skips it (prints "Skipping `<fruit>`", exits 0) — check its log if a fruit's YOLO confusion matrix doesn't show up afterward.

The single-class fruit detector used by real-time tracking (`trained_models/fruit_yolo_detect/best.pt`) also needs a one-time setup step to (re)produce it:
```bash
python pipeline/fruit_detector/dataset_prep.py
python pipeline/fruit_detector/train.py
```

**Adding a new fruit**: drop its `ripe`/`unripe`/`rotten` photos under `datasets/fruit_ripeness/<fruit>/`, add the fruit name to the `FRUITS` list in *every* `member_apps/*/m*_train.py` and `pipeline/pure_yolo/{dataset_prep,yolo_cls_train}.py`, run `dataset_prep.py` again, then `train_all.py`. Once trained, add the fruit to `FRUITS` in `app.py` and `core_modules/model_lab.py`, and to `SUPPORTED_FRUITS` in `core_modules/fruit_validation.py` to wire it into the live app (dropdowns, Model Lab, training report, fruit-match validation). Also add it to the `FRUITS` list in `pipeline/fruit_detector/dataset_prep.py` — since the detector trains a single generic "fruit" class rather than per-species classes, an already-trained detector will generally still localize a brand-new species reasonably well without retraining, but re-running `dataset_prep.py` + `train.py` after adding it there is recommended for best detection accuracy on that species (not strictly required for the app to function).

**Comparing every model at once**, after training:
```bash
python analyze_member_performance.py
```
Prints a console table of overall accuracy and per-class recall for every model × fruit, and saves a consolidated JSON/CSV plus raw per-fruit accuracy (for future weighted-voting work) under `outputs/training/`.

---

## Testing

```bash
pytest tests/
```

Covers auth/admin routes and guardrails, per-user data isolation (stock/history), YOLO-World fruit validation and its inconclusive-result fallback, the surface-analysis + persistence workflow, marketability logic, PDF report generation, and more.

---

## Security Notes

- **Session signing key**: `app.py` generates a random key on first run and persists it to `.secret_key` (gitignored) — it is *not* hardcoded in source. Deleting this file just forces everyone to log in again; it does not affect any data.
- **Passwords**: hashed with Werkzeug's `generate_password_hash`/`check_password_hash` (scrypt), never stored or logged in plaintext.
- **Session cookies**: `HttpOnly` (not readable from JS) and `SameSite=Lax` (blocks the realistic cross-site CSRF vector). Full CSRF tokens were deliberately not added — this is a local/single-deployment analysis tool, not a public multi-tenant service.
- **Per-user data isolation**: farmers only ever see their own harvest records and stock, enforced server-side on every route (list views *and* direct-by-ID access), not just hidden in the UI.
- **Deactivated accounts**: `verify_login()` rejects a deactivated account even with the correct password, and an already-open session for a just-deactivated user is dropped on its very next request.
- Not implemented: rate limiting on login (brute-force protection) — acceptable for this project's scope, worth adding before any public-facing deployment.

---

## Database

SQLite, one file (`database/fruitvision.db`, gitignored), five tables across three modules, no ORM:

- **`users`** — name, email, password hash, role (`admin`/`farmer`), `is_active`, dark mode preference, timestamps.
- **`activity_log`** — one row per notable action (login, role change, invite, deactivate, export, review, ...), for the Admin Panel.
- **`settings`** — small app-wide key/value store (default model, confidence-flag threshold).
- **`results`** — one row per prediction ever run: model, fruit, label, confidence, surface/blemish metrics, marketability fields, optional operator review, owning `user_id`.
- **`stock_events`** — one row per stock movement (signed quantity, so a manual entry can also record stock leaving), owning `user_id`.

All three modules (`auth_db.py`, `history_db.py`, `stock_db.py`) open a fresh connection per call rather than sharing a pooled connection — simple, safe for SQLite's file-level locking, and fine at this project's scale.
