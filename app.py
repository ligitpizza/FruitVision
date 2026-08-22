"""
FruitiVision — global Flask app.

Moved out of member_apps/member_1_ab/m1_extra_app.py so the web app,
history, dashboard, and PDF export are shared infrastructure instead of
living inside one member's folder. Member folders now contain ONLY their
own pipeline code (preprocessing/detection/calibration/predict/train) --
no app.py, no templates, no database file.

Run with:  python app.py
"""
import os
import sys
import json
import math
import time
import csv
import io
from datetime import datetime
from collections import Counter
from functools import wraps
import cv2
from flask import (
    Flask, request, render_template, send_from_directory, redirect, url_for,
    flash, session, g, Response,
)
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMBER_APPS_DIR = os.path.join(BASE_DIR, "member_apps")
sys.path.append(BASE_DIR)
sys.path.append(MEMBER_APPS_DIR)

# --- Each member's own predict module ---------------------------------
sys.path.append(os.path.join(MEMBER_APPS_DIR, "member_1_ab"))
sys.path.append(os.path.join(MEMBER_APPS_DIR, "member_2_bc"))
sys.path.append(os.path.join(MEMBER_APPS_DIR, "member_3_cd"))
sys.path.append(os.path.join(MEMBER_APPS_DIR, "member_4_da"))
# --- Pure-YOLO pipeline (5th, independent predictor -- NOT part of the
#     4-member soft-voted ensemble; see pipeline/pure_yolo/ for rationale) --
sys.path.append(os.path.join(BASE_DIR, "pipeline", "pure_yolo"))
# --- Merged 1+4 (feature-level fusion of member 1's colour+shape and
#     member 4's gabor+colour into ONE SVM; see member_apps/merged_member_1_4) --
sys.path.append(os.path.join(MEMBER_APPS_DIR, "merged_member_1_4"))
# --- Merged 1+4 v2 (same idea, plus texture (C) on top -- colour+shape+
#     gabor+texture; see member_apps/merged_member_1_4_v2) --
sys.path.append(os.path.join(MEMBER_APPS_DIR, "merged_member_1_4_v2"))

from member_apps.member_1_ab.m1_predict import predict_ripeness as m1_predict_ripeness, NotAFruitError as M1NotAFruitError
from member_apps.member_2_bc.m2_predict import predict_ripeness as m2_predict_ripeness, NotAFruitError as M2NotAFruitError
from member_apps.member_3_cd.m3_predict import predict_ripeness as m3_predict_ripeness, NotAFruitError as M3NotAFruitError
from member_apps.member_4_da.m4_predict import predict_ripeness as m4_predict_ripeness, NotAFruitError as M4NotAFruitError
from member_apps.merged_member_1_4.m14_predict import predict_ripeness as m14_predict_ripeness, NotAFruitError as M14NotAFruitError
from member_apps.merged_member_1_4_v2.m14v2_predict import predict_ripeness as m14v2_predict_ripeness, NotAFruitError as M14v2NotAFruitError

from pipeline.pure_yolo.yolo_cls_predict import predict_ripeness as yolo_pure_predict_ripeness, NotAFruitError as YoloPureNotAFruitError

# --- 4-member ensemble (soft-voting across all members) -----------------
from member_apps.predict_ensemble import predict_ensemble

# --- Shared infrastructure (used to live inside member_1_ab) -----------
from core_modules.pdf_report import generate_pdf_report, generate_pdf_report_batch
from core_modules.blemish_analysis import analyze_surface
from core_modules.fruit_validation import validate_selected_fruit, FruitValidationError
from core_modules.multi_fruit_detect import supports_multi_fruit, detect_fruit_boxes
from core_modules.dashboard_charts import (
    generate_trend_chart,
    generate_history_chart,
    generate_fruit_breakdown_chart,
    generate_confidence_trend_chart,
)
from database.history_db import (
    log_result,
    get_recent,
    get_paginated,
    get_all as get_all_results,
    get_by_id,
    update_result,
    delete_result,
    get_stats,
    get_stats_since,
)
from database import auth_db

FRUITS = ["apple", "banana", "orange", "mango"]
RIPENESS_CLASSES = ["ripe", "unripe", "rotten"]
HISTORY_PAGE_SIZE = 15

app = Flask(__name__)
app.secret_key = "fruitivision-dev-key"  # only used for flash(); replace for real deployment
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
TRAINING_DIR = os.path.join(OUTPUTS_DIR, "training")

from realtime.stream_routes import realtime_bp
app.register_blueprint(realtime_bp)

# --------------------------------------------------------------------------
# Auth: session-based login, gating every route except /login and static
# assets. g.user is loaded on every request so templates/route code can
# read it without an extra query.
# --------------------------------------------------------------------------
PUBLIC_PATHS = {"/login"}


@app.before_request
def load_logged_in_user():
    user_id = session.get("user_id")
    g.user = auth_db.get_user_by_id(user_id) if user_id else None

    if request.path.startswith("/static/") or request.path.startswith("/outputs/"):
        return
    if request.path in PUBLIC_PATHS:
        return
    if g.user is None:
        return redirect(url_for("login"))


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not g.user or g.user["role"] != "admin":
            flash("Admin access required.")
            return redirect(url_for("dashboard_home"))
        return view(*args, **kwargs)
    return wrapped


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("dashboard_home"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = auth_db.verify_login(email, password)
        if user:
            session["user_id"] = user["id"]
            auth_db.touch_last_active(user["id"])
            auth_db.log_activity(user["id"], "login")
            return redirect(url_for("dashboard_home"))
        flash("Invalid email or password.")

    show_seed_hint = len(auth_db.list_users()) == 2
    return render_template("login.html", show_seed_hint=show_seed_hint)


@app.route("/logout", methods=["POST"])
def logout():
    if g.user:
        auth_db.log_activity(g.user["id"], "logout")
    session.clear()
    return redirect(url_for("login"))


# --------------------------------------------------------------------------
# Unified model registry: every selectable option across every route maps
# to an entry here. "all_four" is handled separately (it calls
# predict_ensemble instead of a single predict_ripeness function).
#
# "yolo_pure" is a 5th, fully independent predictor (YOLOv8 classification
# head, no SVM/hand-crafted features involved). It slots into this same
# dict because yolo_cls_predict.predict_ripeness() matches the exact
# (label, confidence, bbox, cleaned_img, proba_dict) signature every member
# uses -- but it is deliberately NOT folded into predict_ensemble.py's
# soft vote. That ensemble's whole design point is averaging 4
# complementary hand-crafted feature pairs; YOLO's CNN features aren't
# that, and naively equal-weighting a 5th very-different predictor into it
# would need the (currently unimplemented) weighted-voting work first.
# yolo_pure stays visible as its own always-selectable option instead.
# --------------------------------------------------------------------------
PREDICTORS = {
    "ab": {
        "fn": m1_predict_ripeness,
        "not_fruit_err": M1NotAFruitError,
        "label": "Ensemble AB (Colour + Shape)",
    },
    "bc": {
        "fn": m2_predict_ripeness,
        "not_fruit_err": M2NotAFruitError,
        "label": "Ensemble BC (Shape + Texture)",
    },
    "cd": {
        "fn": m3_predict_ripeness,
        "not_fruit_err": M3NotAFruitError,
        "label": "Ensemble CD (Texture + Gabor)",
    },
    "da": {
        "fn": m4_predict_ripeness,
        "not_fruit_err": M4NotAFruitError,
        "label": "Ensemble DA (Gabor + Colour)",
    },
    "yolo_pure": {
        "fn": yolo_pure_predict_ripeness,
        "not_fruit_err": YoloPureNotAFruitError,
        "label": "YOLOv8 Classification (pure CNN, no SVM)",
    },
    "merged_1_4": {
        "fn": m14_predict_ripeness,
        "not_fruit_err": M14NotAFruitError,
        "label": "Merged 1+4 (Colour + Shape + Gabor, single SVM)",
    },
    "m14v2": {
        "fn": m14v2_predict_ripeness,
        "not_fruit_err": M14v2NotAFruitError,
        "label": "Merged 1+4 v2 (Colour + Shape + Gabor + Texture, single SVM)",
    },
}
MODEL_CHOICES = list(PREDICTORS.keys()) + ["all_four"]

# Model keys that don't follow the "ensemble_<key>" DB/chart tag convention
# (used by ab/bc/cd/da, the 4 members that make up the all_four soft-vote).
_MEMBER_TAG_OVERRIDES = {
    "yolo_pure": "yolo_pure",
    "merged_1_4": "merged_1_4",
    "m14v2": "m14v2",
}


def _member_tag(model_key):
    """DB `member` column value + chart filename tag for a given model key."""
    return _MEMBER_TAG_OVERRIDES.get(model_key, f"ensemble_{model_key}")


@app.route("/outputs/<path:filename>")
def outputs_file(filename):
    """Serves annotated images and charts saved under outputs/."""
    return send_from_directory(OUTPUTS_DIR, filename)


# --------------------------------------------------------------------------
# Dashboard (overview) + Fruit Classification (was "/")
# --------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def dashboard_home():
    stats = get_stats()
    stats_today = get_stats_since(hours=24)
    recent = get_recent(limit=5)

    fruit_chart = generate_fruit_breakdown_chart(None, file_tag="all")
    history_chart = generate_history_chart(None, file_tag="all")
    confidence_chart = generate_confidence_trend_chart(None, file_tag="all")

    return render_template(
        "dashboard.html",
        stats=stats,
        stats_today=stats_today,
        recent=recent,
        fruit_chart=fruit_chart is not None,
        history_chart=history_chart is not None,
        confidence_chart=confidence_chart is not None,
        chart_tag="all",
        active_page="dashboard",
    )


@app.route("/classify", methods=["GET"])
def classify():
    default_model = auth_db.get_setting("default_model", "ab")
    return render_template(
        "classify.html", fruits=FRUITS, models=MODEL_CHOICES, predictors=PREDICTORS,
        default_model=default_model, active_page="classify",
    )


def _is_flagged(confidence_pct):
    """True if confidence falls below the admin-configured review threshold
    (Settings > Vision Model Configuration). Threshold of 0 disables flagging."""
    try:
        threshold = float(auth_db.get_setting("confidence_threshold", "0"))
    except (TypeError, ValueError):
        threshold = 0
    return threshold > 0 and confidence_pct < threshold


def _save_annotated(img, bbox, filename):
    """Draws the detected bbox on a copy of the image and saves it under
    outputs/annotated/. Returns the relative path, or None if there's no
    bbox to draw."""
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    annotated = img.copy()
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 200, 0), 3)
    annotated_dir = os.path.join(OUTPUTS_DIR, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)
    cv2.imwrite(os.path.join(annotated_dir, filename), annotated)
    return f"annotated/{filename}"


def _analyse_surface_and_save(img, bbox, filename):
    """Run the shared post-prediction analysis once and save its overlay."""
    result = analyze_surface(img, bbox=bbox)
    result["surface_path"] = None
    overlay = result.get("surface_overlay")
    if overlay is not None:
        safe_name = secure_filename(filename) or "fruit.jpg"
        stem, extension = os.path.splitext(safe_name)
        extension = extension if extension.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
        surface_dir = os.path.join(OUTPUTS_DIR, "surface")
        os.makedirs(surface_dir, exist_ok=True)
        surface_name = f"{stem}_surface{extension}"
        if cv2.imwrite(os.path.join(surface_dir, surface_name), overlay):
            result["surface_path"] = f"surface/{surface_name}"
    return result


def _surface_db_fields(surface):
    return {
        "fruit_area_px": surface.get("fruit_area_px") or None,
        "blemish_area_px": surface.get("blemish_area_px") if surface.get("blemish_percentage") is not None else None,
        "blemish_percentage": surface.get("blemish_percentage"),
        "quality_grade": surface.get("quality_grade") if surface.get("blemish_percentage") is not None else None,
        "surface_path": surface.get("surface_path"),
    }


def _surface_payload(surface):
    return {
        "fruit_area_px": surface.get("fruit_area_px"),
        "blemish_area_px": surface.get("blemish_area_px"),
        "blemish_percentage": surface.get("blemish_percentage"),
        "quality_grade": surface.get("quality_grade", "Unknown"),
        "surface_path": surface.get("surface_path"),
        "surface_analysis_error": surface.get("surface_analysis_error"),
    }


# --------------------------------------------------------------------------
# Multi-fruit-per-photo batch detection ("this photo may contain multiple
# fruits") -- separate from every member's single-largest-contour detect().
# See core_modules/multi_fruit_detect.py for the YOLO localisation step.
# --------------------------------------------------------------------------
_MULTI_FRUIT_BOX_COLOURS = {"ripe": (0, 200, 0), "unripe": (0, 165, 255), "rotten": (0, 0, 200)}


def _classify_multi_fruit_photo(img, fruit_type, classify_crop, filename):
    """
    Detects every fruit_type box in one photo, classifies each crop via
    classify_crop(crop, fruit_type) -> (label, confidence_0_to_1) (or
    (None, None) on a per-crop failure), draws every box on one annotated
    image, and returns a dict with the majority label/confidence, a
    {label: count} breakdown, the annotated path, and the per-fruit detail
    list -- or None if this photo isn't a multi-fruit candidate (mango,
    or no boxes found), signaling the caller to fall back to the existing
    single-fruit path.
    """
    if not supports_multi_fruit(fruit_type):
        return None
    boxes = detect_fruit_boxes(img, fruit_type)
    if not boxes:
        return None

    annotated = img.copy()
    per_fruit = []
    for (x0, y0, x1, y1) in boxes:
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        try:
            label, confidence = classify_crop(crop, fruit_type)
        except Exception:
            label, confidence = None, None
        if label is None:
            continue

        per_fruit.append({"bbox": (x0, y0, x1, y1), "label": label, "confidence": confidence})
        colour = _MULTI_FRUIT_BOX_COLOURS.get(label, (200, 200, 200))
        cv2.rectangle(annotated, (x0, y0), (x1, y1), colour, 3)
        cv2.putText(annotated, f"{label} {confidence * 100:.0f}%", (x0, max(y0 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    if not per_fruit:
        return None

    breakdown = Counter(r["label"] for r in per_fruit)
    majority_label, _ = breakdown.most_common(1)[0]
    majority_confidences = [r["confidence"] for r in per_fruit if r["label"] == majority_label]
    majority_confidence_pct = round(sum(majority_confidences) / len(majority_confidences) * 100, 1)

    annotated_dir = os.path.join(OUTPUTS_DIR, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)
    cv2.imwrite(os.path.join(annotated_dir, filename), annotated)

    return {
        "label": majority_label,
        "confidence": majority_confidence_pct,
        "breakdown": dict(breakdown),
        "annotated_path": f"annotated/{filename}",
        "per_fruit": per_fruit,
        "fruit_count": len(per_fruit),
    }


def _ensemble_crop_classify(crop, fruit_type):
    """classify_crop adapter for _classify_multi_fruit_photo: predict_ensemble
    returns confidence on a 0-100 scale, but the multi-fruit helper's
    contract (matching every member's predict_ripeness()) expects 0-1."""
    label, confidence_pct, _per_member, _bbox = predict_ensemble(crop, fruit_type)
    return label, confidence_pct / 100


def _single_model_crop_classify(entry):
    """classify_crop adapter for a single PREDICTORS[model_choice] entry."""
    def _classify(crop, fruit_type):
        label, confidence, _bbox, _cleaned, _proba = entry["fn"](crop, fruit_type)
        return label, confidence
    return _classify


# --------------------------------------------------------------------------
# Single-image prediction
# --------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """Legacy single-model endpoint, kept for backwards compatibility.
    Defaults to the AB model. New frontend code should use
    /predict_unified instead, which supports all 4 models + the ensemble."""
    fruit_type = request.form.get("fruit", "apple")
    files = request.files.getlist("image")
    if not files or files[0].filename == "":
        return {"error": "No image uploaded"}, 400

    entry = PREDICTORS["ab"]
    results = []
    for f in files:
        path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(path)
        img = cv2.imread(path)
        if img is None:
            return {"error": "Uploaded image could not be read", "filename": f.filename}, 400
        try:
            input_validation = validate_selected_fruit(img, fruit_type)
        except FruitValidationError as e:
            return {"error": str(e), "filename": f.filename}, 422

        t0 = time.perf_counter()
        try:
            label, confidence, bbox, cleaned, proba_dict = entry["fn"](img, fruit_type)
        except entry["not_fruit_err"] as e:
            return {"error": str(e), "filename": f.filename}, 422
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        annotated_rel = _save_annotated(img, bbox, f.filename)
        surface = _analyse_surface_and_save(img, bbox, f.filename)
        confidence_pct = round(confidence * 100, 1)

        log_result(
            member=_member_tag("ab"),
            fruit=fruit_type,
            label=label,
            confidence=confidence_pct,
            filename=f.filename,
            annotated_path=annotated_rel,
            source="predict",
            **_surface_db_fields(surface),
            user_id=g.user["id"] if g.user else None,
            latency_ms=latency_ms,
            flagged=_is_flagged(confidence_pct),
        )

        results.append({
            "filename": f.filename,
            "fruit": fruit_type,
            "ripeness": label,
            "confidence": round(confidence * 100, 1),
            "input_validation": input_validation,
            **_surface_payload(surface),
        })

    if len(results) == 1:
        return results[0]
    return {"results": results}


@app.route("/predict_unified", methods=["POST"])
def predict_unified():
    """
    Single entry point for the model-selector UI on index.html.
    Accepts a 'model' field: one of "ab", "bc", "cd", "da", "yolo_pure",
    "all_four". Returns a consistent JSON shape regardless of which model ran.
    """
    fruit_type = request.form.get("fruit", "apple")
    model_choice = request.form.get("model", "ab")
    files = request.files.getlist("image")
    if not files or files[0].filename == "":
        return {"error": "No image uploaded"}, 400

    f = files[0]
    path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(path)
    img = cv2.imread(path)
    if img is None:
        return {"error": "Uploaded image could not be read", "filename": f.filename}, 400
    try:
        input_validation = validate_selected_fruit(img, fruit_type)
    except FruitValidationError as e:
        return {"error": str(e), "filename": f.filename}, 422

    if model_choice == "all_four":
        t0 = time.perf_counter()
        try:
            label, confidence, per_member, bbox = predict_ensemble(img, fruit_type)
        except RuntimeError as e:
            return {"error": str(e), "filename": f.filename}, 422
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        annotated_rel = _save_annotated(img, bbox, f.filename)
        surface = _analyse_surface_and_save(img, bbox, f.filename)

        log_result(
            member="ensemble_all_four",
            fruit=fruit_type,
            label=label,
            confidence=confidence,
            filename=f.filename,
            annotated_path=annotated_rel,
            source="predict_unified",
            **_surface_db_fields(surface),
            user_id=g.user["id"] if g.user else None,
            latency_ms=latency_ms,
            flagged=_is_flagged(confidence),
        )

        return {
            "model": "all_four",
            "fruit": fruit_type,
            "ripeness": label,
            "confidence": confidence,
            "per_member": per_member,
            "input_validation": input_validation,
            **_surface_payload(surface),
        }

    entry = PREDICTORS.get(model_choice)
    if not entry:
        return {"error": f"Unknown model '{model_choice}'"}, 400

    t0 = time.perf_counter()
    try:
        label, confidence, bbox, cleaned, proba_dict = entry["fn"](img, fruit_type)
    except entry["not_fruit_err"] as e:
        return {"error": str(e), "filename": f.filename}, 422
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    annotated_rel = _save_annotated(img, bbox, f.filename)
    surface = _analyse_surface_and_save(img, bbox, f.filename)
    confidence_pct = round(confidence * 100, 1)

    log_result(
        member=_member_tag(model_choice),
        fruit=fruit_type,
        label=label,
        confidence=confidence_pct,
        filename=f.filename,
        annotated_path=annotated_rel,
        source="predict_unified",
        **_surface_db_fields(surface),
        user_id=g.user["id"] if g.user else None,
        latency_ms=latency_ms,
        flagged=_is_flagged(confidence_pct),
    )

    return {
        "model": model_choice,
        "fruit": fruit_type,
        "ripeness": label,
        "confidence": confidence_pct,
        "per_member": None,
        "proba": {cls: round(p * 100, 1) for cls, p in proba_dict.items()},
        "input_validation": input_validation,
        **_surface_payload(surface),
    }


# --------------------------------------------------------------------------
# Data Analysis Dashboard — batch upload + per-member analytics.
# Every member (ab/bc/cd/da/yolo_pure) shares this same route/template; the
# only difference is which model_choice was posted / is in the URL.
# --------------------------------------------------------------------------
@app.route("/dashboard/<model_key>", methods=["GET"])
def dashboard(model_key):
    if model_key == ALL_FOUR_KEY:
        member_filter = "ensemble_all_four"
        model_label = ALL_FOUR_LABEL
    else:
        entry = PREDICTORS.get(model_key)
        if not entry:
            flash(f"Unknown model '{model_key}'.")
            return redirect(url_for("classify"))
        member_filter = _member_tag(model_key)
        model_label = entry["label"]

    history_chart_path = generate_history_chart(member_filter, file_tag=model_key)

    return render_template(
        "member_dashboard.html",
        results=[],
        chart=False,
        history_chart=history_chart_path is not None,
        results_json=None,
        model_choice=model_key,
        model_label=model_label,
        history_member_tag=member_filter,
        predictors=PREDICTORS_WITH_ENSEMBLE,
        fruits=FRUITS,
        active_page="classify",
    )

ALL_FOUR_LABEL = "Ensemble (All 4 members, soft-voted)"
ALL_FOUR_KEY = "all_four"

# Same PREDICTORS dict, plus a display-only entry for the ensemble, so
# dashboard/analyse templates can list it alongside ab/bc/cd/da/yolo_pure
# without giving it a fake "fn"/"not_fruit_err" (predict_ensemble has a
# different signature and is called directly instead).
PREDICTORS_WITH_ENSEMBLE = dict(PREDICTORS)
PREDICTORS_WITH_ENSEMBLE[ALL_FOUR_KEY] = {"label": ALL_FOUR_LABEL}

@app.route("/analyse", methods=["POST"])
def analyse():
    """Multi-image batch analysis. `model` is one of ab/bc/cd/da/yolo_pure for
    a single model, or 'all_four' to run the full soft-voted ensemble on every image."""
    fruit_type = request.form.get("fruit_type", "apple")
    model_choice = request.form.get("model", "ab")
    files = request.files.getlist("images")
    # "This photo may contain multiple fruits" -- see core_modules/multi_fruit_detect.py.
    # Only apple/banana/orange support it (no COCO mango class); a mango
    # photo silently keeps using the existing single-fruit path below.
    multi_fruit = request.form.get("multi_fruit") == "on"

    if model_choice == ALL_FOUR_KEY:
        member_tag = "ensemble_all_four"
        model_label = ALL_FOUR_LABEL
        results = []

        for f in files:
            path = os.path.join(UPLOAD_DIR, f.filename)
            f.save(path)
            img = cv2.imread(path)
            if img is None:
                results.append({"filename": f.filename, "label": None, "confidence": None, "error": "Uploaded image could not be read"})
                continue
            try:
                input_validation = validate_selected_fruit(img, fruit_type)
            except FruitValidationError as e:
                results.append({"filename": f.filename, "label": None, "confidence": None, "error": str(e)})
                continue

            if multi_fruit:
                multi = _classify_multi_fruit_photo(img, fruit_type, _ensemble_crop_classify, f.filename)
                if multi:
                    log_result(
                        member=member_tag,
                        fruit=fruit_type,
                        label=multi["label"],
                        confidence=multi["confidence"],
                        filename=f.filename,
                        annotated_path=multi["annotated_path"],
                        source="analyse_multi_fruit",
                        user_id=g.user["id"] if g.user else None,
                        flagged=_is_flagged(multi["confidence"]),
                        detection_breakdown=json.dumps(multi["breakdown"]),
                    )
                    results.append({
                        "filename": f.filename,
                        "fruit": fruit_type,
                        "label": multi["label"],
                        "confidence": multi["confidence"],
                        "annotated_path": multi["annotated_path"],
                        "input_validation": input_validation,
                        "detection_breakdown": multi["breakdown"],
                        "fruit_count": multi["fruit_count"],
                        **_surface_payload({}),
                    })
                    continue

            t0 = time.perf_counter()
            try:
                label, confidence, per_member, bbox = predict_ensemble(img, fruit_type)
            except RuntimeError as e:
                results.append({"filename": f.filename, "label": None, "confidence": None, "error": str(e)})
                continue
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            annotated_rel = _save_annotated(img, bbox, f.filename)
            surface = _analyse_surface_and_save(img, bbox, f.filename)

            log_result(
                member=member_tag,
                fruit=fruit_type,
                label=label,
                confidence=confidence,
                filename=f.filename,
                annotated_path=annotated_rel,
                source="analyse",
                **_surface_db_fields(surface),
                user_id=g.user["id"] if g.user else None,
                latency_ms=latency_ms,
                flagged=_is_flagged(confidence),
            )

            results.append({
                "filename": f.filename,
                "fruit": fruit_type,
                "label": label,
                "confidence": confidence,
                "annotated_path": annotated_rel,
                "multi_fruit_fallback": multi_fruit,
                "per_member": per_member,
                "input_validation": input_validation,
                **_surface_payload(surface),
            })
    else:
        entry = PREDICTORS.get(model_choice)
        if not entry:
            flash(f"Unknown model '{model_choice}'.")
            return redirect(url_for("classify"))

        member_tag = _member_tag(model_choice)
        model_label = entry["label"]
        results = []

        for f in files:
            path = os.path.join(UPLOAD_DIR, f.filename)
            f.save(path)
            img = cv2.imread(path)
            if img is None:
                results.append({"filename": f.filename, "label": None, "confidence": None, "error": "Uploaded image could not be read"})
                continue
            try:
                input_validation = validate_selected_fruit(img, fruit_type)
            except FruitValidationError as e:
                results.append({"filename": f.filename, "label": None, "confidence": None, "error": str(e)})
                continue

            if multi_fruit:
                multi = _classify_multi_fruit_photo(img, fruit_type, _single_model_crop_classify(entry), f.filename)
                if multi:
                    confidence_pct = multi["confidence"]
                    log_result(
                        member=member_tag,
                        fruit=fruit_type,
                        label=multi["label"],
                        confidence=confidence_pct,
                        filename=f.filename,
                        annotated_path=multi["annotated_path"],
                        source="analyse_multi_fruit",
                        user_id=g.user["id"] if g.user else None,
                        flagged=_is_flagged(confidence_pct),
                        detection_breakdown=json.dumps(multi["breakdown"]),
                    )
                    results.append({
                        "filename": f.filename,
                        "fruit": fruit_type,
                        "label": multi["label"],
                        "confidence": confidence_pct,
                        "annotated_path": multi["annotated_path"],
                        "input_validation": input_validation,
                        "detection_breakdown": multi["breakdown"],
                        "fruit_count": multi["fruit_count"],
                        **_surface_payload({}),
                    })
                    continue

            t0 = time.perf_counter()
            try:
                label, confidence, bbox, cleaned, proba_dict = entry["fn"](img, fruit_type)
            except entry["not_fruit_err"] as e:
                results.append({"filename": f.filename, "label": None, "confidence": None, "error": str(e)})
                continue
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            annotated_rel = _save_annotated(img, bbox, f.filename)
            surface = _analyse_surface_and_save(img, bbox, f.filename)
            confidence_pct = round(confidence * 100, 1)

            log_result(
                member=member_tag,
                fruit=fruit_type,
                label=label,
                confidence=confidence_pct,
                filename=f.filename,
                annotated_path=annotated_rel,
                source="analyse",
                **_surface_db_fields(surface),
                user_id=g.user["id"] if g.user else None,
                latency_ms=latency_ms,
                flagged=_is_flagged(confidence_pct),
            )

            results.append({
                "filename": f.filename,
                "fruit": fruit_type,
                "label": label,
                "confidence": confidence_pct,
                "annotated_path": annotated_rel,
                "multi_fruit_fallback": multi_fruit,
                "input_validation": input_validation,
                **_surface_payload(surface),
            })

    chart_path = generate_trend_chart(results, file_tag=model_choice) if results else None
    history_chart_path = generate_history_chart(member_tag, file_tag=model_choice)

    results_for_pdf = [
        {
            "filename": r.get("filename"),
            "fruit": r.get("fruit"),
            "label": r.get("label"),
            "confidence": r.get("confidence"),
            "image_path": os.path.join(OUTPUTS_DIR, r["annotated_path"]) if r.get("annotated_path") else None,
            "surface_image_path": os.path.join(OUTPUTS_DIR, r["surface_path"]) if r.get("surface_path") else None,
            "fruit_area_px": r.get("fruit_area_px"),
            "blemish_area_px": r.get("blemish_area_px"),
            "blemish_percentage": r.get("blemish_percentage"),
            "quality_grade": r.get("quality_grade", "Unknown"),
            "surface_analysis_error": r.get("surface_analysis_error"),
            "detection_breakdown": r.get("detection_breakdown"),
            "fruit_count": r.get("fruit_count"),
        }
        for r in results
    ]

    return render_template(
        "member_dashboard.html",
        results=results,
        chart=chart_path is not None,
        history_chart=history_chart_path is not None,
        results_json=json.dumps(results_for_pdf),
        model_choice=model_choice,
        model_label=model_label,
        history_member_tag=member_tag,
        predictors=PREDICTORS_WITH_ENSEMBLE,
        fruits=FRUITS,
        OUTPUTS_DIR=OUTPUTS_DIR,
        active_page="classify",
    )


# --------------------------------------------------------------------------
# PDF export (shared by every member's dashboard)
# --------------------------------------------------------------------------
@app.route("/extra_export_pdf", methods=["POST"])
def extra_export_pdf():
    label = request.form["label"]
    confidence = float(request.form["confidence"]) / 100
    image_path = request.form.get("image_path")
    model_tag = request.form.get("model_tag", "ab")
    breakdown_raw = request.form.get("detection_breakdown")
    surface_data = {
        "fruit": request.form.get("fruit"),
        "surface_image_path": request.form.get("surface_image_path"),
        "fruit_area_px": request.form.get("fruit_area_px", type=int),
        "blemish_area_px": request.form.get("blemish_area_px", type=int),
        "blemish_percentage": request.form.get("blemish_percentage", type=float),
        "quality_grade": request.form.get("quality_grade") or "Unknown",
        "detection_breakdown": json.loads(breakdown_raw) if breakdown_raw else None,
        "fruit_count": request.form.get("fruit_count", type=int),
    }
    out_path = generate_pdf_report(
        image_path, label, confidence, model_tag=model_tag, surface_data=surface_data
    )
    return send_from_directory(os.path.dirname(out_path), os.path.basename(out_path), as_attachment=True)


@app.route("/extra_export_pdf_batch", methods=["POST"])
def extra_export_pdf_batch():
    """Exports every result currently shown on a dashboard into ONE combined PDF."""
    try:
        results = json.loads(request.form["results_json"])
    except (KeyError, json.JSONDecodeError):
        return {"error": "No results to export."}, 400

    model_tag = request.form.get("model_tag", "ab")
    out_path = generate_pdf_report_batch(results, model_tag=model_tag)
    return send_from_directory(os.path.dirname(out_path), os.path.basename(out_path), as_attachment=True)


@app.route("/history/<int:record_id>/export_pdf", methods=["POST"])
def history_export_pdf(record_id):
    """Exports a single already-logged Harvest Record as a PDF, reusing the
    same generate_pdf_report() the classify page uses right after a fresh
    prediction -- the only difference is the image/surface paths and
    metrics come from the stored DB row instead of an in-request result."""
    record = get_by_id(record_id)
    if not record:
        flash("That record no longer exists.")
        return redirect(url_for("history"))

    image_path = os.path.join(OUTPUTS_DIR, record["annotated_path"]) if record.get("annotated_path") else None
    surface_image_path = os.path.join(OUTPUTS_DIR, record["surface_path"]) if record.get("surface_path") else None
    breakdown_raw = record.get("detection_breakdown")
    surface_data = {
        "fruit": record.get("fruit"),
        "surface_image_path": surface_image_path,
        "fruit_area_px": record.get("fruit_area_px"),
        "blemish_area_px": record.get("blemish_area_px"),
        "blemish_percentage": record.get("blemish_percentage"),
        "quality_grade": record.get("quality_grade") or "Unknown",
        "detection_breakdown": json.loads(breakdown_raw) if breakdown_raw else None,
    }
    out_path = generate_pdf_report(
        image_path, record["label"], record["confidence"] / 100,
        model_tag=record["member"], surface_data=surface_data,
    )

    if g.user:
        auth_db.log_activity(g.user["id"], "export_pdf", detail=f"record {record_id}")

    return send_from_directory(os.path.dirname(out_path), os.path.basename(out_path), as_attachment=True)


# --------------------------------------------------------------------------
# History (global — every member logs into the same table)
# --------------------------------------------------------------------------
def _valid_date_arg(name):
    """Reads a YYYY-MM-DD query arg, ignoring it if malformed rather than
    500ing on a hand-edited URL."""
    raw = request.args.get(name) or None
    if raw is None:
        return None
    try:
        datetime.strptime(raw, "%Y-%m-%d")
    except ValueError:
        return None
    return raw


@app.route("/history")
def history():
    fruit_filter = request.args.get("fruit") or None
    member_filter = request.args.get("member") or None
    date_from = _valid_date_arg("date_from")
    date_to = _valid_date_arg("date_to")
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1

    rows, total = get_paginated(
        member=member_filter, fruit=fruit_filter, date_from=date_from, date_to=date_to,
        page=page, per_page=HISTORY_PAGE_SIZE,
    )
    total_pages = max(1, math.ceil(total / HISTORY_PAGE_SIZE))
    page = min(page, total_pages)

    # Parse the multi-fruit breakdown JSON here so the template can just
    # iterate a dict instead of needing a JSON filter.
    for row in rows:
        raw = row.get("detection_breakdown")
        row["detection_breakdown"] = json.loads(raw) if raw else None

    # member_options = [_member_tag(k) for k in PREDICTORS] + ["ensemble_all_four"] do not remove
    # member_options = [_member_tag(k) for k in PREDICTORS] + ["ensemble_all_four", "realtime_yolo"] do not remove
    member_options = [_member_tag(k) for k in PREDICTORS] + ["ensemble_all_four", "realtime_yolo", "yolo_pure_realtime"]

    stats = get_stats(member=member_filter, fruit=fruit_filter, date_from=date_from, date_to=date_to)

    return render_template(
        "history.html",
        results=rows,
        fruit_filter=fruit_filter,
        member_filter=member_filter,
        date_from=date_from,
        date_to=date_to,
        member_options=member_options,
        fruits=FRUITS,
        page=page,
        total_pages=total_pages,
        total=total,
        stats=stats,
        active_page="history",
    )


@app.route("/history/export.csv")
def history_export_csv():
    fruit_filter = request.args.get("fruit") or None
    member_filter = request.args.get("member") or None
    date_from = _valid_date_arg("date_from")
    date_to = _valid_date_arg("date_to")
    rows = get_all_results(member=member_filter, fruit=fruit_filter, date_from=date_from, date_to=date_to)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "created_at", "member", "fruit", "label", "confidence",
        "source", "flagged", "latency_ms", "filename",
    ])
    for r in rows:
        writer.writerow([
            r["id"], r["created_at"], r["member"], r["fruit"], r["label"], r["confidence"],
            r["source"], r.get("flagged", 0), r.get("latency_ms", ""), r["filename"] or "",
        ])

    if g.user:
        auth_db.log_activity(g.user["id"], "export_csv", detail=f"{len(rows)} records")

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=harvest_records.csv"},
    )


@app.route("/history/<int:record_id>/edit", methods=["GET", "POST"])
def history_edit(record_id):
    record = get_by_id(record_id)
    if not record:
        flash("That record no longer exists.")
        return redirect(url_for("history"))

    if request.method == "POST":
        update_result(
            record_id,
            fruit=request.form.get("fruit"),
            label=request.form.get("label"),
            confidence=float(request.form["confidence"]) if request.form.get("confidence") else None,
            source=request.form.get("source"),
        )
        flash("Record updated.")
        return redirect(url_for("history"))

    return render_template(
        "history_edit.html", record=record, fruits=FRUITS, classes=RIPENESS_CLASSES,
        active_page="history",
    )


@app.route("/history/<int:record_id>/delete", methods=["POST"])
def history_delete(record_id):
    deleted = delete_result(record_id)
    flash("Record deleted." if deleted else "Record not found.")
    return redirect(url_for("history", page=request.form.get("page", 1)))


# --------------------------------------------------------------------------
# Global analytics dashboard (all-time, all members)
# --------------------------------------------------------------------------
@app.route("/analytics")
def analytics():
    stats = get_stats(None)
    fruit_chart = generate_fruit_breakdown_chart(None, file_tag="all")
    confidence_chart = generate_confidence_trend_chart(None, file_tag="all")
    history_chart = generate_history_chart(None, file_tag="all")

    return render_template(
        "analytics_dashboard.html",
        stats=stats,
        fruit_chart=fruit_chart is not None,
        confidence_chart=confidence_chart is not None,
        history_chart=history_chart is not None,
        active_page="dashboard",
    )


# --------------------------------------------------------------------------
# Training report (per member)
# --------------------------------------------------------------------------
@app.route("/training-report/<model_key>")
@app.route("/training-report", defaults={"model_key": "ab"})
def training_report(model_key):
    entry = PREDICTORS.get(model_key)
    if not entry:
        flash(f"Unknown model '{model_key}'.")
        return redirect(url_for("classify"))

    model_training_dir = os.path.join(TRAINING_DIR, model_key)
    graphs = []
    for fruit in FRUITS:
        cm_path = os.path.join(model_training_dir, f"{fruit}_confusion_matrix.png")
        dist_path = os.path.join(model_training_dir, f"{fruit}_class_distribution.png")
        if os.path.exists(cm_path) or os.path.exists(dist_path):
            graphs.append({
                "fruit": fruit,
                "confusion_matrix": f"{fruit}_confusion_matrix.png" if os.path.exists(cm_path) else None,
                "class_distribution": f"{fruit}_class_distribution.png" if os.path.exists(dist_path) else None,
            })
    summary_exists = os.path.exists(os.path.join(model_training_dir, "accuracy_summary.png"))

    meta_path = os.path.join(model_training_dir, "training_meta.json")
    training_time_display = None
    per_fruit_time_display = {}
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)

        def _fmt(seconds):
            if seconds is None:
                return "—"
            minutes, secs = divmod(seconds, 60)
            return f"{int(minutes)}m {secs:.1f}s" if minutes >= 1 else f"{secs:.1f}s"

        training_time_display = _fmt(meta.get("total_seconds"))
        per_fruit_time_display = {f: _fmt(s) for f, s in meta.get("per_fruit_seconds", {}).items()}

    return render_template(
        "training_report.html",
        graphs=graphs,
        summary_exists=summary_exists,
        training_time=training_time_display,
        per_fruit_time=per_fruit_time_display,
        model_key=model_key,
        model_label=entry["label"],
        predictors=PREDICTORS,
        active_page="classify",
    )


# --------------------------------------------------------------------------
# Admin Panel (admin-only): model registry, personnel access, activity log.
# --------------------------------------------------------------------------
@app.route("/admin")
@admin_required
def admin_panel():
    stats_24h = get_stats_since(hours=24)

    registry = {}
    for key, entry in PREDICTORS.items():
        registry[key] = {**entry, "stats": get_stats(member=_member_tag(key))}

    return render_template(
        "admin.html",
        stats_24h=stats_24h,
        predictors=registry,
        users=auth_db.list_users(),
        activity=auth_db.get_recent_activity(limit=12),
        active_page="admin",
    )


@app.route("/admin/users/invite", methods=["POST"])
@admin_required
def admin_invite_user():
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    role = request.form.get("role", "farmer")

    if not name or not email:
        flash("Name and email are required.")
        return redirect(url_for("admin_panel"))

    temp_password = auth_db.generate_temp_password()
    try:
        auth_db.create_user(name, email, temp_password, role=role)
    except Exception:
        flash(f"Could not invite {email} — that email may already be in use.")
        return redirect(url_for("admin_panel"))

    auth_db.log_activity(g.user["id"], "invite_user", detail=email)
    flash(f"Invited {name} ({email}). Temporary password: {temp_password}")
    return redirect(url_for("admin_panel"))


@app.route("/admin/users/<int:user_id>/role", methods=["POST"])
@admin_required
def admin_update_role(user_id):
    role = request.form.get("role", "farmer")
    if user_id == g.user["id"] and role != "admin" and auth_db.admin_count() <= 1:
        flash("Can't demote the only remaining admin.")
        return redirect(url_for("admin_panel"))

    auth_db.update_user_role(user_id, role)
    auth_db.log_activity(g.user["id"], "change_role", detail=f"user {user_id} -> {role}")
    flash("Role updated.")
    return redirect(url_for("admin_panel"))


@app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    if user_id == g.user["id"]:
        flash("You can't remove your own account.")
        return redirect(url_for("admin_panel"))

    target = auth_db.get_user_by_id(user_id)
    if target and target["role"] == "admin" and auth_db.admin_count() <= 1:
        flash("Can't remove the only remaining admin.")
        return redirect(url_for("admin_panel"))

    auth_db.delete_user(user_id)
    auth_db.log_activity(g.user["id"], "remove_user", detail=target["email"] if target else str(user_id))
    flash("User removed.")
    return redirect(url_for("admin_panel"))


# --------------------------------------------------------------------------
# Settings: default model + confidence threshold, account security.
# --------------------------------------------------------------------------
@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        form = request.form.get("form")

        if form == "model_config":
            auth_db.set_setting("default_model", request.form.get("default_model", "ab"))
            auth_db.set_setting("confidence_threshold", request.form.get("confidence_threshold", "0"))
            auth_db.log_activity(g.user["id"], "update_settings")
            flash("Configuration saved.")

        elif form == "password":
            current = request.form.get("current_password", "")
            new = request.form.get("new_password", "")
            if not auth_db.verify_login(g.user["email"], current):
                flash("Current password is incorrect.")
            elif len(new) < 6:
                flash("New password must be at least 6 characters.")
            else:
                auth_db.set_password(g.user["id"], new)
                auth_db.log_activity(g.user["id"], "change_password")
                flash("Password updated.")

        return redirect(url_for("settings"))

    return render_template(
        "settings.html",
        predictors=PREDICTORS,
        settings=auth_db.get_all_settings(),
        active_page="settings",
    )


# --------------------------------------------------------------------------
# Profile: current user's own info, stats, and recent activity.
# --------------------------------------------------------------------------
@app.route("/profile", methods=["GET", "POST"])
def profile():
    if request.method == "POST" and request.form.get("form") == "name":
        name = request.form.get("name", "").strip()
        if name:
            auth_db.update_user_name(g.user["id"], name)
            auth_db.log_activity(g.user["id"], "update_profile")
            flash("Profile updated.")
        return redirect(url_for("profile"))

    my_stats = get_stats(user_id=g.user["id"])
    my_activity = auth_db.get_recent_activity(limit=8, user_id=g.user["id"])

    return render_template(
        "profile.html",
        my_stats=my_stats,
        my_activity=my_activity,
        active_page="profile",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
