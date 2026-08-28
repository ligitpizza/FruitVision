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
# --- Merged 1+4 v3 (same 4-feature set as v2, but detection combines
#     member 1's Otsu box + member 4's HSV-saturation box (union) and
#     calibration uses member 4's deskew; see member_apps/merged_member_1_4_v3) --
sys.path.append(os.path.join(MEMBER_APPS_DIR, "merged_member_1_4_v3"))

from member_apps.member_1_ab.m1_predict import predict_ripeness as m1_predict_ripeness, NotAFruitError as M1NotAFruitError
from member_apps.member_2_bc.m2_predict import predict_ripeness as m2_predict_ripeness, NotAFruitError as M2NotAFruitError
from member_apps.member_3_cd.m3_predict import predict_ripeness as m3_predict_ripeness, NotAFruitError as M3NotAFruitError
from member_apps.member_4_da.m4_predict import predict_ripeness as m4_predict_ripeness, NotAFruitError as M4NotAFruitError
from member_apps.merged_member_1_4.m14_predict import predict_ripeness as m14_predict_ripeness, NotAFruitError as M14NotAFruitError
from member_apps.merged_member_1_4_v2.m14v2_predict import predict_ripeness as m14v2_predict_ripeness, NotAFruitError as M14v2NotAFruitError
from member_apps.merged_member_1_4_v3.m14v3_predict import predict_ripeness as m14v3_predict_ripeness, NotAFruitError as M14v3NotAFruitError

from pipeline.pure_yolo.yolo_cls_predict import predict_ripeness as yolo_pure_predict_ripeness, NotAFruitError as YoloPureNotAFruitError

# --- 4-member ensemble (soft-voting across all members) -----------------
from member_apps.predict_ensemble import predict_ensemble

# --- Shared infrastructure (used to live inside member_1_ab) -----------
from core_modules.pdf_report import generate_pdf_report, generate_pdf_report_batch, generate_stock_report_pdf
from core_modules.blemish_analysis import analyze_surface
from core_modules.marketability import estimate_marketability, average_member_probabilities, stock_eligible
from core_modules.fruit_validation import validate_selected_fruit, FruitValidationError
from core_modules.multi_fruit_detect import supports_multi_fruit, detect_fruit_boxes
from core_modules.mixed_fruit_m14 import analyze_mixed_fruit_m14
from core_modules.filter_photos import (
    FILTER_LABELS,
    ENSEMBLE_MEMBER_TO_MODEL_KEY,
    filter_photos_single,
    filter_photos_ensemble,
    pop_member_cleaned_images,
)
from core_modules import model_lab
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
    get_fruit_label_breakdown,
)
from database import auth_db, stock_db

FRUITS = ["apple", "banana", "orange", "mango"]
RIPENESS_CLASSES = ["ripe", "unripe", "rotten"]
HISTORY_PAGE_SIZE = 15
STOCK_PAGE_SIZE = 15
MARKETABILITY_PAGE_SIZE = 25

def _load_or_create_secret_key():
    """The secret key signs session cookies (not just flash()) -- a
    hardcoded value here would let anyone who reads the source forge a
    session cookie for any user_id, including an admin, without ever
    knowing a password. Persist a random one outside git instead."""
    import secrets as _secrets
    key_path = os.path.join(BASE_DIR, ".secret_key")
    if os.path.exists(key_path):
        with open(key_path, "r") as fh:
            key = fh.read().strip()
        if key:
            return key
    key = _secrets.token_hex(32)
    with open(key_path, "w") as fh:
        fh.write(key)
    return key


app = Flask(__name__)
app.secret_key = _load_or_create_secret_key()
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
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

    if g.user is not None and not g.user.get("is_active", 1):
        # Deactivated mid-session: drop them immediately instead of
        # honoring a still-valid session cookie for a disabled account.
        session.clear()
        g.user = None

    if request.path.startswith("/static/") or request.path.startswith("/outputs/") or request.path.startswith("/uploads/"):
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


def _scope_user_id():
    """Farmers only ever see their own predictions/stock; admins see
    everything across every account (system-wide oversight, same as the
    Admin Panel's stats). Returns the id to filter records by, or None for
    "no filter"."""
    if g.user and g.user["role"] != "admin":
        return g.user["id"]
    return None


def _owns_record(record):
    """True if the current user may view/edit/delete this history or stock
    row. Admins always can. A farmer can if it's theirs, or if it predates
    per-user tracking (user_id is NULL) so old data isn't suddenly
    inaccessible to everyone -- but never if it belongs to someone else."""
    if not g.user:
        return False
    if g.user["role"] == "admin":
        return True
    owner_id = record.get("user_id")
    return owner_id is None or owner_id == g.user["id"]


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
        flash("Invalid email or password.", "error")

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
    "m14v3": {
        "fn": m14v3_predict_ripeness,
        "not_fruit_err": M14v3NotAFruitError,
        "label": "Merged 1+4 v3 (Otsu+HSV union detect, deskew calibrate, Colour + Shape + Gabor + Texture)",
    },
}
MODEL_CHOICES = list(PREDICTORS.keys()) + ["all_four"]

# Model keys that don't follow the "ensemble_<key>" DB/chart tag convention
# (used by ab/bc/cd/da, the 4 members that make up the all_four soft-vote).
_MEMBER_TAG_OVERRIDES = {
    "yolo_pure": "yolo_pure",
    "merged_1_4": "merged_1_4",
    "m14v3": "m14v3",
    "m14v2": "m14v2",
    "m14v3": "m14v3",
}


def _member_tag(model_key):
    """DB `member` column value + chart filename tag for a given model key."""
    return _MEMBER_TAG_OVERRIDES.get(model_key, f"ensemble_{model_key}")


@app.route("/outputs/<path:filename>")
def outputs_file(filename):
    """Serves annotated images and charts saved under outputs/."""
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/uploads/<path:filename>")
def uploads_file(filename):
    """Serves the original user-uploaded photos saved under uploads/."""
    return send_from_directory(UPLOAD_DIR, filename)


def _marketability_db_fields(estimate):
    """Map a marketability payload to optional history columns."""
    return {
        "marketability_status": estimate.get("status"),
        "dispatch_priority": estimate.get("dispatch_priority"),
        "marketability_min_days": estimate.get("min_days"),
        "marketability_max_days": estimate.get("max_days"),
        "marketability_action": estimate.get("action"),
        "marketability_reliability": estimate.get("reliability"),
        "marketability_storage_assumption": estimate.get("storage_assumption"),
    }


def _marketability_for_record(record):
    """Return a current, display-only view of a historical scan estimate.

    New rows use the exact estimate stored at prediction time. Older rows are
    reconstructed from their unchanged label/confidence/surface fields. Day
    ranges count down from the scan date; expired estimates require re-scan
    instead of continuing to claim that old fruit is market-ready.
    """
    if record.get("marketability_status"):
        estimate = {
            "status": record.get("marketability_status"),
            "dispatch_priority": record.get("dispatch_priority") or "unknown",
            "min_days": record.get("marketability_min_days"),
            "max_days": record.get("marketability_max_days"),
            "action": record.get("marketability_action") or "Inspect this fruit before marketing.",
            "reliability": record.get("marketability_reliability") or "unavailable",
            "storage_assumption": record.get("marketability_storage_assumption"),
            "blemish_percentage": record.get("blemish_percentage"),
            "disclaimer": "Image-based operational estimate only; inspect fruit before sale or disposal.",
        }
    else:
        estimate = estimate_marketability(
            fruit=record.get("fruit"),
            ripeness=record.get("label"),
            confidence=record.get("confidence"),
            blemish_percentage=record.get("blemish_percentage"),
            quality_grade=record.get("quality_grade"),
        )

    estimate = dict(estimate)
    elapsed_days = 0
    try:
        scanned_at = datetime.fromisoformat(record.get("created_at"))
        elapsed_days = max(0, int((datetime.now() - scanned_at).total_seconds() // 86400))
    except (TypeError, ValueError):
        pass
    estimate["elapsed_days"] = elapsed_days

    min_days = estimate.get("min_days")
    max_days = estimate.get("max_days")
    if min_days is not None and max_days is not None and estimate.get("status") != "remove":
        min_days = max(0, int(min_days) - elapsed_days)
        max_days = max(0, int(max_days) - elapsed_days)
        if max_days == 0:
            estimate.update({
                "status": "inspect",
                "dispatch_priority": "urgent",
                "min_days": None,
                "max_days": None,
                "window": None,
                "reliability": "low",
                "action": "The estimate from this scan has expired. Re-scan or inspect before marketing.",
            })
        else:
            estimate["min_days"] = min_days
            estimate["max_days"] = max_days
            estimate["window"] = f"{min_days}-{max_days} days" if min_days != max_days else f"{max_days} days"
    elif min_days is not None and max_days is not None:
        estimate["window"] = "0 days"
    else:
        estimate["window"] = None
    return estimate


def _marketability_sort_key(record):
    estimate = record["marketability"]
    status_rank = {
        "remove": 0,
        "sort": 1,
        "isolate": 2,
        "inspect": 3,
        "ready": 4,
        "hold": 5,
    }
    priority_rank = {"remove": 0, "urgent": 1, "high": 2, "normal": 3, "unknown": 4}
    return (
        status_rank.get(estimate.get("status"), 5),
        priority_rank.get(estimate.get("dispatch_priority"), 5),
        estimate.get("max_days") if estimate.get("max_days") is not None else 999999,
        -int(record.get("id") or 0),
    )


def _review_for_record(record, breakdown, estimate):
    """Return the operator-review view without changing the model result."""
    stored_status = record.get("review_status")
    if stored_status in {"confirmed", "corrected"}:
        return {
            "status": stored_status,
            "fruit": record.get("review_fruit"),
            "label": record.get("review_label"),
            "reason": record.get("review_reason"),
            "reviewed_by": record.get("reviewed_by"),
            "reviewed_at": record.get("reviewed_at"),
            "triggers": [],
        }

    triggers = []
    if record.get("flagged"):
        triggers.append("Low confidence")
    if breakdown:
        present_classes = [label for label, count in breakdown.items() if int(count) > 0]
        if len(present_classes) > 1:
            triggers.append("Mixed classifications")
    if estimate.get("status") in {"remove", "inspect", "isolate"}:
        triggers.append("Handling decision requires inspection")
    elif estimate.get("dispatch_priority") in {"remove", "urgent"}:
        triggers.append("Urgent handling decision")

    return {
        "status": "needs_review" if triggers else "not_required",
        "fruit": None,
        "label": None,
        "reason": None,
        "reviewed_by": None,
        "reviewed_at": None,
        "triggers": triggers,
    }


def _decorate_marketability_rows(rows):
    decorated = []
    for row in rows:
        item = dict(row)
        raw_breakdown = item.get("detection_breakdown")
        if isinstance(raw_breakdown, str):
            try:
                breakdown = json.loads(raw_breakdown)
            except (TypeError, ValueError, json.JSONDecodeError):
                breakdown = None
        elif isinstance(raw_breakdown, dict):
            breakdown = raw_breakdown
        else:
            breakdown = None
        item["detection_breakdown"] = breakdown

        source = item.get("source") or ""
        if source == "analyse_mixed_fruit_m14":
            item["analysis_type"] = "YOLOv8n + M14 mixed fruit"
            item["analysis_kind"] = "mixed_fruit_m14"
        elif source == "analyse_multi_fruit":
            item["analysis_type"] = "Multi-fruit batch"
            item["analysis_kind"] = "multi_fruit_batch"
        elif source == "analyse":
            item["analysis_type"] = "Batch analysis"
            item["analysis_kind"] = "batch"
        elif source.startswith("realtime") or source.endswith("realtime"):
            item["analysis_type"] = "Real-time"
            item["analysis_kind"] = "realtime"
        else:
            item["analysis_type"] = "Single classification"
            item["analysis_kind"] = "single"

        estimate = _marketability_for_record(item)
        if breakdown:
            detected_count = sum(max(0, int(count)) for count in breakdown.values())
            present_classes = [label for label, count in breakdown.items() if int(count) > 0]
            item["detected_count"] = detected_count
            # One majority label cannot safely represent a genuinely mixed
            # group. Preserve every classifier result in the breakdown and
            # issue a sorting action only in the dashboard layer.
            if len(present_classes) > 1:
                estimate = dict(estimate)
                rotten_count = max(0, int(breakdown.get("rotten", 0)))
                ripe_count = max(0, int(breakdown.get("ripe", 0)))
                unripe_count = max(0, int(breakdown.get("unripe", 0)))
                action_parts = []
                if rotten_count:
                    action_parts.append(f"isolate {rotten_count} rotten")
                if ripe_count:
                    action_parts.append(f"dispatch {ripe_count} ripe")
                if unripe_count:
                    action_parts.append(f"hold {unripe_count} unripe")
                estimate.update({
                    "status": "sort",
                    "dispatch_priority": "urgent" if rotten_count else "high",
                    "min_days": None,
                    "max_days": None,
                    "window": None,
                    "reliability": "per-fruit",
                    "action": "Sort this mixed batch: " + ", ".join(action_parts) + ".",
                })
        else:
            item["detected_count"] = None
        item["marketability"] = estimate
        item["review"] = _review_for_record(item, breakdown, estimate)
        decorated.append(item)
    return sorted(decorated, key=_marketability_sort_key)


# --------------------------------------------------------------------------
# Dashboard (overview) + Fruit Classification (was "/")
# --------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def dashboard_home():
    scope = _scope_user_id()
    stats = get_stats(user_id=scope)
    stats_today = get_stats_since(hours=24, user_id=scope)
    recent = get_recent(limit=5, user_id=scope)

    confidence_chart = generate_confidence_trend_chart(None, file_tag="all")
    fruit_label_breakdown = get_fruit_label_breakdown(user_id=scope)
    marketability_rows = _decorate_marketability_rows(get_recent(limit=100, user_id=scope))
    marketability_alert_count = sum(
        row["marketability"].get("dispatch_priority") in {"urgent", "remove"}
        for row in marketability_rows
    )

    return render_template(
        "dashboard.html",
        stats=stats,
        stats_today=stats_today,
        recent=recent,
        fruits=FRUITS,
        fruit_label_breakdown_json=json.dumps(fruit_label_breakdown),
        by_label_json=json.dumps(stats["by_label"]),
        confidence_chart=confidence_chart is not None,
        marketability_alert_count=marketability_alert_count,
        chart_tag="all",
        active_page="dashboard",
    )


@app.route("/classify", methods=["GET"])
def classify():
    default_model = auth_db.get_setting("default_model", "ab")
    return render_template(
        "classify.html", fruits=FRUITS, models=MODEL_CHOICES, predictors=PREDICTORS,
        default_model=default_model, active_page="classify",
        filter_technique_labels_json=json.dumps(FILTER_LABELS),
        ensemble_member_map_json=json.dumps(ENSEMBLE_MEMBER_TO_MODEL_KEY),
        predictor_labels_json=json.dumps({k: v["label"] for k, v in PREDICTORS.items()}),
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
# Per-member filter-technique photos: after classification, render each
# filter module's intermediate image (not just its numeric features) from
# the calibrated `cleaned_img` every predict_ripeness() already returns.
# Generation logic itself lives in core_modules/filter_photos.py so the
# realtime/*.py trackers can reuse it too (they can't import app.py).
# --------------------------------------------------------------------------
def _filter_photos_single(cleaned_img, model_key, filename):
    return filter_photos_single(cleaned_img, model_key, filename, OUTPUTS_DIR)


def _filter_photos_ensemble(cleaned_by_member, filename):
    return filter_photos_ensemble(cleaned_by_member, filename, OUTPUTS_DIR)


def _filter_member_label(member_key):
    """Human-readable label for a filter_photos member key -- either a model
    key ("ab") for a single-model prediction, or predict_ensemble's member
    key ("member_1_ab") for an All-Four ensemble prediction."""
    model_key = ENSEMBLE_MEMBER_TO_MODEL_KEY.get(member_key, member_key)
    entry = PREDICTORS.get(model_key)
    return entry["label"] if entry else model_key.upper()


def _filter_photos_display(filter_photos):
    """Reshapes a stored filter_photos dict into a list the templates can
    loop over directly, with human-readable labels. Handles both shapes:
    {member_key: {technique: path}} (single-model / All-Four ensemble), and
    {"per_fruit": [{"index", "label", "filter_photos"}, ...]} (multi-fruit
    batch mode -- one group per (fruit, member) pair)."""
    if not filter_photos:
        return []
    if "per_fruit" in filter_photos:
        groups = []
        for item in filter_photos["per_fruit"]:
            fruit_tag = f"Fruit #{item['index'] + 1} ({item['label'].upper()})"
            for member_key, techniques in item.get("filter_photos", {}).items():
                groups.append({
                    "member_label": f"{fruit_tag} — {_filter_member_label(member_key)}",
                    "techniques": [
                        {"label": FILTER_LABELS.get(step, step), "path": path}
                        for step, path in techniques.items()
                    ],
                })
        return groups
    return [
        {
            "member_label": _filter_member_label(member_key),
            "techniques": [
                {"label": FILTER_LABELS.get(step, step), "path": path}
                for step, path in techniques.items()
            ],
        }
        for member_key, techniques in filter_photos.items()
    ]


def _mark_filter_photo_availability(filter_photos_display):
    """Checks each technique's file against disk, for a persisted record
    being viewed later (history_detail) rather than right after it was
    generated -- outputs/filters/ is gitignored, so a teammate's fresh
    clone (or anyone who's cleared their outputs/ folder) has DB rows that
    reference filter photos with nothing on disk to back them. Marks each
    technique "exists": False instead of leaving the template to render a
    broken <img> for a file that was never there."""
    for group in filter_photos_display:
        for technique in group["techniques"]:
            technique["exists"] = os.path.exists(os.path.join(OUTPUTS_DIR, technique["path"]))
    return filter_photos_display


def _filter_photos_for_pdf(filter_photos_display):
    """pdf_report.py's PDF generators take filter photo paths as absolute
    filesystem paths (same convention as image_path/surface_image_path),
    not the outputs/-relative paths stored/displayed everywhere else --
    resolve them here right before handing the list to a generator."""
    return [
        {
            "member_label": group["member_label"],
            "techniques": [
                {"label": t["label"], "path": os.path.join(OUTPUTS_DIR, t["path"])}
                for t in group["techniques"]
            ],
        }
        for group in (filter_photos_display or [])
    ]


# --------------------------------------------------------------------------
# Multi-fruit-per-photo batch detection ("this photo may contain multiple
# fruits") -- separate from every member's single-largest-contour detect().
# See core_modules/multi_fruit_detect.py for the YOLO localisation step.
# --------------------------------------------------------------------------
_MULTI_FRUIT_BOX_COLOURS = {"ripe": (0, 200, 0), "unripe": (0, 165, 255), "rotten": (0, 0, 200)}


def _classify_multi_fruit_photo(img, fruit_type, classify_crop, filename):
    """
    Detects every fruit_type box in one photo, classifies each crop via
    classify_crop(crop, fruit_type, crop_tag) -> (label, confidence_0_to_1,
    filter_photos) (or (None, None, {}) on a per-crop failure), draws every
    box on one annotated image, and returns a dict with the majority
    label/confidence, a {label: count} breakdown, the annotated path, the
    per-fruit detail list, and a filter_photos payload (one entry per
    detected fruit, see app._filter_photos_display) -- or None if this photo
    isn't a multi-fruit candidate (mango, or no boxes found), signaling the
    caller to fall back to the existing single-fruit path.
    """
    if not supports_multi_fruit(fruit_type):
        return None
    boxes = detect_fruit_boxes(img, fruit_type)
    if not boxes:
        return None

    safe_name = secure_filename(filename) or "fruit.jpg"
    stem, extension = os.path.splitext(safe_name)
    extension = extension if extension.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"

    annotated = img.copy()
    per_fruit = []
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        # Unique per-fruit tag so each detected fruit's filter photos get
        # their own filenames instead of overwriting one another.
        crop_tag = f"{stem}_fruit{idx}{extension}"
        try:
            label, confidence, crop_filter_photos = classify_crop(crop, fruit_type, crop_tag)
        except Exception:
            label, confidence, crop_filter_photos = None, None, {}
        if label is None:
            continue

        per_fruit.append({
            "bbox": (x0, y0, x1, y1), "label": label, "confidence": confidence,
            "filter_photos": crop_filter_photos,
        })
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

    filter_photos_payload = {
        "per_fruit": [
            {"index": i, "label": r["label"], "filter_photos": r["filter_photos"]}
            for i, r in enumerate(per_fruit)
        ]
    } if any(r["filter_photos"] for r in per_fruit) else {}

    return {
        "label": majority_label,
        "confidence": majority_confidence_pct,
        "breakdown": dict(breakdown),
        "annotated_path": f"annotated/{filename}",
        "per_fruit": per_fruit,
        "fruit_count": len(per_fruit),
        "filter_photos": filter_photos_payload,
    }


def _ensemble_crop_classify(crop, fruit_type, crop_tag):
    """classify_crop adapter for _classify_multi_fruit_photo: predict_ensemble
    returns confidence on a 0-100 scale, but the multi-fruit helper's
    contract (matching every member's predict_ripeness()) expects 0-1."""
    label, confidence_pct, per_member, _bbox = predict_ensemble(crop, fruit_type)
    cleaned_by_member = pop_member_cleaned_images(per_member)
    filter_photos = _filter_photos_ensemble(cleaned_by_member, crop_tag)
    return label, confidence_pct / 100, filter_photos


def _single_model_crop_classify(entry, model_key):
    """classify_crop adapter for a single PREDICTORS[model_choice] entry."""
    def _classify(crop, fruit_type, crop_tag):
        label, confidence, _bbox, cleaned, _proba = entry["fn"](crop, fruit_type)
        filter_photos = _filter_photos_single(cleaned, model_key, crop_tag)
        return label, confidence, filter_photos
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
        marketability = estimate_marketability(
            fruit=fruit_type,
            ripeness=label,
            confidence=confidence_pct,
            probabilities=proba_dict,
            blemish_percentage=surface.get("blemish_percentage"),
            quality_grade=surface.get("quality_grade"),
        )
        filter_photos = _filter_photos_single(cleaned, "ab", f.filename)

        log_result(
            member=_member_tag("ab"),
            fruit=fruit_type,
            label=label,
            confidence=confidence_pct,
            filename=f.filename,
            annotated_path=annotated_rel,
            source="predict",
            **_surface_db_fields(surface),
            **_marketability_db_fields(marketability),
            user_id=g.user["id"] if g.user else None,
            latency_ms=latency_ms,
            flagged=_is_flagged(confidence_pct),
            filter_photos=json.dumps(filter_photos) if filter_photos else None,
        )
        _log_stock_result(True, fruit_type, label, marketability_status=marketability["status"], source="single")

        results.append({
            "filename": f.filename,
            "fruit": fruit_type,
            "ripeness": label,
            "confidence": round(confidence * 100, 1),
            "marketability": marketability,
            "input_validation": input_validation,
            "filter_photos": filter_photos,
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
    validate_upload = request.form.get("validate", "1") != "0"
    files = request.files.getlist("image")
    if not files or files[0].filename == "":
        return {"error": "No image uploaded"}, 400

    f = files[0]
    path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(path)
    img = cv2.imread(path)
    if img is None:
        return {"error": "Uploaded image could not be read", "filename": f.filename}, 400
    if validate_upload:
        try:
            input_validation = validate_selected_fruit(img, fruit_type)
        except FruitValidationError as e:
            return {"error": str(e), "filename": f.filename}, 422
    else:
        input_validation = {
            "selected_fruit": fruit_type,
            "detected_fruit": None,
            "confidence": None,
            "validation_method": "skipped",
        }

    if model_choice == "all_four":
        t0 = time.perf_counter()
        try:
            label, confidence, per_member, bbox = predict_ensemble(img, fruit_type)
        except RuntimeError as e:
            return {"error": str(e), "filename": f.filename}, 422
        cleaned_by_member = pop_member_cleaned_images(per_member)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        annotated_rel = _save_annotated(img, bbox, f.filename)
        surface = _analyse_surface_and_save(img, bbox, f.filename)
        ensemble_proba = average_member_probabilities(per_member)
        marketability = estimate_marketability(
            fruit=fruit_type,
            ripeness=label,
            confidence=confidence,
            probabilities=ensemble_proba,
            blemish_percentage=surface.get("blemish_percentage"),
            quality_grade=surface.get("quality_grade"),
        )
        filter_photos = _filter_photos_ensemble(cleaned_by_member, f.filename)

        log_result(
            member="ensemble_all_four",
            fruit=fruit_type,
            label=label,
            confidence=confidence,
            filename=f.filename,
            annotated_path=annotated_rel,
            source="predict_unified",
            **_surface_db_fields(surface),
            **_marketability_db_fields(marketability),
            user_id=g.user["id"] if g.user else None,
            latency_ms=latency_ms,
            flagged=_is_flagged(confidence),
            filter_photos=json.dumps(filter_photos) if filter_photos else None,
        )
        _log_stock_result(True, fruit_type, label, marketability_status=marketability["status"], source="single")

        return {
            "model": "all_four",
            "fruit": fruit_type,
            "ripeness": label,
            "confidence": confidence,
            "per_member": per_member,
            "proba": ensemble_proba,
            "marketability": marketability,
            "input_validation": input_validation,
            "filter_photos": filter_photos,
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
    marketability = estimate_marketability(
        fruit=fruit_type,
        ripeness=label,
        confidence=confidence_pct,
        probabilities=proba_dict,
        blemish_percentage=surface.get("blemish_percentage"),
        quality_grade=surface.get("quality_grade"),
    )
    filter_photos = _filter_photos_single(cleaned, model_choice, f.filename)

    log_result(
        member=_member_tag(model_choice),
        fruit=fruit_type,
        label=label,
        confidence=confidence_pct,
        filename=f.filename,
        annotated_path=annotated_rel,
        source="predict_unified",
        **_surface_db_fields(surface),
        **_marketability_db_fields(marketability),
        user_id=g.user["id"] if g.user else None,
        latency_ms=latency_ms,
        flagged=_is_flagged(confidence_pct),
        filter_photos=json.dumps(filter_photos) if filter_photos else None,
    )
    _log_stock_result(True, fruit_type, label, marketability_status=marketability["status"], source="single")

    return {
        "model": model_choice,
        "fruit": fruit_type,
        "ripeness": label,
        "confidence": confidence_pct,
        "per_member": None,
        "proba": {cls: round(p * 100, 1) for cls, p in proba_dict.items()},
        "marketability": marketability,
        "input_validation": input_validation,
        "filter_photos": filter_photos,
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

def _log_stock_result(should_log, fruit_type, label, marketability_status=None, per_fruit=None, source="batch"):
    """Log one prediction's result(s) as stock movements, if should_log is
    true (the batch page's 'add to stock' checkbox, or always-on for single
    uploads/realtime). unripe/rotten always count -- their disposition (hold
    to ripen / remove) is already unambiguous from the raw label; a 'ripe'
    result only counts if it's actually market-ready (see
    core_modules.marketability.stock_eligible), so a low-confidence or
    heavily blemished 'ripe' classification doesn't inflate the sellable
    stock count.

    In multi-fruit mode, per_fruit is the per-detection list from
    _classify_multi_fruit_photo -- each entry is gated on its own confidence
    (no per-fruit blemish data exists there) instead of the photo's majority
    verdict. Otherwise marketability_status is the caller's already-computed
    verdict (with blemish/quality folded in) for the whole image."""
    if not should_log:
        return
    user_id = g.user["id"] if g.user else None
    if per_fruit:
        counts = Counter()
        for r in per_fruit:
            if stock_eligible(fruit_type, r["label"], r["confidence"]):
                counts[r["label"]] += 1
        for detected_label, count in counts.items():
            stock_db.log_stock_event(
                fruit=fruit_type, label=detected_label, quantity=count,
                source=source, user_id=user_id,
            )
    elif label != "ripe" or marketability_status == "ready":
        stock_db.log_stock_event(
            fruit=fruit_type, label=label, quantity=1,
            source=source, user_id=user_id,
        )


ALL_FOUR_LABEL = "Ensemble (All 4 members, soft-voted)"
ALL_FOUR_KEY = "all_four"

# Same PREDICTORS dict, plus a display-only entry for the ensemble, so
# dashboard/analyse templates can list it alongside ab/bc/cd/da/yolo_pure
# without giving it a fake "fn"/"not_fruit_err" (predict_ensemble has a
# different signature and is called directly instead).
PREDICTORS_WITH_ENSEMBLE = dict(PREDICTORS)
PREDICTORS_WITH_ENSEMBLE[ALL_FOUR_KEY] = {"label": ALL_FOUR_LABEL}


@app.route("/analyse-mixed-fruit-m14", methods=["POST"])
def analyse_mixed_fruit_m14():
    """Detect apple/banana/orange together and route every crop through M14."""
    uploaded = request.files.get("image")
    if not uploaded or not uploaded.filename:
        flash("Choose an image containing apple, banana, or orange fruit.")
        return redirect(url_for("classify"))

    original_name = secure_filename(uploaded.filename) or "mixed_fruit.jpg"
    stem, extension = os.path.splitext(original_name)
    extension = extension if extension.lower() in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"
    unique_tag = time.time_ns()
    stored_name = f"{stem}_mixed_m14_{unique_tag}{extension}"
    upload_path = os.path.join(UPLOAD_DIR, stored_name)
    uploaded.save(upload_path)
    image = cv2.imread(upload_path)
    if image is None:
        flash("The uploaded mixed-fruit image could not be read.")
        return redirect(url_for("classify"))

    try:
        analysis = analyze_mixed_fruit_m14(image)
    except Exception as exc:
        return render_template(
            "mixed_fruit_m14.html",
            result={"filename": original_name, "error": str(exc)},
            active_page="classify",
        )

    annotated_name = f"{stem}_mixed_m14_{unique_tag}_annotated{extension}"
    annotated_dir = os.path.join(OUTPUTS_DIR, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)
    annotated_path = os.path.join(annotated_dir, annotated_name)
    annotated_rel = None
    if cv2.imwrite(annotated_path, analysis["annotated_image"]):
        annotated_rel = f"annotated/{annotated_name}"

    add_to_stock = request.form.get("add_to_stock") == "on"
    for detection in analysis["detections"]:
        if not detection.get("label"):
            detection["marketability"] = None
            continue

        marketability = estimate_marketability(
            fruit=detection["fruit"],
            ripeness=detection["label"],
            confidence=detection["ripeness_confidence"],
            probabilities=detection.get("probabilities"),
        )
        detection["marketability"] = marketability
        log_result(
            member="merged_1_4",
            fruit=detection["fruit"],
            label=detection["label"],
            confidence=detection["ripeness_confidence"],
            filename=original_name,
            annotated_path=annotated_rel,
            source="analyse_mixed_fruit_m14",
            user_id=g.user["id"] if g.user else None,
            latency_ms=analysis["latency_ms"],
            flagged=_is_flagged(detection["ripeness_confidence"]),
            **_marketability_db_fields(marketability),
        )
        _log_stock_result(
            add_to_stock,
            detection["fruit"],
            detection["label"],
            marketability_status=marketability["status"],
            source="mixed_fruit_m14",
        )

    analysis.update({
        "filename": original_name,
        "annotated_path": annotated_rel,
    })
    return render_template(
        "mixed_fruit_m14.html",
        result=analysis,
        active_page="classify",
    )


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
    add_to_stock = request.form.get("add_to_stock") == "on"

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
                    marketability = estimate_marketability(
                        fruit=fruit_type,
                        ripeness=multi["label"],
                        confidence=multi["confidence"],
                    )
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
                        **_marketability_db_fields(marketability),
                        filter_photos=json.dumps(multi["filter_photos"]) if multi["filter_photos"] else None,
                    )
                    _log_stock_result(add_to_stock, fruit_type, multi["label"], per_fruit=multi["per_fruit"])
                    results.append({
                        "filename": f.filename,
                        "fruit": fruit_type,
                        "label": multi["label"],
                        "confidence": multi["confidence"],
                        "annotated_path": multi["annotated_path"],
                        "input_validation": input_validation,
                        "detection_breakdown": multi["breakdown"],
                        "fruit_count": multi["fruit_count"],
                        "marketability": marketability,
                        "filter_photos": _filter_photos_display(multi["filter_photos"]),
                        **_surface_payload({}),
                    })
                    continue

            t0 = time.perf_counter()
            try:
                label, confidence, per_member, bbox = predict_ensemble(img, fruit_type)
            except RuntimeError as e:
                results.append({"filename": f.filename, "label": None, "confidence": None, "error": str(e)})
                continue
            cleaned_by_member = pop_member_cleaned_images(per_member)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            annotated_rel = _save_annotated(img, bbox, f.filename)
            surface = _analyse_surface_and_save(img, bbox, f.filename)
            ensemble_proba = average_member_probabilities(per_member)
            marketability = estimate_marketability(
                fruit=fruit_type,
                ripeness=label,
                confidence=confidence,
                probabilities=ensemble_proba,
                blemish_percentage=surface.get("blemish_percentage"),
                quality_grade=surface.get("quality_grade"),
            )
            filter_photos = _filter_photos_ensemble(cleaned_by_member, f.filename)

            log_result(
                member=member_tag,
                fruit=fruit_type,
                label=label,
                confidence=confidence,
                filename=f.filename,
                annotated_path=annotated_rel,
                source="analyse",
                **_surface_db_fields(surface),
                **_marketability_db_fields(marketability),
                user_id=g.user["id"] if g.user else None,
                latency_ms=latency_ms,
                flagged=_is_flagged(confidence),
                filter_photos=json.dumps(filter_photos) if filter_photos else None,
            )
            _log_stock_result(add_to_stock, fruit_type, label, marketability_status=marketability["status"])

            results.append({
                "filename": f.filename,
                "fruit": fruit_type,
                "label": label,
                "confidence": confidence,
                "annotated_path": annotated_rel,
                "multi_fruit_fallback": multi_fruit,
                "per_member": per_member,
                "input_validation": input_validation,
                "marketability": marketability,
                "filter_photos": _filter_photos_display(filter_photos),
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
                multi = _classify_multi_fruit_photo(img, fruit_type, _single_model_crop_classify(entry, model_choice), f.filename)
                if multi:
                    confidence_pct = multi["confidence"]
                    marketability = estimate_marketability(
                        fruit=fruit_type,
                        ripeness=multi["label"],
                        confidence=confidence_pct,
                    )
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
                        **_marketability_db_fields(marketability),
                        filter_photos=json.dumps(multi["filter_photos"]) if multi["filter_photos"] else None,
                    )
                    _log_stock_result(add_to_stock, fruit_type, multi["label"], per_fruit=multi["per_fruit"])
                    results.append({
                        "filename": f.filename,
                        "fruit": fruit_type,
                        "label": multi["label"],
                        "confidence": confidence_pct,
                        "annotated_path": multi["annotated_path"],
                        "input_validation": input_validation,
                        "detection_breakdown": multi["breakdown"],
                        "fruit_count": multi["fruit_count"],
                        "marketability": marketability,
                        "filter_photos": _filter_photos_display(multi["filter_photos"]),
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
            marketability = estimate_marketability(
                fruit=fruit_type,
                ripeness=label,
                confidence=confidence_pct,
                probabilities=proba_dict,
                blemish_percentage=surface.get("blemish_percentage"),
                quality_grade=surface.get("quality_grade"),
            )
            filter_photos = _filter_photos_single(cleaned, model_choice, f.filename)

            log_result(
                member=member_tag,
                fruit=fruit_type,
                label=label,
                confidence=confidence_pct,
                filename=f.filename,
                annotated_path=annotated_rel,
                source="analyse",
                **_surface_db_fields(surface),
                **_marketability_db_fields(marketability),
                user_id=g.user["id"] if g.user else None,
                latency_ms=latency_ms,
                flagged=_is_flagged(confidence_pct),
                filter_photos=json.dumps(filter_photos) if filter_photos else None,
            )
            _log_stock_result(add_to_stock, fruit_type, label, marketability_status=marketability["status"])

            results.append({
                "filename": f.filename,
                "fruit": fruit_type,
                "label": label,
                "confidence": confidence_pct,
                "annotated_path": annotated_rel,
                "multi_fruit_fallback": multi_fruit,
                "filter_photos": _filter_photos_display(filter_photos),
                "input_validation": input_validation,
                "marketability": marketability,
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
            "filter_photos": _filter_photos_for_pdf(r.get("filter_photos")),
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
    filter_photos_raw = request.form.get("filter_photos_json")
    filter_photos = _filter_photos_for_pdf(json.loads(filter_photos_raw)) if filter_photos_raw else None
    out_path = generate_pdf_report(
        image_path, label, confidence, model_tag=model_tag, surface_data=surface_data,
        filter_photos=filter_photos,
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
    if not record or not _owns_record(record):
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
    filter_photos_raw = record.get("filter_photos")
    filter_photos = (
        _filter_photos_for_pdf(_filter_photos_display(json.loads(filter_photos_raw)))
        if filter_photos_raw else None
    )
    out_path = generate_pdf_report(
        image_path, record["label"], record["confidence"] / 100,
        model_tag=record["member"], surface_data=surface_data,
        filter_photos=filter_photos,
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


@app.route("/marketability")
def marketability_dashboard():
    fruit_filter = request.args.get("fruit") or None
    model_filter = request.args.get("model") or None
    ripeness_filter = request.args.get("ripeness") or None
    analysis_filter = request.args.get("analysis") or None
    status_filter = request.args.get("status") or None
    priority_filter = request.args.get("priority") or None
    review_filter = request.args.get("review") or None
    date_from = _valid_date_arg("date_from")
    date_to = _valid_date_arg("date_to")
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1

    rows = _decorate_marketability_rows(get_all_results(
        member=model_filter,
        fruit=fruit_filter,
        user_id=_scope_user_id(),
        date_from=date_from,
        date_to=date_to,
    ))
    if ripeness_filter:
        rows = [row for row in rows if row.get("label") == ripeness_filter]
    if analysis_filter:
        rows = [row for row in rows if row.get("analysis_kind") == analysis_filter]
    if status_filter:
        rows = [row for row in rows if row["marketability"].get("status") == status_filter]
    if priority_filter:
        rows = [
            row for row in rows
            if row["marketability"].get("dispatch_priority") == priority_filter
        ]
    if review_filter:
        rows = [row for row in rows if row["review"].get("status") == review_filter]

    summary = {
        "total": len(rows),
        "urgent": sum(
            row["marketability"].get("dispatch_priority") in {"urgent", "remove"}
            for row in rows
        ),
        "ready": sum(row["marketability"].get("status") == "ready" for row in rows),
        "hold": sum(row["marketability"].get("status") == "hold" for row in rows),
        "inspect": sum(
            row["marketability"].get("status") in {"inspect", "isolate"}
            for row in rows
        ),
        "remove": sum(row["marketability"].get("status") == "remove" for row in rows),
        "batch": sum(
            row.get("analysis_kind") in {"batch", "multi_fruit_batch", "mixed_fruit_m14"}
            for row in rows
        ),
        "needs_review": sum(
            row["review"].get("status") == "needs_review" for row in rows
        ),
    }
    model_options = [
        (_member_tag(key), entry["label"])
        for key, entry in PREDICTORS.items()
    ] + [("ensemble_all_four", ALL_FOUR_LABEL)]

    # Rows are sorted worst-first (see _marketability_sort_key) so removal/
    # inspection items surface first, but that means a hard cutoff here
    # would silently drop every "ready"/"hold" row past it -- paginate
    # instead so every filtered row stays reachable.
    total = len(rows)
    total_pages = max(1, math.ceil(total / MARKETABILITY_PAGE_SIZE))
    page = min(page, total_pages)
    start = (page - 1) * MARKETABILITY_PAGE_SIZE
    page_rows = rows[start:start + MARKETABILITY_PAGE_SIZE]

    return render_template(
        "marketability.html",
        results=page_rows,
        summary=summary,
        page=page,
        total_pages=total_pages,
        total=total,
        fruits=FRUITS,
        model_options=model_options,
        model_labels=dict(model_options),
        fruit_filter=fruit_filter,
        model_filter=model_filter,
        ripeness_filter=ripeness_filter,
        analysis_filter=analysis_filter,
        status_filter=status_filter,
        priority_filter=priority_filter,
        review_filter=review_filter,
        date_from=date_from,
        date_to=date_to,
        active_page="marketability",
    )


@app.route("/marketability/<int:record_id>/review", methods=["POST"])
def marketability_review(record_id):
    """Store a human decision alongside, never over, the model prediction."""
    record = get_by_id(record_id)
    if not record:
        flash("That prediction record no longer exists.")
        return redirect(url_for("marketability_dashboard"))

    decision = request.form.get("decision", "")
    reason = request.form.get("reason", "").strip()[:500] or None
    review_fields = {
        "reviewed_by": g.user["name"],
        "reviewed_at": datetime.now().isoformat(timespec="seconds"),
        "review_reason": reason,
    }
    if decision == "confirm":
        review_fields.update({
            "review_status": "confirmed",
            "review_fruit": None,
            "review_label": None,
        })
        message = "Model result confirmed after inspection."
    elif decision == "correct":
        review_fruit = request.form.get("review_fruit")
        review_label = request.form.get("review_label")
        if review_fruit not in FRUITS or review_label not in RIPENESS_CLASSES:
            flash("Choose a valid observed fruit and ripeness classification.")
            return redirect(url_for("marketability_dashboard"))
        review_fields.update({
            "review_status": "corrected",
            "review_fruit": review_fruit,
            "review_label": review_label,
        })
        message = "Operator correction saved separately from the model result."
    else:
        flash("Choose whether to confirm or correct the model result.")
        return redirect(url_for("marketability_dashboard"))

    if not update_result(record_id, **review_fields):
        flash("The review could not be saved because the record is no longer available.")
        return redirect(url_for("marketability_dashboard"))
    auth_db.log_activity(
        g.user["id"], "review_marketability",
        detail=f"record {record_id}: {review_fields['review_status']}",
    )
    flash(message)
    return redirect(url_for("marketability_dashboard"))


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

    scope = _scope_user_id()
    rows, total = get_paginated(
        member=member_filter, fruit=fruit_filter, user_id=scope, date_from=date_from, date_to=date_to,
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

    stats = get_stats(member=member_filter, fruit=fruit_filter, user_id=scope, date_from=date_from, date_to=date_to)

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
    rows = get_all_results(
        member=member_filter, fruit=fruit_filter, user_id=_scope_user_id(),
        date_from=date_from, date_to=date_to,
    )

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


@app.route("/history/<int:record_id>")
def history_detail(record_id):
    record = get_by_id(record_id)
    if not record or not _owns_record(record):
        flash("That record no longer exists.")
        return redirect(url_for("history"))

    raw = record.get("detection_breakdown")
    record["detection_breakdown"] = json.loads(raw) if raw else None
    input_exists = bool(record.get("filename")) and os.path.exists(os.path.join(UPLOAD_DIR, record["filename"]))
    # outputs/annotated/, outputs/surface/, and outputs/filters/ are all
    # gitignored (generated artifacts, not source) -- a persisted record can
    # reference a path that simply isn't there on this machine (a
    # teammate's fresh clone, or a locally cleared outputs/ folder), so
    # check disk rather than trusting the stored path is still backed by a
    # real file.
    annotated_exists = bool(record.get("annotated_path")) and os.path.exists(os.path.join(OUTPUTS_DIR, record["annotated_path"]))
    surface_exists = bool(record.get("surface_path")) and os.path.exists(os.path.join(OUTPUTS_DIR, record["surface_path"]))

    filter_photos_raw = record.get("filter_photos")
    filter_photos = (
        _mark_filter_photo_availability(_filter_photos_display(json.loads(filter_photos_raw)))
        if filter_photos_raw else []
    )

    return render_template(
        "history_detail.html", record=record, input_exists=input_exists,
        annotated_exists=annotated_exists, surface_exists=surface_exists,
        filter_photos=filter_photos, active_page="history",
    )


@app.route("/history/<int:record_id>/edit", methods=["GET", "POST"])
def history_edit(record_id):
    record = get_by_id(record_id)
    if not record or not _owns_record(record):
        flash("That record no longer exists.")
        return redirect(url_for("history"))

    if request.method == "POST":
        edited_fruit = request.form.get("fruit")
        edited_label = request.form.get("label")
        edited_confidence = float(request.form["confidence"]) if request.form.get("confidence") else None
        marketability = estimate_marketability(
            fruit=edited_fruit,
            ripeness=edited_label,
            confidence=edited_confidence,
            blemish_percentage=record.get("blemish_percentage"),
            quality_grade=record.get("quality_grade"),
        )
        update_result(
            record_id,
            fruit=edited_fruit,
            label=edited_label,
            confidence=edited_confidence,
            source=request.form.get("source"),
            **_marketability_db_fields(marketability),
        )
        flash("Record updated.")
        return redirect(url_for("history"))

    return render_template(
        "history_edit.html", record=record, fruits=FRUITS, classes=RIPENESS_CLASSES,
        active_page="history",
    )


@app.route("/history/<int:record_id>/delete", methods=["POST"])
def history_delete(record_id):
    record = get_by_id(record_id)
    if not record or not _owns_record(record):
        flash("Record not found.")
        return redirect(url_for("history", page=request.form.get("page", 1)))

    delete_result(record_id)
    flash("Record deleted.")
    return redirect(url_for("history", page=request.form.get("page", 1)))


# --------------------------------------------------------------------------
# Fruit Stock — CRUD ledger of stock movements (manual entries, plus
# automatic entries from batch analysis; see _log_stock_result above).
# --------------------------------------------------------------------------
@app.route("/stock")
def stock():
    fruit_filter = request.args.get("fruit") or None
    label_filter = request.args.get("label") or None
    source_filter = request.args.get("source") or None
    date_from = _valid_date_arg("date_from")
    date_to = _valid_date_arg("date_to")
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1

    scope = _scope_user_id()
    rows, total = stock_db.get_paginated(
        fruit=fruit_filter, label=label_filter, source=source_filter, user_id=scope,
        date_from=date_from, date_to=date_to, page=page, per_page=STOCK_PAGE_SIZE,
    )
    total_pages = max(1, math.ceil(total / STOCK_PAGE_SIZE))
    page = min(page, total_pages)

    summary = stock_db.get_summary(fruit=fruit_filter, user_id=scope, date_from=date_from, date_to=date_to)

    return render_template(
        "stock.html",
        results=rows,
        fruit_filter=fruit_filter,
        label_filter=label_filter,
        source_filter=source_filter,
        date_from=date_from,
        date_to=date_to,
        fruits=FRUITS,
        classes=RIPENESS_CLASSES,
        page=page,
        total_pages=total_pages,
        total=total,
        summary=summary,
        active_page="stock",
    )


def _stock_filters_from_args():
    return {
        "fruit": request.args.get("fruit") or None,
        "label": request.args.get("label") or None,
        "source": request.args.get("source") or None,
        "date_from": _valid_date_arg("date_from"),
        "date_to": _valid_date_arg("date_to"),
    }


@app.route("/stock/export.csv")
def stock_export_csv():
    rows = stock_db.get_all(user_id=_scope_user_id(), **_stock_filters_from_args())

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "created_at", "fruit", "label", "quantity", "source", "note"])
    for r in rows:
        writer.writerow([
            r["id"], r["created_at"], r["fruit"], r["label"], r["quantity"],
            r["source"], r.get("note") or "",
        ])

    if g.user:
        auth_db.log_activity(g.user["id"], "export_stock_csv", detail=f"{len(rows)} entries")

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=fruit_stock.csv"},
    )


@app.route("/stock/export_pdf")
def stock_export_pdf():
    filters = _stock_filters_from_args()
    scope = _scope_user_id()
    rows = stock_db.get_all(user_id=scope, **filters)
    summary = stock_db.get_summary(fruit=filters["fruit"], user_id=scope, date_from=filters["date_from"], date_to=filters["date_to"])
    out_path = generate_stock_report_pdf(summary, rows)

    if g.user:
        auth_db.log_activity(g.user["id"], "export_stock_pdf", detail=f"{len(rows)} entries")

    return send_from_directory(os.path.dirname(out_path), os.path.basename(out_path), as_attachment=True)


@app.route("/stock/add", methods=["POST"])
def stock_add():
    fruit = request.form.get("fruit")
    label = request.form.get("label")
    note = request.form.get("note") or None
    try:
        quantity = int(request.form.get("quantity", ""))
    except ValueError:
        flash("Quantity must be a whole number.")
        return redirect(url_for("stock"))
    if fruit not in FRUITS or label not in RIPENESS_CLASSES or quantity == 0:
        flash("Enter a valid fruit, ripeness, and non-zero quantity.")
        return redirect(url_for("stock"))

    stock_db.log_stock_event(
        fruit=fruit, label=label, quantity=quantity, source="manual",
        note=note, user_id=g.user["id"] if g.user else None,
    )
    flash("Stock entry added.")
    return redirect(url_for("stock"))


@app.route("/stock/<int:event_id>/edit", methods=["GET", "POST"])
def stock_edit(event_id):
    record = stock_db.get_by_id(event_id)
    if not record or not _owns_record(record):
        flash("That stock entry no longer exists.")
        return redirect(url_for("stock"))

    if request.method == "POST":
        try:
            quantity = int(request.form.get("quantity", ""))
        except ValueError:
            flash("Quantity must be a whole number.")
            return redirect(url_for("stock_edit", event_id=event_id))

        stock_db.update_stock_event(
            event_id,
            fruit=request.form.get("fruit"),
            label=request.form.get("label"),
            quantity=quantity,
            note=request.form.get("note") or None,
        )
        flash("Stock entry updated.")
        return redirect(url_for("stock"))

    return render_template(
        "stock_edit.html", record=record, fruits=FRUITS, classes=RIPENESS_CLASSES,
        active_page="stock",
    )


@app.route("/stock/<int:event_id>/delete", methods=["POST"])
def stock_delete(event_id):
    record = stock_db.get_by_id(event_id)
    if not record or not _owns_record(record):
        flash("Stock entry not found.")
        return redirect(url_for("stock", page=request.form.get("page", 1)))

    stock_db.delete_stock_event(event_id)
    flash("Stock entry deleted.")
    return redirect(url_for("stock", page=request.form.get("page", 1)))


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
# Model Lab — compare every trained model side by side (accuracy, macro-F1,
# balanced accuracy, weakest-class recall, latency, size), swap between
# per-fruit confusion matrices, and set the Classify page's default model.
# --------------------------------------------------------------------------
def _model_lab_label(model_key):
    entry = PREDICTORS.get(model_key)
    return entry["label"] if entry else model_key.upper()


@app.route("/model-lab")
def model_lab_dashboard():
    default_model = auth_db.get_setting("default_model", "ab")

    rows = []
    for key in model_lab.MODEL_ORDER:
        summary = model_lab.get_model_summary(key)
        stats = get_stats(member=_member_tag(key))
        summary["label"] = _model_lab_label(key)
        summary["avg_latency_ms"] = stats.get("avg_latency_ms")
        summary["size_display"] = model_lab.format_size(summary["size_bytes"])
        summary["is_active"] = key == default_model
        rows.append(summary)

    trained_rows = [r for r in rows if r["has_data"]]
    recommended = max(trained_rows, key=lambda r: r["accuracy"], default=None)

    confusion_data = {
        key: {fruit: model_lab.get_confusion_matrix(key, fruit) for fruit in model_lab.FRUITS}
        for key in model_lab.MODEL_ORDER
    }
    per_fruit_recall = {key: model_lab.get_per_fruit_recall(key) for key in model_lab.MODEL_ORDER}
    yolo_history = {fruit: model_lab.get_yolo_training_history(fruit) for fruit in model_lab.FRUITS}

    return render_template(
        "model_lab.html",
        rows=rows,
        trained_count=len(trained_rows),
        total_models=len(model_lab.MODEL_ORDER),
        recommended=recommended,
        default_model=default_model,
        confusion_data_json=json.dumps(confusion_data),
        per_fruit_recall_json=json.dumps(per_fruit_recall),
        yolo_history_json=json.dumps(yolo_history),
        fruits=model_lab.FRUITS,
        model_order=model_lab.MODEL_ORDER,
        active_page="model_lab",
    )


@app.route("/model-lab/activate", methods=["POST"])
def model_lab_activate():
    model_key = request.form.get("model_key")
    if model_key not in PREDICTORS:
        flash("Unknown model.")
        return redirect(url_for("model_lab_dashboard"))
    auth_db.set_setting("default_model", model_key)
    flash(f"{_model_lab_label(model_key)} set as the active default model.")
    return redirect(url_for("model_lab_dashboard"))


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
    password = request.form.get("password", "")
    role = request.form.get("role", "farmer")

    if not name or not email:
        flash("Name and email are required.")
        return redirect(url_for("admin_panel"))
    if len(password) < 8:
        flash("Password must be at least 8 characters.")
        return redirect(url_for("admin_panel"))

    try:
        auth_db.create_user(name, email, password, role=role)
    except Exception:
        flash(f"Could not create {email} — that email may already be in use.")
        return redirect(url_for("admin_panel"))

    auth_db.log_activity(g.user["id"], "invite_user", detail=email)
    flash(f"Created {name} ({email}).")
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


@app.route("/admin/users/<int:user_id>/deactivate", methods=["POST"])
@admin_required
def admin_deactivate_user(user_id):
    if user_id == g.user["id"]:
        flash("You can't deactivate your own account.")
        return redirect(url_for("admin_panel"))

    target = auth_db.get_user_by_id(user_id)
    if target and target["role"] == "admin" and auth_db.active_admin_count() <= 1:
        flash("Can't deactivate the only remaining active admin.")
        return redirect(url_for("admin_panel"))

    auth_db.set_active(user_id, False)
    auth_db.log_activity(g.user["id"], "deactivate_user", detail=target["email"] if target else str(user_id))
    flash("User deactivated.")
    return redirect(url_for("admin_panel"))


@app.route("/admin/users/<int:user_id>/reactivate", methods=["POST"])
@admin_required
def admin_reactivate_user(user_id):
    target = auth_db.get_user_by_id(user_id)
    auth_db.set_active(user_id, True)
    auth_db.log_activity(g.user["id"], "reactivate_user", detail=target["email"] if target else str(user_id))
    flash("User reactivated.")
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

        elif form == "appearance":
            # Per-user, not the app-wide `settings` table above -- this
            # follows whichever account is logged in, starting next page load.
            dark_mode = request.form.get("dark_mode") == "on"
            auth_db.set_dark_mode(g.user["id"], dark_mode)
            auth_db.log_activity(g.user["id"], "update_appearance", detail="dark_mode=" + str(dark_mode))
            flash("Appearance updated.")

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
