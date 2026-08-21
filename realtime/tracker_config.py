"""
Shared YOLO detector/tracker configuration for every real-time engine
(svm_yolo_tracker.py, yolo_cls_tracker.py, ensemble_ab/bc/cd/da_tracker.py).

Centralized here (MUST-DO #4, item 5) instead of each engine hardcoding its
own copy of the same values -- pulled forward while touching both files
anyway for items 1-2 (baseline benchmarking + conf/iou tuning), so tuning
only has to happen in ONE place and can't silently drift out of sync
between engines.

Tuning workflow (MUST-DO #4, in order):
  1. Benchmark current baseline (defaults below) using the FPS logging
     that's now built into every tracker -- get a "before" number first.
  2. Tune YOLO_CONF_THRESHOLD / YOLO_IOU_THRESHOLD (cheapest lever, no
     model swap).
  3. Try YOLO_MODEL_NAME = "yolov8s.pt" (test in isolation from imgsz).
  4. Try YOLO_IMGSZ = 960 LAST, and specifically re-check the combined
     compute cost with BoT-SORT -- this combination was flagged as
     potentially expensive, don't assume it's free just because each
     piece was cheap on its own.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))

# --- Detector model --------------------------------------------------------
# Baseline is yolov8n.pt. Swap to "yolov8s.pt" here to test the n->s
# upgrade (step 3) -- every engine picks up the change automatically.
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_WEIGHTS_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "trained_models", "svm_yolo"))
YOLO_WEIGHTS_PATH = os.path.join(YOLO_WEIGHTS_DIR, YOLO_MODEL_NAME)

# --- Inference size ---------------------------------------------------------
# Baseline is 640 (ultralytics' own default). Step 4 in the plan is testing
# 960 -- change this ONLY after benchmarking the 640 baseline, since imgsz
# is the change most likely to hurt real-time frame rate.
YOLO_IMGSZ = 640

# --- Confidence / IoU thresholds ---------------------------------------------
# Raise CONF to cut false positives (spurious "fruit" boxes on background
# clutter); lower it to catch more dim/occluded fruit. Lower IOU to let
# tighter/overlapping boxes both survive non-max suppression; raise it to
# suppress more aggressively.
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.7

# --- Tracker ------------------------------------------------------------
# BoT-SORT -- named here so every engine references the same value instead
# of each hardcoding the "botsort.yaml" string separately.
TRACKER_CONFIG = "botsort.yaml"

# --- Benchmarking -----------------------------------------------------------
# How many frames between FPS log lines.
# NOTE: this value is a reconstructed default (30) -- if you remember
# setting this differently originally, change it back.
FPS_LOG_EVERY_N_FRAMES = 30