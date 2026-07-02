import os
import sys
import glob
import time
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from m1_train_report import (
    plot_confusion_matrix,
    plot_class_distribution,
    plot_accuracy_summary,
    save_training_time,
    format_duration,
)

# --- Optional GPU training path ---------------------------------------------
# Set USE_GPU=1 as an environment variable to try GPU-accelerated training via
# RAPIDS cuML. Falls back to CPU (scikit-learn) automatically if cuML isn't
# installed/available, so this is safe to leave on by default.
#
# IMPORTANT CAVEAT: RAPIDS cuML only installs on Linux with an NVIDIA GPU +
# CUDA toolkit (via conda, not plain pip) -- it does NOT work on Windows.
# Also, for this project's feature set (~13 handcrafted colour+shape numbers
# per image, likely a few hundred/thousand samples), an SVM trains in well
# under a second on CPU already. GPU won't meaningfully speed this up -- the
# real bottleneck if any is the per-image OpenCV preprocessing loop, which
# this doesn't accelerate. This flag exists so you can experiment, not
# because it's expected to give a big speedup here.
USE_GPU = os.environ.get("USE_GPU", "0") == "1"

_gpu_ready = False
if USE_GPU:
    try:
        from cuml.svm import SVC as GPUSVC
        from cuml.preprocessing import StandardScaler as GPUScaler
        _gpu_ready = True
        print("[train] USE_GPU=1 and cuML is available -- training on GPU.")
    except ImportError:
        print("[train] USE_GPU=1 but cuML is not installed (it requires Linux + "
              "NVIDIA CUDA via conda). Falling back to CPU.")

if _gpu_ready:
    SVC = GPUSVC
    StandardScaler = GPUScaler
else:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
# -----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..', '..'))

from core_modules.preprocessing import preprocess, load_image
from core_modules.calibration import calibrate
from core_modules.ma_colour_space import extract_colour
from core_modules.mb_shape_contours import extract_shape

DATASET_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "datasets", "fruit_ripeness"))
MODEL_OUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "trained_models", "ensemble_ab"))

FRUITS = ["apple", "banana", "orange", "mango"]
CLASSES = ["ripe", "rotten", "unripe"]


def build_dataset(fruit):
    X, y = [], []
    for label in CLASSES:
        folder = os.path.join(DATASET_ROOT, fruit, label)
        if not os.path.isdir(folder):
            continue
        for path in glob.glob(os.path.join(folder, "*.*")):
            try:
                img = load_image(path)
                cropped, bbox = preprocess(img)
                cleaned, _calib_info = calibrate(cropped, bbox, target_size=(256, 256))
                vec_a = extract_colour(cleaned)
                vec_b = extract_shape(cleaned)
                vec = np.concatenate([vec_a, vec_b])
                X.append(vec)
                y.append(label)
            except Exception as e:
                print(f"Skipping {path}: {e}")
    return np.array(X), np.array(y)


if __name__ == "__main__":
    accuracies = {}
    per_fruit_seconds = {}
    run_start = time.time()

    for fruit in FRUITS:
        fruit_start = time.time()

        print(f"\nBuilding feature dataset for {fruit}...")
        X, y = build_dataset(fruit)
        print(f"Loaded {len(X)} samples for {fruit} across classes: {set(y)}")

        if len(X) < 10:
            print(f"Not enough images for {fruit}. Add images to "
                  f"datasets/fruit_ripeness/{fruit}/<class>/ folders. Skipping.")
            continue

        plot_class_distribution(y, fruit)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = SVC(kernel='rbf', probability=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print(f"Test set performance for {fruit}:")
        print(classification_report(y_test, y_pred))

        accuracies[fruit] = accuracy_score(y_test, y_pred)
        plot_confusion_matrix(y_test, y_pred, classes=CLASSES, fruit=fruit)

        # retrain on full dataset before saving, so the saved model uses all available data
        clf.fit(X_scaled, y)

        os.makedirs(MODEL_OUT_DIR, exist_ok=True)
        out_path = os.path.join(MODEL_OUT_DIR, f"{fruit}_ensemble_ab.pkl")
        joblib.dump({"model": clf, "scaler": scaler}, out_path)
        print(f"Model saved to {out_path}")

        per_fruit_seconds[fruit] = time.time() - fruit_start
        print(f"[time] {fruit} took {format_duration(per_fruit_seconds[fruit])}")

    total_seconds = time.time() - run_start
    save_training_time(total_seconds, per_fruit_seconds)
    print(f"\n[time] Total training run took {format_duration(total_seconds)}")

    if accuracies:
        summary_path = plot_accuracy_summary(accuracies)
        print(f"\nAll graphs saved to outputs/training/")
        print(f"Accuracy summary: {summary_path}")