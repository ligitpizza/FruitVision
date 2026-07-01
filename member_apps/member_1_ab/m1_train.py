import os
import sys
import glob
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..', '..'))

from core_modules.preprocessing import preprocess, load_image
from core_modules.ma_colour_space import extract_colour
from core_modules.mb_shape_contours import extract_shape

DATASET_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "datasets", "fruit_ripeness"))
MODEL_OUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "trained_models"))

FRUITS = ["apple", "banana", "orange"]
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
                cleaned, _ = preprocess(img)
                vec_a = extract_colour(cleaned)
                vec_b = extract_shape(cleaned)
                vec = np.concatenate([vec_a, vec_b])
                X.append(vec)
                y.append(label)
            except Exception as e:
                print(f"Skipping {path}: {e}")
    return np.array(X), np.array(y)


if __name__ == "__main__":
    for fruit in FRUITS:
        print(f"\nBuilding feature dataset for {fruit}...")
        X, y = build_dataset(fruit)
        print(f"Loaded {len(X)} samples for {fruit} across classes: {set(y)}")

        if len(X) < 10:
            print(f"Not enough images for {fruit}. Add images to "
                  f"datasets/fruit_ripeness/{fruit}/<class>/ folders. Skipping.")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = SVC(kernel='rbf', probability=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        clf.fit(X_train, y_train)

        print(f"Test set performance for {fruit}:")
        print(classification_report(y_test, clf.predict(X_test)))

        # retrain on full dataset before saving, so the saved model uses all available data
        clf.fit(X_scaled, y)

        os.makedirs(MODEL_OUT_DIR, exist_ok=True)
        out_path = os.path.join(MODEL_OUT_DIR, f"{fruit}_ensemble_ab.pkl")
        joblib.dump({"model": clf, "scaler": scaler}, out_path)
        print(f"Model saved to {out_path}")