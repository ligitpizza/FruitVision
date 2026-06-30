import os
import sys
import glob
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core_modules.preprocessing import preprocess, load_image
from core_modules.ma_colour_space import extract_colour
from core_modules.mb_shape_contours import extract_shape

FRUITS = ["apple", "banana", "orange"]
CLASSES = ["unripe", "ripe", "rotten"]


def build_dataset(fruit):
    X, y = [], []
    for label in CLASSES:
        folder = f"../../datasets/self_prepared/{fruit}/{label}"
        if not os.path.isdir(folder):
            continue
        for path in glob.glob(f"{folder}/*.*"):
            try:
                img = load_image(path)
                cleaned, _ = preprocess(img)
                vec = np.concatenate([extract_colour(cleaned), extract_shape(cleaned)])
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
                  f"datasets/self_prepared/{fruit}/<class>/ folders. Skipping.")
            continue

        clf = SVC(kernel='rbf', probability=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf.fit(X_train, y_train)

        print(f"Test set performance for {fruit}:")
        print(classification_report(y_test, clf.predict(X_test)))

        # retrain on full dataset before saving, so the saved model uses all available data
        clf.fit(X, y)

        os.makedirs("../../trained_models", exist_ok=True)
        out_path = f"../../trained_models/{fruit}_ensemble_ab.pkl"
        joblib.dump(clf, out_path)
        print(f"Model saved to {out_path}")