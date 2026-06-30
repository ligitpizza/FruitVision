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

DATASET_DIRS = [
    "../../datasets/asadullah",
    "../../datasets/ryanpark",
    "../../datasets/self_prepared",
]
CLASSES = ["unripe", "ripe", "overripe"]
MODEL_OUT = "../../trained_models/ensemble_ab.pkl"


def build_dataset():
    X, y = [], []
    for dataset_dir in DATASET_DIRS:
        for label in CLASSES:
            folder = os.path.join(dataset_dir, label)
            if not os.path.isdir(folder):
                continue
            for path in glob.glob(os.path.join(folder, "*.*")):
                try:
                    img = load_image(path)
                    cleaned, _ = preprocess(img)
                    vec_a = extract_colour(cleaned)
                    vec_b = extract_shape(cleaned)
                    combined = np.concatenate([vec_a, vec_b])
                    X.append(combined)
                    y.append(label)
                except Exception as e:
                    print(f"Skipping {path}: {e}")
    return np.array(X), np.array(y)


if __name__ == "__main__":
    print("Building feature dataset...")
    X, y = build_dataset()
    print(f"Loaded {len(X)} samples across classes: {set(y)}")

    if len(X) < 10:
        print("Not enough images found. Add more to datasets/*/<class>/ folders.")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    clf.fit(X_train, y_train)

    print("\nTest set performance:")
    print(classification_report(y_test, clf.predict(X_test)))

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")