"""Manual smoke check for all four trained member predictors plus ensemble."""
import os
import sys

import cv2

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
for folder in ("member_1_ab", "member_2_bc", "member_3_cd", "member_4_da"):
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "member_apps", folder))

from core_modules.blemish_analysis import analyze_surface  # noqa: E402
from member_apps.member_1_ab.m1_predict import predict_ripeness as predict_ab  # noqa: E402
from member_apps.member_2_bc.m2_predict import predict_ripeness as predict_bc  # noqa: E402
from member_apps.member_3_cd.m3_predict import predict_ripeness as predict_cd  # noqa: E402
from member_apps.member_4_da.m4_predict import predict_ripeness as predict_da  # noqa: E402
from member_apps.predict_ensemble import predict_ensemble  # noqa: E402


def main():
    path = os.path.join(PROJECT_ROOT, "uploads", "banana.png")
    image = cv2.imread(path)
    if image is None:
        raise RuntimeError(f"Could not read {path}")

    for name, predictor in (("ab", predict_ab), ("bc", predict_bc), ("cd", predict_cd), ("da", predict_da)):
        label, confidence, bbox, _, _ = predictor(image, "banana")
        surface = analyze_surface(image, bbox=bbox)
        print(name, label, round(confidence * 100, 1), surface["blemish_percentage"], surface["quality_grade"])

    label, confidence, per_member, bbox = predict_ensemble(image, "banana")
    surface = analyze_surface(image, bbox=bbox)
    print("all_four", label, confidence, surface["blemish_percentage"], surface["quality_grade"], len(per_member))


if __name__ == "__main__":
    main()
