"""
Combines all four members' predictions (A+B, B+C, C+D, D+A) into one final
ripeness verdict for a single uploaded photo.

Design notes:
- Each member's predict_ripeness() is loaded independently and wrapped in a
  try/except. If a member's model isn't trained yet, or their predict.py
  still has bugs, the ensemble skips that member instead of crashing.
- Combination strategy: SOFT VOTING. Each member returns its full 3-class
  probability distribution (ripe/unripe/rotten), not just its top label.
  We average those distributions across every member that succeeded, and
  the final label is whichever class has the highest averaged probability.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))

MEMBER_MODULES = {
    "member_1_ab": ("member_1_ab", "m1_predict"),
    "member_2_bc": ("member_2_bc", "m2_predict"),
    "member_3_cd": ("member_3_cd", "m3_predict"),
    "member_4_da": ("member_4_da", "m4_predict"),
}

_MEMBER_PREDICTORS = {}
_LOAD_ERRORS = {}


def _load_all_members():
    """Imports each member's predict module in isolation so one broken member
    doesn't stop the others from loading."""
    for member, (folder, module_name) in MEMBER_MODULES.items():
        member_dir = os.path.join(PROJECT_ROOT, "member_apps", folder)
        if member_dir not in sys.path:
            sys.path.insert(0, member_dir)
        try:
            module = __import__(module_name)
            _MEMBER_PREDICTORS[member] = module.predict_ripeness
        except Exception as e:
            _LOAD_ERRORS[member] = str(e)
            print(f"[ensemble] Skipping {member} (failed to load): {e}")


_load_all_members()


def _run_member(member, fn, raw_img, fruit_type):
    """Calls a member's predict_ripeness, tolerating both signatures."""
    try:
        label, confidence, bbox, _, proba_dict = fn(raw_img, fruit_type)
    except TypeError:
        label, confidence, bbox, _, proba_dict = fn(raw_img)
    return label, float(confidence), bbox, proba_dict


def predict_ensemble(raw_img, fruit_type):
    """
    Runs every successfully-loaded member model on the same image and
    combines them via soft voting.

    Returns:
        final_label (str)
        final_confidence (float, 0-100)
        per_member (dict) -- each member's individual result or error
        bbox (tuple or None)
    """
    per_member = {}
    bbox = None

    for member, fn in _MEMBER_PREDICTORS.items():
        try:
            label, confidence, member_bbox, proba_dict = _run_member(member, fn, raw_img, fruit_type)
            per_member[member] = {
                "label": label,
                "confidence": round(confidence * 100, 1),
                "proba": {cls: round(p * 100, 1) for cls, p in proba_dict.items()},
            }
            bbox = bbox or member_bbox
        except Exception as e:
            per_member[member] = {"label": None, "confidence": None, "error": str(e)}

    for member in _LOAD_ERRORS:
        per_member.setdefault(member, {"label": None, "confidence": None, "error": _LOAD_ERRORS[member]})

    valid = {m: r for m, r in per_member.items() if r.get("label") and r.get("proba")}

    if not valid:
        raise RuntimeError(
            "No member models could produce a prediction. "
            "Check that trained_models/ has the right .pkl files for at least one member."
        )

    all_classes = set()
    for r in valid.values():
        all_classes.update(r["proba"].keys())

    averaged = {}
    for cls in all_classes:
        probs = [r["proba"].get(cls, 0.0) for r in valid.values()]
        averaged[cls] = sum(probs) / len(probs)

    top_label = max(averaged, key=averaged.get)
    final_confidence = round(averaged[top_label], 1)

    return top_label, final_confidence, per_member, bbox


if __name__ == "__main__":
    import cv2
    if len(sys.argv) < 3:
        print("Usage: python predict_ensemble.py <image_path> <fruit_type>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)

    label, confidence, per_member, bbox = predict_ensemble(img, sys.argv[2])

    print(f"\nFinal ensemble result (soft voting): {label.upper()} ({confidence}% confidence)\n")
    print("Per-member breakdown:")
    for member, result in per_member.items():
        if result.get("error"):
            print(f"  {member}: SKIPPED — {result['error']}")
        else:
            proba_str = ", ".join(f"{cls}: {p}%" for cls, p in result["proba"].items())
            print(f"  {member}: top={result['label']} ({result['confidence']}%)  [{proba_str}]")
