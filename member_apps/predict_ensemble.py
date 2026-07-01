# ---------------------------------------------------------------------------
# Upgrade path (do this once all 4 members are trained):
#
# Change each member's predict_ripeness() to also return the full probability
# array (not just the max), e.g.:
#     return label, confidence, bbox, cleaned, proba_dict
# where proba_dict = {"ripe": 0.7, "unripe": 0.2, "rotten": 0.1}
#
# Then in predict_ensemble(), instead of majority-voting on labels, average
# the proba_dict across all members class-by-class and pick the highest —
# that's true soft voting and is usually more accurate than hard voting.
# ---------------------------------------------------------------------------

"""
Combines all four members' predictions (A+B, B+C, C+D, D+A) into one final
ripeness verdict for a single uploaded photo.

Design notes:
- Each member's predict_ripeness() is loaded independently and wrapped in a
  try/except. If a member's model isn't trained yet, or their predict.py
  still has bugs, the ensemble skips that member instead of crashing.
- Combination strategy: MAJORITY VOTE across members that succeeded, with
  average confidence (of the members who voted for the winning label) used
  as the final confidence score.
- This is intentionally simple/robust rather than "true" soft-voting, because
  right now member predict_ripeness() functions only return the top class's
  confidence, not a full probability vector over all 3 classes. See the
  "Upgrade path" note at the bottom of this file for how to make it soft-vote.
"""
import os
import sys
from collections import Counter

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
    """Calls a member's predict_ripeness, tolerating both signatures:
    member_1 takes (raw_img, fruit_type); members 2-4 currently only take (raw_img)
    since they haven't been updated to be fruit-aware yet."""
    try:
        label, confidence, bbox, _ = fn(raw_img, fruit_type)
    except TypeError:
        label, confidence, bbox, _ = fn(raw_img)
    return label, float(confidence), bbox


def predict_ensemble(raw_img, fruit_type):
    """
    Runs every successfully-loaded member model on the same image.

    Returns:
        final_label (str)
        final_confidence (float, 0-100)
        per_member (dict) -- each member's individual result or error, for transparency
        bbox (tuple or None)
    """
    per_member = {}
    bbox = None

    for member, fn in _MEMBER_PREDICTORS.items():
        try:
            label, confidence, member_bbox = _run_member(member, fn, raw_img, fruit_type)
            per_member[member] = {"label": label, "confidence": round(confidence * 100, 1)}
            bbox = bbox or member_bbox
        except Exception as e:
            per_member[member] = {"label": None, "confidence": None, "error": str(e)}

    for member in _LOAD_ERRORS:
        per_member.setdefault(member, {"label": None, "confidence": None, "error": _LOAD_ERRORS[member]})

    valid = {m: r for m, r in per_member.items() if r.get("label")}

    if not valid:
        raise RuntimeError(
            "No member models could produce a prediction. "
            "Check that trained_models/ has the right .pkl files for at least one member."
        )

    votes = Counter(r["label"] for r in valid.values())
    top_label, _ = votes.most_common(1)[0]

    supporting = [r["confidence"] for r in valid.values() if r["label"] == top_label]
    final_confidence = round(sum(supporting) / len(supporting), 1)

    return top_label, final_confidence, per_member, bbox


# ---------------------------------------------------------------------------
# Upgrade path (do this once all 4 members are trained):
#
# Change each member's predict_ripeness() to also return the full probability
# array (not just the max), e.g.:
#     return label, confidence, bbox, cleaned, proba_dict
# where proba_dict = {"ripe": 0.7, "unripe": 0.2, "rotten": 0.1}
#
# Then in predict_ensemble(), instead of majority-voting on labels, average
# the proba_dict across all members class-by-class and pick the highest —
# that's true soft voting and is usually more accurate than hard voting.
# ---------------------------------------------------------------------------


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

    print(f"\nFinal ensemble result: {label.upper()} ({confidence}% confidence)\n")
    print("Per-member breakdown:")
    for member, result in per_member.items():
        if result.get("error"):
            print(f"  {member}: SKIPPED — {result['error']}")
        else:
            print(f"  {member}: {result['label']} ({result['confidence']}%)")