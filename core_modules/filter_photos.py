"""
Shared "filter technique" photo generation -- renders + saves each SVM
member's intermediate filtered image (colour/shape/texture/gabor) from a
prediction's calibrated cleaned_img.

Lives here (not in app.py) because both app.py's classify/batch routes AND
the realtime/*.py trackers need it, and realtime/*.py must never import
app.py (app.py already imports realtime.stream_routes -- importing the
other way would be circular).
"""
import os
import cv2
from werkzeug.utils import secure_filename

from core_modules.ma_colour_space import visualize_colour
from core_modules.mb_shape_contours import visualize_shape
from core_modules.mc_texture_glmc import visualize_texture
from core_modules.md_gabor_filters import visualize_gabor

# Which filter modules each SVM-based model's pipeline actually runs.
# yolo_pure has no entry -- it's a CNN, none of these hand-crafted filters
# run in its pipeline, so it gets no filter photos.
FILTER_STEPS = {
    "ab": ["colour", "shape"],
    "bc": ["shape", "texture"],
    "cd": ["texture", "gabor"],
    "da": ["gabor", "colour"],
    "merged_1_4": ["colour", "shape", "gabor"],
    "m14v2": ["colour", "shape", "gabor", "texture"],
    "m14v3": ["colour", "shape", "gabor", "texture"],
}
FILTER_VISUALIZERS = {
    "colour": visualize_colour,
    "shape": visualize_shape,
    "texture": visualize_texture,
    "gabor": visualize_gabor,
}
FILTER_LABELS = {
    "colour": "Colour Space (Lab A-channel)",
    "shape": "Shape / Contour",
    "texture": "Texture (GLCM)",
    "gabor": "Gabor Filter",
}
# predict_ensemble.py's per_member keys ("member_1_ab") -> FILTER_STEPS keys.
ENSEMBLE_MEMBER_TO_MODEL_KEY = {
    "member_1_ab": "ab", "member_2_bc": "bc", "member_3_cd": "cd", "member_4_da": "da",
}


def save_filter_photos_for_model(cleaned_img, model_key, filename, outputs_dir, name_suffix=""):
    """Renders + saves one photo per filter technique model_key's pipeline
    applies, from that prediction's calibrated cleaned_img. Returns
    {technique: relative_output_path}; {} if model_key has no filter steps
    (e.g. yolo_pure) or cleaned_img is unavailable."""
    steps = FILTER_STEPS.get(model_key)
    if not steps or cleaned_img is None:
        return {}
    safe_name = secure_filename(filename) or "fruit.jpg"
    stem, extension = os.path.splitext(safe_name)
    extension = extension if extension.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
    filters_dir = os.path.join(outputs_dir, "filters")
    os.makedirs(filters_dir, exist_ok=True)
    paths = {}
    for step in steps:
        try:
            viz_img = FILTER_VISUALIZERS[step](cleaned_img)
        except Exception:
            continue
        out_name = f"{stem}_{step}_{model_key}{name_suffix}{extension}"
        if cv2.imwrite(os.path.join(filters_dir, out_name), viz_img):
            paths[step] = f"filters/{out_name}"
    return paths


def pop_member_cleaned_images(per_member):
    """predict_ensemble() embeds each member's raw cleaned_img ndarray in
    per_member so filter photos can be generated from it; that ndarray can't
    survive JSON-serializing per_member (a Flask response) or being handed
    to a template, so pull it out into a separate {member: ndarray} dict
    first -- used by both app.py and the All-Four realtime tracker."""
    return {member: result.pop("cleaned_img", None) for member, result in per_member.items()}


def filter_photos_single(cleaned_img, model_key, filename, outputs_dir):
    """Filter photos for a single-model prediction, wrapped under its own
    model_key so the stored shape is uniform with the ensemble's per-member
    shape: {model_key: {technique: path}}."""
    paths = save_filter_photos_for_model(cleaned_img, model_key, filename, outputs_dir)
    return {model_key: paths} if paths else {}


def filter_photos_ensemble(cleaned_by_member, filename, outputs_dir):
    """Filter photos for an All-Four ensemble prediction: one sub-dict per
    member that actually produced a cleaned_img, keyed the same as
    predict_ensemble()'s per_member dict."""
    filter_photos = {}
    for member, cleaned_img in cleaned_by_member.items():
        model_key = ENSEMBLE_MEMBER_TO_MODEL_KEY.get(member)
        if not model_key or cleaned_img is None:
            continue
        paths = save_filter_photos_for_model(cleaned_img, model_key, filename, outputs_dir, name_suffix=f"_{member}")
        if paths:
            filter_photos[member] = paths
    return filter_photos
