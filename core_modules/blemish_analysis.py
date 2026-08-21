"""Shared classical surface-quality analysis for FruitiVision.

The ripeness predictors only provide a bounding box, so this module creates a
pixel-accurate fruit mask inside that box before measuring surface anomalies.
It deliberately runs after prediction and does not alter any member pipeline.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# Project-defined grades. These are configurable assignment thresholds, not
# agricultural or industry standards.
GRADE_A_MAX = 5.0
GRADE_B_MAX = 15.0


@dataclass(frozen=True)
class BlemishConfig:
    """Centralised detector parameters, expressed in OpenCV Lab units."""

    min_fruit_area_px: int = 250
    min_fruit_fraction: float = 0.02
    max_fruit_fraction: float = 0.98
    local_window_fraction: float = 0.09
    local_window_min: int = 15
    local_window_max: int = 51
    local_colour_deviation: float = 24.0
    local_luminance_drop: float = 10.0
    local_chroma_deviation: float = 28.0
    global_colour_deviation: float = 45.0
    global_luminance_drop: float = 28.0
    edge_erosion_fraction: float = 0.012
    edge_erosion_min_px: int = 2
    morphology_open_size: int = 3
    morphology_close_size: int = 5
    min_component_px: int = 20
    min_component_fraction: float = 0.001
    mask_close_size: int = 7


DEFAULT_CONFIG = BlemishConfig()


def quality_grade(blemish_percentage: Optional[float]) -> str:
    """Return the project-defined surface grade for a valid percentage."""
    if blemish_percentage is None or not np.isfinite(blemish_percentage):
        return "Unknown"
    if blemish_percentage <= GRADE_A_MAX:
        return "Grade A"
    if blemish_percentage <= GRADE_B_MAX:
        return "Grade B"
    return "Grade C"


def calculate_blemish_percentage(fruit_mask: np.ndarray, blemish_mask: np.ndarray) -> Optional[float]:
    """Calculate blemished visible area, always ignoring pixels off the fruit."""
    if fruit_mask is None or blemish_mask is None or fruit_mask.shape != blemish_mask.shape:
        return None
    fruit = fruit_mask > 0
    fruit_area = int(np.count_nonzero(fruit))
    if fruit_area == 0:
        return None
    blemish_area = int(np.count_nonzero((blemish_mask > 0) & fruit))
    return 100.0 * blemish_area / fruit_area


def failure_result(message: str):
    """Canonical result for an analysis that could not be performed."""
    return {
        "fruit_area_px": 0,
        "blemish_area_px": 0,
        "blemish_percentage": None,
        "quality_grade": "Unknown",
        "fruit_mask": None,
        "blemish_mask": None,
        "surface_overlay": None,
        "surface_analysis_error": message,
    }


def _odd_kernel(value: int, minimum: int, maximum: int) -> int:
    maximum = max(3, maximum)
    minimum = min(maximum, max(3, minimum))
    value = max(minimum, min(maximum, value))
    if value % 2 == 0:
        value -= 1
    return max(3, value)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    binary = np.uint8(mask > 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros_like(binary, dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.uint8(labels == largest) * 255


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def _fallback_colour_mask(roi: np.ndarray) -> np.ndarray:
    """Border-colour segmentation used only when GrabCut cannot initialise."""
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]), axis=0)
    background = np.median(border, axis=0)
    colour_distance = np.linalg.norm(lab - background, axis=2)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Colour difference handles white/black studio backgrounds; Otsu offers a
    # fallback for low-chroma fruit while the largest-component rule suppresses
    # scattered background texture.
    colour_candidate = np.uint8(colour_distance >= 18.0) * 255
    candidates = [colour_candidate, otsu, cv2.bitwise_not(otsu)]
    centre = (roi.shape[0] // 2, roi.shape[1] // 2)
    plausible = []
    for candidate in candidates:
        component = _largest_component(candidate)
        fraction = np.count_nonzero(component) / component.size
        includes_centre = bool(component[centre])
        if 0.02 <= fraction <= 0.98:
            plausible.append((includes_centre, fraction, component))
    if not plausible:
        return np.zeros(roi.shape[:2], dtype=np.uint8)
    return max(plausible, key=lambda item: (item[0], item[1]))[2]


def build_fruit_mask(roi: np.ndarray, config: BlemishConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Create a filled, single-component fruit mask for a BGR fruit ROI."""
    h, w = roi.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if h < 5 or w < 5:
        return mask

    grabcut_labels = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            roi,
            grabcut_labels,
            (1, 1, w - 2, h - 2),
            bg_model,
            fg_model,
            5,
            cv2.GC_INIT_WITH_RECT,
        )
        mask = np.uint8(
            (grabcut_labels == cv2.GC_FGD) | (grabcut_labels == cv2.GC_PR_FGD)
        ) * 255
    except cv2.error:
        mask = np.zeros((h, w), dtype=np.uint8)

    grabcut_mask = _largest_component(mask)
    fallback_mask = _fallback_colour_mask(roi)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, saturation_mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    saturation_mask = cv2.morphologyEx(
        saturation_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    saturation_mask = _largest_component(saturation_mask)

    border_saturation = np.median(
        np.concatenate((saturation[0], saturation[-1], saturation[:, 0], saturation[:, -1]))
    )

    def candidate_score(candidate):
        area = int(np.count_nonzero(candidate))
        fraction = area / candidate.size
        if not config.min_fruit_fraction <= fraction <= config.max_fruit_fraction:
            return float("-inf")
        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return float("-inf")
        contour = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 0.0
        x, y, cw, ch = cv2.boundingRect(contour)
        touched_sides = sum((x == 0, y == 0, x + cw >= w, y + ch >= h))
        centre_support = 1.0 if candidate[h // 2, w // 2] else 0.0
        mean_saturation = float(np.mean(saturation[candidate > 0]))
        saturation_contrast = mean_saturation - float(border_saturation)
        return saturation_contrast + 25.0 * solidity + 15.0 * centre_support - 20.0 * touched_sides

    candidates = [grabcut_mask, fallback_mask, saturation_mask]
    compact_candidates = []
    for candidate in candidates:
        fraction = np.count_nonzero(candidate) / candidate.size
        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not (config.min_fruit_fraction <= fraction <= config.max_fruit_fraction and contours):
            continue
        contour = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = np.count_nonzero(candidate) / hull_area if hull_area > 0 else 0.0
        x, y, cw, ch = cv2.boundingRect(contour)
        touched_sides = sum((x == 0, y == 0, x + cw >= w, y + ch >= h))
        if solidity >= 0.65 and touched_sides <= 1:
            compact_candidates.append(candidate)

    # The largest compact, non-border object best preserves low-saturation
    # damage (for example pale mould) as visible fruit. If no candidate meets
    # that geometric test, fall back to colour/centre/border scoring.
    mask = (
        max(compact_candidates, key=np.count_nonzero)
        if compact_candidates
        else max(candidates, key=candidate_score)
    )
    if not np.isfinite(candidate_score(mask)):
        return np.zeros((h, w), dtype=np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.mask_close_size, config.mask_close_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return _fill_holes(_largest_component(mask))


def _normalise_bbox(bbox, image_shape) -> Optional[Tuple[int, int, int, int]]:
    if bbox is None or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (int(round(float(v))) for v in bbox)
    except (TypeError, ValueError, OverflowError):
        return None
    height, width = image_shape[:2]
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    return x0, y0, x1, y1


def _clean_components(mask: np.ndarray, valid_mask: np.ndarray, minimum_area: int) -> np.ndarray:
    binary = np.uint8((mask > 0) & (valid_mask > 0))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    cleaned = np.zeros_like(binary)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[labels == label] = 255
    return cv2.bitwise_and(cleaned, np.uint8(valid_mask > 0) * 255)


def detect_blemishes(
    roi: np.ndarray,
    fruit_mask: np.ndarray,
    config: BlemishConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Segment abnormal dark/discoloured regions inside a known fruit mask."""
    fruit = np.uint8(fruit_mask > 0) * 255
    fruit_area = int(np.count_nonzero(fruit))
    if fruit_area == 0:
        return np.zeros_like(fruit)

    min_dim = min(roi.shape[:2])
    erosion_px = max(config.edge_erosion_min_px, int(round(min_dim * config.edge_erosion_fraction)))
    erosion_size = 2 * erosion_px + 1
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
    valid = cv2.erode(fruit, erosion_kernel)
    if np.count_nonzero(valid) < config.min_fruit_area_px:
        valid = fruit.copy()

    lab_u8 = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    lab = lab_u8.astype(np.float32)
    window = _odd_kernel(
        int(round(min_dim * config.local_window_fraction)),
        config.local_window_min,
        min(config.local_window_max, min_dim if min_dim % 2 else min_dim - 1),
    )
    local = np.dstack([cv2.medianBlur(lab_u8[:, :, channel], window) for channel in range(3)]).astype(np.float32)

    difference = lab - local
    local_colour = np.linalg.norm(difference, axis=2)
    local_chroma = np.linalg.norm(difference[:, :, 1:3], axis=2)
    local_dark = local[:, :, 0] - lab[:, :, 0]

    sample = lab[valid > 0]
    global_median = np.median(sample, axis=0)
    global_difference = lab - global_median
    global_colour = np.linalg.norm(global_difference, axis=2)
    global_dark = global_median[0] - lab[:, :, 0]

    dark_local = (
        (local_colour >= config.local_colour_deviation)
        & (local_dark >= config.local_luminance_drop)
    )
    discoloured_local = (
        (local_colour >= config.local_colour_deviation)
        & (local_chroma >= config.local_chroma_deviation)
    )
    dark_global = (
        (global_colour >= config.global_colour_deviation)
        & (global_dark >= config.global_luminance_drop)
    )
    candidate = np.uint8((dark_local | discoloured_local | dark_global) & (valid > 0)) * 255

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.morphology_open_size, config.morphology_open_size)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.morphology_close_size, config.morphology_close_size)
    )
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, open_kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, close_kernel)
    minimum_area = max(config.min_component_px, int(round(fruit_area * config.min_component_fraction)))
    return _clean_components(candidate, fruit, minimum_area)


def _analyze_surface(
    image: np.ndarray,
    bbox=None,
    fruit_mask: Optional[np.ndarray] = None,
    config: BlemishConfig = DEFAULT_CONFIG,
):
    """Measure blemishes on an original BGR image using its final localisation."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return failure_result("Unreadable or empty image")
    if image.ndim != 3 or image.shape[2] != 3:
        return failure_result("Surface analysis requires a three-channel BGR image")
    if image.dtype != np.uint8:
        image = np.uint8(np.clip(image, 0, 255))

    normalised_bbox = _normalise_bbox(bbox, image.shape)
    if normalised_bbox is None:
        return failure_result("No valid fruit region detected")
    x0, y0, x1, y1 = normalised_bbox
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return failure_result("Empty fruit region")

    if fruit_mask is None:
        roi_fruit_mask = build_fruit_mask(roi, config)
    else:
        supplied = np.asarray(fruit_mask)
        if supplied.shape == image.shape[:2]:
            supplied = supplied[y0:y1, x0:x1]
        if supplied.shape != roi.shape[:2]:
            return failure_result("Fruit mask dimensions do not match the image or fruit region")
        roi_fruit_mask = np.uint8(supplied > 0) * 255

    fruit_area = int(np.count_nonzero(roi_fruit_mask))
    if fruit_area < config.min_fruit_area_px:
        return failure_result("Fruit region is too small for reliable surface analysis")

    roi_blemish_mask = detect_blemishes(roi, roi_fruit_mask, config)
    # This final intersection is an invariant, even if future detection code
    # changes: a blemish can never be counted outside the visible fruit.
    roi_blemish_mask = cv2.bitwise_and(roi_blemish_mask, roi_fruit_mask)
    blemish_area = int(np.count_nonzero(roi_blemish_mask))
    percentage = calculate_blemish_percentage(roi_fruit_mask, roi_blemish_mask)

    full_fruit_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    full_blemish_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    full_fruit_mask[y0:y1, x0:x1] = roi_fruit_mask
    full_blemish_mask[y0:y1, x0:x1] = roi_blemish_mask

    overlay = image.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    blemished = full_blemish_mask > 0
    overlay[blemished] = cv2.addWeighted(overlay, 0.35, red, 0.65, 0)[blemished]
    contours, _ = cv2.findContours(full_fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 200, 0), 2)

    return {
        "fruit_area_px": fruit_area,
        "blemish_area_px": blemish_area,
        "blemish_percentage": round(float(percentage), 2),
        "quality_grade": quality_grade(percentage),
        "fruit_mask": full_fruit_mask,
        "blemish_mask": full_blemish_mask,
        "surface_overlay": overlay,
        "surface_analysis_error": None,
    }


def analyze_surface(
    image: np.ndarray,
    bbox=None,
    fruit_mask: Optional[np.ndarray] = None,
    config: BlemishConfig = DEFAULT_CONFIG,
):
    """Safe public entry point; surface-analysis failures never break prediction."""
    try:
        return _analyze_surface(image, bbox=bbox, fruit_mask=fruit_mask, config=config)
    except (cv2.error, ValueError, TypeError, OverflowError) as exc:
        return failure_result(f"Surface analysis failed: {exc}")
