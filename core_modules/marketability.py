"""Conservative post-prediction marketability estimates for whole fruit.

This module does not alter, retrain, calibrate, or override any ripeness model.
It converts an existing model result into a farmer-facing handling suggestion.
The day ranges are project heuristics for produce kept near the listed
recommended post-harvest conditions; they are not food-safety guarantees.
"""

from math import isfinite


# Conservative operational starting ranges for remaining *marketable* life.
# They intentionally represent wider ranges because cultivar, harvest history,
# temperature, humidity, handling, and internal defects are unknown to an image.
# Storage-condition references: UC Davis Postharvest Research and Extension
# Center produce fact sheets for apple, banana, mango, and orange.
MARKETABLE_LIFE_DAYS = {
    "apple": {"unripe": (14, 28), "ripe": (7, 14)},
    "banana": {"unripe": (10, 21), "ripe": (3, 7)},
    "mango": {"unripe": (7, 14), "ripe": (3, 7)},
    "orange": {"unripe": (21, 45), "ripe": (14, 30)},
}

STORAGE_ASSUMPTIONS = {
    "apple": "whole fruit in suitable cold storage near 0-4°C and 90-95% RH",
    "banana": "whole fruit stored near 13-14°C and 90-95% RH",
    "mango": "whole fruit stored near 13°C when mature-green or near 10°C when ripe, at 90-95% RH",
    "orange": "whole fruit stored near 3-8°C and 90-95% RH",
}

MIN_RELIABLE_CONFIDENCE = 60.0
HIGH_CONFIDENCE = 80.0
ELEVATED_ROTTEN_PROBABILITY = 20.0
HIGH_ROTTEN_PROBABILITY = 40.0


def _number(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def _normalise_probabilities(probabilities):
    """Return class probabilities on a 0-100 scale without changing a model."""
    if not probabilities:
        return {}
    cleaned = {}
    for label in ("unripe", "ripe", "rotten"):
        value = _number(probabilities.get(label))
        if value is None:
            continue
        # Member predictors use 0-1; API/ensemble payloads use 0-100.
        cleaned[label] = max(0.0, min(100.0, value * 100 if value <= 1 else value))
    return cleaned


def average_member_probabilities(per_member):
    """Average existing ensemble-member probabilities for post-processing."""
    totals = {label: [] for label in ("unripe", "ripe", "rotten")}
    for result in (per_member or {}).values():
        probabilities = _normalise_probabilities(result.get("proba") or {})
        for label, value in probabilities.items():
            totals[label].append(value)
    return {
        label: round(sum(values) / len(values), 1)
        for label, values in totals.items()
        if values
    }


def _result(
    *, fruit, status, priority, action, reliability, min_days, max_days,
    confidence, rotten_probability, blemish_percentage, reasons,
):
    if min_days is None or max_days is None:
        window = None
    elif min_days == max_days:
        window = f"{min_days} day" if min_days == 1 else f"{min_days} days"
    else:
        window = f"{min_days}-{max_days} days"

    return {
        "status": status,
        "dispatch_priority": priority,
        "min_days": min_days,
        "max_days": max_days,
        "window": window,
        "action": action,
        "reliability": reliability,
        "storage_assumption": STORAGE_ASSUMPTIONS.get(fruit),
        "confidence": round(confidence, 1) if confidence is not None else None,
        "rotten_probability": round(rotten_probability, 1) if rotten_probability is not None else None,
        "blemish_percentage": round(blemish_percentage, 2) if blemish_percentage is not None else None,
        "reasons": reasons,
        "disclaimer": "Image-based operational estimate only; inspect fruit before sale or disposal.",
    }


def estimate_marketability(
    fruit,
    ripeness,
    confidence,
    probabilities=None,
    blemish_percentage=None,
    quality_grade=None,
):
    """Estimate remaining marketable life without modifying model output.

    Safety/consistency invariants:
    - A model label of ``rotten`` can never receive a positive day range.
    - Strong contradictory decay evidence causes isolation/manual inspection,
      never a "fresh" or normal-market recommendation.
    - Low-confidence non-rotten results withhold the day estimate.
    """
    fruit = str(fruit or "").strip().lower()
    ripeness = str(ripeness or "").strip().lower()
    # Application routes consistently supply confidence on a 0-100 scale.
    # Keeping one explicit unit avoids treating a genuine 1% result as 100%.
    confidence_pct = _number(confidence)
    probability_pct = _normalise_probabilities(probabilities)
    rotten_probability = probability_pct.get("rotten")
    blemish = _number(blemish_percentage)
    grade = str(quality_grade or "Unknown")

    if fruit not in MARKETABLE_LIFE_DAYS:
        return _result(
            fruit=fruit, status="inspect", priority="unknown",
            action="No marketability profile is available for this fruit; inspect manually.",
            reliability="unavailable", min_days=None, max_days=None,
            confidence=confidence_pct, rotten_probability=rotten_probability,
            blemish_percentage=blemish, reasons=["unsupported fruit type"],
        )

    # The predicted class remains authoritative. Post-processing never turns a
    # rotten classification into a saleable result, even at low confidence.
    if ripeness == "rotten":
        return _result(
            fruit=fruit, status="remove", priority="remove",
            action="Do not market this fruit. Isolate it and inspect before disposal.",
            reliability="high" if confidence_pct is not None and confidence_pct >= HIGH_CONFIDENCE else "moderate",
            min_days=0, max_days=0, confidence=confidence_pct,
            rotten_probability=rotten_probability, blemish_percentage=blemish,
            reasons=["ripeness model classified the fruit as rotten"],
        )

    if ripeness not in ("unripe", "ripe"):
        return _result(
            fruit=fruit, status="inspect", priority="unknown",
            action="Ripeness result is not recognised; inspect manually.",
            reliability="unavailable", min_days=None, max_days=None,
            confidence=confidence_pct, rotten_probability=rotten_probability,
            blemish_percentage=blemish, reasons=["unrecognised ripeness result"],
        )

    if rotten_probability is not None and rotten_probability >= HIGH_ROTTEN_PROBABILITY:
        return _result(
            fruit=fruit, status="isolate", priority="urgent",
            action="Possible decay detected. Isolate this fruit and inspect it before marketing.",
            reliability="low", min_days=None, max_days=None,
            confidence=confidence_pct, rotten_probability=rotten_probability,
            blemish_percentage=blemish,
            reasons=["high rotten-class probability conflicts with the final label"],
        )

    if confidence_pct is None or confidence_pct < MIN_RELIABLE_CONFIDENCE:
        return _result(
            fruit=fruit, status="inspect", priority="urgent",
            action="Prediction is uncertain. Re-scan in better lighting or inspect manually before marketing.",
            reliability="low", min_days=None, max_days=None,
            confidence=confidence_pct, rotten_probability=rotten_probability,
            blemish_percentage=blemish, reasons=["prediction confidence is below 60%"],
        )

    min_days, max_days = MARKETABLE_LIFE_DAYS[fruit][ripeness]
    reasons = [f"{ripeness} model classification"]
    multiplier = 1.0
    priority = "normal" if ripeness == "unripe" else "high"
    status = "hold" if ripeness == "unripe" else "ready"

    if rotten_probability is not None and rotten_probability >= ELEVATED_ROTTEN_PROBABILITY:
        multiplier *= 0.5
        priority = "urgent"
        reasons.append("elevated rotten-class probability")

    if blemish is not None:
        if blemish > 30:
            return _result(
                fruit=fruit, status="inspect", priority="urgent",
                action="Severe visible surface damage detected. Isolate and inspect before marketing.",
                reliability="low", min_days=None, max_days=None,
                confidence=confidence_pct, rotten_probability=rotten_probability,
                blemish_percentage=blemish, reasons=["visible blemish exceeds 30%"],
            )
        if blemish > 15 or grade == "Grade C":
            multiplier *= 0.5
            priority = "urgent"
            reasons.append("Grade C or greater than 15% visible blemish")
        elif blemish > 5 or grade == "Grade B":
            multiplier *= 0.8
            if ripeness == "ripe":
                priority = "urgent"
            reasons.append("Grade B or greater than 5% visible blemish")
    else:
        reasons.append("surface condition unavailable")

    min_days = max(1, int(round(min_days * multiplier)))
    max_days = max(min_days, int(round(max_days * multiplier)))
    reliability = "high" if confidence_pct >= HIGH_CONFIDENCE and blemish is not None else "moderate"

    if ripeness == "unripe":
        action = f"Hold for ripening and monitor. Estimated marketable window: {min_days}-{max_days} days."
    elif priority == "urgent":
        action = f"Prioritise inspection and dispatch. Estimated marketable window: {min_days}-{max_days} days."
    else:
        action = f"Ready for market. Plan dispatch within approximately {min_days}-{max_days} days."

    return _result(
        fruit=fruit, status=status, priority=priority, action=action,
        reliability=reliability, min_days=min_days, max_days=max_days,
        confidence=confidence_pct, rotten_probability=rotten_probability,
        blemish_percentage=blemish, reasons=reasons,
    )
