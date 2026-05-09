# src/algorithm/validate.py
"""
Validation and repair helpers for the Week 4 pipeline.

Purpose:
- catch critical invalid metadata
- repair safe numeric issues
- clamp values where appropriate
- normalize LVLM structured fields
"""

import math
from typing import Any, Dict, List
from src.lvlm.parse import (ensure_string,
                            ensure_string_list) #helpers

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def is_bad_number(value: Any) -> bool:
    """
    Return True if value is missing, non-numeric, NaN, or infinite.
    """
    try:
        number = float(value)
        return math.isnan(number) or math.isinf(number)
    except (TypeError, ValueError):
        return True


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convert value to float. If invalid, return default.
    """
    if is_bad_number(value):
        return default
    return float(value)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


# ---------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------


def validate_video_features(video_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and lightly repair the raw VideoFeatures.json structure.

    Critical rule:
    - duration_seconds must be greater than 0.

    Safe repairs:
    - missing shots.count -> 0
    - missing objects fields -> 0.0
    - human_presence_ratio clamped to [0, 1]
    """
    duration_seconds = safe_float(video_features.get("duration_seconds"), default=0.0)

    if duration_seconds <= 0:
        raise ValueError("Invalid video features: duration_seconds must be greater than 0.")

    shots = video_features.setdefault("shots", {})
    objects = video_features.setdefault("objects", {})

    shots["count"] = int(safe_float(shots.get("count"), default=0.0))

    objects["object_entropy"] = safe_float(
        objects.get("object_entropy"),
        default=0.0,
    )

    objects["interaction_density_tracks_per_sec"] = safe_float(
        objects.get("interaction_density_tracks_per_sec"),
        default=0.0,
    )

    objects["human_presence_ratio"] = clamp(
        safe_float(objects.get("human_presence_ratio"), default=0.0),
        0.0,
        1.0,
    )

    return video_features


def validate_raw_features(raw_features: Dict[str, float]) -> Dict[str, float]:
    """
    Validate and repair raw algorithm features.

    These values are allowed to be 0, but not NaN/inf/invalid.
    human_presence_ratio is clamped to [0, 1].
    """
    cleaned = {
        "shot_frequency": safe_float(raw_features.get("shot_frequency"), 0.0),
        "object_entropy": safe_float(raw_features.get("object_entropy"), 0.0),
        "interaction_density": safe_float(raw_features.get("interaction_density"), 0.0),
        "human_presence_ratio": safe_float(raw_features.get("human_presence_ratio"), 0.0),
    }

    cleaned["human_presence_ratio"] = clamp(cleaned["human_presence_ratio"], 0.0, 1.0)

    return cleaned


def validate_normalized_features(norm_features: Dict[str, float]) -> Dict[str, float]:
    """
    Validate normalized features.

    All normalized features should be in [0, 1].
    Invalid values are repaired to 0.0.
    """
    return {
        "shot_frequency": clamp(safe_float(norm_features.get("shot_frequency"), 0.0), 0.0, 1.0),
        "object_entropy": clamp(safe_float(norm_features.get("object_entropy"), 0.0), 0.0, 1.0),
        "interaction_density": clamp(safe_float(norm_features.get("interaction_density"), 0.0), 0.0, 1.0),
        "human_presence_ratio": clamp(safe_float(norm_features.get("human_presence_ratio"), 0.0), 0.0, 1.0),
    }


def validate_lvlm_structured(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate LVLM structured output.

    Expected schema:
    {
      "setting": str,
      "main_entities": list[str],
      "actions": list[str],
      "emotion_words": list[str],
      "interaction_level": int in [0, 3],
      "summary": str
    }
    """
    interaction_level = data.get("interaction_level", 0)

    try:
        interaction_level = int(interaction_level)
    except (TypeError, ValueError):
        interaction_level = 0

    interaction_level = int(clamp(interaction_level, 0, 3))

    return {
        "setting": ensure_string(data.get("setting")),
        "main_entities": ensure_string_list(data.get("main_entities")),
        "actions": ensure_string_list(data.get("actions")),
        "emotion_words": ensure_string_list(data.get("emotion_words")),
        "interaction_level": interaction_level,
        "summary": ensure_string(data.get("summary")),
    }