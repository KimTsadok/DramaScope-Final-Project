# src/algorithm/features.py
"""
Compute raw features from existing metadata (VideoFeatures.json).
* compute shot_frequency
* pass through the other 3 features
* return a clean flat dictionary

Algorithm Features → cleaned / derived features used for scoring
VideoFeatures.json → clean ML-style feature vector
Prepare features for scoring + classification
"""

from typing import Any, Dict


def safe_divide(numerator: float, denominator: float) -> float: # no division with 0
    if denominator == 0:
        return 0.0
    return numerator / denominator


def build_raw_features(video_features: Dict[str, Any]) -> Dict[str, float]:
    duration_seconds = float(video_features.get("duration_seconds", 0.0))

    shots = video_features.get("shots", {})
    shot_count = float(shots.get("count", 0.0))

    objects = video_features.get("objects", {})
    object_entropy = float(objects.get("object_entropy", 0.0))
    interaction_density = float(objects.get("interaction_density_tracks_per_sec", 0.0))
    human_presence_ratio = float(objects.get("human_presence_ratio", 0.0))

    shot_frequency = safe_divide(shot_count, duration_seconds)

    return {
        "shot_frequency": shot_frequency,
        "object_entropy": object_entropy,
        "interaction_density": interaction_density,
        "human_presence_ratio": human_presence_ratio,
    } # build dict for VideoInterpertation.json