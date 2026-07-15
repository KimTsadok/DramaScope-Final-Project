# src/gcp/feature_engineering.py
"""
This file handles:
* time conversion
* interval merging
* entropy calculation
* video duration estimation
* feature engineering
* JSON saving

This is your richest file, because this is where raw annotations
become your VideoFeatures.json.
GCP Features  → raw metadata extracted from the video
RAW VIDEO → GCP → structured metadata → VideoFeatures.json
"""

import json
from collections import Counter
from math import log2
from pathlib import Path


def duration_to_seconds(d) -> float:
    """
    Convert one of the following into float seconds:
    - protobuf Duration (seconds + nanos)
    - datetime.timedelta (total_seconds)
    - numeric

    This prevents crashes when offsets are returned as timedelta.
    """
    if d is None:
        return 0.0

    if hasattr(d, "total_seconds"):
        return float(d.total_seconds())

    if hasattr(d, "seconds"):
        nanos = getattr(d, "nanos", 0)
        return float(d.seconds) + float(nanos) / 1e9

    return float(d)


def merge_intervals(intervals):
    """
    Merge overlapping intervals [(start, end), ...] and return merged intervals.
    """
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def intervals_total_length(intervals):
    """
    Sum the total length of intervals.
    """
    return sum(max(0.0, end - start) for start, end in intervals)


def shannon_entropy_from_counts(counts: Counter) -> float:
    """
    Compute Shannon entropy:
        H = -sum(p_i * log2(p_i))
    """
    total = sum(counts.values())
    if total <= 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * log2(probability)

    return entropy


def compute_video_duration_seconds(annotations) -> float:
    """
    Estimate video duration.

    Use the latest available end time across shot and object annotations.
    """
    shot_ends = [
        duration_to_seconds(shot.end_time_offset)
        for shot in getattr(annotations, "shot_annotations", [])
    ]
    object_ends = [
        duration_to_seconds(obj.segment.end_time_offset)
        for obj in getattr(annotations, "object_annotations", [])
    ]

    return max(shot_ends + object_ends, default=0.0)


def compute_features(annotations, video_uri: str) -> dict:
    """
    Compute the raw video metadata / feature JSON structure.

    Output includes:
    - duration
    - shot statistics
    - labels
    - object statistics
    - human presence ratio
    - object entropy
    - interaction density
    """
    duration_seconds = compute_video_duration_seconds(annotations)

    # ---- Shots ----
    shot_lengths = []
    for shot in getattr(annotations, "shot_annotations", []):
        start = duration_to_seconds(shot.start_time_offset)
        end = duration_to_seconds(shot.end_time_offset)
        if end > start:
            shot_lengths.append(end - start)

    if shot_lengths:
        avg_shot_length = sum(shot_lengths) / len(shot_lengths)
        shot_variance = sum((x - avg_shot_length) ** 2 for x in shot_lengths) / len(shot_lengths)
    else:
        avg_shot_length = 0.0
        shot_variance = 0.0

    # ---- Labels ----
    labels = []
    for label in getattr(annotations, "segment_label_annotations", []):
        description = getattr(label.entity, "description", None)
        if description:
            labels.append(description)

    # ---- Objects ----
    object_descriptions = []
    person_intervals = []

    for obj in getattr(annotations, "object_annotations", []):
        description = getattr(obj.entity, "description", None)
        if not description:
            continue

        object_descriptions.append(description)

        if description.lower() == "person":
            start = duration_to_seconds(obj.segment.start_time_offset)
            end = duration_to_seconds(obj.segment.end_time_offset)
            if end > start:
                person_intervals.append((start, end))

    object_counts = Counter(object_descriptions)

    # ---- Human presence ratio ----
    if duration_seconds > 0 and person_intervals:
        merged_person_intervals = merge_intervals(person_intervals)
        human_presence_ratio = intervals_total_length(merged_person_intervals) / duration_seconds
        human_presence_ratio = max(0.0, min(1.0, human_presence_ratio))
    else:
        human_presence_ratio = 0.0

    # ---- Object entropy ----
    object_entropy = shannon_entropy_from_counts(object_counts)

    # ---- Interaction density ----
    total_object_tracks = len(getattr(annotations, "object_annotations", []))
    interaction_density = (total_object_tracks / duration_seconds) if duration_seconds > 0 else 0.0

    object_frequency_distribution = dict(object_counts.most_common())

    return {
        "video_uri": video_uri,
        "duration_seconds": round(duration_seconds, 6),
        "shots": {
            "count": len(shot_lengths),
            "avg_shot_length": round(avg_shot_length, 6),
            "shot_variance": round(shot_variance, 6),
            "shot_lengths": [round(length, 6) for length in shot_lengths],
        },
        "labels": {
            "unique": sorted(set(labels)),
            "count_unique": len(set(labels)),
        },
        "objects": {
            "total_tracks": total_object_tracks,
            "object_frequency_distribution": object_frequency_distribution,
            "top_objects": [name for name, _ in object_counts.most_common(10)],
            "object_entropy": round(object_entropy, 6),
            "human_presence_ratio": round(human_presence_ratio, 6),
            "interaction_density_tracks_per_sec": round(interaction_density, 6),
        },
    }


def save_json(data: dict, out_path: str | Path, label: str = "JSON") -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"\nSaved {label} to: {out_file.resolve()}")
