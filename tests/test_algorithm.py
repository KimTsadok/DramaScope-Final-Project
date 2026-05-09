# Testing algorithm scripts (complexity.py, features.py, normalize.py, phase.py).

import json
import sys
from pathlib import Path

from src.algorithm.complexity import compute_scene_complexity
from src.algorithm.features import build_raw_features
from src.algorithm.normalize import normalize_features
from src.algorithm.phase import classify_narrative_phase
from src.config import OUTPUT_FILES


def extract_video_id(video_uri: str) -> str:
    return Path(video_uri).stem if video_uri else "unknown_video"


def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def auto_find_features_file() -> Path | None:
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return None

    candidates = list(outputs_dir.rglob(OUTPUT_FILES.raw_features_filename))
    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_input_path() -> Path:
    if len(sys.argv) >= 2:
        return Path(sys.argv[1])

    auto_path = auto_find_features_file()
    if auto_path:
        print(f"No input path provided. Using latest: {auto_path}")
        return auto_path

    print("Usage:")
    print("python -m src.pipeline.test_algorithm <path_to_VideoFeatures.json>")
    print("or run without args after generating outputs/*.json")
    sys.exit(1)


def main() -> None:
    input_path = resolve_input_path()

    if not input_path.exists():
        print(f"Error: file not found -> {input_path}")
        sys.exit(1)

    with input_path.open("r", encoding="utf-8") as f:
        video_features = json.load(f)

    video_uri = video_features.get("video_uri", "")
    video_id = extract_video_id(video_uri)

    raw_features = build_raw_features(video_features)
    norm_features = normalize_features(raw_features)
    complexity_score, breakdown = compute_scene_complexity(norm_features)
    narrative_phase, phase_reasons = classify_narrative_phase(norm_features)

    interpretation = {
        "video_id": video_id,
        "video_uri": video_uri,
        "features_raw": raw_features,
        "features_norm": norm_features,
        "scene_complexity_score": complexity_score,
        "scene_complexity_breakdown": breakdown,
        "narrative_phase": narrative_phase,
        "phase_reasons": phase_reasons,
    }

    output_path = input_path.parent / OUTPUT_FILES.interpretation_filename
    save_json(interpretation, output_path)

    print("\nRaw Features:")
    print(json.dumps(raw_features, indent=2))
    print("\nNormalized Features:")
    print(json.dumps(norm_features, indent=2))
    print("\nScene Complexity Score:")
    print(round(complexity_score, 6))
    print("\nBreakdown:")
    print(json.dumps(breakdown, indent=2))
    print("\nNarrative Phase:")
    print(narrative_phase)
    print("\nPhase Reasons:")
    print(json.dumps(phase_reasons, indent=2))
    print(f"\nSaved interpretation JSON to: {output_path}")


if __name__ == "__main__":
    main()
