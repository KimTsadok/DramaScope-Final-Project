# src/pipeline/run_full_pipeline.py
"""
This script takes one local video and runs the full video analysis pipeline.

Stage 1 — raw video metadata extraction
    - uploads the video to Google Cloud Storage if needed
    - analyzes the video with Google Video Intelligence
    - converts the raw API annotations into a structured metadata file
    - creates VideoFeatures.json
VideoFeatures.json format:
        * video_uri
        * duration_seconds
        shots:
            * count
            * avg_shot_length
            * shot_variance
            * shot_lengths
        labels:
            * unique
            * count_unique
        objects:
            * total_tracks
            * object_frequency_distribution
            * top_objects
            * object_entropy
            * human_presence_ratio
            * interaction_density_tracks_per_sec

Stage 2 — algorithmic scene interpretation
    - takes VideoFeatures.json from Stage 1
    - computes the 4 Week 3 raw features:
        * shot_frequency
        * object_entropy
        * interaction_density
        * human_presence_ratio
    - normalizes the features to [0, 1]
    - computes the scene complexity score
    - classifies the narrative phase
    - creates VideoInterpretation.json
VideoInterpretation.json algorithm fields:
        * video_id
        * video_uri
        * features_raw
        * features_norm
        * scene_complexity_score
        * scene_complexity_breakdown
        * narrative_phase
        * phase_reasons

Stage 3 — LVLM semantic interpretation
    - optionally extracts shared keyframes from the local video
    - sends the frames to the LVLM using two prompts:
        1. free-text scene summary
        2. structured JSON scene interpretation
    - parses and validates the structured LVLM response
    - adds semantic metadata to VideoInterpretation.json
VideoInterpretation.json LVLM fields:
        * lvlm_summary
        * lvlm_structured_raw
        * lvlm_structured
"""

import argparse
import json
from pathlib import Path

from src.gcp.upload import (
    DEFAULT_BUCKET_NAME,
    DEFAULT_GCS_PREFIX,
    upload_to_gcs,
)

from src.gcp.analyze import analyze_video_uri
from src.gcp.feature_engineering import compute_features, save_json
from src.algorithm.features import build_raw_features
from src.algorithm.normalize import normalize_features
from src.algorithm.complexity import compute_scene_complexity
from src.algorithm.phase import classify_narrative_phase
from src.lvlm.parse import parse_structured_response
from src.config import OUTPUT_FILES

from src.pipeline.run_gcp_features import (
    extract_video_id,
    build_features_output_path,
    print_gcp_summary,
)
from src.lvlm.client import (
    extract_frames,
    run_summary_inference_from_frames,
    run_structured_inference_from_frames,
)
from src.algorithm.validate import (
    validate_video_features,
    validate_raw_features,
    validate_normalized_features,
    validate_lvlm_structured,
)

def build_interpretation_output_path(video_id: str) -> Path:
    """
    Build the VideoInterpretation.json output path.
    """
    return Path("outputs") / video_id / OUTPUT_FILES.interpretation_filename


def print_interpretation_summary(
    raw_features: dict,
    norm_features: dict,
    complexity_score: float,
    breakdown: dict,
    narrative_phase: str,
    phase_reasons: list[str],
) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full Week 3 pipeline: upload a local video if needed, "
            "analyze it with Google Video Intelligence, save VideoFeatures.json, "
            "then compute and save VideoInterpretation.json."
        )
    )
    parser.add_argument("--video", type=str, help="Local path to video file (mp4, etc.)")
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET_NAME, help="GCS bucket name")
    parser.add_argument("--prefix", type=str, default=DEFAULT_GCS_PREFIX, help="GCS folder/prefix")
    parser.add_argument("--timeout", type=int, default=600, help="Analysis timeout in seconds")
    parser.add_argument( #lvlm
        "--with_lvlm",
        action="store_true",
        help="Run LVLM summary and structured scene interpretation",
    )
    parser.add_argument(
        "--print_summary",
        action="store_true",
        help="Print raw feature summary and interpretation summary to console",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    local_video_path = args.video
    if not local_video_path:
        local_video_path = input(
            r'Enter local path to video (e.g. C:\path\video.mp4): '
        ).strip().strip('"')

    # Stage 1: GCP feature extraction
    video_uri = upload_to_gcs(
        local_video_path=local_video_path,
        bucket_name=args.bucket,
        prefix=args.prefix,
    )
    print(f"Using URI: {video_uri}")

    annotations = analyze_video_uri(
        video_uri=video_uri,
        timeout_seconds=args.timeout,
    )

    video_features = compute_features(annotations, video_uri)
    video_features = validate_video_features(video_features) #lvlm
    video_id = extract_video_id(video_uri)

    features_output_path = build_features_output_path(video_id)
    save_json(video_features, features_output_path, label="features JSON")

    # Stage 2: Interpretation
    raw_features = build_raw_features(video_features)
    raw_features = validate_raw_features(raw_features) #lvlm
    norm_features = normalize_features(raw_features)
    norm_features = validate_normalized_features(norm_features) #lvlm
    complexity_score, breakdown = compute_scene_complexity(norm_features)
    narrative_phase, phase_reasons = classify_narrative_phase(norm_features)

    # lvlm integration layer (with safe fallbacks in case of overload)
    lvlm_summary = None
    lvlm_structured_raw = None
    lvlm_structured = None
    lvlm_error = None

    if args.with_lvlm:
        try:
            print("\nExtracting shared frames for LVLM inference...")
            frames_b64 = extract_frames(local_video_path)

            print("\nRunning LVLM summary inference...")
            lvlm_summary = run_summary_inference_from_frames(frames_b64)

            print("\nRunning LVLM structured inference...")
            lvlm_structured_raw = run_structured_inference_from_frames(frames_b64)

            print("\nParsing LVLM structured response...")
            parsed_structured = parse_structured_response(lvlm_structured_raw["text"])
            lvlm_structured = validate_lvlm_structured(parsed_structured)

        except Exception as exc:
            lvlm_error = f"{type(exc).__name__}: {exc}"
            print(f"\nLVLM inference failed, continuing without LVLM output.")
            print(f"LVLM error: {lvlm_error}")


    # updated with lvlm (with error fallback)
    interpretation = {
    "video_id": video_id,
    "video_uri": video_uri,
    "features_raw": raw_features,
    "features_norm": norm_features,
    "scene_complexity_score": complexity_score,
    "scene_complexity_breakdown": breakdown,
    "narrative_phase": narrative_phase,
    "phase_reasons": phase_reasons,
    "lvlm_summary": lvlm_summary,
    "lvlm_structured_raw": lvlm_structured_raw,
    "lvlm_structured": lvlm_structured,
    "lvlm_error": lvlm_error,
}

    interpretation_output_path = build_interpretation_output_path(video_id)
    save_json(interpretation, interpretation_output_path, label="interpretation JSON")

    if args.print_summary:
        print_gcp_summary(video_features)
        print_interpretation_summary(
            raw_features=raw_features,
            norm_features=norm_features,
            complexity_score=complexity_score,
            breakdown=breakdown,
            narrative_phase=narrative_phase,
            phase_reasons=phase_reasons,
        )

    print(f"\nSaved features JSON to: {features_output_path}")
    print(f"Saved interpretation JSON to: {interpretation_output_path}")


if __name__ == "__main__":
    main()