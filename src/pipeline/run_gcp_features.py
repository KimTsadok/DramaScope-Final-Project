# src/pipeline/run_gcp_features.py
"""
That file will:
* ask for video path if needed
* call upload_to_gcs(...)
* call analyze_video_uri(...)
* call compute_features(...)
* derive video_id
* save to: VideoFeatures.json
"""

import argparse
from pathlib import Path

from src.gcp.upload import (
    DEFAULT_BUCKET_NAME,
    DEFAULT_GCS_PREFIX,
    upload_to_gcs,
)
from src.gcp.analyze import analyze_video_uri
from src.gcp.feature_engineering import compute_features, save_json
from src.config import OUTPUT_FILES

def extract_video_id(video_uri: str) -> str:
    """
    Extract video ID from a bucket URI.

    Example:
    gs://final-project-bucket1/uploads/ACCEDE09230.mp4 -> ACCEDE09230
    """
    return Path(video_uri).stem


def build_features_output_path(video_id: str) -> Path:
    """
    Build the output path for VideoFeatures.json.

    Example:
    outputs/ACCEDE09230/VideoFeatures.json
    """
    return Path("outputs") / video_id / OUTPUT_FILES.raw_features_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a local video to GCS if needed, analyze it with "
            "Google Video Intelligence, and save VideoFeatures.json"
        )
    )
    parser.add_argument("--video", type=str, help="Local path to video file (mp4, etc.)")
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET_NAME, help="GCS bucket name")
    parser.add_argument("--prefix", type=str, default=DEFAULT_GCS_PREFIX, help="GCS folder/prefix")
    parser.add_argument("--timeout", type=int, default=600, help="Analysis timeout in seconds")
    parser.add_argument(
        "--print_summary",
        action="store_true",
        help="Print a short feature summary to console",
    )
    return parser.parse_args()


def print_gcp_summary(features: dict) -> None:
    print("\n=== Feature Summary ===")
    print(f"Duration: {features['duration_seconds']}s")
    print(
        f"Shots: {features['shots']['count']} | "
        f"avg={features['shots']['avg_shot_length']} | "
        f"var={features['shots']['shot_variance']}"
    )
    print(f"Top objects: {features['objects']['top_objects']}")
    print(f"Human presence ratio: {features['objects']['human_presence_ratio']}")
    print(f"Object entropy: {features['objects']['object_entropy']}")
    print(
        "Interaction density (tracks/sec): "
        f"{features['objects']['interaction_density_tracks_per_sec']}"
    )


def main() -> None:
    args = parse_args()

    local_video_path = args.video
    if not local_video_path:
        local_video_path = input(
            r'Enter local path to video (e.g. C:\path\video.mp4): '
        ).strip().strip('"')

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

    features = compute_features(annotations, video_uri)

    video_id = extract_video_id(video_uri)
    output_path = build_features_output_path(video_id)

    save_json(features, output_path, label="features JSON")

    if args.print_summary:
        print_gcp_summary(features)


if __name__ == "__main__":
    main()
