# src/pipeline/run_batch_pipeline.py
"""
Batch runner for the video-analysis pipeline.

This script is responsible only for batch orchestration:

1. Discover local video files.
2. Select all videos, a limited number, or one specific video.
3. Decide whether each video is already complete for the requested mode.
4. Call src.pipeline.run_full_pipeline once per video.
5. Track processed, skipped, and failed videos.

All actual video-processing logic belongs in run_full_pipeline.py.

Supported modes
---------------

gcp:
    Run GCP feature extraction and algorithmic interpretation.
    Does not run LVLM.

lvlm:
    Load an existing VideoFeatures.json and run:
    algorithmic interpretation + LVLM.

full:
    Run the complete pipeline:
    GCP + algorithmic interpretation + LVLM.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.config import OUTPUT_FILES
from src.gcp.upload import (
    DEFAULT_BUCKET_NAME,
    DEFAULT_GCS_PREFIX,
)


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
}

PIPELINE_MODES = {
    "gcp",
    "lvlm",
    "full",
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def build_features_path(video_path: Path) -> Path:
    """
    Build the expected VideoFeatures.json path.

    Example:
        videos/videos/ACCEDE09230.mp4
        -> outputs/ACCEDE09230/VideoFeatures.json
    """
    return (
        Path("outputs")
        / video_path.stem
        / OUTPUT_FILES.raw_features_filename
    )


def build_interpretation_path(video_path: Path) -> Path:
    """
    Build the expected VideoInterpretation.json path.

    Example:
        videos/videos/ACCEDE09230.mp4
        -> outputs/ACCEDE09230/VideoInterpretation.json
    """
    return (
        Path("outputs")
        / video_path.stem
        / OUTPUT_FILES.interpretation_filename
    )


def load_json_object(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file and verify that its root value is an object.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a JSON file, received: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, "
            f"received {type(data).__name__}."
        )

    return data


# ---------------------------------------------------------------------------
# Video discovery and selection
# ---------------------------------------------------------------------------

def find_video_files(
    videos_dir: Path,
) -> List[Path]:
    """
    Find supported video files in the selected directory.

    The search is non-recursive for predictable batch behavior.
    """
    if not videos_dir.exists():
        raise FileNotFoundError(
            f"Videos directory not found: {videos_dir}"
        )

    if not videos_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, received: {videos_dir}"
        )

    videos = [
        path
        for path in videos_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
        )
    ]

    return sorted(
        videos,
        key=lambda path: path.name.lower(),
    )


def select_videos(
    all_videos: List[Path],
    video_id: str | None,
    limit: int | None,
) -> List[Path]:
    """
    Select videos using the following priority:

    1. --video_id
    2. --limit
    3. all discovered videos
    """
    # CHANGED:
    # Support selecting one specific video without relying on --limit.
    if video_id:
        normalized_video_id = Path(
            video_id
        ).stem.lower()

        selected = [
            video_path
            for video_path in all_videos
            if video_path.stem.lower()
            == normalized_video_id
        ]

        if not selected:
            raise FileNotFoundError(
                f"Video ID not found in the selected directory: "
                f"{video_id}"
            )

        return selected

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "--limit must be greater than zero."
            )

        return all_videos[:limit]

    return all_videos


# ---------------------------------------------------------------------------
# Output completeness checks
# ---------------------------------------------------------------------------

def has_valid_algorithm_output(
    interpretation: Dict[str, Any],
) -> bool:
    """
    Check whether VideoInterpretation.json contains the required
    algorithmic output fields.
    """
    required_fields = (
        "video_id",
        "features_raw",
        "features_norm",
        "scene_complexity_score",
        "scene_complexity_breakdown",
        "narrative_phase",
        "phase_reasons",
    )

    return all(
        field in interpretation
        and interpretation[field] is not None
        for field in required_fields
    )


def has_valid_lvlm_output(
    interpretation: Dict[str, Any],
) -> bool:
    """
    Check whether VideoInterpretation.json contains complete LVLM output.
    """
    lvlm_summary = interpretation.get(
        "lvlm_summary"
    )

    lvlm_structured_raw = interpretation.get(
        "lvlm_structured_raw"
    )

    lvlm_structured = interpretation.get(
        "lvlm_structured"
    )

    lvlm_error = interpretation.get(
        "lvlm_error"
    )

    return (
        isinstance(lvlm_summary, dict)
        and isinstance(lvlm_structured_raw, dict)
        and isinstance(lvlm_structured, dict)
        and lvlm_error is None
    )


def should_skip_video(
    video_path: Path,
    mode: str,
    force: bool,
) -> bool:
    """
    Decide whether a video is already complete for the selected mode.

    Behavior
    --------

    gcp:
        Skip only when both VideoFeatures.json and a valid algorithmic
        VideoInterpretation.json already exist.

    lvlm:
        Skip only when VideoFeatures.json exists and
        VideoInterpretation.json contains complete LVLM output.

    full:
        Skip only when VideoFeatures.json exists,
        algorithmic output is valid,
        and LVLM output is complete.

    --force:
        Always rerun.
    """
    if force:
        return False

    features_path = build_features_path(
        video_path
    )

    interpretation_path = (
        build_interpretation_path(
            video_path
        )
    )

    if mode == "gcp":
        if (
            not features_path.exists()
            or not interpretation_path.exists()
        ):
            return False

        try:
            interpretation = load_json_object(
                interpretation_path
            )
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
        ):
            return False

        return has_valid_algorithm_output(
            interpretation
        )

    if mode == "lvlm":
        # LVLM mode requires existing GCP features.
        if not features_path.exists():
            return False

        if not interpretation_path.exists():
            return False

        try:
            interpretation = load_json_object(
                interpretation_path
            )
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
        ):
            return False

        return has_valid_lvlm_output(
            interpretation
        )

    if mode == "full":
        if (
            not features_path.exists()
            or not interpretation_path.exists()
        ):
            return False

        try:
            interpretation = load_json_object(
                interpretation_path
            )
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
        ):
            return False

        return (
            has_valid_algorithm_output(
                interpretation
            )
            and has_valid_lvlm_output(
                interpretation
            )
        )

    return False


def explain_skip(
    video_path: Path,
    mode: str,
) -> None:
    """
    Print a mode-specific skip message.
    """
    if mode == "gcp":
        print(
            "Skipping: existing GCP features and "
            "algorithmic interpretation are complete."
        )

    elif mode == "lvlm":
        print(
            "Skipping: existing LVLM output is complete "
            "and contains no LVLM error."
        )

    elif mode == "full":
        print(
            "Skipping: GCP, algorithmic, and LVLM "
            "outputs are already complete."
        )

    print(
        "Interpretation path: "
        f"{build_interpretation_path(video_path)}"
    )
    print(
        "Use --force to rerun this video."
    )


# ---------------------------------------------------------------------------
# Single-video pipeline command
# ---------------------------------------------------------------------------

def build_pipeline_command(
    video_path: Path,
    mode: str,
    bucket_name: str,
    prefix: str,
    timeout_seconds: int,
    print_summary: bool,
) -> List[str]:
    """
    Build the command that invokes run_full_pipeline.py for one video.

    CHANGED:
    Every mode now delegates to the same single-video pipeline script.
    The batch runner no longer implements algorithm or LVLM logic itself.
    """
    command = [
        sys.executable,
        "-m",
        "src.pipeline.run_full_pipeline",
        "--video",
        str(video_path),
        "--mode",
        mode,
    ]

    # GCP configuration is relevant for gcp and full modes.
    if mode in {"gcp", "full"}:
        command.extend(
            [
                "--bucket",
                bucket_name,
                "--prefix",
                prefix,
                "--timeout",
                str(timeout_seconds),
            ]
        )

    if print_summary:
        command.append(
            "--print_summary"
        )

    return command


def format_command(
    command: List[str],
) -> str:
    """
    Format a subprocess command for readable console output.
    """
    return " ".join(
        f'"{part}"' if " " in part else part
        for part in command
    )


def run_pipeline_for_video(
    video_path: Path,
    mode: str,
    bucket_name: str,
    prefix: str,
    timeout_seconds: int,
    print_summary: bool,
) -> bool:
    """
    Invoke the single-video pipeline and return whether it succeeded.
    """
    command = build_pipeline_command(
        video_path=video_path,
        mode=mode,
        bucket_name=bucket_name,
        prefix=prefix,
        timeout_seconds=timeout_seconds,
        print_summary=print_summary,
    )

    print("Running pipeline command:")
    print(
        format_command(command)
    )

    try:
        result = subprocess.run(
            command,
            check=False,
        )

    except OSError as exc:
        print(
            "Could not launch the single-video pipeline."
        )
        print(
            f"Error: {type(exc).__name__}: {exc}"
        )
        return False

    if result.returncode != 0:
        print(
            "Single-video pipeline returned "
            f"exit code {result.returncode}."
        )
        return False

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse batch-runner arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the single-video analysis pipeline "
            "over a batch of local videos."
        )
    )

    parser.add_argument(
        "--videos_dir",
        type=str,
        default="videos/videos",
        help=(
            "Directory containing local video files."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of videos to process. "
            "If omitted, process all discovered videos."
        ),
    )

    parser.add_argument(
        "--video_id",
        type=str,
        default=None,
        help=(
            "Process one specific video ID, such as "
            "ACCEDE09250 or ACCEDE09250.mp4."
        ),
    )

    # CHANGED:
    # Mode names now match run_full_pipeline.py exactly.
    parser.add_argument(
        "--mode",
        choices=sorted(
            PIPELINE_MODES
        ),
        default="full",
        help=(
            "'gcp': GCP + algorithm only; "
            "'lvlm': existing features + algorithm + LVLM; "
            "'full': GCP + algorithm + LVLM."
        ),
    )

    parser.add_argument(
        "--bucket",
        type=str,
        default=DEFAULT_BUCKET_NAME,
        help=(
            "Google Cloud Storage bucket used by "
            "gcp and full modes."
        ),
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_GCS_PREFIX,
        help=(
            "Google Cloud Storage prefix used by "
            "gcp and full modes."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help=(
            "Google Video Intelligence timeout in seconds."
        ),
    )

    parser.add_argument(
        "--print_summary",
        action="store_true",
        help=(
            "Print detailed per-video pipeline summaries."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rerun videos even when the requested "
            "mode appears complete."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Run the requested pipeline mode over the selected videos.

    Returns:
        0 when every selected video succeeds or is skipped.
        1 when one or more videos fail.
    """
    args = parse_args()

    videos_dir = Path(
        args.videos_dir
    )

    try:
        all_videos = find_video_files(
            videos_dir
        )

        selected_videos = select_videos(
            all_videos=all_videos,
            video_id=args.video_id,
            limit=args.limit,
        )

    except (
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
    ) as exc:
        print(
            "Batch setup failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1

    print(
        f"Found {len(all_videos)} video files in: "
        f"{videos_dir}"
    )
    print(
        f"Selected {len(selected_videos)} "
        "video files for this batch"
    )
    print(
        f"Mode: {args.mode}"
    )

    if args.mode == "gcp":
        print(
            "Stages: GCP + algorithmic interpretation"
        )

    elif args.mode == "lvlm":
        print(
            "Stages: existing features + "
            "algorithmic interpretation + LVLM"
        )

    elif args.mode == "full":
        print(
            "Stages: GCP + algorithmic interpretation + LVLM"
        )

    if args.force:
        print(
            "Force mode enabled: existing outputs "
            "will not be skipped."
        )

    processed = 0
    skipped = 0
    failed = 0

    failed_videos: List[str] = []

    for index, video_path in enumerate(
        selected_videos,
        start=1,
    ):
        print("\n" + "=" * 80)
        print(
            f"[{index}/{len(selected_videos)}] "
            f"Video: {video_path.name}"
        )

        if should_skip_video(
            video_path=video_path,
            mode=args.mode,
            force=args.force,
        ):
            explain_skip(
                video_path=video_path,
                mode=args.mode,
            )

            skipped += 1
            continue

        # CHANGED:
        # lvlm mode depends on an existing VideoFeatures.json.
        # Fail early with a clear message rather than launching
        # the child process unnecessarily.
        if args.mode == "lvlm":
            features_path = build_features_path(
                video_path
            )

            if not features_path.exists():
                print(
                    "Cannot run lvlm mode because "
                    "VideoFeatures.json is missing:"
                )
                print(
                    features_path
                )

                failed += 1
                failed_videos.append(
                    video_path.name
                )
                continue

        success = run_pipeline_for_video(
            video_path=video_path,
            mode=args.mode,
            bucket_name=args.bucket,
            prefix=args.prefix,
            timeout_seconds=args.timeout,
            print_summary=args.print_summary,
        )

        if success:
            print(
                "Finished successfully: "
                f"{video_path.name}"
            )
            processed += 1

        else:
            print(
                f"Failed: {video_path.name}"
            )
            failed += 1
            failed_videos.append(
                video_path.name
            )

    print("\n" + "=" * 80)
    print("Batch run complete.")
    print(
        f"Mode: {args.mode}"
    )
    print(
        f"Processed: {processed}"
    )
    print(
        f"Skipped: {skipped}"
    )
    print(
        f"Failed: {failed}"
    )

    if failed_videos:
        print("\nFailed videos:")

        for video_name in failed_videos:
            print(
                f"- {video_name}"
            )

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())