# src/pipeline/run_batch_pipeline.py
"""
Batch runner for the video analysis pipeline.

Purpose:
- avoid manually running run_full_pipeline.py for every video
- process X videos from a local video folder
- skip videos that already have VideoInterpretation.json unless --force is used
- run without LVLM by default for cost-efficient Week 5 numeric tuning

Modes:
    full:
        Run the full GCP pipeline for local videos.
        Use this for new videos that do not have VideoFeatures.json yet.

    gcp_tuning:
        Recompute VideoInterpretation.json from existing VideoFeatures.json.
        Use this after changing normalization ranges or phase thresholds in config.py.
        This mode does not call GCS, Google Video Intelligence, or LVLM.

    lvlm_tuning:
        Recompute VideoInterpretation.json from existing VideoFeatures.json
        and refresh LVLM semantic fields from local video frames.
        Use this after changing LVLM prompts, parsing, or validation.
"""


import argparse
import subprocess
import sys
import json
from pathlib import Path
from typing import List
from typing import Any, Dict

from src.config import OUTPUT_FILES

# newly added
from src.algorithm.complexity import compute_scene_complexity
from src.algorithm.features import build_raw_features
from src.algorithm.normalize import normalize_features
from src.algorithm.phase import classify_narrative_phase
from src.algorithm.validate import (
    validate_video_features,
    validate_raw_features,
    validate_normalized_features,
    validate_lvlm_structured,
)
from src.gcp.feature_engineering import save_json
from src.lvlm.client import (
    extract_frames,
    run_summary_inference_from_frames,
    run_structured_inference_from_frames,
)
from src.lvlm.parse import parse_structured_response


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

#---------------- Helpers ----------------

def build_features_path(video_path: Path) -> Path:
    """
    Build expected path to existing VideoFeatures.json.
    """
    video_id = video_path.stem
    return Path("outputs") / video_id / OUTPUT_FILES.raw_features_filename


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load JSON from disk.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_video_id_from_features(video_features: Dict[str, Any], video_path: Path) -> str:
    """
    Prefer video_uri filename stem. Fallback to local video filename stem.
    """
    video_uri = video_features.get("video_uri", "")
    if video_uri:
        return Path(video_uri).stem

    return video_path.stem


def build_algorithm_interpretation_base(
    video_features: Dict[str, Any],
    video_id: str,
) -> Dict[str, Any]:
    """
    Build the algorithmic part of VideoInterpretation.json.
    """
    video_features = validate_video_features(video_features)

    raw_features = build_raw_features(video_features)
    raw_features = validate_raw_features(raw_features)

    norm_features = normalize_features(raw_features)
    norm_features = validate_normalized_features(norm_features)

    complexity_score, breakdown = compute_scene_complexity(norm_features)
    narrative_phase, phase_reasons = classify_narrative_phase(norm_features)

    return {
        "video_id": video_id,
        "video_uri": video_features.get("video_uri", ""),
        "features_raw": raw_features,
        "features_norm": norm_features,
        "scene_complexity_score": complexity_score,
        "scene_complexity_breakdown": breakdown,
        "narrative_phase": narrative_phase,
        "phase_reasons": phase_reasons,
    }

#---------------- Helpers 2 ----------------

def find_video_files(videos_dir: Path) -> List[Path]:
    """
    Find supported video files in the given directory.
    Non-recursive by default for safety and predictability.
    """
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    if not videos_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {videos_dir}")

    videos = [
        path
        for path in videos_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    return sorted(videos)


def build_interpretation_path(video_path: Path) -> Path:
    """
    Build expected output path for a video's VideoInterpretation.json.

    Example:
        videos/ACCEDE09230.mp4
        -> outputs/ACCEDE09230/VideoInterpretation.json
    """
    video_id = video_path.stem
    return Path("outputs") / video_id / OUTPUT_FILES.interpretation_filename


def should_skip_video(video_path: Path, force: bool) -> bool:
    """
    Decide whether to skip a video.
    By default, skip if VideoInterpretation.json already exists.
    """
    if force:
        return False

    interpretation_path = build_interpretation_path(video_path)
    return interpretation_path.exists()


#---------------- GCP tuning mode ----------------

def run_gcp_tuning_mode(video_path: Path) -> bool:
    """
    Regenerate VideoInterpretation.json from existing VideoFeatures.json only.

    Cost-efficient mode for Week 5 numeric tuning.
    """
    features_path = build_features_path(video_path)
    interpretation_path = build_interpretation_path(video_path)

    if not features_path.exists():
        print(f"Missing VideoFeatures.json: {features_path}")
        return False

    try:
        video_features = load_json(features_path)
        video_id = extract_video_id_from_features(video_features, video_path)

        interpretation = build_algorithm_interpretation_base(
            video_features=video_features,
            video_id=video_id,
        )

        interpretation.update(
            {
                "lvlm_summary": None,
                "lvlm_structured_raw": None,
                "lvlm_structured": None,
                "lvlm_error": "LVLM not run in gcp_tuning mode",
            }
        )

        save_json(interpretation, interpretation_path, label="interpretation JSON")
        print(f"Regenerated interpretation from existing features: {interpretation_path}")
        return True

    except Exception as exc:
        print(f"Failed gcp_tuning mode for: {video_path.name}")
        print(f"Error: {type(exc).__name__}: {exc}")
        return False
    
#---------------- LVLM tuning mode ----------------

def run_lvlm_tuning_mode(video_path: Path) -> bool:
    """
    Regenerate VideoInterpretation.json from existing VideoFeatures.json,
    then refresh LVLM semantic fields using the local video.

    Cost-aware mode for Week 5 LVLM prompt/interaction tuning.
    """
    features_path = build_features_path(video_path)
    interpretation_path = build_interpretation_path(video_path)

    if not features_path.exists():
        print(f"Missing VideoFeatures.json: {features_path}")
        return False

    try:
        video_features = load_json(features_path)
        video_id = extract_video_id_from_features(video_features, video_path)

        interpretation = build_algorithm_interpretation_base(
            video_features=video_features,
            video_id=video_id,
        )

        lvlm_summary = None
        lvlm_structured_raw = None
        lvlm_structured = None
        lvlm_error = None

        try:
            print("Extracting shared frames for LVLM inference...")
            frames_b64 = extract_frames(str(video_path))

            print("Running LVLM summary inference...")
            lvlm_summary = run_summary_inference_from_frames(frames_b64)

            print("Running LVLM structured inference...")
            lvlm_structured_raw = run_structured_inference_from_frames(frames_b64)

            print("Parsing LVLM structured response...")
            parsed_structured = parse_structured_response(lvlm_structured_raw["text"])
            lvlm_structured = validate_lvlm_structured(parsed_structured)

        except Exception as exc:
            lvlm_error = f"{type(exc).__name__}: {exc}"
            print("LVLM failed, saving interpretation without LVLM output.")
            print(f"LVLM error: {lvlm_error}")

        interpretation.update(
            {
                "lvlm_summary": lvlm_summary,
                "lvlm_structured_raw": lvlm_structured_raw,
                "lvlm_structured": lvlm_structured,
                "lvlm_error": lvlm_error,
            }
        )

        save_json(interpretation, interpretation_path, label="interpretation JSON")
        print(f"Regenerated interpretation with LVLM tuning fields: {interpretation_path}")
        return True

    except Exception as exc:
        print(f"Failed lvlm_tuning mode for: {video_path.name}")
        print(f"Error: {type(exc).__name__}: {exc}")
        return False
    


def build_command(
    video_path: Path,
    with_lvlm: bool,
    print_summary: bool,
) -> List[str]:
    """
    Build the command that runs run_full_pipeline.py for one video.
    """
    command = [
        sys.executable,
        "-m",
        "src.pipeline.run_full_pipeline",
        "--video",
        str(video_path),
    ]

    if with_lvlm:
        command.append("--with_lvlm")

    if print_summary:
        command.append("--print_summary")

    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the video analysis pipeline on a batch of local videos."
    )

    parser.add_argument(
        "--videos_dir",
        type=str,
        default="videos/videos",
        help="Directory containing local video files",
    )

    # if u wish to run all videos - remove --limit
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of videos to process. If omitted, process all found videos.",
    )

    parser.add_argument(
    "--mode",
    choices=["full", "gcp_tuning", "lvlm_tuning"],
    default="gcp_tuning",
    help=(
        "Batch mode: "
        "'full' runs the complete GCP pipeline, "
        "'gcp_tuning' regenerates VideoInterpretation.json from existing VideoFeatures.json, "
        "'lvlm_tuning' regenerates VideoInterpretation.json and refreshes LVLM fields."
    ),
)
    # in order to run full mode with LVLM too:
    parser.add_argument(
        "--with_lvlm",
        action="store_true",
        help="Also run LVLM. Disabled by default for cost-efficient numeric tuning.",
    )

    parser.add_argument(
        "--print_summary",
        action="store_true",
        help="Pass --print_summary to run_full_pipeline.py in full mode.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    videos_dir = Path(args.videos_dir)
    all_videos = find_video_files(videos_dir)

    if args.limit is not None:
        selected_videos = all_videos[: args.limit]
    else:
        selected_videos = all_videos

    print(f"Found {len(all_videos)} video files in: {videos_dir}")
    print(f"Selected {len(selected_videos)} video files for this batch")
    print(f"Mode: {args.mode}")
    print(f"LVLM enabled in full mode: {args.with_lvlm}")

    processed = 0
    skipped = 0
    failed = 0

    for index, video_path in enumerate(selected_videos, start=1):
        print("\n" + "=" * 80)
        print(f"[{index}/{len(selected_videos)}] Video: {video_path.name}")

        # In full mode, we may want to skip videos that already have interpretation output.
        # In tuning modes, we usually want to overwrite VideoInterpretation.json
        # because config.py / prompts may have changed.
        # if args.mode == "full" and should_skip_video(video_path, force=args.force):
        #    print(
        #        f"Skipping: {build_interpretation_path(video_path)} already exists. "
        #        "Use --force to rerun."
        #    )
        #    skipped += 1
        #    continue

        if args.mode == "gcp_tuning":
            success = run_gcp_tuning_mode(video_path)

        elif args.mode == "lvlm_tuning":
            success = run_lvlm_tuning_mode(video_path)

        elif args.mode == "full":
            command = build_command(
                video_path=video_path,
                with_lvlm=args.with_lvlm,
                print_summary=args.print_summary,
            )

            print("Running command:")
            print(" ".join(f'"{part}"' if " " in part else part for part in command))

            result = subprocess.run(command)
            success = result.returncode == 0

        else:
            print(f"Unknown mode: {args.mode}")
            success = False

        if success:
            print(f"Finished successfully: {video_path.name}")
            processed += 1
        else:
            print(f"Failed: {video_path.name}")
            failed += 1

    print("\n" + "=" * 80)
    print("Batch run complete.")
    print(f"Mode: {args.mode}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()