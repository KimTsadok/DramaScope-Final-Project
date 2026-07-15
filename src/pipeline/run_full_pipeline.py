# src/pipeline/run_full_pipeline.py
"""
Single-video entry point for the video-analysis pipeline.

Supported modes
---------------

gcp:
    Run GCP feature extraction and algorithmic interpretation.
    Does not run LVLM.

lvlm:
    Load an existing VideoFeatures.json.
    Recompute algorithmic interpretation.
    Run LVLM semantic interpretation.
    Does not call GCS or Google Video Intelligence.

full:
    Run GCP feature extraction.
    Run algorithmic interpretation.
    Run LVLM semantic interpretation.

Important behavior
------------------
- VideoInterpretation.json is assembled fully and saved once.
- LVLM failures are recorded in lvlm_error.
- Saved output is reloaded and verified.
- Large LVLM payloads are saved to JSON but are not printed to the terminal.
- The script returns exit code 1 when an expected LVLM stage fails.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

from src.algorithm.complexity import compute_scene_complexity
from src.algorithm.features import build_raw_features
from src.algorithm.normalize import normalize_features
from src.algorithm.phase import classify_narrative_phase
from src.algorithm.validate import (
    validate_lvlm_structured,
    validate_normalized_features,
    validate_raw_features,
    validate_video_features,
)
from src.config import OUTPUT_FILES
from src.gcp.analyze import analyze_video_uri
from src.gcp.feature_engineering import compute_features, save_json
from src.gcp.upload import (
    DEFAULT_BUCKET_NAME,
    DEFAULT_GCS_PREFIX,
    upload_to_gcs,
)
from src.lvlm.client import (
    extract_frames,
    run_structured_inference_from_frames,
    run_summary_inference_from_frames,
)
from src.lvlm.parse import parse_structured_response
from src.pipeline.run_gcp_features import (
    build_features_output_path,
    extract_video_id,
    print_gcp_summary,
)


PIPELINE_MODES = {
    "gcp",
    "lvlm",
    "full",
}


# ---------------------------------------------------------------------------
# Path and JSON helpers
# ---------------------------------------------------------------------------

def build_interpretation_output_path(
    video_id: str,
) -> Path:
    """
    Build the output path for VideoInterpretation.json.
    """
    return (
        Path("outputs")
        / video_id
        / OUTPUT_FILES.interpretation_filename
    )


def load_json_object(
    path: Path,
) -> Dict[str, Any]:
    """
    Load a JSON file and verify that its root value is an object.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file, received: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, "
            f"received {type(data).__name__}."
        )

    return data


def validate_local_video_path(
    video_argument: str,
) -> Path:
    """
    Normalize and validate the local video path.
    """
    video_path = Path(
        video_argument
    ).expanduser()

    if not video_path.exists():
        raise FileNotFoundError(
            f"Local video file not found: {video_path}"
        )

    if not video_path.is_file():
        raise ValueError(
            f"Video path is not a file: {video_path}"
        )

    return video_path


def extract_video_id_from_features(
    video_features: Dict[str, Any],
    video_path: Path,
) -> str:
    """
    Prefer video_id derived from video_uri.

    Fall back to the local video filename stem.
    """
    video_uri = str(
        video_features.get("video_uri", "")
    ).strip()

    if video_uri:
        return extract_video_id(
            video_uri
        )

    return video_path.stem


# ---------------------------------------------------------------------------
# Stage 1: GCP features
# ---------------------------------------------------------------------------

def run_gcp_stage(
    video_path: Path,
    bucket_name: str,
    prefix: str,
    timeout_seconds: int,
) -> Tuple[
    Dict[str, Any],
    str,
    Path,
]:
    """
    Upload and analyze one video, validate features,
    and save VideoFeatures.json.
    """
    print(
        "\nStage 1/3 — GCP feature extraction"
    )

    video_uri = upload_to_gcs(
        local_video_path=str(video_path),
        bucket_name=bucket_name,
        prefix=prefix,
    )

    print(f"Using URI: {video_uri}")

    annotations = analyze_video_uri(
        video_uri=video_uri,
        timeout_seconds=timeout_seconds,
    )

    video_features = compute_features(
        annotations,
        video_uri,
    )

    video_features = validate_video_features(
        video_features
    )

    video_id = extract_video_id(
        video_uri
    )

    features_output_path = (
        build_features_output_path(
            video_id
        )
    )

    save_json(
        video_features,
        features_output_path,
        label="features JSON",
    )

    return (
        video_features,
        video_id,
        features_output_path,
    )


def load_existing_features_stage(
    video_path: Path,
) -> Tuple[
    Dict[str, Any],
    str,
    Path,
]:
    """
    Load existing VideoFeatures.json for LVLM mode.
    """
    print(
        "\nStage 1/3 — Loading existing GCP features"
    )

    expected_video_id = video_path.stem

    features_output_path = (
        build_features_output_path(
            expected_video_id
        )
    )

    video_features = load_json_object(
        features_output_path
    )

    video_features = validate_video_features(
        video_features
    )

    video_id = extract_video_id_from_features(
        video_features,
        video_path,
    )

    print(
        "Loaded existing features JSON from: "
        f"{features_output_path}"
    )

    return (
        video_features,
        video_id,
        features_output_path,
    )


# ---------------------------------------------------------------------------
# Stage 2: Algorithm
# ---------------------------------------------------------------------------

def run_algorithm_stage(
    video_features: Dict[str, Any],
    video_id: str,
) -> Dict[str, Any]:
    """
    Build the algorithmic interpretation fields.
    """
    print(
        "\nStage 2/3 — Algorithmic interpretation"
    )

    validated_video_features = (
        validate_video_features(
            video_features
        )
    )

    raw_features = build_raw_features(
        validated_video_features
    )

    raw_features = validate_raw_features(
        raw_features
    )

    normalized_features = normalize_features(
        raw_features
    )

    normalized_features = (
        validate_normalized_features(
            normalized_features
        )
    )

    complexity_score, breakdown = (
        compute_scene_complexity(
            normalized_features
        )
    )

    narrative_phase, phase_reasons = (
        classify_narrative_phase(
            normalized_features
        )
    )

    return {
        "video_id": video_id,
        "video_uri": validated_video_features.get(
            "video_uri",
            "",
        ),
        "features_raw": raw_features,
        "features_norm": normalized_features,
        "scene_complexity_score": complexity_score,
        "scene_complexity_breakdown": breakdown,
        "narrative_phase": narrative_phase,
        "phase_reasons": phase_reasons,
    }


# ---------------------------------------------------------------------------
# Stage 3: LVLM
# ---------------------------------------------------------------------------

def build_empty_lvlm_fields(
    reason: str | None,
) -> Dict[str, Any]:
    """
    Build a consistent empty LVLM result.
    """
    return {
        "lvlm_summary": None,
        "lvlm_structured_raw": None,
        "lvlm_structured": None,
        "lvlm_error": reason,
    }


def run_lvlm_stage(
    video_path: Path,
) -> Dict[str, Any]:
    """
    Run summary and structured LVLM inference.

    Returns fields ready to merge into VideoInterpretation.json.
    """
    print(
        "\nStage 3/3 — LVLM semantic interpretation"
    )

    print(
        "Extracting shared frames for LVLM inference..."
    )

    frames_b64 = extract_frames(
        str(video_path)
    )

    if not frames_b64:
        raise RuntimeError(
            f"No frames were extracted from: {video_path}"
        )

    print(
        "\nRunning LVLM summary inference..."
    )

    lvlm_summary = (
        run_summary_inference_from_frames(
            frames_b64
        )
    )

    # CHANGED:
    # Keep response validation, but do not print the full response payload.
    if not isinstance(
        lvlm_summary,
        dict,
    ):
        raise TypeError(
            "LVLM summary inference must return "
            f"a dictionary, received "
            f"{type(lvlm_summary).__name__}."
        )

    print(
        "\nRunning LVLM structured inference..."
    )

    lvlm_structured_raw = (
        run_structured_inference_from_frames(
            frames_b64
        )
    )

    # CHANGED:
    # Keep structured-response validation without terminal JSON dumps.
    if not isinstance(
        lvlm_structured_raw,
        dict,
    ):
        raise TypeError(
            "LVLM structured inference must return "
            f"a dictionary, received "
            f"{type(lvlm_structured_raw).__name__}."
        )

    structured_text = (
        lvlm_structured_raw.get("text")
    )

    if not isinstance(
        structured_text,
        str,
    ):
        raise ValueError(
            "LVLM structured response is missing "
            "a string 'text' field."
        )

    if not structured_text.strip():
        finish_reason = (
            lvlm_structured_raw.get(
                "finish_reason"
            )
        )

        raise ValueError(
            "LVLM structured response contains empty text. "
            f"finish_reason={finish_reason!r}"
        )

    print(
        "\nParsing LVLM structured response..."
    )

    parsed_structured = (
        parse_structured_response(
            structured_text
        )
    )

    lvlm_structured = (
        validate_lvlm_structured(
            parsed_structured
        )
    )

    # CHANGED:
    # The complete payload is returned for saving but is not printed.
    return {
        "lvlm_summary": lvlm_summary,
        "lvlm_structured_raw": lvlm_structured_raw,
        "lvlm_structured": lvlm_structured,
        "lvlm_error": None,
    }


def run_lvlm_stage_with_fallback(
    video_path: Path,
) -> Tuple[
    Dict[str, Any],
    bool,
]:
    """
    Run LVLM and preserve algorithm output on failure.
    """
    try:
        lvlm_fields = run_lvlm_stage(
            video_path
        )

        return lvlm_fields, True

    except Exception as exc:
        lvlm_error = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "\nLVLM inference failed. "
            "The algorithmic interpretation "
            "will still be saved."
        )

        print(
            f"LVLM error: {lvlm_error}"
        )

        traceback.print_exc()

        return (
            build_empty_lvlm_fields(
                reason=lvlm_error
            ),
            False,
        )


# ---------------------------------------------------------------------------
# Output verification
# ---------------------------------------------------------------------------

def has_complete_lvlm_output(
    interpretation: Dict[str, Any],
) -> bool:
    """
    Check whether an interpretation contains complete LVLM output.

    CHANGED:
    This replaces the temporary full-payload terminal diagnostics with
    a compact reusable validation function.
    """
    return (
        isinstance(
            interpretation.get("lvlm_summary"),
            dict,
        )
        and isinstance(
            interpretation.get(
                "lvlm_structured_raw"
            ),
            dict,
        )
        and isinstance(
            interpretation.get(
                "lvlm_structured"
            ),
            dict,
        )
        and interpretation.get(
            "lvlm_error"
        ) is None
    )


def save_interpretation(
    interpretation: Dict[str, Any],
    output_path: Path,
    expect_lvlm: bool,
) -> None:
    """
    Save VideoInterpretation.json and verify the result.

    When expect_lvlm=True, require valid saved LVLM dictionaries.
    """
    # CHANGED:
    # Removed the large pre-save LVLM output dump.
    save_json(
        interpretation,
        output_path,
        label="interpretation JSON",
    )

    saved_interpretation = (
        load_json_object(
            output_path
        )
    )

    expected_video_id = interpretation.get(
        "video_id"
    )

    saved_video_id = (
        saved_interpretation.get(
            "video_id"
        )
    )

    if saved_video_id != expected_video_id:
        raise RuntimeError(
            "Saved interpretation verification failed: "
            f"expected video_id={expected_video_id!r}, "
            f"saved video_id={saved_video_id!r}."
        )

    # CHANGED:
    # Keep strict LVLM verification, but no longer print the full saved data.
    if (
        expect_lvlm
        and not has_complete_lvlm_output(
            saved_interpretation
        )
    ):
        saved_error = saved_interpretation.get(
            "lvlm_error"
        )

        raise RuntimeError(
            "Saved interpretation is missing complete LVLM output. "
            f"lvlm_error={saved_error!r}"
        )

    print(
        "Verified saved interpretation JSON: "
        f"{output_path.resolve()}"
    )


def print_interpretation_summary(
    interpretation: Dict[str, Any],
) -> None:
    """
    Print algorithm summary and compact LVLM status.
    """
    print("\nRaw Features:")
    print(
        json.dumps(
            interpretation.get(
                "features_raw"
            ),
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nNormalized Features:")
    print(
        json.dumps(
            interpretation.get(
                "features_norm"
            ),
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nScene Complexity Score:")
    print(
        round(
            float(
                interpretation.get(
                    "scene_complexity_score",
                    0.0,
                )
            ),
            6,
        )
    )

    print("\nBreakdown:")
    print(
        json.dumps(
            interpretation.get(
                "scene_complexity_breakdown"
            ),
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nNarrative Phase:")
    print(
        interpretation.get(
            "narrative_phase"
        )
    )

    print("\nPhase Reasons:")
    print(
        json.dumps(
            interpretation.get(
                "phase_reasons"
            ),
            indent=2,
            ensure_ascii=False,
        )
    )

    # CHANGED:
    # Print only compact LVLM status instead of the full model responses.
    print("\nLVLM Status:")
    print(
        json.dumps(
            {
                "summary_available": isinstance(
                    interpretation.get(
                        "lvlm_summary"
                    ),
                    dict,
                ),
                "structured_raw_available": isinstance(
                    interpretation.get(
                        "lvlm_structured_raw"
                    ),
                    dict,
                ),
                "structured_available": isinstance(
                    interpretation.get(
                        "lvlm_structured"
                    ),
                    dict,
                ),
                "lvlm_error": interpretation.get(
                    "lvlm_error"
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run one video through GCP-only, "
            "LVLM-only, or full processing."
        )
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Local path to the video file.",
    )

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
        help="Google Cloud Storage bucket name.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_GCS_PREFIX,
        help="Google Cloud Storage prefix.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help=(
            "Google Video Intelligence timeout "
            "in seconds."
        ),
    )

    parser.add_argument(
        "--print_summary",
        action="store_true",
        help=(
            "Print GCP and interpretation summaries."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Run the requested single-video pipeline mode.
    """
    args = parse_args()

    try:
        video_path = (
            validate_local_video_path(
                args.video
            )
        )

        print(f"Video: {video_path}")
        print(f"Mode: {args.mode}")

        if args.mode in {
            "gcp",
            "full",
        }:
            (
                video_features,
                video_id,
                features_output_path,
            ) = run_gcp_stage(
                video_path=video_path,
                bucket_name=args.bucket,
                prefix=args.prefix,
                timeout_seconds=args.timeout,
            )

        elif args.mode == "lvlm":
            (
                video_features,
                video_id,
                features_output_path,
            ) = load_existing_features_stage(
                video_path
            )

        else:
            raise ValueError(
                f"Unsupported pipeline mode: "
                f"{args.mode}"
            )

        interpretation = run_algorithm_stage(
            video_features=video_features,
            video_id=video_id,
        )

        lvlm_succeeded = True

        if args.mode in {
            "lvlm",
            "full",
        }:
            (
                lvlm_fields,
                lvlm_succeeded,
            ) = run_lvlm_stage_with_fallback(
                video_path
            )

        else:
            print(
                "\nStage 3/3 — LVLM skipped by gcp mode"
            )

            lvlm_fields = (
                build_empty_lvlm_fields(
                    reason=(
                        "LVLM not run in gcp mode"
                    )
                )
            )

        # CHANGED:
        # Merge LVLM fields without printing the full payload.
        interpretation.update(
            lvlm_fields
        )

        interpretation_output_path = (
            build_interpretation_output_path(
                video_id
            )
        )

        save_interpretation(
            interpretation=interpretation,
            output_path=interpretation_output_path,
            expect_lvlm=(
                args.mode
                in {"lvlm", "full"}
                and lvlm_succeeded
            ),
        )

        if args.print_summary:
            print_gcp_summary(
                video_features
            )

            print_interpretation_summary(
                interpretation
            )

        print(
            f"\nFeatures JSON: "
            f"{features_output_path}"
        )

        print(
            f"Interpretation JSON: "
            f"{interpretation_output_path}"
        )

        if not lvlm_succeeded:
            print(
                "\nPipeline completed with "
                "an LVLM error."
            )
            return 1

        print(
            "\nPipeline completed successfully."
        )
        return 0

    except Exception as exc:
        print(
            f"\nPipeline failed: "
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )