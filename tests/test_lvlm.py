# src/pipeline/test_lvlm.py
"""
Testing lvlm pipeline standalone before integretion with the existing GCP pipeline
"""

import argparse
import json
from pathlib import Path

from src.gcp.feature_engineering import save_json
from src.lvlm.client import (
    extract_frames,
    run_summary_inference_from_frames,
    run_structured_inference_from_frames,
)
from src.lvlm.parse import parse_structured_response

def extract_video_id(video_path: str) -> str:
    """
    Extract video ID from the local video path.

    Example:
    C:/videos/ACCEDE09230.mp4 -> ACCEDE09230
    """
    return Path(video_path).stem


def build_output_path(video_id: str) -> Path:
    """
    Build the test output path for LVLM debugging.
    """
    return Path("outputs") / video_id / "LVLMTestOutput.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the LVLM summary and structured prompts on a local video file."
        )
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Local path to video file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = args.video
    if not video_path:
        video_path = input(
            r'Enter local path to video (e.g. C:\path\video.mp4): '
        ).strip().strip('"')

    video_id = extract_video_id(video_path)

    print("\nExtracting shared frames for LVLM inference...")
    frames_b64 = extract_frames(video_path)

    print("\nRunning LVLM summary inference...")
    summary_result = run_summary_inference_from_frames(frames_b64)

    print("\nRunning LVLM structured inference...")
    structured_result_raw = run_structured_inference_from_frames(frames_b64)
    # -----------------------------
    # 3) Parse structured response
    # -----------------------------
    print("\nParsing structured response...")
    structured_result_parsed = parse_structured_response(structured_result_raw["text"])

    # -----------------------------
    # 4) Build debug output
    # -----------------------------
    test_output = {
        "video_id": video_id,
        "video_path": video_path,
        "summary_result": summary_result,
        "shared_frame_count": len(frames_b64),
        "structured_result_raw": structured_result_raw,
        "structured_result_parsed": structured_result_parsed,
    }

    output_path = build_output_path(video_id)
    save_json(test_output, output_path, label="LVLM test output JSON")

    # -----------------------------
    # 5) Print results
    # -----------------------------
    print("\n=== LVLM SUMMARY RESULT ===")
    print(summary_result["text"])

    print("\n=== LVLM STRUCTURED RAW TEXT ===")
    print(structured_result_raw["text"])

    print("\n=== LVLM STRUCTURED PARSED ===")
    print(json.dumps(structured_result_parsed, indent=2, ensure_ascii=False))

    print(f"\nSaved LVLM test output to: {output_path}")


if __name__ == "__main__":
    main()