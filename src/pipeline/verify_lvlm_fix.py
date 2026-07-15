# src/pipeline/verify_lvlm_fix.py
"""
Quick end-to-end check of the LVLM structured path.

Runs the same steps as run_lvlm_tuning_mode, but for a single video and
without needing an existing VideoFeatures.json:

    frame extraction
    -> summary inference
    -> structured inference
    -> parse
    -> validate

Usage:
    python -m src.pipeline.verify_lvlm_fix --video videos/videos/ACCEDE09230.mp4

Background:
    glm-5v-turbo is a thinking model. With thinking enabled (the API default)
    its internal reasoning could consume the whole max_tokens budget, so the
    API returned finish_reason="length" with EMPTY content, and parsing failed
    with "No JSON object start ('{') found in response".
    The fix disables thinking via config.py (ModelSettings.lvlm_enable_thinking)
    and client.py (extra_body). This script confirms the full path works.
"""

import argparse
import json

from src.algorithm.validate import validate_lvlm_structured
from src.lvlm.client import (
    extract_frames,
    run_structured_inference_from_frames,
    run_summary_inference_from_frames,
)
from src.lvlm.parse import parse_structured_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the LVLM summary + structured + parse path on one video."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to a local video file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Extracting shared frames for LVLM inference...")
    frames_b64 = extract_frames(args.video)

    print("Running LVLM summary inference...")
    lvlm_summary = run_summary_inference_from_frames(frames_b64)
    print(f"  finish_reason: {lvlm_summary['finish_reason']}")
    print(f"  usage: {lvlm_summary['usage']}")
    print(f"  summary head: {lvlm_summary['text'][:150]!r}")

    print("\nRunning LVLM structured inference...")
    lvlm_structured_raw = run_structured_inference_from_frames(frames_b64)
    print(f"  finish_reason: {lvlm_structured_raw['finish_reason']}")
    print(f"  usage: {lvlm_structured_raw['usage']}")

    print("\nParsing LVLM structured response...")
    parsed = parse_structured_response(lvlm_structured_raw["text"])
    validated = validate_lvlm_structured(parsed)
    print(json.dumps(validated, indent=2, ensure_ascii=False))

    print("\nOK: full LVLM structured path succeeded.")


if __name__ == "__main__":
    main()
