# src/pipeline/collect_feature_stats.py
"""
Collect Week 5 feature statistics from existing VideoInterpretation.json files.

This script scans:
    outputs/*/VideoInterpretation.json (for about 20 videos)

It reads:
    features_raw

Then it produces:
    evaluation/week5_feature_stats.md

Purpose:
    Help tune normalization ranges in config.py using a small project dataset.
"""

import argparse
import json
import statistics
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any
from src.config import NORMALIZATION_RANGES, OUTPUT_FILES


FEATURE_NAMES = [
    "shot_frequency",
    "object_entropy",
    "interaction_density",
    "human_presence_ratio",
]

PHASE_NAMES = [
    "Calm",
    "Dynamic",
    "Dense",
    "Static",
    "Unknown",
]

def find_interpretation_files(outputs_dir: Path) -> List[Path]:
    """
    Find all VideoInterpretation.json files under the outputs directory.
    """
    if not outputs_dir.exists():
        return []

    return sorted(outputs_dir.rglob(OUTPUT_FILES.interpretation_filename))


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load JSON from disk.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def safe_float(value: Any) -> float | None:
    """
    Convert value to float. Return None if invalid.
    """
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_markdown_section(existing_text: str, heading: str) -> str | None:
    """
    Extract a markdown section from an existing markdown file.

    It starts at the requested heading line and continues until the next level-2 heading.
    Example heading:
        "## Observations"

    Returns:
        the full section text, including the heading
    """
    lines = existing_text.splitlines()

    start_index = None
    for index, line in enumerate(lines):
        if line.strip() == heading:
            start_index = index
            break

    if start_index is None:
        return None

    end_index = len(lines)
    for index in range(start_index + 1, len(lines)):
        line = lines[index].strip()

        if line.startswith("## ") and line != heading:
            end_index = index
            break

    return "\n".join(lines[start_index:end_index]).strip()


def collect_phase_counts(files: List[Path]) -> Counter:
    """
    Count how many videos were classified into each narrative phase.

    Reads:
        narrative_phase

    from each VideoInterpretation.json file.
    """
    phase_counts: Counter = Counter()

    for file_path in files:
        data = load_json(file_path)
        phase = data.get("narrative_phase", "Unknown")

        if not phase:
            phase = "Unknown"

        phase_counts[str(phase)] += 1

    return phase_counts

def collect_raw_feature_values(files: List[Path]) -> Dict[str, List[float]]:
    """
    Collect features_raw values from VideoInterpretation.json files.
    """
    values: Dict[str, List[float]] = {feature: [] for feature in FEATURE_NAMES}

    for file_path in files:
        data = load_json(file_path)
        raw_features = data.get("features_raw", {})

        for feature in FEATURE_NAMES:
            value = safe_float(raw_features.get(feature))
            if value is not None:
                values[feature].append(value)

    return values


def summarize_values(values: List[float]) -> Dict[str, float | int | None]:
    """
    Compute count, min, median, mean, and max for a list of values.
    """
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }

    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def get_current_range(feature: str) -> str:
    """
    Return the current normalization range from config.py.
    """
    if feature == "shot_frequency":
        return (
            f"{NORMALIZATION_RANGES.shot_frequency_min}"
            f"–{NORMALIZATION_RANGES.shot_frequency_max}"
        )

    if feature == "object_entropy":
        return (
            f"{NORMALIZATION_RANGES.object_entropy_min}"
            f"–{NORMALIZATION_RANGES.object_entropy_max}"
        )

    if feature == "interaction_density":
        return (
            f"{NORMALIZATION_RANGES.interaction_density_min}"
            f"–{NORMALIZATION_RANGES.interaction_density_max}"
        )

    if feature == "human_presence_ratio":
        return (
            f"{NORMALIZATION_RANGES.human_presence_ratio_min}"
            f"–{NORMALIZATION_RANGES.human_presence_ratio_max}"
        )

    return "unknown"


def format_number(value: float | int | None) -> str:
    """
    Format numbers for markdown.
    """
    if value is None:
        return "N/A"

    if isinstance(value, int):
        return str(value)

    return f"{value:.4f}"


def build_feature_note(feature: str, summary: Dict[str, float | int | None]) -> str:
    """
    Generate a simple tuning note for each feature.
    """
    max_value = summary["max"]
    median_value = summary["median"]

    if max_value is None or median_value is None:
        return "No data available"

    if feature == "human_presence_ratio":
        return "Already bounded in [0, 1]; usually keep range unchanged"

    if feature == "interaction_density":
        current_max = NORMALIZATION_RANGES.interaction_density_max
        if max_value < current_max * 0.5:
            return "Observed values are much lower than current max; consider lowering max range"
        return "Current range may be acceptable"

    if feature == "shot_frequency":
        current_max = NORMALIZATION_RANGES.shot_frequency_max
        if max_value < current_max * 0.5:
            return "Observed shot frequency is low; current max may be too wide"
        return "Current range may be acceptable"

    if feature == "object_entropy":
        current_max = NORMALIZATION_RANGES.object_entropy_max
        if max_value > current_max:
            return "Observed value exceeds current max; range should be increased"
        if max_value < current_max * 0.5:
            return "Observed entropy is far below current max; range may be too wide"
        return "Current range may be acceptable"

    return ""


def build_markdown(
    files: List[Path],
    feature_values: Dict[str, List[float]],
    phase_counts: Counter,
    existing_markdown: str | None = None,
) -> str:
    """
    Build the full markdown report.
    """
    lines: List[str] = []

    existing_observations = None
    existing_suggested_next_step = None
    existing_notes = None

    if existing_markdown:
        existing_observations = extract_markdown_section(
            existing_markdown,
            "## Observations",
        )
        existing_suggested_next_step = extract_markdown_section(
            existing_markdown,
            "## Suggested Next Tuning Step (Suggested Range Updates)",
        )
        existing_notes = extract_markdown_section(
            existing_markdown,
            "## Notes",
        )

    lines.append("# Week 5 Feature Statistics")
    lines.append("")
    lines.append("## Goal")
    lines.append(
        "Collect raw feature distributions from analyzed videos and use them to evaluate "
        "whether the current normalization ranges in `config.py` are realistic."
    )

# ------- Dataset -------
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"Number of `VideoInterpretation.json` files analyzed: {len(files)}")

# ------- Feature Statistics -------
    lines.append("")
    lines.append("## Feature Statistics")
    lines.append("")
    lines.append(
        "| feature | count | min | median | mean | max | current_range | notes |"
    )
    lines.append(
        "|---------|-------|-----|--------|------|-----|---------------|-------|"
    )

    for feature in FEATURE_NAMES:
        summary = summarize_values(feature_values[feature])

        lines.append(
            "| "
            f"{feature} | "
            f"{summary['count']} | "
            f"{format_number(summary['min'])} | "
            f"{format_number(summary['median'])} | "
            f"{format_number(summary['mean'])} | "
            f"{format_number(summary['max'])} | "
            f"{get_current_range(feature)} | "
            f"{build_feature_note(feature, summary)} |"
    )
# ------- Observations ------- (updated to not be overriden in testing)
    lines.append("")

    if existing_observations:
        lines.append(existing_observations)
    else:
        lines.append("## Observations")
        lines.append("- TODO: Write manual observations after reviewing the generated statistics.")
        lines.append("- TODO: Compare phase distribution before and after tuning changes.")
        lines.append("- TODO: Decide whether more threshold tuning is needed.")
    
# ------- Phase Distribution -------
    lines.append("")
    lines.append("## Phase Distribution")
    lines.append("")
    lines.append("| phase | count |")
    lines.append("|-------|-------|")

    for phase in PHASE_NAMES:
        lines.append(f"| {phase} | {phase_counts.get(phase, 0)} |")

    extra_phases = sorted(
        phase for phase in phase_counts.keys()
        if phase not in PHASE_NAMES
    )

    for phase in extra_phases:
        lines.append(f"| {phase} | {phase_counts[phase]} |")

    lines.append("")

    non_calm_count = sum(
        count for phase, count in phase_counts.items()
        if phase != "Calm"
    )

    if phase_counts.get("Calm", 0) == len(files):
        lines.append(
            "- All analyzed videos were classified as `Calm`. "
            "This suggests that the current normalization ranges and/or phase thresholds "
            "may be too restrictive for `Dense`, `Dynamic`, and `Static`."
        )
    elif non_calm_count == 0:
        lines.append(
            "- No non-Calm phases were detected. "
            "This suggests that the phase rules may need tuning."
        )
    else:
        lines.append(
            "- Multiple phase categories were detected, but distribution should still be reviewed "
            "for imbalance."
        )
    
# ------- Suggested ------- (updated to not be overriden in testing)
    lines.append("")

    if existing_suggested_next_step:
        lines.append(existing_suggested_next_step)
    else:
        lines.append("## Suggested Next Tuning Step (Suggested Range Updates)")
        lines.append("")
        lines.append("| component | current_value | suggested_value | reason |")
        lines.append("|----------|---------------|-----------------|--------|")
        lines.append("| TODO | TODO | TODO | TODO |")

# ------- Notes ------- (updated to not be overriden in testing)
    lines.append("")

    if existing_notes:
        lines.append(existing_notes)
    else:
        lines.append("## Notes")
        lines.append("- This report contains generated statistics plus manual tuning analysis.")
        lines.append("- Generated sections may update after each script run.")
        lines.append("- Manual sections are preserved from the existing markdown file when possible.")

    return "\n".join(lines)


def save_text(content: str, output_path: Path) -> None:
    """
    Save markdown text to disk.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        file.write(content)

    print(f"Saved feature statistics markdown to: {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect raw feature statistics from VideoInterpretation.json files."
    )

    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Directory containing per-video output folders",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="evaluation/week5_feature_stats.md",
        help="Output markdown path",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outputs_dir = Path(args.outputs_dir)
    output_path = Path(args.out)

    interpretation_files = find_interpretation_files(outputs_dir)

    if not interpretation_files:
        print(f"No {OUTPUT_FILES.interpretation_filename} files found under {outputs_dir}")
        return

    feature_values = collect_raw_feature_values(interpretation_files)
    phase_counts = collect_phase_counts(interpretation_files) # addition to count phases per video

    # takes care of existing sections and does not override them if they exist
    existing_markdown = None
    if output_path.exists():
        existing_markdown = output_path.read_text(encoding="utf-8")

    markdown = build_markdown(
        files=interpretation_files,
        feature_values=feature_values,
        phase_counts=phase_counts,
        existing_markdown=existing_markdown,
)

    save_text(markdown, output_path)

    print("\nFiles analyzed:")
    for file_path in interpretation_files:
        print(f"- {file_path}")


if __name__ == "__main__":
    main()