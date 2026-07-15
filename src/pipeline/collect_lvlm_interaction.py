# src/pipeline/collect_lvlm_interaction.py
"""
Collect Week 6 LVLM interaction-level evaluation results.

This script:
1. Reads the existing After Prompt Update evaluation Markdown file.
2. Extracts video IDs and expected interaction levels from:
   - the optional initial ## Evaluation Table
   - the current ## After Prompt Update table
3. Scans outputs/*/VideoInterpretation.json.
4. Reads lvlm_structured.interaction_level and interaction_evidence.
5. Replaces only the ## After Prompt Update section.

This avoids hardcoding video IDs inside the script.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from src.config import OUTPUT_FILES
from src.pipeline.collect_feature_stats import (
    find_interpretation_files,
    load_json,
)


def safe_int(value: Any) -> int | None:
    """
    Convert a value to int when possible.

    Returns None for missing, invalid, or placeholder values such as TBD.
    """
    try:
        if value is None:
            return None

        # CHANGED:
        # Explicitly support placeholder strings used in the Markdown table.
        if isinstance(value, str):
            cleaned = value.strip()

            if not cleaned:
                return None

            if cleaned.upper() in {
                "TBD",
                "N/A",
                "NONE",
                "NULL",
                "?",
            }:
                return None

            return int(cleaned)

        return int(value)

    except (TypeError, ValueError):
        return None


def extract_video_id(
    data: Dict[str, Any],
    file_path: Path,
) -> str:
    """
    Prefer video_id from JSON. Fall back to the parent folder name.
    """
    video_id = data.get("video_id")

    if video_id:
        return str(video_id)

    return file_path.parent.name


def get_lvlm_structured(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return lvlm_structured safely.
    """
    structured = data.get("lvlm_structured")

    if isinstance(structured, dict):
        return structured

    return {}


def extract_markdown_section(
    existing_text: str,
    heading: str,
) -> str | None:
    """
    Extract a level-2 Markdown section.

    Example:
        heading = "## After Prompt Update"

    The section ends at the next level-2 heading.
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
        if lines[index].strip().startswith("## "):
            end_index = index
            break

    return "\n".join(
        lines[start_index:end_index]
    ).strip()


# CHANGED:
# This generic parser replaces the old function that read only
# the first "## Evaluation Table" section.
def parse_expected_levels_from_section(
    markdown_text: str,
    heading: str,
) -> Dict[str, int]:
    """
    Parse video IDs and expected interaction levels from a Markdown table
    inside the requested level-2 section.

    The first two columns must be:
        video_id | expected_interaction_level

    Invalid expected values such as TBD are ignored.
    """
    section = extract_markdown_section(
        markdown_text,
        heading,
    )

    if not section:
        return {}

    expected_levels: Dict[str, int] = {}

    for raw_line in section.splitlines():
        line = raw_line.strip()

        if not line.startswith("|"):
            continue

        # Skip Markdown separator rows.
        if "---" in line:
            continue

        cells = [
            cell.strip()
            for cell in line.strip("|").split("|")
        ]

        if len(cells) < 2:
            continue

        # Skip table header.
        if (
            cells[0].lower() == "video_id"
            or cells[1].lower()
            == "expected_interaction_level"
        ):
            continue

        video_id = cells[0]
        expected_value = safe_int(cells[1])

        if not video_id:
            continue

        if video_id.upper().startswith("TODO"):
            continue

        # CHANGED:
        # Rows with TBD expected values are ignored until the user fills them.
        if expected_value is None:
            continue

        expected_levels[video_id] = expected_value

    return expected_levels


# CHANGED:
# New function that merges expected values from both Markdown tables.
def parse_all_expected_levels(
    markdown_text: str,
) -> Dict[str, int]:
    """
    Read expected interaction levels from both evaluation tables.

    The initial Evaluation Table may contain the original tuning set.
    The After Prompt Update table may contain the complete expanded set.

    If the same video appears in both tables, the value from
    After Prompt Update takes precedence.
    """
    initial_levels = parse_expected_levels_from_section(
        markdown_text,
        "## Evaluation Table",
    )

    updated_levels = parse_expected_levels_from_section(
        markdown_text,
        "## After Prompt Update",
    )

    # CHANGED:
    # Merge both dictionaries so added videos from the After Prompt Update
    # table are also collected.
    expected_levels = {
        **initial_levels,
        **updated_levels,
    }

    # CHANGED:
    # Diagnostic output helps verify which section supplied the labels.
    print(
        "Expected levels parsed from "
        f"'## Evaluation Table': {len(initial_levels)}"
    )
    print(
        "Expected levels parsed from "
        f"'## After Prompt Update': {len(updated_levels)}"
    )

    if not expected_levels:
        raise ValueError(
            "No expected interaction levels were found in either "
            "'## Evaluation Table' or "
            "'## After Prompt Update'."
        )

    return expected_levels


def build_interpretation_file_map(
    files: List[Path],
) -> Dict[str, Path]:
    """
    Build a mapping:
        video_id -> VideoInterpretation.json path
    """
    result: Dict[str, Path] = {}

    for file_path in files:
        data = load_json(file_path)
        video_id = extract_video_id(
            data,
            file_path,
        )
        result[video_id] = file_path

    return result


def compute_match(
    expected: int,
    predicted: int | None,
) -> str:
    """
    Compare expected and predicted interaction levels.

    Yes:
        Exact match.

    Partial:
        Prediction differs by one level.

    No:
        Prediction is missing or differs by more than one level.
    """
    if predicted is None:
        return "No"

    if predicted == expected:
        return "Yes"

    if abs(predicted - expected) == 1:
        return "Partial"

    return "No"


def build_auto_note(
    expected: int,
    predicted: int | None,
) -> str:
    """
    Build a short general note from the expected and predicted levels.
    """
    if predicted is None:
        return "LVLM interaction_level missing"

    if predicted == expected:
        return "Correct after prompt update"

    if expected > 0 and predicted == 0:
        return (
            "Still too conservative; "
            "missed visible interaction"
        )

    if expected == 0 and predicted > 0:
        return "May be over-detecting interaction"

    if abs(predicted - expected) == 1:
        return "Close result; needs manual inspection"

    return "Mismatch; needs review"


# CHANGED:
# Added to prevent evidence text containing "|" or newlines
# from breaking the Markdown table.
def sanitize_markdown_cell(value: Any) -> str:
    """
    Convert a value into a safe single-line Markdown table cell.

    Escapes vertical pipes because they otherwise split table columns.
    """
    text = str(value)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("|", "\\|")

    return " ".join(text.split()).strip()


def collect_eval_rows(
    expected_levels: Dict[str, int],
    interpretation_files: List[Path],
) -> List[Dict[str, Any]]:
    """
    Collect LVLM interaction-level evaluation rows.
    """
    file_map = build_interpretation_file_map(
        interpretation_files
    )

    rows: List[Dict[str, Any]] = []

    # CHANGED:
    # Sort video IDs so the console and Markdown output remain predictable.
    for video_id in sorted(expected_levels):
        expected = expected_levels[video_id]
        file_path = file_map.get(video_id)

        if file_path is None:
            rows.append(
                {
                    "video_id": video_id,
                    "expected": expected,
                    "predicted": None,
                    "match": "No",
                    "evidence": (
                        "VideoInterpretation.json not found"
                    ),
                    "notes": "Missing output file",
                }
            )
            continue

        data = load_json(file_path)
        structured = get_lvlm_structured(data)

        predicted = safe_int(
            structured.get("interaction_level")
        )

        evidence = (
            structured.get("interaction_evidence")
            or structured.get("summary")
            or "N/A"
        )

        rows.append(
            {
                "video_id": video_id,
                "expected": expected,
                "predicted": predicted,
                "match": compute_match(
                    expected,
                    predicted,
                ),
                # CHANGED:
                # Sanitize evidence before writing it into Markdown.
                "evidence": sanitize_markdown_cell(
                    evidence
                ),
                "notes": build_auto_note(
                    expected,
                    predicted,
                ),
            }
        )

    return rows


def build_after_prompt_update_section(
    rows: List[Dict[str, Any]],
) -> str:
    """
    Build the Markdown section that replaces
    ## After Prompt Update.
    """
    lines: List[str] = []

    lines.append("## After Prompt Update")
    lines.append("")

    lines.append(
        "| video_id | expected_interaction_level | "
        "predicted_interaction_level_v2 | match | "
        "lvlm_interaction_evidence | notes |"
    )

    lines.append(
        "|---------|----------------------------|"
        "--------------------------------|-------|"
        "---------------------------|------|"
    )

    for row in rows:
        predicted = (
            "N/A"
            if row["predicted"] is None
            else str(row["predicted"])
        )

        lines.append(
            "| "
            f"{sanitize_markdown_cell(row['video_id'])} | "
            f"{row['expected']} | "
            f"{predicted} | "
            f"{row['match']} | "
            f"{sanitize_markdown_cell(row['evidence'])} | "
            f"{sanitize_markdown_cell(row['notes'])} |"
        )

    lines.append("")
    lines.append("### After Prompt Update Summary")
    lines.append("")

    total = len(rows)

    yes_count = sum(
        1
        for row in rows
        if row["match"] == "Yes"
    )

    partial_count = sum(
        1
        for row in rows
        if row["match"] == "Partial"
    )

    no_count = sum(
        1
        for row in rows
        if row["match"] == "No"
    )

    lines.append(
        f"- Total evaluated videos: {total}"
    )
    lines.append(
        f"- Exact matches: {yes_count}"
    )
    lines.append(
        f"- Partial matches: {partial_count}"
    )
    lines.append(
        f"- Mismatches: {no_count}"
    )

    # CHANGED:
    # The summary now includes the calculated percentage.
    if total > 0:
        accuracy = (
            yes_count / total
        ) * 100

        lines.append(
            f"- Exact match rate: "
            f"{yes_count}/{total} = "
            f"{accuracy:.1f}%"
        )

    return "\n".join(lines).strip()


def replace_markdown_section(
    existing_text: str,
    heading: str,
    new_section: str,
) -> str:
    """
    Replace one level-2 Markdown section.

    If the section does not exist, append it to the end.
    """
    lines = existing_text.splitlines()

    start_index = None

    for index, line in enumerate(lines):
        if line.strip() == heading:
            start_index = index
            break

    if start_index is None:
        return (
            existing_text.rstrip()
            + "\n\n"
            + new_section
            + "\n"
        )

    end_index = len(lines)

    for index in range(
        start_index + 1,
        len(lines),
    ):
        if lines[index].strip().startswith("## "):
            end_index = index
            break

    new_lines = (
        lines[:start_index]
        + new_section.splitlines()
        + [""]
        + lines[end_index:]
    )

    return (
        "\n".join(new_lines).rstrip()
        + "\n"
    )


def save_text(
    content: str,
    output_path: Path,
) -> None:
    """
    Save Markdown text to disk.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(content)

    print(
        "Saved LVLM interaction evaluation "
        "Markdown to: "
        f"{output_path.resolve()}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect LVLM interaction-level "
            "evaluation results."
        )
    )

    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help=(
            "Directory containing per-video "
            "output folders."
        ),
    )

    # CHANGED:
    # The script now reads and updates the separate
    # After_Prompt_Update_table.md file by default.
    parser.add_argument(
        "--out",
        type=str,
        default=(
            "evaluation/"
            "After_Prompt_Update_table.md"
        ),
        help=(
            "Markdown file containing the "
            "'## After Prompt Update' table. "
            "The same file is read and updated in place."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outputs_dir = Path(args.outputs_dir)
    output_path = Path(args.out)

    # CHANGED:
    # Print the exact input/output Markdown path to avoid accidentally
    # reading an older or different evaluation file.
    print(
        "Reading evaluation Markdown from: "
        f"{output_path.resolve()}"
    )

    if not output_path.exists():
        print(
            "Evaluation Markdown file not found: "
            f"{output_path}"
        )

        print(
            "Create it first with an "
            "'## After Prompt Update' table "
            "containing numeric expected levels."
        )

        return

    existing_markdown = output_path.read_text(
        encoding="utf-8"
    )

    # CHANGED:
    # Previously this called
    # parse_expected_levels_from_evaluation_table(),
    # which only read the initial evaluation table.
    #
    # This version reads expected values from both sections,
    # including the new standalone After Prompt Update file.
    expected_levels = parse_all_expected_levels(
        existing_markdown
    )

    # CHANGED:
    # Added debug output so it is obvious how many expected labels were found.
    print(
        f"Found {len(expected_levels)} videos "
        "with expected interaction levels."
    )

    # CHANGED:
    # Print every parsed expected value before modifying the Markdown file.
    print("\nExpected interaction levels parsed:")

    for video_id in sorted(expected_levels):
        print(
            f"- {video_id}: "
            f"{expected_levels[video_id]}"
        )

    interpretation_files = (
        find_interpretation_files(
            outputs_dir
        )
    )

    if not interpretation_files:
        print(
            f"No "
            f"{OUTPUT_FILES.interpretation_filename} "
            f"files found under {outputs_dir}"
        )
        return

    # CHANGED:
    # Added debug output for the number of discovered interpretation files.
    print(
        f"\nFound {len(interpretation_files)} "
        "VideoInterpretation.json files."
    )

    rows = collect_eval_rows(
        expected_levels=expected_levels,
        interpretation_files=interpretation_files,
    )

    after_prompt_update_section = (
        build_after_prompt_update_section(
            rows
        )
    )

    markdown = replace_markdown_section(
        existing_text=existing_markdown,
        heading="## After Prompt Update",
        new_section=(
            after_prompt_update_section
        ),
    )

    save_text(
        markdown,
        output_path,
    )

    print(
        "\nLVLM interaction rows collected:"
    )

    for row in rows:
        print(
            f"- {row['video_id']}: "
            f"expected={row['expected']} | "
            f"predicted={row['predicted']} | "
            f"match={row['match']}"
        )


if __name__ == "__main__":
    main()