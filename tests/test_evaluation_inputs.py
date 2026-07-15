import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.collect_feature_stats import (
    collect_raw_feature_values,
    load_json,
    safe_float,
)
from src.pipeline.collect_lvlm_interaction import (
    build_interpretation_file_map,
    collect_eval_rows,
    parse_all_expected_levels,
    parse_expected_levels_from_section,
    safe_interaction_level,
)


class InteractionEvaluationInputTests(unittest.TestCase):
    def test_expected_levels_must_be_between_zero_and_three(self):
        for value in (-1, 4):
            markdown = (
                "## Evaluation Table\n\n"
                "| video_id | expected_interaction_level |\n"
                "|---|---|\n"
                f"| video | {value} |"
            )
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Expected 0 through 3"):
                    parse_expected_levels_from_section(
                        markdown,
                        "## Evaluation Table",
                    )

    def test_updated_section_still_overrides_initial_section(self):
        markdown = (
            "## Evaluation Table\n\n"
            "| video_id | expected_interaction_level |\n"
            "|---|---|\n"
            "| video | 1 |\n\n"
            "## After Prompt Update\n\n"
            "| video_id | expected_interaction_level |\n"
            "|---|---|\n"
            "| video | 2 |"
        )

        self.assertEqual(parse_all_expected_levels(markdown), {"video": 2})

    def test_duplicate_video_ids_from_different_files_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for folder in ("first", "second"):
                path = root / folder / "VideoInterpretation.json"
                path.parent.mkdir()
                path.write_text(
                    json.dumps({"video_id": "duplicate"}),
                    encoding="utf-8",
                )
                paths.append(path)

            with self.assertRaisesRegex(ValueError, "Duplicate video_id"):
                build_interpretation_file_map(paths)

    def test_out_of_range_prediction_is_reported_as_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video" / "VideoInterpretation.json"
            path.parent.mkdir()
            path.write_text(
                json.dumps(
                    {
                        "video_id": "video",
                        "lvlm_structured": {
                            "interaction_level": 9,
                            "interaction_evidence": "evidence",
                        },
                    }
                ),
                encoding="utf-8",
            )

            rows = collect_eval_rows({"video": 2}, [path])

            self.assertIsNone(rows[0]["predicted"])
            self.assertEqual(rows[0]["match"], "No")
            self.assertIn("Invalid", rows[0]["notes"])

    def test_non_integral_and_boolean_levels_are_rejected(self):
        self.assertIsNone(safe_interaction_level(1.5))
        self.assertIsNone(safe_interaction_level(True))


class FeatureStatisticsInputTests(unittest.TestCase):
    def test_non_finite_values_are_not_accepted(self):
        for value in (float("nan"), float("inf"), float("-inf"), "nan"):
            with self.subTest(value=value):
                self.assertIsNone(safe_float(value))

    def test_json_root_must_be_an_object(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Expected a JSON object"):
                load_json(path)

    def test_features_raw_must_be_an_object(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "VideoInterpretation.json"
            path.write_text(
                json.dumps({"features_raw": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "features_raw"):
                collect_raw_feature_values([path])


if __name__ == "__main__":
    unittest.main()
