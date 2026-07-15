import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pipeline.run_batch_pipeline import (
    has_valid_algorithm_output,
    has_valid_lvlm_output,
    should_skip_video,
)
from src.pipeline.run_full_pipeline import run_lvlm_stage_with_fallback


def valid_interpretation(video_id="video"):
    return {
        "video_id": video_id,
        "features_raw": {
            "shot_frequency": 0.2,
            "object_entropy": 1.0,
            "interaction_density": 0.5,
            "human_presence_ratio": 0.4,
        },
        "features_norm": {
            "shot_frequency": 0.2,
            "object_entropy": 0.25,
            "interaction_density": 0.1,
            "human_presence_ratio": 0.4,
        },
        "scene_complexity_score": 0.25,
        "scene_complexity_breakdown": {
            "shot_frequency": 0.07,
            "object_entropy": 0.0875,
            "interaction_density": 0.02,
            "human_presence_ratio": 0.04,
        },
        "narrative_phase": "Calm",
        "phase_reasons": ["Defaulted to Calm"],
        "lvlm_summary": {"text": "summary"},
        "lvlm_structured_raw": {
            "text": '{"interaction_level": 2}'
        },
        "lvlm_structured": {"interaction_level": 2},
        "lvlm_error": None,
    }


class LvlmFallbackTests(unittest.TestCase):
    @patch("src.pipeline.run_full_pipeline.extract_frames", return_value=["frame"])
    @patch(
        "src.pipeline.run_full_pipeline.run_summary_inference_from_frames",
        return_value={"text": "summary"},
    )
    @patch(
        "src.pipeline.run_full_pipeline.run_structured_inference_from_frames",
        return_value={
            "text": '{"summary":"structured","interaction_level":2}'
        },
    )
    def test_complete_lvlm_result_succeeds(self, _structured, _summary, _frames):
        fields, succeeded = run_lvlm_stage_with_fallback(Path("video.mp4"))

        self.assertTrue(succeeded)
        self.assertIsNone(fields["lvlm_error"])
        self.assertEqual(fields["lvlm_structured"]["interaction_level"], 2)

    @patch("src.pipeline.run_full_pipeline.extract_frames", return_value=["frame"])
    @patch(
        "src.pipeline.run_full_pipeline.run_summary_inference_from_frames",
        return_value={"text": "summary"},
    )
    @patch(
        "src.pipeline.run_full_pipeline.run_structured_inference_from_frames",
        side_effect=RuntimeError("structured failed"),
    )
    def test_summary_is_preserved_when_structured_stage_fails(
        self, _structured, _summary, _frames
    ):
        fields, succeeded = run_lvlm_stage_with_fallback(Path("video.mp4"))

        self.assertFalse(succeeded)
        self.assertEqual(fields["lvlm_summary"], {"text": "summary"})
        self.assertIsNone(fields["lvlm_structured_raw"])
        self.assertIn("structured failed", fields["lvlm_error"])

    @patch("src.pipeline.run_full_pipeline.extract_frames", return_value=["frame"])
    @patch(
        "src.pipeline.run_full_pipeline.run_summary_inference_from_frames",
        side_effect=RuntimeError("summary failed"),
    )
    @patch(
        "src.pipeline.run_full_pipeline.run_structured_inference_from_frames",
        return_value={"text": '{"interaction_level":1}'},
    )
    def test_structured_result_is_preserved_when_summary_fails(
        self, _structured, _summary, _frames
    ):
        fields, succeeded = run_lvlm_stage_with_fallback(Path("video.mp4"))

        self.assertFalse(succeeded)
        self.assertIsNone(fields["lvlm_summary"])
        self.assertEqual(fields["lvlm_structured"]["interaction_level"], 1)
        self.assertIn("summary failed", fields["lvlm_error"])


class BatchCompletenessTests(unittest.TestCase):
    def test_valid_current_shape_is_accepted(self):
        interpretation = valid_interpretation("video")
        self.assertTrue(
            has_valid_algorithm_output(
                interpretation,
                expected_video_id="video",
            )
        )
        self.assertTrue(has_valid_lvlm_output(interpretation))

    def test_mismatched_video_id_and_invalid_types_are_rejected(self):
        interpretation = valid_interpretation("other")
        self.assertFalse(
            has_valid_algorithm_output(
                interpretation,
                expected_video_id="video",
            )
        )

        interpretation = valid_interpretation("video")
        interpretation["features_norm"] = []
        self.assertFalse(has_valid_algorithm_output(interpretation))

        interpretation = valid_interpretation("video")
        interpretation["scene_complexity_score"] = float("nan")
        self.assertFalse(has_valid_algorithm_output(interpretation))

    def test_invalid_lvlm_interaction_level_is_rejected(self):
        interpretation = valid_interpretation("video")
        interpretation["lvlm_structured"]["interaction_level"] = 4
        self.assertFalse(has_valid_lvlm_output(interpretation))

    def test_skip_requires_matching_complete_output_for_every_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features_path = root / "VideoFeatures.json"
            interpretation_path = root / "VideoInterpretation.json"
            features_path.write_text("{}", encoding="utf-8")
            interpretation_path.write_text(
                json.dumps(valid_interpretation("video")),
                encoding="utf-8",
            )

            with patch(
                "src.pipeline.run_batch_pipeline.build_features_path",
                return_value=features_path,
            ), patch(
                "src.pipeline.run_batch_pipeline.build_interpretation_path",
                return_value=interpretation_path,
            ):
                video_path = Path("video.mp4")
                for mode in ("gcp", "lvlm", "full"):
                    with self.subTest(mode=mode):
                        self.assertTrue(
                            should_skip_video(video_path, mode, force=False)
                        )
                        self.assertFalse(
                            should_skip_video(video_path, mode, force=True)
                        )

                interpretation_path.write_text(
                    json.dumps(valid_interpretation("different")),
                    encoding="utf-8",
                )
                for mode in ("gcp", "lvlm", "full"):
                    with self.subTest(mode=mode, mismatch=True):
                        self.assertFalse(
                            should_skip_video(video_path, mode, force=False)
                        )


if __name__ == "__main__":
    unittest.main()
