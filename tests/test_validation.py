import copy
import unittest

from src.algorithm.validate import validate_video_features


class ValidateVideoFeaturesTests(unittest.TestCase):
    def make_features(self):
        return {
            "video_uri": "gs://bucket/video.mp4",
            "duration_seconds": 10.0,
            "shots": {"count": 3, "other": "preserved"},
            "objects": {
                "object_entropy": 1.5,
                "interaction_density_tracks_per_sec": 0.8,
                "human_presence_ratio": 0.4,
                "other": "preserved",
            },
        }

    def test_valid_values_are_preserved_without_mutating_input(self):
        features = self.make_features()
        original = copy.deepcopy(features)

        validated = validate_video_features(features)

        self.assertEqual(validated, original)
        self.assertEqual(features, original)
        self.assertIsNot(validated, features)
        self.assertIsNot(validated["shots"], features["shots"])
        self.assertIsNot(validated["objects"], features["objects"])

    def test_missing_optional_numeric_fields_keep_compatible_defaults(self):
        validated = validate_video_features(
            {"duration_seconds": 2.0, "shots": {}, "objects": {}}
        )

        self.assertEqual(validated["shots"]["count"], 0)
        self.assertEqual(validated["objects"]["object_entropy"], 0.0)
        self.assertEqual(
            validated["objects"]["interaction_density_tracks_per_sec"],
            0.0,
        )
        self.assertEqual(validated["objects"]["human_presence_ratio"], 0.0)

    def test_human_presence_ratio_is_still_clamped(self):
        features = self.make_features()
        features["objects"]["human_presence_ratio"] = 1.5

        validated = validate_video_features(features)

        self.assertEqual(validated["objects"]["human_presence_ratio"], 1.0)

    def test_non_mapping_nested_sections_raise_clear_errors(self):
        for field in ("shots", "objects"):
            with self.subTest(field=field):
                features = self.make_features()
                features[field] = None
                with self.assertRaisesRegex(ValueError, rf"{field} must be a dictionary"):
                    validate_video_features(features)

    def test_negative_counts_and_object_measurements_are_rejected(self):
        cases = (
            ("shots", "count", -1),
            ("objects", "object_entropy", -0.1),
            ("objects", "interaction_density_tracks_per_sec", -0.1),
        )

        for section, field, value in cases:
            with self.subTest(section=section, field=field):
                features = self.make_features()
                features[section][field] = value
                with self.assertRaises(ValueError):
                    validate_video_features(features)

    def test_explicit_non_finite_numeric_fields_are_rejected(self):
        for section, field in (
            ("shots", "count"),
            ("objects", "object_entropy"),
            ("objects", "interaction_density_tracks_per_sec"),
            ("objects", "human_presence_ratio"),
        ):
            with self.subTest(section=section, field=field):
                features = self.make_features()
                features[section][field] = float("nan")
                with self.assertRaisesRegex(ValueError, "finite number"):
                    validate_video_features(features)

    def test_invalid_duration_is_rejected(self):
        for value in (0, -1, float("nan"), float("inf"), None):
            with self.subTest(value=value):
                features = self.make_features()
                features["duration_seconds"] = value
                with self.assertRaisesRegex(ValueError, "duration_seconds"):
                    validate_video_features(features)


if __name__ == "__main__":
    unittest.main()
