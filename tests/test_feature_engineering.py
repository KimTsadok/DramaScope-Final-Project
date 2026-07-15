import unittest
from types import SimpleNamespace

from src.gcp.feature_engineering import compute_video_duration_seconds


def make_annotations(shot_ends=(), object_ends=()):
    return SimpleNamespace(
        shot_annotations=[
            SimpleNamespace(end_time_offset=value)
            for value in shot_ends
        ],
        object_annotations=[
            SimpleNamespace(
                segment=SimpleNamespace(end_time_offset=value)
            )
            for value in object_ends
        ],
    )


class ComputeVideoDurationTests(unittest.TestCase):
    def test_object_end_later_than_shot_is_used(self):
        annotations = make_annotations(shot_ends=(5,), object_ends=(8,))
        self.assertEqual(compute_video_duration_seconds(annotations), 8.0)

    def test_shot_end_later_than_object_is_used(self):
        annotations = make_annotations(shot_ends=(8,), object_ends=(5,))
        self.assertEqual(compute_video_duration_seconds(annotations), 8.0)

    def test_single_annotation_source_is_supported(self):
        self.assertEqual(
            compute_video_duration_seconds(make_annotations(shot_ends=(4,))),
            4.0,
        )
        self.assertEqual(
            compute_video_duration_seconds(make_annotations(object_ends=(6,))),
            6.0,
        )

    def test_empty_annotations_return_zero(self):
        self.assertEqual(
            compute_video_duration_seconds(make_annotations()),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
