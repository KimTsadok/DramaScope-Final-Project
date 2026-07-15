import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.io_utils.env_utils import load_dotenv
from src.io_utils.json_utils import load_json_object
from src.pipeline.run_gcp_features import build_features_output_path


class EnvironmentLoaderTests(unittest.TestCase):
    def test_loads_values_and_preserves_existing_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "# comment\nNEW_VALUE=loaded\nQUOTED=\"hello world\"\n"
                "EXISTING=replaced\ninvalid line\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"EXISTING": "original"}, clear=True):
                load_dotenv(env_path)
                self.assertEqual(os.environ["NEW_VALUE"], "loaded")
                self.assertEqual(os.environ["QUOTED"], "hello world")
                self.assertEqual(os.environ["EXISTING"], "original")

    def test_missing_env_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            load_dotenv(Path(directory) / "missing.env")


class JsonObjectLoaderTests(unittest.TestCase):
    def test_loads_json_object(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps({"value": 1}), encoding="utf-8")
            self.assertEqual(load_json_object(path), {"value": 1})

    def test_rejects_missing_directory_and_non_object_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                load_json_object(root / "missing.json")
            with self.assertRaises(ValueError):
                load_json_object(root)

            path = root / "list.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Expected a JSON object"):
                load_json_object(path)

    def test_feature_path_uses_configured_filename(self):
        self.assertEqual(
            build_features_output_path("video"),
            Path("outputs") / "video" / "VideoFeatures.json",
        )


if __name__ == "__main__":
    unittest.main()
