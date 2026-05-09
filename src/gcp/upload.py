# src/gcp/upload.py
"""
This file handles:
* loading .env
* bucket config defaults
* stable destination naming
* upload-if-missing behavior
"""

import os
from pathlib import Path
from google.cloud import storage    

def load_dotenv(dotenv_path: str = ".env") -> None:
    """
    Load environment variables from a local .env file if it exists.
    Only sets variables that are not already present in os.environ.
    """
    env_path = Path(dotenv_path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


# Load .env immediately when module is imported
load_dotenv()

DEFAULT_BUCKET_NAME = os.getenv("DEFAULT_BUCKET_NAME", "final-project-bucket1")
DEFAULT_GCS_PREFIX = os.getenv("DEFAULT_GCS_PREFIX", "uploads")


def build_destination_name(local_path: Path, prefix: str) -> str:
    """
    Build a stable object name in the bucket.

    Example:
    local_path = C:/videos/ACCEDE09230.mp4
    prefix = uploads

    result:
    uploads/ACCEDE09230.mp4
    """
    prefix = prefix.strip("/")
    return f"{prefix}/{local_path.name}"


def upload_to_gcs(local_video_path: str, bucket_name: str, prefix: str) -> str:
    """
    Upload a local video to GCS(Google Cloud Storage)
    only if it does not already exist.

    Returns:
        gs://<bucket_name>/<prefix>/<filename>
    """
    path = Path(local_video_path).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    destination_blob_name = build_destination_name(path, prefix)
    blob = bucket.blob(destination_blob_name)

    if blob.exists(client=client):
        print(f"Already exists in bucket, skipping upload: gs://{bucket_name}/{destination_blob_name}")
    else:
        print(f"Uploading to bucket: gs://{bucket_name}/{destination_blob_name}")
        blob.upload_from_filename(str(path))

    return f"gs://{bucket_name}/{destination_blob_name}"