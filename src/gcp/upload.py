# src/gcp/upload.py
"""
This file handles:
* loading .env
* bucket config defaults
* stable destination naming
* upload-if-missing behavior
"""

import os
import time
from pathlib import Path
from google.cloud import storage    
from requests.exceptions import ConnectionError, Timeout
from google.api_core.exceptions import GoogleAPIError, ServiceUnavailable, TooManyRequests

from src.io_utils.env_utils import load_dotenv

def sleep_before_retry(attempt: int) -> None:
    """
    Sleep before retrying a temporary GCS network/API failure.
    attempt starts from 0.
    """
    delay = min(20, 2 ** attempt)
    print(f"Temporary GCS connection issue. Retrying in {delay}s...")
    time.sleep(delay)

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
    Upload a local video to GCS (Google Cloud Storage)
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

    # 1) Check whether the file already exists in GCS, with retries.
    
    exists = False
    max_attempts = 5

    for attempt in range(max_attempts):
        try:
            exists = blob.exists(client=client)
            break

        except (ConnectionError, Timeout, GoogleAPIError, ServiceUnavailable, TooManyRequests) as exc:
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Failed to check whether blob exists after {max_attempts} attempts: "
                    f"gs://{bucket_name}/{destination_blob_name}. Error: {exc}"
                ) from exc

            sleep_before_retry(attempt)

    # 2) If it exists, reuse it.
    if exists:
        print(f"Already exists in bucket, skipping upload: gs://{bucket_name}/{destination_blob_name}")
        return f"gs://{bucket_name}/{destination_blob_name}"

    # 3) If it does not exist, upload it, also with retries.
    for attempt in range(max_attempts):
        try:
            print(f"Uploading to bucket: gs://{bucket_name}/{destination_blob_name}")
            blob.upload_from_filename(str(path))
            break

        except (ConnectionError, Timeout, GoogleAPIError, ServiceUnavailable, TooManyRequests) as exc:
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Failed to upload video after {max_attempts} attempts: "
                    f"gs://{bucket_name}/{destination_blob_name}. Error: {exc}"
                ) from exc

            sleep_before_retry(attempt)

    return f"gs://{bucket_name}/{destination_blob_name}"
