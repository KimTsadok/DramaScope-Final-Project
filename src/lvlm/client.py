# src/lvlm/client.py

import base64
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import ffmpeg
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    RateLimitError,
)

from src.config import FRAME_SETTINGS, MODEL_SETTINGS
from src.lvlm.prompts import PROMPT_SUMMARY_V1, PROMPT_STRUCTURED_V1
from src.gcp.upload import load_dotenv

# ---------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------

# Load .env immediately when module is imported
load_dotenv()

# ---------------------------------------------------------------------
# LVLM client configuration
# ---------------------------------------------------------------------
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
GLM_BASE_URL = "https://api.z.ai/api/paas/v4/"

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 20.0
JITTER_RATIO = 0.25
REQUEST_TIMEOUT_SECONDS = 120
MAX_COMPLETION_TOKENS = 1024


# ---------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------
def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds using ffmpeg probe.
    """
    probe = ffmpeg.probe(video_path)
    return float(probe["format"]["duration"])


def extract_frames(
    video_path: str,
    fps: int = FRAME_SETTINGS.frame_rate,
    max_frames: int = FRAME_SETTINGS.max_frames,
) -> List[str]:
    """
    Extract evenly sampled frames from a video file and return them as
    base64-encoded JPEG strings.
    """
    duration = get_video_duration(video_path)
    print(f"Video duration: {duration:.1f}s | Sampling at {fps} fps")

    total_samples = min(max_frames, max(1, int(duration * fps)))
    timestamps = [duration * i / total_samples for i in range(total_samples)]

    frames_b64: List[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for index, timestamp in enumerate(timestamps):
            frame_path = os.path.join(tmpdir, f"frame_{index:04d}.jpg")

            (
                ffmpeg.input(video_path, ss=timestamp)
                .output(frame_path, vframes=1, format="image2", vcodec="mjpeg")
                .overwrite_output()
                .run(quiet=True)
            )

            if os.path.exists(frame_path):
                with open(frame_path, "rb") as file:
                    encoded = base64.b64encode(file.read()).decode("utf-8")
                    frames_b64.append(encoded)
                    print(f"  Captured frame at {timestamp:.1f}s -> sample {len(frames_b64)}")

    print(f"\nExtracted {len(frames_b64)} frames\n")
    return frames_b64


# ---------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------
def build_messages(frames_b64: List[str], prompt: str) -> List[Dict[str, Any]]:
    """
    Build the chat-completions message payload with interleaved image frames
    and one text instruction.
    """
    content: List[Dict[str, Any]] = []

    for encoded_frame in frames_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_frame}"
                },
            }
        )

    content.append(
        {
            "type": "text",
            "text": (
                f"The above {len(frames_b64)} images are evenly sampled frames "
                f"from a video. {prompt}"
            ),
        }
    )

    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------
def is_retryable_error(exc: Exception) -> bool:
    """
    Return True if the exception should trigger a retry.
    """
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True

    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500

    return False


def compute_backoff_seconds(attempt: int) -> float:
    """
    Exponential backoff with jitter for attempt 1..N.
    """
    base_delay = min(
        MAX_BACKOFF_SECONDS,
        INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)),
    )
    jitter_factor = random.uniform(1 - JITTER_RATIO, 1 + JITTER_RATIO)
    return max(0.0, base_delay * jitter_factor)


def call_glm_with_retry(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
):
    """
    Call chat completion with retry/backoff on transient API errors.
    """
    attempt = 1

    while attempt <= MAX_RETRIES:
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        except Exception as exc:
            retryable = is_retryable_error(exc)
            if not retryable:
                raise

            if attempt >= MAX_RETRIES:
                raise RuntimeError(
                    f"GLM API call failed after {MAX_RETRIES} attempts: {exc}"
                ) from exc

            delay = compute_backoff_seconds(attempt)
            print(
                f"Retry {attempt}/{MAX_RETRIES - 1} after transient API error "
                f"({type(exc).__name__}): waiting {delay:.1f}s"
            )
            time.sleep(delay)
            attempt += 1


# ---------------------------------------------------------------------
# Internal inference helpers
# ---------------------------------------------------------------------
def build_result(
    model_name: str,
    prompt_version: str,
    frame_count: int,
    response,
) -> Dict[str, Any]:
    """
    Build the standard result dictionary returned by LVLM inference functions.
    """
    response_text = response.choices[0].message.content or ""

    usage = {}
    if getattr(response, "usage", None):
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }

    return {
        "model": model_name,
        "prompt_version": prompt_version,
        "frame_count": frame_count,
        "text": response_text,
        "usage": usage,
    }


def run_inference_from_frames(
    frames_b64: List[str],
    prompt: str,
    prompt_version: str,
) -> Dict[str, Any]:
    """
    Run LVLM inference using frames that were already extracted.
    """
    if not GLM_API_KEY:
        raise RuntimeError("GLM_API_KEY is not set. Add it to .env and rerun.")

    if not frames_b64:
        raise RuntimeError("No frames were provided for LVLM inference.")

    model_name = MODEL_SETTINGS.lvlm_model_name
    messages = build_messages(frames_b64, prompt)

    client = OpenAI(
        api_key=GLM_API_KEY,
        base_url=GLM_BASE_URL,
    )

    response = call_glm_with_retry(
        client=client,
        model=model_name,
        messages=messages,
        max_tokens=MAX_COMPLETION_TOKENS,
    )

    return build_result(
        model_name=model_name,
        prompt_version=prompt_version,
        frame_count=len(frames_b64),
        response=response,
    )


def run_inference(
    video_path: str,
    prompt: str,
    prompt_version: str,
) -> Dict[str, Any]:
    """
    Run LVLM inference starting from a local video path.
    This is a convenience wrapper around frame extraction + inference.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Processing LVLM inference: {video_path}")
    print(f"Model: {MODEL_SETTINGS.lvlm_model_name}\n")

    frames_b64 = extract_frames(
        video_path=video_path,
        fps=FRAME_SETTINGS.frame_rate,
        max_frames=FRAME_SETTINGS.max_frames,
    )

    return run_inference_from_frames(
        frames_b64=frames_b64,
        prompt=prompt,
        prompt_version=prompt_version,
    )


# -----------------------------------------------------------------------
# Public inference functions - for not duplicating code - extract frames
# -----------------------------------------------------------------------
"""
video_path
→ extract frames once
→ summary inference from those frames
→ structured inference from the same frames
"""
def run_summary_inference(video_path: str) -> Dict[str, Any]:
    """
    Run free-form summary inference for the given video path.
    """
    return run_inference(
        video_path=video_path,
        prompt=PROMPT_SUMMARY_V1,
        prompt_version=MODEL_SETTINGS.lvlm_prompt_version + "_summary",
    )


def run_structured_inference(video_path: str) -> Dict[str, Any]:
    """
    Run structured-JSON inference for the given video path.
    """
    return run_inference(
        video_path=video_path,
        prompt=PROMPT_STRUCTURED_V1,
        prompt_version=MODEL_SETTINGS.lvlm_prompt_version + "_structured",
    )


def run_summary_inference_from_frames(frames_b64: List[str]) -> Dict[str, Any]:
    """
    Run free-form summary inference using already extracted frames.
    """
    return run_inference_from_frames(
        frames_b64=frames_b64,
        prompt=PROMPT_SUMMARY_V1,
        prompt_version=MODEL_SETTINGS.lvlm_prompt_version + "_summary",
    )


def run_structured_inference_from_frames(frames_b64: List[str]) -> Dict[str, Any]:
    """
    Run structured-JSON inference using already extracted frames.
    """
    return run_inference_from_frames(
        frames_b64=frames_b64,
        prompt=PROMPT_STRUCTURED_V1,
        prompt_version=MODEL_SETTINGS.lvlm_prompt_version + "_structured",
    )