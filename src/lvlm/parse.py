# src/lvlm/parse.py
"""
The structured prompt asks the model for JSON only, but models sometimes still return:

* code fences
* extra commentary
* malformed JSON
So this file should clean that up.
"""
import json
import math
import re
from typing import Any, Dict, List


def strip_code_fences(text: str) -> str:
    """
    Remove markdown code fences such as ```json ... ``` or ``` ... ```.
    """
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    return cleaned.strip()


def extract_json_block(text: str) -> str:
    """
    Extract the first top-level JSON object block from text.

    This is a lightweight fallback for cases where the model returns
    extra text before or after the JSON.
    """
    start_index = text.find("{")
    if start_index == -1:
        raise ValueError("No JSON object start ('{') found in response.")

    depth = 0
    end_index = None

    for index in range(start_index, len(text)):
        char = text[index]

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end_index = index
                break

    if end_index is None:
        raise ValueError("Could not find the end of the JSON object in response.")

    return text[start_index:end_index + 1]


def ensure_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def ensure_string_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value]

    return [str(value)]


def coerce_interaction_level(value: Any) -> int:
    """
    Convert interaction_level to an int in [0, 3].
    """
    try:
        if value is None:
            return 0

        if isinstance(value, float) and math.isnan(value):
            return 0

        level = int(value)
        return max(0, min(3, level))

    except (ValueError, TypeError):
        return 0


def normalize_structured_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the structured LVLM response into the expected schema.
    Missing fields are repaired with safe defaults.
    """
    return {
        "setting": ensure_string(data.get("setting")),
        "main_entities": ensure_string_list(data.get("main_entities")),
        "actions": ensure_string_list(data.get("actions")),
        "emotion_words": ensure_string_list(data.get("emotion_words")),
        "interaction_level": coerce_interaction_level(data.get("interaction_level")),
        "summary": ensure_string(data.get("summary")),
    }


def parse_structured_response(response_text: str) -> Dict[str, Any]:
    """
    Parse a structured LVLM response into a validated dictionary.

    Steps:
    1. Strip markdown code fences
    2. Try direct JSON parsing
    3. If that fails, extract the first JSON block and try again
    4. Normalize fields to the expected schema
    """
    cleaned_text = strip_code_fences(response_text)

    try:
        parsed = json.loads(cleaned_text)
        if not isinstance(parsed, dict):
            raise ValueError("Structured LVLM response is not a JSON object.")
        return normalize_structured_fields(parsed)

    except Exception:
        json_block = extract_json_block(cleaned_text)
        parsed = json.loads(json_block)

        if not isinstance(parsed, dict):
            raise ValueError("Structured LVLM response is not a JSON object.")

        return normalize_structured_fields(parsed)
