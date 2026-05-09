# src/lvlm/prompts.py

"""
This script should only define prompt strings.

PROMPT_SUMMARY_V1
Good for:
* readable description
* quick sanity check
* storing a human-readable summary

PROMPT_STRUCTURED_V1
Good for:
* stable fields
* easier downstream usage
* easier validation
* easier future comparison across videos
"""

PROMPT_SUMMARY_V1 = """
Describe what is happening in this video.
Focus on:
- setting
- people or main entities
- visible actions
- emotional tone
- interaction between people if present

Keep the answer concise and factual.
""".strip()


PROMPT_STRUCTURED_V1 = """
Return JSON only.

Schema:
{
  "setting": "string",
  "main_entities": ["string"],
  "actions": ["string"],
  "emotion_words": ["string"],
  "interaction_level": 0,
  "summary": "string"
}

Rules:
- interaction_level must be an integer from 0 to 3
- use short factual strings
- if uncertain, use best effort
- do not include markdown
- do not include explanations outside the JSON
""".strip()