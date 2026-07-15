# src/lvlm/prompts.py

"""
This script should only define prompt strings.

PROMPT_SUMMARY_V1
Good for:
* readable description
* quick sanity check
* storing a human-readable summary

PROMPT_STRUCTURED_V3
Good for:
* stable fields
* interaction-level tuning
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

PROMPT_STRUCTURED_V3 = """
Return exactly one valid JSON object.

Do not include markdown.
Do not include explanations before or after the JSON.
Do not use code fences.
The response must start with { and end with }.

Schema:
{
  "setting": "string",
  "main_entities": ["string"],
  "actions": ["string"],
  "emotion_words": ["string"],
  "interaction_level": 0,
  "interaction_evidence": "string",
  "summary": "string"
}

Interaction level scale:
0 = No visible interaction.
    Use only when entities are alone, separated, or clearly focused on unrelated independent activities.
    If there is no visible social, communicative, responsive, or coordinated cue, use 0.

1 = Weak or indirect interaction.
    Use when there is a subtle social or task-based connection between entities.
    This includes one-sided communication, speaking posture, listening posture, attentive gaze, shared attention, passive observation, brief reaction, or people participating in the same shared situation without strong mutual exchange.
    Level 1 does not require physical contact, large gestures, direct eye contact, or obvious back-and-forth conversation.

2 = Clear mutual interaction.
    Use when entities visibly communicate, respond to each other, gesture toward each other, coordinate actions, or appear engaged in the same exchange.
    This requires clearer mutual involvement than level 1.

3 = Strong or highly active interaction.
    Use when the interaction is intense, urgent, emotionally strong, or involves clearly active coordinated behavior between entities.

Important rules:
- Do not return 0 if there is visible speaking, listening, attentive gaze, reaction, shared attention, or coordinated behavior between entities.
- If one entity appears to be communicating and another entity appears present, attentive, affected, or contextually involved, choose at least interaction_level 1.
- Use interaction_level 0 only when there is no visible social, communicative, responsive, or coordinated cue.
- Prefer level 1 over level 0 when weak interaction cues are visible but not strong enough for level 2.
- Always include interaction_evidence explaining the chosen level.
""".strip()

# Active prompts used by the LVLM client.
# Update these aliases when changing prompt versions.
ACTIVE_SUMMARY_PROMPT = PROMPT_SUMMARY_V1
ACTIVE_STRUCTURED_PROMPT = PROMPT_STRUCTURED_V3