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

#updated LVLM interaction level description
PROMPT_STRUCTURED_V1 = """
Return JSON only.

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
0 = no visible interaction between entities.
    Examples: people are present but doing separate activities; one person walks alone.
1 = weak or indirect interaction.
    Examples: one person speaks while another listens; one person glances at another; people share the same activity but do not clearly respond to each other.
2 = clear mutual interaction.
    Examples: two people talk to each other, react to each other, gesture toward each other, or coordinate actions.
3 = strong or intense interaction.
    Examples: argument, physical struggle, urgent emotional exchange, chase, direct confrontation, intense cooperation.

Rules:
- interaction_level must be based on visible behavior only.
- If there is speaking, looking, gesturing, or reaction between people, do not return 0.
- Return 0 only when entities are present but no direct or indirect interaction is visible.
- Include a short interaction_evidence string explaining the chosen level.
- Do not include markdown.
- Do not include text outside the JSON.
""".strip()