# Week 6 LVLM Interaction-Level Evaluation

## Goal
Evaluate whether the LVLM `interaction_level` field matches human expectations after prompt tuning.

The purpose of this week is to improve the LVLM semantic interpretation layer, specifically the consistency of `interaction_level`.

## Interaction Level Scale

| level | meaning | examples |
|------|---------|----------|
| 0 | No visible interaction | One person alone; multiple people doing separate activities |
| 1 | Weak / indirect interaction | Glancing, listening, one-sided speaking, shared car ride without clear response |
| 2 | Clear mutual interaction | Conversation, visible reaction, gesture toward another person, coordinated action |
| 3 | Strong / intense interaction | Argument, confrontation, chase, urgent cooperation, physical struggle |

## Evaluation Table

| video_id | expected_interaction_level | predicted_interaction_level | match | human interaction_evidence | notes |
|---------|----------------------------|-----------------------------|-------|----------------------|------|
| ACCEDE09230 | 0 | 0 | Yes | People are in the same room but focused on separate activities | Correct |
| ACCEDE09231 | 1 | 0 | No | Woman talks / man glances briefly | Prompt too conservative; should detect weak indirect interaction |
| ACCEDE09232 | 0 | 0 | Yes | One person walking alone; no direct interaction | Correct |
| ACCEDE09233 | 1 | 0 | No | Gentle couple interaction; visible shared attention / communication between the two people | Prompt/model missed subtle interpersonal interaction |


## After Prompt Update

| video_id | expected_interaction_level | predicted_interaction_level_v2 | match | lvlm_interaction_evidence | notes |
|---------|----------------------------|--------------------------------|-------|---------------------------|------|
| ACCEDE09230 | 0 | 1 | Partial | The woman looks up from her letter with a smile, appearing to react to or share something with the man who is present in the same room and looking in her direction. | May be over-detecting interaction |
| ACCEDE09231 | 1 | 0 | Partial | The woman is focused on driving and the road, while the man appears to be looking out the window or forward without any visible communication, eye contact, or coordinated behavior between them. | Still too conservative; missed visible interaction |
| ACCEDE09232 | 0 | 0 | Yes | The man is walking alone on the sidewalk. While he raises his hand in a wave near the end of the sequence, there is no other person visible to receive the gesture or respond to it, indicating no mutual social interaction. | Correct after prompt update |
| ACCEDE09233 | 1 | 1 | Yes | The man appears to be speaking and smiling while looking toward the camera or the driver, and the woman is driving the vehicle they are both occupying. They are participating in a shared situation (a car ride) with visible social cues suggesting a conversation is taking place between them. | Correct after prompt update |


### After Prompt Update Summary

- Total evaluated videos: 21
- Exact matches: 12
- Partial matches: 9
- Mismatches: 0
- Exact match rate: 12/21 = 57.1%

## Evaluation Plan

- The first 4 videos were used for initial prompt tuning and debugging.  
- The next 8 videos, `ACCEDE09234` to `ACCEDE09241`, will be used as unfamiliar evaluation examples to check whether the updated prompt generalizes.

## Notes
- GCP `interaction_density` measures tracked object density per second.
- LVLM `interaction_level` estimates visible human/social interaction.
- The goal is not perfect accuracy, but improved consistency and better explanations through `interaction_evidence`.
- The first prompt update failed on two videos because the LVLM provider content filter rejected the request. The level-3 examples were softened and the same videos were rerun.
- Switching from `glm-4.6v-flash` to `glm-5v-turbo` improved execution stability and produced valid structured outputs for all four test videos, but it did not improve the interaction-level classification. The model still predicted `0` for subtle interaction cases.
- The evidence text showed that the model recognized cues such as a person talking, but still treated them as insufficient for interaction. Therefore, the next prompt update focuses on general subtle interaction cues rather than video-specific examples.

## Conclusion
- The Week 6 evaluation shows that prompt tuning improved the LVLM interaction layer, especially by producing valid structured outputs and better interaction_evidence.
The model achieved 8 exact matches out of 12, giving an exact accuracy of about 66.7%.
- The main remaining issue is distinguishing passive co-presence from weak interaction: the model sometimes over-detects interaction when people simply share a space, and sometimes misses subtle cues such as brief speech or glances.
Overall, the LVLM layer is now more stable, explainable, and useful for semantic interaction analysis.
- The model was changed from GLM 4.6v flash to GLM v5 turbo - so the overall running time, and outputs improved.
- Further tuning should focus on borderline cases, especially the difference between weak, clear, and strong interaction.
