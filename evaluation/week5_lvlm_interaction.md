# Week 5 LVLM Interaction-Level Evaluation

## Goal
Evaluate and tune the LVLM `interaction_level` field using a small batch of videos.

## Evaluation Table

| video_id | expected_interaction_level | predicted_interaction_level | match | interaction_evidence | notes |
|---------|----------------------------|-----------------------------|-------|----------------------|------|
| ACCEDE09230 | 0 | 0 | Yes | People are in the same room but focused on separate activities | Correct |
| ACCEDE09231 | 1 | 0 | No | Woman talks / man glances briefly | Prompt too conservative; should detect weak indirect interaction |
| ACCEDE09232 | 0 | 0 | Yes | One person walking alone; no direct interaction | Correct |
| ACCEDE09233 | 2 | 0 | No | Mutual communication between characters | Prompt missed visible interaction |

## After Prompt Update

| video_id | expected_interaction_level | predicted_interaction_level_v2 | match | notes |
|---------|----------------------------|--------------------------------|-------|------|
| ACCEDE09230 | 0 | TODO | TODO | TODO |
| ACCEDE09231 | 1 | TODO | TODO | TODO |
| ACCEDE09232 | 0 | TODO | TODO | TODO |
| ACCEDE09233 | 2 | TODO | TODO | TODO |