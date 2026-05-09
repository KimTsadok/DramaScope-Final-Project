# Week 4 LVLM Evaluation

## Goal
By the end of Week 4, the system should extend the existing `VideoInterpretation.json` output with LVLM-based semantic metadata.

For each tested video, the pipeline should produce:
- `VideoFeatures.json`
- `VideoInterpretation.json`

and `VideoInterpretation.json` should now include:
- `lvlm_summary`
- `lvlm_structured_raw`
- `lvlm_structured`

The goal of this evaluation is to check whether the LVLM layer gives a reasonable semantic interpretation of the video and whether it supports or contradicts the algorithmic interpretation from Week 3.

---

## Evaluation Table

| video_id | narrative_phase | complexity_score | lvlm_setting | lvlm_main_entities | lvlm_actions | lvlm_emotion_words | lvlm_interaction_level |expected_interation_level| LVLM matches scene? | notes |
|---------|-----------------|------------------|--------------|--------------------|--------------|--------------------|------------------------|------------------------|---------------------|------|
| ACCEDE09230 | Calm | 0.5232 | Living room | Man in police uniform; Woman in pink top | Reading a magazine; Reading a letter | Focused; Smiling; Calm | 0 | 0 | Yes | LVLM correctly identifies a quiet domestic scene. It supports the algorithmic Calm phase. GCP interaction density is moderate, but LVLM semantic interaction level is 0 because the people are not directly interacting. |
| ACCEDE09231 | Calm | 0.3679 | A residential neighborhood, viewed from inside a car | A woman driving; A man in the passenger seat | Driving (holding the steering wheel); Sitting in the passenger seat | Focused; Neutral; Calm | 0 | 1 | Yes | LVLM correctly identifies the car interior/residential driving scene. It supports the algorithmic Calm phase: low shot frequency, low interaction density, and no direct interaction between the two people. |
| ACCEDE09232 | Calm | 0.4226 | Urban park or campus pathway | Man walking; Sidewalk; Trees; Building; Fire hydrant; Bench; Car | Walking; Waving | Calm; Neutral | 0 | 0 | Yes | LVLM correctly identifies an outdoor walking scene with no direct interaction. Algorithm predicts Calm because shot frequency and interaction density are low, but object entropy is high enough to prevent Static classification. |
| ACCEDE09233 | Calm | 0.4130 | Suburban residential neighborhood, viewed from inside a car | Man; Woman | Driving; Smiling; Speaking | Content; Amused; Focused | 0 | 2 | Yes | LVLM identifies a suburban driving scene with relaxed emotional tone. Algorithm predicts Calm because shot frequency and interaction density are low, while object entropy is high enough to avoid Static. |
---

## Field Meanings

### `narrative_phase`
The phase predicted by the rule-based algorithm from Week 3:
Calm ,Dynamic, Dense, Static

This value is taken from:

"narrative_phase"
inside VideoInterpretation.json.

### `complexity_score`
The numeric SceneComplexityScore computed by the algorithm.

This value is taken from:

"scene_complexity_score"

inside VideoInterpretation.json.

### `lvlm_setting`
The scene setting detected by the LVLM.

This value is taken from:

"lvlm_structured.setting"

### `lvlm_main_entities
The main visible people, objects, or entities detected by the LVLM.

This value is taken from:

"lvlm_structured.main_entities"

### `lvlm_actions`
The main actions described by the LVLM.

This value is taken from:

"lvlm_structured.actions"


### `lvlm_emotion_words`
Emotion or tone words returned by the LVLM.

This value is taken from:

"lvlm_structured.emotion_words"

### `lvlm_interaction_level`
A semantic interaction level returned by the LVLM.

Expected scale:

* 0 = no direct interaction
* 1 = weak / indirect interaction
* 2 = clear interaction
* 3 = strong / intense interaction

This value is taken from:

"lvlm_structured.interaction_level"

### `LVLM matches scene?`
Manual evaluation of whether the LVLM semantic output matches the actual video content.

Use:

* Yes
* Partial
* No


## Notes
* This evaluation is qualitative and meant as a Week 4 sanity check.

* The LVLM layer is used as a semantic interpretation layer, not as the main numeric classifier.

* The algorithmic phase is still computed from numeric metadata: shot frequency, object entropy, interaction density, and human presence ratio.

* The LVLM output helps explain the scene in human-readable terms and can sanity-check the GCP metadata.

* interaction_density from GCP and lvlm_interaction_level do not measure the exact same thing:
GCP interaction_density is based on tracked object density per second.

* LVLM interaction_level is a semantic estimate of how much the visible entities interact.

- For example,

 in ACCEDE09230, GCP interaction density is moderate because many objects are tracked, but LVLM interaction level is 0 because the man and woman are doing separate activities without direct interaction.

* a note about video id: ACCEDE09231: it says in the summary: LVLM correctly identifies the car interior/residential driving scene. It supports the algorithmic Calm phase: low shot frequency, low interaction density, and no direct interaction between the two people.
- but, it has a communication between the 2 visible characters, although one sided, interaction is still there being the woman talks, and the man glances at her momentarily. (might need to further tune this) - which means my expectation was interaction score - weak, indirect interaction
- after further testing, it is crystal clear that there is mutual interaction between our characters in video id 09233, yet the LVLM says the interaction level is 0... need further tuning.
