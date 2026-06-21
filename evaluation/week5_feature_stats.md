# Week 5 Feature Statistics

## Goal
Collect raw feature distributions from analyzed videos and use them to evaluate whether the current normalization ranges in `config.py` are realistic.

## Dataset
Number of `VideoInterpretation.json` files analyzed: 21

## Feature Statistics

| feature | count | min | median | mean | max | current_range | notes |
|---------|-------|-----|--------|------|-----|---------------|-------|
| shot_frequency | 21 | 0.0964 | 0.1920 | 0.2073 | 0.4660 | 0.0–1.0 | Observed shot frequency is low; current max may be too wide |
| object_entropy | 21 | 1.5219 | 2.6981 | 2.5640 | 3.4601 | 0.0–4.0 | Current range may be acceptable |
| interaction_density | 21 | 0.2915 | 1.6302 | 1.8693 | 4.8857 | 0.0–5.0 | Current range may be acceptable |
| human_presence_ratio | 21 | 0.4096 | 0.9742 | 0.9185 | 0.9960 | 0.0–1.0 | Already bounded in [0, 1]; usually keep range unchanged |

## Observations
- After v1 normalization range tuning, the phase distribution changed from all Calm to 18 Calm and 3 Dense. This showed that Dense became reachable once `interaction_density_max` was reduced.
- After v2 phase-threshold tuning, all four phase categories became reachable: 15 Calm, 1 Dynamic, 2 Dense, and 3 Static.
- After v3 phase-threshold tuning, the distribution changed to: Calm - 16, Dynamic - 2, Dense - 0, Static - 3.
- After fixing the phase-threshold tuning, we reverted to this final distribution: Calm - 15, Dynamic - 1, Dense - 2, Static - 3.
- Lowering `dynamic_shot_frequency_min` from 0.65 to 0.35 made Dynamic reachable for the highest shot-frequency videos in this dataset.
- Raising `static_entropy_max` from 0.35 to 0.50 made Static reachable, while stricter shot-frequency and density limits kept Static selective.
- Raising `dense_entropy_min` from 0.65 to 0.70 made Dense slightly stricter, reducing Dense predictions from 3 to 2.

## Phase Distribution

| phase | count |
|-------|-------|
| Calm | 15 |
| Dynamic | 1 |
| Dense | 2 |
| Static | 3 |
| Unknown | 0 |

- Multiple phase categories were detected, but distribution should still be reviewed for imbalance.

## Suggested Next Tuning Step (Suggested Range Updates)

| component | current_value | suggested_value | reason |
|----------|---------------|-----------------|--------|
| dense_entropy_min | 0.65 | 0.70 | Object entropy median is already high, so Dense should require stronger entropy. |
| dynamic_shot_frequency_min | 0.65 | 0.35 | Current value is unreachable because max observed normalized shot frequency is 0.4660. |
| dynamic_density_max | 0.65 | 0.75 | Allows moderately busy fast-cut scenes to still be Dynamic. |
| static_shot_frequency_max | 0.35 | 0.20 | Static should require very slow cutting. |
| static_density_max | 0.35 | 0.30 | Static should require low tracked-object density. |
| static_entropy_max | 0.35 | 0.50 | Current value is too strict because even the lowest observed entropy is above 0.35. |

## Notes
- LVLM semantic interaction tuning is handled separately from numeric feature range tuning.
- Manual inspection showed that the classifier still over-predicts `Calm` compared to human labels. 
- Human labels suggest that many clips are better described as `Static` or `Dynamic`, while the algorithm still assigns many borderline cases to `Calm` because `Calm` is the fallback class.
- The v2 threshold update improved the classifier by making all four categories reachable
- The v3 threshold update slightly increased Dynamic predictions, but worsened the overall distribution by increasing Calm to 16 and reducing Dense to 0. Therefore, v3 was not accepted as a final improvement.
- This suggests that the current `Dynamic` rule still relies too heavily on shot frequency, while the `Static` rule is sensitive to object entropy even in visually slow scenes.
- Further improvement likely requires either additional motion/activity features or using LVLM semantic fields to support borderline cases.