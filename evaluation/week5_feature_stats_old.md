# Week 5 Feature Statistics

## Goal
Collect raw feature distributions from analyzed videos and use them to evaluate whether the current normalization ranges in `config.py` are realistic.

## Dataset
Number of `VideoInterpretation.json` files analyzed: 21

## Feature Statistics

| feature | count | min | median | mean | max | current_range | notes |
|---------|-------|-----|--------|------|-----|---------------|-------|
| shot_frequency | 21 | 0.0964 | 0.1920 | 0.2073 | 0.4660 | 0.0–2.0 | Observed shot frequency is low; current max may be too wide |
| object_entropy | 21 | 1.5219 | 2.6981 | 2.5640 | 3.4601 | 0.0–4.0 | Current range may be acceptable |
| interaction_density | 21 | 0.2915 | 1.6302 | 1.8693 | 4.8857 | 0.0–10.0 | Observed values are much lower than current max; consider lowering max range |
| human_presence_ratio | 21 | 0.4096 | 0.9742 | 0.9185 | 0.9960 | 0.0–1.0 | Already bounded in [0, 1]; usually keep range unchanged |

## Observations
- TODO: Describe which features appear compressed or too widely normalized.
- TODO: Check whether `interaction_density` rarely approaches the current max.
- TODO: Check whether `object_entropy` is frequently high enough to prevent Static classification.

## Suggested Range Updates

| feature | old_range | suggested_range | reason |
|---------|-----------|-----------------|--------|
| shot_frequency | 0.0–2.0 | 0.0–1.0 | Observed max is 0.4660, so the current max compresses all values too strongly. A max of 1.0 is less compressed while still leaving room for faster clips. |
| object_entropy | 0.0–4.0 | 0.0–4.0 | Observed max is 3.4601, so the current range is still reasonable. Static threshold may need later tuning instead. |
| interaction_density | 0.0–10.0 | 0.0–5.0 | Observed max is 4.8857, so a max of 5.0 better matches the current dataset and reduces compression. |
| human_presence_ratio | 0.0–1.0 | 0.0–1.0 | This feature is already naturally bounded. |

## Phase Distribution

| phase | count |
|------|-------|
| Calm | 21 |
| Dynamic | 0 |
| Dense | 0 |
| Static | 0 |

## Notes
- This report should be generated before changing `config.py`.
- After updating ranges, rerun selected videos and compare phase/score changes.
- LVLM semantic interaction tuning is handled separately from numeric feature range tuning.
- Number of categorized videos by phases: Dense - 0, Dynamic - 0, Static - 0, Calm - 21.