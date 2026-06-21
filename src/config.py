# src/config.py
"""
One place for:
* ranges
* thresholds
* weights
* model name
* frame settings
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class NormalizationRanges: # Used later in normalize.py
# These are your v1 expected ranges:
    shot_frequency_min: float = 0.0
    shot_frequency_max: float = 1.0 # was 2.0

    object_entropy_min: float = 0.0
    object_entropy_max: float = 4.0

    interaction_density_min: float = 0.0
    interaction_density_max: float = 5.0 # was 10.0 

    human_presence_ratio_min: float = 0.0
    human_presence_ratio_max: float = 1.0


@dataclass(frozen=True)
class ComplexityWeights: # Used later in complexity.py (sums up to 1.0)
# This is your scoring formula:
    shot_frequency: float = 0.35
    object_entropy: float = 0.35
    interaction_density: float = 0.20
    human_presence_ratio: float = 0.10


@dataclass(frozen=True)
class PhaseThresholds: # Used later in phase.py 
# (represent the ruling bounds for Dense, Dynamic, Static, else Calm)

    # Dense - visually busy scene: high object variety + high tracked-object density
    dense_entropy_min: float = 0.70 # old 0.65 (v2), old 0.70 (v3) - was 0.72
    dense_density_min: float = 0.65 # old 0.65 (v3) - was 0.70

    # Dynamic - higher pace/activity relative to this dataset
    dynamic_shot_frequency_min: float = 0.35 # old 0.65 (v2), old 0.35 (v3) - kept
    dynamic_density_max: float = 0.75 # old 0.65 (v2), old 0.75 (v3)


    # Static - very slow pace + low tracked-object density 
    # entropy is allowed to be moderate because GCP detects many objects even in quiet scenes
    static_shot_frequency_max: float = 0.20 # old 0.35 (v2), old 0.20 (v3) - was 0.22
    static_density_max: float = 0.30 # old 0.35 (v2), old 0.30 (v3) 
    static_entropy_max: float = 0.50 # old 0.35 (v2), old 0.50 (v3) - was 0.60


@dataclass(frozen=True) 
# it belongs in central config
# later both video extraction and LVLM client can use it
class FrameSettings:
    frame_rate: int = 1
    max_frames: int = 10


@dataclass(frozen=True)
class ModelSettings:
    lvlm_model_name: str = "glm-4.6v-flash"
    lvlm_prompt_version: str = "v1"


@dataclass(frozen=True)
class OutputFiles:
    raw_features_filename: str = "VideoFeatures.json"
    interpretation_filename: str = "VideoInterpretation.json"


NORMALIZATION_RANGES = NormalizationRanges()
COMPLEXITY_WEIGHTS = ComplexityWeights()
PHASE_THRESHOLDS = PhaseThresholds()
FRAME_SETTINGS = FrameSettings()
MODEL_SETTINGS = ModelSettings()
OUTPUT_FILES = OutputFiles()