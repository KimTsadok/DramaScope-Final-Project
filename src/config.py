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
    shot_frequency_max: float = 2.0

    object_entropy_min: float = 0.0
    object_entropy_max: float = 4.0

    interaction_density_min: float = 0.0
    interaction_density_max: float = 10.0

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
    dense_entropy_min: float = 0.65
    dense_density_min: float = 0.65

    dynamic_shot_frequency_min: float = 0.65
    dynamic_density_max: float = 0.65

    static_shot_frequency_max: float = 0.35
    static_density_max: float = 0.35
    static_entropy_max: float = 0.35


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