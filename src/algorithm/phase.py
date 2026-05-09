# src/algorithm/phase.py
"""
Apply your rule-based classifier.
* read normalized features
* apply the rules in order

return:
phase string (Dynamic, Dense, Static or Calm)
reasons list
"""

from typing import Dict, List, Tuple
from src.config import PHASE_THRESHOLDS


def format_bool(condition: bool) -> str:
    return "True" if condition else "False"


def classify_narrative_phase(norm_features: Dict[str, float]) -> Tuple[str, List[str]]:
    sf = norm_features["shot_frequency"]
    ent = norm_features["object_entropy"]
    dens = norm_features["interaction_density"]

    reasons: List[str] = [
        f"Normalized values -> shot_frequency={sf:.4f}, object_entropy={ent:.4f}, interaction_density={dens:.4f}"
    ]

    # Dense: ent > 0.65 and dens > 0.65
    dense_ent_ok = ent > PHASE_THRESHOLDS.dense_entropy_min
    dense_dens_ok = dens > PHASE_THRESHOLDS.dense_density_min

    if dense_ent_ok and dense_dens_ok:
        reasons.append(
            "Dense rule satisfied: "
            f"object_entropy > {PHASE_THRESHOLDS.dense_entropy_min} is {format_bool(dense_ent_ok)}, "
            f"interaction_density > {PHASE_THRESHOLDS.dense_density_min} is {format_bool(dense_dens_ok)}"
        )
        return "Dense", reasons

    reasons.append(
        "Dense rule not satisfied: "
        f"object_entropy > {PHASE_THRESHOLDS.dense_entropy_min} is {format_bool(dense_ent_ok)}, "
        f"interaction_density > {PHASE_THRESHOLDS.dense_density_min} is {format_bool(dense_dens_ok)}"
    )

    # Dynamic: sf > 0.65 and dens <= 0.65
    dynamic_sf_ok = sf > PHASE_THRESHOLDS.dynamic_shot_frequency_min
    dynamic_dens_ok = dens <= PHASE_THRESHOLDS.dynamic_density_max

    if dynamic_sf_ok and dynamic_dens_ok:
        reasons.append(
            "Dynamic rule satisfied: "
            f"shot_frequency > {PHASE_THRESHOLDS.dynamic_shot_frequency_min} is {format_bool(dynamic_sf_ok)}, "
            f"interaction_density <= {PHASE_THRESHOLDS.dynamic_density_max} is {format_bool(dynamic_dens_ok)}"
        )
        return "Dynamic", reasons

    reasons.append(
        "Dynamic rule not satisfied: "
        f"shot_frequency > {PHASE_THRESHOLDS.dynamic_shot_frequency_min} is {format_bool(dynamic_sf_ok)}, "
        f"interaction_density <= {PHASE_THRESHOLDS.dynamic_density_max} is {format_bool(dynamic_dens_ok)}"
    )

    # Static: sf < 0.35 and dens < 0.35 and ent < 0.35
    static_sf_ok = sf < PHASE_THRESHOLDS.static_shot_frequency_max
    static_dens_ok = dens < PHASE_THRESHOLDS.static_density_max
    static_ent_ok = ent < PHASE_THRESHOLDS.static_entropy_max

    if static_sf_ok and static_dens_ok and static_ent_ok:
        reasons.append(
            "Static rule satisfied: "
            f"shot_frequency < {PHASE_THRESHOLDS.static_shot_frequency_max} is {format_bool(static_sf_ok)}, "
            f"interaction_density < {PHASE_THRESHOLDS.static_density_max} is {format_bool(static_dens_ok)}, "
            f"object_entropy < {PHASE_THRESHOLDS.static_entropy_max} is {format_bool(static_ent_ok)}"
        )
        return "Static", reasons

    reasons.append(
        "Static rule not satisfied: "
        f"shot_frequency < {PHASE_THRESHOLDS.static_shot_frequency_max} is {format_bool(static_sf_ok)}, "
        f"interaction_density < {PHASE_THRESHOLDS.static_density_max} is {format_bool(static_dens_ok)}, "
        f"object_entropy < {PHASE_THRESHOLDS.static_entropy_max} is {format_bool(static_ent_ok)}"
    )

    reasons.append("Defaulted to Calm")
    return "Calm", reasons