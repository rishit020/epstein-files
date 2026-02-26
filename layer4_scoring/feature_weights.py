# layer4_scoring/feature_weights.py — Feature weight definitions
# PRD §6.1
#
# Weights must sum to 1.0 ± 1e-6.
# All values sourced from config.py — no magic numbers here.

from dataclasses import dataclass

import config


@dataclass(frozen=True)
class FeatureWeights:
    """Immutable feature weight set for the composite distraction scorer.

    PRD §6.1 — W1 through W4 must sum to 1.0.
    Based on NHTSA 100-Car Study (Klauer et al. 2006) and
    PERCLOS validation studies (Wierwille 1994).
    """
    gaze: float     # W1 — gaze off-road fraction
    head: float     # W2 — head deviation
    perclos: float  # W3 — PERCLOS drowsiness
    blink: float    # W4 — blink rate anomaly

    def __post_init__(self) -> None:
        total = self.gaze + self.head + self.perclos + self.blink
        if abs(total - 1.0) >= 1e-6:
            raise ValueError(
                f"Feature weights must sum to 1.0 ± 1e-6, got {total:.10f}"
            )


DEFAULT_WEIGHTS = FeatureWeights(
    gaze=config.WEIGHT_GAZE,       # 0.45
    head=config.WEIGHT_HEAD,       # 0.30
    perclos=config.WEIGHT_PERCLOS, # 0.20
    blink=config.WEIGHT_BLINK,     # 0.05
)
