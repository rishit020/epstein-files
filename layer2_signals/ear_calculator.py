# layer2_signals/ear_calculator.py — Eye Aspect Ratio + Baseline Calibration
# PRD §5.2 — Eye Aspect Ratio (EAR)
#
# EAR formula: Soukupová & Čech (2016)
# EAR = (||p3-p6|| + ||p4-p5||) / (2 * ||p1-p2||)
# where p1,p2 = horizontal endpoints; p3,p4 = upper lid; p5,p6 = lower lid
#
# Landmark index convention (iBUG 68-point, 0-based):
#   Left eye  (36-41): outer-corner, upper-outer, upper-inner,
#                      inner-corner, lower-inner, lower-outer
#   Right eye (42-47): inner-corner, upper-inner, upper-outer,
#                      outer-corner, lower-outer, lower-inner
#
# Per-session EAR baseline calibration is managed here and exposed to
# signal_processor.py via set_baseline / load_baseline.

import numpy as np

import config

# ── iBUG 68-pt eye indices ─────────────────────────────────────────────────────
# Each tuple: (p1, p2, p3, p4, p5, p6) for the EAR formula.
# p1 = outer corner, p2 = inner corner (horizontal axis)
# p3 = upper-outer lid, p4 = upper-inner lid  (pair → row 1 of vertical distance)
# p5 = lower-inner lid, p6 = lower-outer lid  (p3↔p6 and p4↔p5 vertical pairs)
# iBUG 68-point convention (0-based): left eye = 36-41, right eye = 42-47
_LEFT_EYE_IDX  = (36, 39, 37, 38, 40, 41)   # outer, inner, up-outer, up-inner, lo-inner, lo-outer
_RIGHT_EYE_IDX = (45, 42, 44, 43, 47, 46)   # outer, inner, up-outer, up-inner, lo-inner, lo-outer


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _ear_six_points(lm: np.ndarray, idx: tuple) -> float:
    """Compute EAR for one eye given 6 landmark indices (p1..p6).

    Returns EAR value in [0.0, ~0.4]; returns 0.0 if denominator is zero.
    """
    p1, p2, p3, p4, p5, p6 = (lm[i] for i in idx)
    denom = 2.0 * _euclidean(p1, p2)
    if denom < 1e-6:
        return 0.0
    return (_euclidean(p3, p6) + _euclidean(p4, p5)) / denom


class EARCalculator:
    """Computes EAR values and maintains per-session baseline calibration.

    PRD §5.2
    """

    def __init__(self) -> None:
        self._calibration_samples: list[float] = []
        self._calibration_complete: bool = False
        # Before calibration completes use population defaults (PRD §5.2)
        self._baseline_EAR: float = config.EAR_DEFAULT_CLOSE_THRESHOLD / config.EAR_CALIBRATION_MULTIPLIER
        self._close_threshold: float = config.EAR_DEFAULT_CLOSE_THRESHOLD

    # ── Public API ───────────────────────────────────────────────────────────

    def compute(self, landmarks_norm: np.ndarray) -> tuple[float, float, float]:
        """Compute left EAR, right EAR, mean EAR from normalised landmarks.

        Args:
            landmarks_norm: Shape (68, 2), coordinates normalised [0, 1].

        Returns:
            (left_EAR, right_EAR, mean_EAR)
        """
        left_ear  = _ear_six_points(landmarks_norm, _LEFT_EYE_IDX)
        right_ear = _ear_six_points(landmarks_norm, _RIGHT_EYE_IDX)
        mean_ear  = (left_ear + right_ear) / 2.0
        return left_ear, right_ear, mean_ear

    def update_calibration(self, mean_ear: float, is_driving: bool) -> None:
        """Feed one EAR sample into the calibration window.

        Calibration accumulates EAR samples while driving until
        EAR_CALIBRATION_DURATION_S seconds of data are collected.
        Once complete, baseline_EAR and close_threshold are updated.

        Args:
            mean_ear:   Mean EAR for this frame.
            is_driving: True if vehicle speed > V_MIN (calibration only runs
                        during driving to avoid closed-eye resting samples).
        """
        if self._calibration_complete:
            return
        if not is_driving:
            return

        self._calibration_samples.append(mean_ear)

        required_samples = int(config.EAR_CALIBRATION_DURATION_S * config.CAPTURE_FPS)
        if len(self._calibration_samples) >= required_samples:
            self._baseline_EAR = float(np.mean(self._calibration_samples))
            self._close_threshold = self._baseline_EAR * config.EAR_CALIBRATION_MULTIPLIER
            self._calibration_complete = True

    def load_baseline(self, baseline_ear: float, close_threshold: float) -> None:
        """Load persisted calibration from session_state.json (PRD §24.2).

        Marks calibration as complete immediately — skips the 30s warm-up.
        """
        self._baseline_EAR = baseline_ear
        self._close_threshold = close_threshold
        self._calibration_complete = True

    def reset_calibration(self) -> None:
        """Clear calibration state (e.g., on VIN change or forced re-cal)."""
        self._calibration_samples = []
        self._calibration_complete = False
        self._baseline_EAR = config.EAR_DEFAULT_CLOSE_THRESHOLD / config.EAR_CALIBRATION_MULTIPLIER
        self._close_threshold = config.EAR_DEFAULT_CLOSE_THRESHOLD

    @property
    def baseline_EAR(self) -> float:
        return self._baseline_EAR

    @property
    def close_threshold(self) -> float:
        return self._close_threshold

    @property
    def calibration_complete(self) -> bool:
        return self._calibration_complete

    @property
    def calibration_samples_count(self) -> int:
        return len(self._calibration_samples)
