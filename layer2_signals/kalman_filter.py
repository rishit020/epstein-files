# layer2_signals/kalman_filter.py — 1D Constant-Velocity Kalman Filter
# PRD §5.6 — Kalman Filter for Pose and Gaze Smoothing
#
# Applied independently to: head_yaw, head_pitch, head_roll, gaze_world_yaw, gaze_world_pitch
# State vector: [angle, angular_velocity] (2D constant-velocity model)
# Observation: angle only (H = [1, 0])

import numpy as np

import config


class KalmanFilter1D:
    """1D constant-velocity Kalman filter for angle smoothing.

    PRD §5.6
    State: [angle, angular_velocity]
    Measurement: angle only
    """

    def __init__(self) -> None:
        # State: [angle, angular_velocity]
        self._x = np.zeros(2, dtype=np.float64)
        # State covariance
        self._P = np.eye(2, dtype=np.float64) * config.KALMAN_INITIAL_COVARIANCE
        # Measurement noise covariance (scalar, ±2° PnP variance)
        self._R = np.array([[config.KALMAN_MEASUREMENT_NOISE_R]], dtype=np.float64)
        # Process noise covariance (same Q on both state dimensions)
        self._Q = np.array(
            [[config.KALMAN_PROCESS_NOISE_Q, 0.0],
             [0.0, config.KALMAN_PROCESS_NOISE_Q]],
            dtype=np.float64,
        )
        # Measurement matrix H = [1, 0] — observe angle only
        self._H = np.array([[1.0, 0.0]], dtype=np.float64)
        self._initialized = False

    # ── Public API ──────────────────────────────────────────────────────────

    def update(self, measurement: float, dt: float = 1.0 / config.CAPTURE_FPS) -> float:
        """Predict + update step.  Returns filtered angle estimate.

        Args:
            measurement: Raw observed angle (degrees).
            dt: Time since last frame (seconds).  Defaults to 1/FPS.

        Returns:
            Filtered angle estimate (degrees).
        """
        if not self._initialized:
            self._x[0] = measurement
            self._x[1] = 0.0
            self._initialized = True
            return float(self._x[0])

        # ── Predict step ────────────────────────────────────────────────────
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=np.float64)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._Q

        # ── Update step ─────────────────────────────────────────────────────
        z = np.array([measurement], dtype=np.float64)
        y = z - self._H @ self._x                         # Innovation
        S = self._H @ self._P @ self._H.T + self._R       # Innovation covariance
        K = self._P @ self._H.T @ np.linalg.inv(S)        # Kalman gain
        self._x = self._x + K @ y
        self._P = (np.eye(2, dtype=np.float64) - K @ self._H) @ self._P

        return float(self._x[0])

    def reset(self) -> None:
        """Reset filter state.

        Called when: head_pose.valid → False, face absent > 10 frames,
        or source restart (PRD §5.6).
        """
        self._x[:] = 0.0
        self._P = np.eye(2, dtype=np.float64) * config.KALMAN_INITIAL_COVARIANCE
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized
