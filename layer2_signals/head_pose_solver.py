# layer2_signals/head_pose_solver.py — Head Pose via PnP Solve
# PRD §5.1 — Head Pose via PnP Solve
#
# Recovers head pose from 2D landmarks + 3D mean face model using cv2.solvePnP.
# Returns raw Euler angles (yaw, pitch, roll) and reprojection error.
# Kalman filtering and neutral-pose correction are applied by signal_processor.py.

from math import atan2, pi, sqrt

import cv2
import numpy as np

import config

# ── 3D Mean Face Model (6 reference points, millimetres) ──────────────────────
# Points: nose tip, chin, left eye outer corner, right eye outer corner,
#         left mouth corner, right mouth corner.
# Coordinate system: OpenCV camera convention — X right, Y down, Z into scene.
# Converted from the widely-cited Y-up/Z-out reference by negating Y and Z,
# so PnP returns a near-identity rotation for a neutral face (no 180° roll artefact).
_FACE_MODEL_3D = np.array([
    [  0.0,    0.0,    0.0],   # Nose tip
    [  0.0,   63.6,   12.5],   # Chin
    [-43.3,  -32.7,   26.0],   # Left eye outer corner  (subject's left)
    [ 43.3,  -32.7,   26.0],   # Right eye outer corner (subject's right)
    [-28.9,   28.9,   24.1],   # Left mouth corner
    [ 28.9,   28.9,   24.1],   # Right mouth corner
], dtype=np.float64)

# ── iBUG 68-point landmark indices for the 6 PnP keypoints ───────────────────
# Index mapping (0-based) follows the iBUG/300W annotation convention:
#   Face contour: 0-16, Eyebrows: 17-26, Nose: 27-35,
#   Left eye: 36-41, Right eye: 42-47, Mouth: 48-67
#
# Nose tip:             30  (tip of nose)
# Chin:                  8  (bottom-centre of face contour)
# Left eye outer:       36  (lateral canthus, subject's left)
# Right eye outer:      45  (lateral canthus, subject's right)
# Left mouth corner:    48  (left commissure)
# Right mouth corner:   54  (right commissure)
_PNP_LANDMARK_INDICES = [30, 8, 36, 45, 48, 54]


class HeadPoseSolver:
    """Recovers raw head pose (yaw, pitch, roll) via PnP from PFLD landmarks.

    PRD §5.1
    """

    def solve(
        self,
        landmarks_norm: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float, float, float, bool]:
        """Compute raw Euler angles from 68-point normalized landmarks.

        Args:
            landmarks_norm: Shape (68, 2), coordinates normalized to [0, 1].
            frame_width:    Frame width in pixels (used for camera matrix).
            frame_height:   Frame height in pixels.

        Returns:
            (raw_yaw_deg, raw_pitch_deg, raw_roll_deg, reprojection_error_px, valid)
            valid is False if reprojection_error >= PNP_REPROJECTION_ERR_MAX (8.0 px).

        Raises:
            Nothing — all exceptions caught internally; returns (0, 0, 0, 999.0, False).
        """
        try:
            # Convert normalized landmarks → pixel coordinates
            lm_px = landmarks_norm.copy().astype(np.float64)
            lm_px[:, 0] *= frame_width
            lm_px[:, 1] *= frame_height

            # Extract the 6 PnP keypoints
            image_points = lm_px[_PNP_LANDMARK_INDICES]   # shape (6, 2)

            # Camera intrinsic matrix (pinhole approximation, PRD §5.1)
            focal_len = float(frame_width)
            cx = frame_width / 2.0
            cy = frame_height / 2.0
            camera_matrix = np.array([
                [focal_len, 0.0, cx],
                [0.0, focal_len, cy],
                [0.0, 0.0,  1.0],
            ], dtype=np.float64)
            dist_coeffs = np.zeros(4, dtype=np.float64)

            # PnP solve — SOLVEPNP_SQPNP (more robust than ITERATIVE for
            # quasi-coplanar face landmarks; avoids flipped-solution ambiguity).
            success, rvec, tvec = cv2.solvePnP(
                _FACE_MODEL_3D,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP,
            )
            if not success or tvec[2, 0] < 0:
                # Face must be in front of camera (positive Z in camera coords)
                return 0.0, 0.0, 0.0, 999.0, False

            # Rotation vector → rotation matrix
            R_mat, _ = cv2.Rodrigues(rvec)

            # YXZ Euler decomposition: R = Ry(yaw) * Rx(pitch) * Rz(roll)
            # Gives head-pose-meaningful angles directly with the OpenCV-convention
            # 3D model (Y-down, Z-into-scene).
            raw_yaw_deg   = atan2(R_mat[0, 2], R_mat[2, 2]) * (180.0 / pi)
            raw_pitch_deg = atan2(-R_mat[1, 2], sqrt(R_mat[0, 2] ** 2 + R_mat[2, 2] ** 2)) * (180.0 / pi)
            raw_roll_deg  = atan2(R_mat[1, 0], R_mat[1, 1]) * (180.0 / pi)

            # Reprojection error (mean pixel error over 6 keypoints)
            reprojection_error = self._reprojection_error(
                _FACE_MODEL_3D, image_points, rvec, tvec, camera_matrix, dist_coeffs
            )
            valid = reprojection_error < config.PNP_REPROJECTION_ERR_MAX

            return raw_yaw_deg, raw_pitch_deg, raw_roll_deg, reprojection_error, valid

        except Exception:
            return 0.0, 0.0, 0.0, 999.0, False

    # ── Private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _reprojection_error(
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> float:
        """Mean reprojection error in pixels over all keypoints."""
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(image_points - projected, axis=1)
        return float(np.mean(errors))
