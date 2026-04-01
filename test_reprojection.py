#!/usr/bin/env python3
"""Debug reprojection error — check if PnP camera matrix is accurate."""

import sys
import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

import cv2
import numpy as np
import config
from layer0_video.webcam_source import WebcamSource
from layer1_perception.face_detector import FaceDetector
from layer1_perception.landmark_model import LandmarkModel
from layer1_perception.gaze_model import GazeModel
from layer1_perception.perception_stack import PerceptionStack
from layer2_signals.head_pose_solver import HeadPoseSolver

class _Stub:
    def infer(self, frame):
        from layer1_perception.messages import PhoneDetectionOutput
        return PhoneDetectionOutput(detected=False, max_confidence=0.0, bbox_norm=None)

source = WebcamSource(device_index=0)
stack = PerceptionStack(FaceDetector(), LandmarkModel(), GazeModel(), _Stub())
head_solver = HeadPoseSolver()

print("=== REPROJECTION ERROR DIAGNOSTIC ===\n")
print(f"Frame size: {config.CAPTURE_WIDTH}x{config.CAPTURE_HEIGHT}")
print(f"PnP threshold: {config.PNP_REPROJECTION_ERR_MAX} pixels\n")

reprojection_errors = []

for i in range(20):
    raw = source.read()
    if raw is None:
        continue

    # Perception
    bundle = stack.infer(raw.data, raw.frame_id)

    if not bundle.face.present:
        print(f"Frame {i}: No face detected")
        continue

    # Get landmarks
    if not bundle.landmarks or not bundle.landmarks.pose_valid:
        print(f"Frame {i}: Landmarks not valid")
        continue

    # Solve PnP
    yaw, pitch, roll, reproj_err, valid = head_solver.solve(
        bundle.landmarks.landmarks,
        config.CAPTURE_WIDTH,
        config.CAPTURE_HEIGHT,
    )

    reprojection_errors.append(reproj_err)
    status = "✓ PASS" if valid else "✗ FAIL"
    print(f"Frame {i}: reproj_err={reproj_err:.2f}px {status} | yaw={yaw:+.1f}° pitch={pitch:+.1f}° roll={roll:+.1f}°")

source.release()

if reprojection_errors:
    avg_err = np.mean(reprojection_errors)
    min_err = np.min(reprojection_errors)
    max_err = np.max(reprojection_errors)

    print(f"\n=== SUMMARY ===")
    print(f"Average reprojection error: {avg_err:.2f}px")
    print(f"Min: {min_err:.2f}px, Max: {max_err:.2f}px")
    print(f"Threshold: {config.PNP_REPROJECTION_ERR_MAX}px")
    print(f"\nIf avg error > threshold:")
    print(f"  Problem: Camera intrinsic matrix is inaccurate for your webcam")
    print(f"  Solution: Need to calibrate camera (compute actual focal length, principal point)")
    print(f"  Temporary fix: Relax PNP_REPROJECTION_ERR_MAX in config.py (e.g., 15.0 or 20.0)")
    print(f"                 ⚠️  This is a workaround, not a proper fix!")
