#!/usr/bin/env python3
"""Debug why signals are invalid — check each step of the pipeline."""

import sys
import logging

logging.basicConfig(level=logging.DEBUG, format='%(name)-20s %(levelname)-8s %(message)s')

import cv2
import numpy as np
import config
from layer0_video.webcam_source import WebcamSource
from layer1_perception.face_detector import FaceDetector
from layer1_perception.landmark_model import LandmarkModel
from layer1_perception.gaze_model import GazeModel
from layer1_perception.perception_stack import PerceptionStack
from layer1_perception.phone_detector import PhoneDetector
from layer2_signals.signal_processor import SignalProcessor

_log = logging.getLogger(__name__)

# Stub phone detector
class _Stub:
    def infer(self, frame):
        from layer1_perception.messages import PhoneDetectionOutput
        return PhoneDetectionOutput(detected=False, max_confidence=0.0, bbox_norm=None)

source = WebcamSource(device_index=0)
stack = PerceptionStack(FaceDetector(), LandmarkModel(), GazeModel(), _Stub())
processor = SignalProcessor()

_log.info("=== SIGNAL DEBUGGING ===\n")

for frame_num in range(15):
    raw = source.read()
    if raw is None:
        continue

    # Step 1: Perception
    bundle = stack.infer(raw.data, raw.frame_id)

    print(f"\n[Frame {frame_num}]")
    print(f"  Face present: {bundle.face.present}")
    print(f"  Face conf: {bundle.face.confidence:.3f}")

    if bundle.landmarks:
        print(f"  Landmarks conf: {bundle.landmarks.confidence:.3f}")
        print(f"  Pose valid: {bundle.landmarks.pose_valid}")
    else:
        print(f"  Landmarks: NONE")

    if bundle.gaze:
        print(f"  Gaze valid: {bundle.gaze.valid}")
        print(f"  Gaze yaw/pitch: {bundle.gaze.combined_yaw:.1f}° / {bundle.gaze.combined_pitch:.1f}°")
    else:
        print(f"  Gaze: NONE")

    # Step 2: Signal processing
    signal_frame = processor.process(bundle, speed_mps=5.0, speed_stale=False)

    print(f"  ▶︎ Signal frame computed:")
    print(f"    - signals_valid: {signal_frame.signals_valid}")
    print(f"    - head_pose: {signal_frame.head_pose}")
    if signal_frame.head_pose:
        print(f"      yaw_deg={signal_frame.head_pose.yaw_deg:.1f}°, valid={signal_frame.head_pose.valid}")
    print(f"    - gaze_world: {signal_frame.gaze_world}")
    if signal_frame.gaze_world:
        print(f"      yaw_deg={signal_frame.gaze_world.yaw_deg:.1f}°, valid={signal_frame.gaze_world.valid}")

source.release()
print(f"\n=== SUMMARY ===")
print(f"If all frames show signals_valid=False, head_pose.valid=False, and gaze_world.valid=False,")
print(f"then BOTH head pose solving AND gaze estimation are failing.")
print(f"Check: PnP reprojection error, gaze model output validity, landmarks quality.")
