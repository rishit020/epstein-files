# Attentia Drive v2 — Project

## Vision
Driver monitoring system that detects distraction in real-time using a camera pipeline and audio alerts.

## Stack
- Python 3.10+
- ONNX runtime (Mac dev), RKNN (Phase 8 hardware)
- OpenCV for webcam/video capture
- Typed dataclasses for all inter-layer communication
- pytest for testing

## Principles
- All values from config.py — no magic numbers
- PRD_v2.md is the golden source of truth
- Hardware (RKNN, V4L2, IMX219) is Phase 8 only
- Mac dev: use ONNX, OpenCV webcam, afplay audio

## Current Milestone
Mac development build — full pipeline running on Mac webcam before hardware handoff.
