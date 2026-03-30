---
phase: 6
slug: video-output-wiring
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-29
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.0.2 |
| **Config file** | none — Wave 0 installs test stubs |
| **Quick run command** | `python3 -m pytest tests/unit/test_webcam_source.py tests/unit/test_audio_handler.py tests/unit/test_event_logger.py -x -q` |
| **Full suite command** | `python3 -m pytest tests/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python3 -m pytest tests/unit/test_webcam_source.py tests/unit/test_audio_handler.py tests/unit/test_event_logger.py -x -q`
- **After every plan wave:** Run `python3 -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 6-W0-01 | W0 | 0 | FR-0.x | unit stub | `pytest tests/unit/test_webcam_source.py -x` | ❌ W0 | ⬜ pending |
| 6-W0-02 | W0 | 0 | FR-6.1 | unit stub | `pytest tests/unit/test_audio_handler.py -x` | ❌ W0 | ⬜ pending |
| 6-W0-03 | W0 | 0 | FR-6.3/§9 | unit stub | `pytest tests/unit/test_event_logger.py -x` | ❌ W0 | ⬜ pending |
| 6-W0-04 | W0 | 0 | §3.3 | integration stub | `pytest tests/integration/test_pipeline_wiring.py -x` | ❌ W0 | ⬜ pending |
| 6-01-01 | 01 | 1 | FR-0.1 | unit | `pytest tests/unit/test_webcam_source.py -x` | ❌ W0 | ⬜ pending |
| 6-01-02 | 01 | 1 | FR-0.2 | unit | `pytest tests/unit/test_webcam_source.py -x` | ❌ W0 | ⬜ pending |
| 6-01-03 | 01 | 1 | FR-0.3 | unit | `pytest tests/unit/test_webcam_source.py -x` | ❌ W0 | ⬜ pending |
| 6-02-01 | 02 | 1 | FR-6.1 | unit | `pytest tests/unit/test_audio_handler.py -x` | ❌ W0 | ⬜ pending |
| 6-02-02 | 02 | 1 | FR-6.1 NFR-P4 | unit | `pytest tests/unit/test_audio_handler.py -x` | ❌ W0 | ⬜ pending |
| 6-03-01 | 03 | 1 | FR-6.3/§9 | unit | `pytest tests/unit/test_event_logger.py -x` | ❌ W0 | ⬜ pending |
| 6-03-02 | 03 | 1 | §9 schemas | unit | `pytest tests/unit/test_event_logger.py -x` | ❌ W0 | ⬜ pending |
| 6-04-01 | 04 | 2 | §3.3 | integration | `pytest tests/integration/test_pipeline_wiring.py -x` | ❌ W0 | ⬜ pending |
| 6-04-02 | 04 | 2 | §3.3 stale | integration | `pytest tests/integration/test_pipeline_wiring.py -x` | ❌ W0 | ⬜ pending |
| 6-04-03 | 04 | 2 | §3.3 shutdown | integration | `pytest tests/integration/test_pipeline_wiring.py -x` | ❌ W0 | ⬜ pending |
| 6-05-01 | 05 | 2 | full pipeline | integration | `pytest tests/integration/test_pipeline_wiring.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_webcam_source.py` — stubs for FR-0.1–FR-0.4 (mock cv2.VideoCapture)
- [ ] `tests/unit/test_audio_handler.py` — stubs for FR-6.1 (mock subprocess.Popen)
- [ ] `tests/unit/test_event_logger.py` — stubs for FR-6.3/§9 (write to tempdir, validate JSONL)
- [ ] `tests/integration/test_pipeline_wiring.py` — threaded pipeline integration stubs (mock models, synthetic frames)
- [ ] `tests/integration/` directory — create if not exists

*Wave 0 creates all test files so later tasks have runnable (initially failing) tests to satisfy.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Live webcam preview renders in window | FR-0.4 / `--display` flag | Requires physical display + webcam | Run `python main.py --display`, confirm cv2.imshow window opens |
| Audio alert plays audibly on alert | FR-6.1 | Requires speakers | Trigger alert, confirm Ping.aiff heard |
| URGENT alert plays different sound | FR-6.1 | Requires speakers | Trigger URGENT, confirm Sosumi.aiff heard |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
