---
status: partial
phase: 06-video-output-wiring
source: [06-VERIFICATION.md]
started: 2026-03-31
updated: 2026-03-31
---

## Current Test

[awaiting human testing]

## Tests

### 1. Live webcam run
expected: `python3 main.py --device 0` starts, opens webcam, logs frame processing, responds to Ctrl-C with clean shutdown

### 2. Audio playback confirmation
expected: Alert fires when distraction threshold exceeded → afplay plays Ping.aiff (HIGH) or Sosumi.aiff (URGENT), non-blocking

### 3. JSONL log rotation under load
expected: attentia_events.jsonl rotates at LOG_MAX_BYTES, retains LOG_BACKUP_COUNT backups; each line valid JSON

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
