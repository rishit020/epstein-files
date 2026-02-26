# layer2_signals/phone_signal.py — Phone Signal Extractor
# PRD §4.3 — PhoneSignal (Layer 2 output)
#
# Converts PhoneDetectionOutput (Layer 1) into a PhoneSignal (Layer 2 output).
# Applies the PHONE_CONFIDENCE_THRESHOLD gate from config.
# Preserves the staleness flag from the T-2 phone detection thread.

import config
from layer1_perception.messages import PhoneDetectionOutput
from layer2_signals.messages import PhoneSignal


def extract_phone_signal(phone: PhoneDetectionOutput, result_stale: bool) -> PhoneSignal:
    """Convert raw phone detection output to a gated PhoneSignal.

    A phone is considered 'detected' only if:
      - phone.detected is True, AND
      - phone.max_confidence >= PHONE_CONFIDENCE_THRESHOLD (0.70)

    The stale flag is passed through unchanged from the T-2 sync result.

    Args:
        phone:        PhoneDetectionOutput from Layer 1.
        result_stale: True if T-2 timed out and last valid result was used.

    Returns:
        PhoneSignal for Layer 3 consumption.
    """
    detected = phone.detected and phone.max_confidence >= config.PHONE_CONFIDENCE_THRESHOLD
    return PhoneSignal(
        detected=detected,
        confidence=phone.max_confidence,
        stale=result_stale,
    )
