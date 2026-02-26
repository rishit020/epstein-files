# layer3_temporal/thermal_monitor.py — Thermal monitor
# PRD §FR-3.7
#
# On RK3568: polls /sys/class/thermal/thermal_zone0/temp every
# THERMAL_CHECK_INTERVAL_S and signals throttle / DEGRADED states.
#
# On Mac (development): thermal path does not exist → stub only.
# All properties return nominal values (throttle_active=False, temperature_c=0.0).

import logging
import os
import threading
from typing import Optional

import config

logger = logging.getLogger(__name__)


class ThermalMonitor:
    """Monitors device temperature and signals thermal throttle states.

    PRD §FR-3.7
    Mac stub: path absent → always nominal.
    """

    def __init__(self) -> None:
        self._throttle_active: bool = False
        self._temperature_c: float = 0.0
        self._on_hardware: bool = os.path.exists(config.THERMAL_MONITOR_PATH)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the background polling thread (no-op on Mac)."""
        if not self._on_hardware:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name='ThermalMonitorThread',
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=config.THERMAL_CHECK_INTERVAL_S + 1.0)
        self._thread = None

    @property
    def throttle_active(self) -> bool:
        """True if thermal conditions require reduced processing load."""
        return self._throttle_active

    @property
    def temperature_c(self) -> float:
        """Most recent temperature reading in °C (0.0 on Mac)."""
        return self._temperature_c

    # ── Private ───────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._read_temperature()
            self._stop_event.wait(timeout=config.THERMAL_CHECK_INTERVAL_S)

    def _read_temperature(self) -> None:
        try:
            with open(config.THERMAL_MONITOR_PATH) as fh:
                raw = fh.read().strip()
            temp_c = int(raw) / 1000.0
            self._temperature_c = temp_c

            if temp_c >= config.THERMAL_CRITICAL_TEMP_C:
                if not self._throttle_active:
                    logger.critical(
                        'THERMAL_DEGRADED: temp=%.1f°C >= critical threshold %d°C',
                        temp_c,
                        config.THERMAL_CRITICAL_TEMP_C,
                    )
                self._throttle_active = True
            elif temp_c >= config.THERMAL_WARN_TEMP_C:
                if not self._throttle_active:
                    logger.warning(
                        'THERMAL_WARNING: temp=%.1f°C >= warn threshold %d°C',
                        temp_c,
                        config.THERMAL_WARN_TEMP_C,
                    )
                self._throttle_active = True
            else:
                if self._throttle_active:
                    logger.info(
                        'Thermal recovery: temp=%.1f°C below warn threshold', temp_c
                    )
                self._throttle_active = False

        except OSError as exc:
            logger.warning('ThermalMonitor: could not read temperature: %s', exc)
        except (ValueError, TypeError) as exc:
            logger.warning('ThermalMonitor: unexpected temperature format: %s', exc)
