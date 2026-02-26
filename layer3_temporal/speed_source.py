# layer3_temporal/speed_source.py — Speed signal acquisition
# PRD §22
#
# Priority stack: OBD2 → CAN → GPS → NONE
# On Mac (development): all hardware sources are unavailable.
# Returns (None, True) and source_type 'NONE' → triggers FM-05 URBAN fallback.
#
# Hardware source implementations (OBD2/CAN/GPS) are stubs here;
# they will be completed in the hardware phase when real interfaces exist.

import logging

import config

logger = logging.getLogger(__name__)

_HARDWARE_SOURCES = {'OBD2', 'CAN', 'GPS'}


class SpeedSource:
    """Speed signal acquisition with priority-stack source selection.

    PRD §22
    """

    def __init__(self) -> None:
        self._source_type: str = 'NONE'
        self._speed_mps: float | None = None
        self._is_stale: bool = True
        self._discover()

    def _discover(self) -> None:
        """Attempt to connect to each source in priority order.

        On Mac, all hardware sources fail silently — defaults to NONE.
        """
        for source in config.SPEED_SOURCE_PRIORITY:
            if source in _HARDWARE_SOURCES:
                if self._try_connect(source):
                    self._source_type = source
                    logger.info('SpeedSource: using %s', source)
                    return
            else:
                # 'NONE' — emergency fallback
                self._source_type = 'NONE'
                logger.warning(
                    'SpeedSource: no hardware source available, defaulting to URBAN fallback (FM-05)'
                )
                return

        self._source_type = 'NONE'

    def _try_connect(self, source: str) -> bool:
        """Attempt to connect to a hardware speed source.

        Returns True if the source is available, False otherwise.
        Hardware stubs always return False on Mac (no device files present).
        """
        try:
            if source == 'OBD2':
                import os
                return os.path.exists(config.OBD2_PORT)
            if source == 'CAN':
                import os
                return os.path.exists(f'/sys/class/net/{config.CAN_INTERFACE}')
            if source == 'GPS':
                import os
                return os.path.exists(config.GPS_PORT)
        except Exception as exc:  # noqa: BLE001
            logger.debug('SpeedSource._try_connect(%s) failed: %s', source, exc)
        return False

    def get_speed_mps(self) -> tuple[float | None, bool]:
        """Return (speed_mps, is_stale).

        Returns (None, True) when no source is available (triggers FM-05).
        """
        if self._source_type == 'NONE':
            return None, True
        # Hardware polling would go here; stub returns stale for now
        return self._speed_mps, self._is_stale

    def get_source_type(self) -> str:
        """Return active source name: 'OBD2' | 'CAN' | 'GPS' | 'NONE'."""
        return self._source_type

    def is_available(self) -> bool:
        """True if a non-NONE speed source is active."""
        return self._source_type != 'NONE'
