"""
Logging and timing helpers of the validation launchers.

configure_logging : console logging (INFO with `verbose`, WARNING otherwise)
timetracker       : stopwatch logging elapsed times (replaces toolkit `dtime`)
"""

from __future__ import annotations

# General libraries
import logging
import time

LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    """
    Configure console logging.

    Parameters
    ----------
    verbose : bool, default=False
        Use informational logging when enabled and warning-level logging
        otherwise.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


class timetracker:
    """
    Stopwatch timer.

    `tic()` starts an interval, `toc()` returns the elapsed seconds since
    the last `tic()` / `toc()` and logs it (INFO) when `verbose`.

    Parameters
    ----------
    name : str | None, default=None
        Label used in the log message.
    verbose : bool, default=False
        Log elapsed times.
    """

    def __init__(
        self,
        name: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.verbose = verbose
        self._last_time = time.time()

    def tic(self) -> None:
        """Start a time interval."""
        self._last_time = time.time()

    def toc(self, log: bool = True) -> float:
        """
        Return (and log when `verbose`) the seconds elapsed since the last
        `tic()` / `toc()`, then restart the interval.
        """
        current_time = time.time()
        elapsed = current_time - self._last_time
        self._last_time = current_time

        if log and self.verbose:
            LOGGER.info(
                "%s | elapsed time: %.1f s.",
                self.name or "timer",
                elapsed,
            )

        return elapsed
