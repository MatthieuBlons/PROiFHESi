"""
Logging and timing helpers of the validation launchers.

configure_logging : console logging (INFO with `verbose`, WARNING otherwise)
timetracker       : stopwatch logging elapsed times (replaces toolkit `dtime`)
"""

from __future__ import annotations

# General libraries
import logging
import time
import torch 
from argparse import Namespace

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

def resolve_device(args: Namespace) -> str:
    """
    Resolve the compute device passed to each training run.

    Explicit --device takes precedence. Otherwise, --gpu requests
    CUDA, optionally with a selected device index. If no device option is
    provided, CUDA is used when available and CPU otherwise.

    Parameters
    ----------
    args : argparse.Namespace
        Launcher arguments.

    Returns
    -------
    str
        Resolved PyTorch device string.
    """
    if args.device is not None:
        if "cuda" in args.device and not torch.cuda.is_available():
            LOGGER.warning("CUDA was requested but is unavailable; using CPU.")
            return "cpu"
        return args.device

    if args.gpu:
        if not torch.cuda.is_available():
            LOGGER.warning("CUDA was requested but is unavailable; using CPU.")
            return "cpu"

        if args.gpu_id is None:
            return "cuda"

        if args.gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {args.gpu_id} was requested, but only "
                f"{torch.cuda.device_count()} device(s) are available."
            )

        return f"cuda:{args.gpu_id}"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"