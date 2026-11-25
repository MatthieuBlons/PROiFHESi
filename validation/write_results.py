"""
Cross-validation result helpers.

Job layout written by `validation/cross_val.py`
-----------------------------------------------
<job_dir>/test_<fold>/rep_<repeat>/
    model.pt.tar, best_model.pt.tar   checkpoints (`best_` with validation)
    test_predictions.csv              written for the selected repeat
    test_metrics.csv                  written for the selected repeat
    tensorboard/                      with `--log`

Files written in <job_dir>
--------------------------
all_validation_results.csv       best validation metrics of every run
mean_validation_over_repeats.csv mean of the above over repeats, per fold
best_model_test_<f>_repeat_<r>.pt.tar
                                 copy of the selected checkpoint per fold
all_test_results.csv             tile-level test predictions of all folds
test_metrics.csv                 test metrics, one row per fold
"""

from __future__ import annotations

# General libraries
import logging
import shutil
from pathlib import Path
from typing import Any

# Data libraries
import pandas as pd
import torch

LOGGER = logging.getLogger(__name__)

BEST_CHECKPOINT_NAME = "best_model.pt.tar"
LAST_CHECKPOINT_NAME = "model.pt.tar"
TENSORBOARD_DIR_NAME = "tensorboard"

VALIDATION_RESULTS_FILE = "all_validation_results.csv"
MEAN_VALIDATION_FILE = "mean_validation_over_repeats.csv"
TEST_RESULTS_FILE = "all_test_results.csv"
TEST_METRICS_FILE = "test_metrics.csv"
TENSORBOARD_ROOT_NAME = "model_best_events"


# -----------------------------------------------------------------------------
# Validation results
# -----------------------------------------------------------------------------


def find_run_checkpoints(job_dir: str | Path) -> dict[tuple[int, int], Path]:
    """
    Find the checkpoint of every (fold, repeat) run of a job.

    The best checkpoint (`best_model.pt.tar`) is used when it exists
    (training with validation), the last one (`model.pt.tar`) otherwise.

    Parameters
    ----------
    job_dir : str | Path
        Cross-validation job directory.

    Returns
    -------
    dict[tuple[int, int], Path]
        Checkpoint path per `(fold, repeat)`, sorted.
    """
    checkpoints = {}

    for run_dir in sorted(Path(job_dir).glob("test_*/rep_*")):
        fold = int(run_dir.parent.name.split("_")[-1])
        repeat = int(run_dir.name.split("_")[-1])

        for checkpoint_name in (BEST_CHECKPOINT_NAME, LAST_CHECKPOINT_NAME):
            checkpoint_path = run_dir / checkpoint_name

            if checkpoint_path.is_file():
                checkpoints[(fold, repeat)] = checkpoint_path
                break

        else:
            LOGGER.warning("No checkpoint in %s; run skipped.", run_dir)

    return checkpoints


def read_checkpoint_summary(checkpoint_path: str | Path) -> dict[str, Any]:
    """
    Read the run identifiers and best validation metrics of a checkpoint.

    The checkpoint is memory-mapped, so the network weights are not loaded.

    Returns
    -------
    dict
        `test`, `repeat`, `checkpoint`, `ref_metric`, `metric_mode`, then the
        `best_metrics` of the run (empty without validation).
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    args = checkpoint["args"]

    return {
        "test": int(args.test_fold),
        "repeat": int(args.repeat),
        "checkpoint": str(checkpoint_path),
        "ref_metric": args.ref_metric,
        "metric_mode": args.metric_mode,
        **(checkpoint.get("best_metrics") or {}),
    }


def collect_validation_results(job_dir: str | Path) -> pd.DataFrame:
    """
    Gather the best validation metrics of every run of a job.

    Returns
    -------
    pd.DataFrame
        One row per (fold, repeat), see `read_checkpoint_summary`.

    Raises
    ------
    FileNotFoundError
        If the job has no checkpoint.
    """
    rows = []

    for (fold, repeat), checkpoint_path in find_run_checkpoints(job_dir).items():
        try:
            rows.append(read_checkpoint_summary(checkpoint_path))
        except Exception:
            LOGGER.exception(
                "Cannot read checkpoint %s (fold=%d, repeat=%d); run skipped.",
                checkpoint_path,
                fold,
                repeat,
            )

    if not rows:
        raise FileNotFoundError(f"No readable checkpoint in `job_dir`: {job_dir}.")

    return pd.DataFrame(rows).sort_values(["test", "repeat"], ignore_index=True)


def mean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average the numeric metrics over repeats, for every test fold.

    Parameters
    ----------
    df : pd.DataFrame
        Results with `test` and `repeat` columns.

    Returns
    -------
    pd.DataFrame
        One row per test fold, without the `repeat` column.
    """
    return (
        df.drop(columns="repeat")
        .groupby("test", as_index=False)
        .mean(numeric_only=True)
    )


# -----------------------------------------------------------------------------
# Model selection
# -----------------------------------------------------------------------------


def select_best_repeat(
    df: pd.DataFrame,
    reference_metric: str = "mean_val_loss",
    metric_mode: str = "min",
    n_best: int = 1,
) -> list[tuple[int, int]]:
    """
    Select the best repeat(s) of every test fold on a validation metric.

    Parameters
    ----------
    df : pd.DataFrame
        Validation results (`collect_validation_results`).
    reference_metric : str, default="mean_val_loss"
        Selection metric (training `ref_metric`).
    metric_mode : {"min", "max"}, default="min"
        Whether lower or higher values are better.
    n_best : int, default=1
        Number of repeats kept per fold.

    Returns
    -------
    list[tuple[int, int]]
        Selected `(test, repeat)` pairs.

    Raises
    ------
    KeyError
        If `reference_metric` is not a column of `df`.
    """
    if reference_metric not in df.columns:
        raise KeyError(
            f"`reference_metric={reference_metric}` is not in the validation "
            f"results. Available columns: {sorted(df.columns)}."
        )

    selection = []

    for test_fold, df_fold in df.groupby("test"):
        best = df_fold.sort_values(
            reference_metric,
            ascending=metric_mode == "min",
        ).head(n_best)

        selection.extend((int(test_fold), int(repeat)) for repeat in best["repeat"])

    return selection


def copy_best_to_root(
    job_dir: str | Path,
    selection: list[tuple[int, int]],
    checkpoints: dict[tuple[int, int], Path],
) -> dict[int, dict[int, Path]]:
    """
    Copy the selected checkpoints (and TensorBoard events) to the job root.

    Parameters
    ----------
    job_dir : str | Path
        Cross-validation job directory.
    selection : list[tuple[int, int]]
        Selected `(test, repeat)` pairs (`select_best_repeat`).
    checkpoints : dict[tuple[int, int], Path]
        Checkpoint per run (`find_run_checkpoints`).

    Returns
    -------
    dict[int, dict[int, Path]]
        `{test: {repeat: copied checkpoint path}}`.
    """
    job_dir = Path(job_dir)
    events_dir = job_dir / TENSORBOARD_ROOT_NAME
    best_in_root: dict[int, dict[int, Path]] = {}

    for test_fold, repeat in selection:
        source_checkpoint = checkpoints[(test_fold, repeat)]
        destination_checkpoint = (
            job_dir / f"best_model_test_{test_fold}_repeat_{repeat}.pt.tar"
        )
        shutil.copy(
            source_checkpoint,
            destination_checkpoint,
        )
        best_in_root.setdefault(test_fold, {})[repeat] = destination_checkpoint

        source_events = source_checkpoint.parent / TENSORBOARD_DIR_NAME

        if source_events.is_dir():
            shutil.copytree(
                source_events,
                events_dir / f"event_test_{test_fold}_repeat_{repeat}",
                dirs_exist_ok=True,
            )

    return best_in_root
