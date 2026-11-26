"""
Repeated k-fold cross-validation of PROiFHESi.

1. Training: `finetuning.train.main` runs once per (fold, repeat) in
   `<output_dir>/<job>/test_<fold>/rep_<repeat>/`, with a shared YAML
   configuration, the fold / repeat indices, the resolved device and the
   seed `fold * n_repeats + repeat` (or `--seed`).
2. Selection: the best validation metrics of every run are gathered and
   the best repeat of each fold is selected on the training `ref_metric`.
   Without validation (`use_val=False`), every repeat is kept.
3. Test: each selected model is tested on its excluded fold with
   `finetuning.test.test_model`.

No ensembling and no Monte Carlo dropout: one model per fold.

Outputs in <output_dir>/<job>/ are listed in `validation.write_results`.

Usage
-----
python -m validation.cross_val --config <config.yaml> --output_dir <dir> \
    --n_folds 5 [--n_repeats 3] [--job <name>] [--gpu] [--log] [--verbose]
"""

from __future__ import annotations

# General libraries
import logging
import warnings
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace,
)
from datetime import datetime
from pathlib import Path

# Data libraries
import pandas as pd
import torch
from tqdm.auto import tqdm

# Project modules
from finetuning.arguments import load_yaml_config
from finetuning.test import test_model
from finetuning.train import main as train_single_run
from validation.logger import configure_logging, timetracker, resolve_device
from validation.write_results import (
    MEAN_VALIDATION_FILE,
    TEST_METRICS_FILE,
    TEST_RESULTS_FILE,
    VALIDATION_RESULTS_FILE,
    collect_validation_results,
    copy_best_to_root,
    find_run_checkpoints,
    mean_dataframe,
    select_best_repeat,
)

# For the sklearn warnings
warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)

SUMMARY_METRICS = (
    "test_loss",
    "mean_pearson",
    "mean_r2",
    "mean_ccc",
)


# -----------------------------------------------------------------------------
# Launcher arguments
# -----------------------------------------------------------------------------


def build_job_parser() -> ArgumentParser:
    """
    Build the cross-validation launcher argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured command-line parser.
    """
    parser = ArgumentParser(
        description="Run repeated cross-validation of PROiFHESi fine-tuning.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # Experiment tracking

    parser.add_argument(
        "--project",
        type=str,
        default="PROiFHESi",
        help="W&B project name.",
    )
    parser.add_argument(
        "--job",
        type=str,
        default=None,
        help=(
            "Experiment group and output-directory name. "
            "Defaults to a date-stamped name."
        ),
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable W&B and TensorBoard logging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed of every run. Defaults to fold * n_repeats + repeat.",
    )

    # Cross-validation

    parser.add_argument(
        "--n_folds",
        type=int,
        required=True,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=1,
        help="Number of repeated training runs per fold.",
    )

    # Configuration and outputs

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration passed to each training run.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Parent directory used to store the cross-validation outputs.",
    )

    # Runtime

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "PyTorch device passed to each run, for example cpu, cuda, "
            "cuda:0, or mps. When omitted, the device is inferred."
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Request a CUDA device when --device is not explicitly set. "
            "Falls back to CPU if CUDA is unavailable."
        ),
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="CUDA device index used with --gpu.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display detailed logging and progress bars.",
    )
    parser.add_argument(
        "--clock",
        action="store_true",
        help="Log training and test durations.",
    )

    return parser


def parse_job_arguments(
    known_args: list[str] | None = None,
    parser: ArgumentParser | None = None,
) -> Namespace:
    """
    Parse and validate cross-validation launcher arguments.

    Parameters
    ----------
    known_args : list of str, optional
        Explicit argument sequence. When None, use command-line
        arguments.

    parser : argparse.ArgumentParser, optional
        Launcher parser extending `build_job_parser`. When None, use
        `build_job_parser()`.

    Returns
    -------
    argparse.Namespace
        Validated launcher configuration.
    """
    if parser is None:
        parser = build_job_parser()

    args = parser.parse_args(known_args)

    if args.n_folds < 2:
        parser.error("--n_folds must be at least 2.")

    if args.n_repeats < 1:
        parser.error("--n_repeats must be at least 1.")

    if not args.config.is_file():
        parser.error(f"Configuration file does not exist: {args.config}")

    if args.device is not None and args.gpu:
        parser.error("Use either --device or --gpu, not both.")

    return args





def resolve_job_name(job_name: str | None) -> str:
    """
    Resolve a date-stamped experiment name.

    Parameters
    ----------
    job_name : str or None
        User-provided experiment name.

    Returns
    -------
    str
        Resolved job name.
    """
    date_tag = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if job_name is None:
        return f"PROiFHESi_CV_{date_tag}"

    return f"{job_name}_{date_tag}"


def log_configuration(
    config: dict,
    title: str = "Resolved base configuration",
) -> None:
    """
    Log the contents of a configuration mapping.

    Parameters
    ----------
    config : dict
        Configuration values.

    title : str, default="Resolved base configuration"
        Heading written before the configuration entries.
    """
    LOGGER.info("%s:", title)

    for key, value in config.items():
        LOGGER.info("  %s: %s", key, value)

def build_run_directory(
    job_dir: Path,
    fold: int,
    repetition: int,
) -> Path:
    """
    Construct and create one fold/repetition output directory.

    Parameters
    ----------
    job_dir : pathlib.Path
        Parent experiment directory.

    fold : int
        Zero-based test-fold index.

    repetition : int
        Zero-based repetition index.

    Returns
    -------
    pathlib.Path
        Created run directory.
    """
    run_dir = job_dir / f"test_{fold}" / f"rep_{repetition}"

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return run_dir


def build_training_arguments(
    args: Namespace,
    device: str,
    run_dir: Path,
    fold: int,
    repetition: int,
    overrides: list[str] | None = None,
) -> list[str]:
    """
    Build command-line arguments for one training run.

    Values provided here override matching YAML configuration values.

    Parameters
    ----------
    args : argparse.Namespace
        Cross-validation launcher configuration.

    device : str
        Resolved PyTorch device.

    run_dir : pathlib.Path
        Output directory for the current run.

    fold : int
        Zero-based test-fold index.

    repetition : int
        Zero-based repetition index.

    overrides : list of str, optional
        Extra command-line arguments appended last, for example the
        hyperparameters of one grid-search configuration.
        Takes priority over --config.

    Returns
    -------
    list[str]
        Argument list passed to the single-run training entry point.
    """
    if args.seed is None:
        training_seed = fold * args.n_repeats + repetition
    else:
        training_seed = args.seed

    return [
        "--config",
        str(args.config),
        "--device",
        device,
        "--num_workers",
        str(args.num_workers),
        "--job_dir",
        str(run_dir),
        "--n_reps",
        str(args.n_repeats),
        "--repeat",
        str(repetition),
        "--n_folds",
        str(args.n_folds),
        "--test_fold",
        str(fold),
        "--seed",
        str(training_seed),
        "--write_config",
        *(overrides or []),
    ]


def train_cross_validation(
    args: Namespace,
    device: str,
) -> Path:
    """
    Train every cross-validation fold and repetition.

    Parameters
    ----------
    args : argparse.Namespace
        Validated launcher configuration.
    device : str
        Resolved PyTorch device.

    Returns
    -------
    pathlib.Path
        Job directory holding the `test_<fold>/rep_<repeat>` runs.
    """
    job_name = resolve_job_name(args.job)

    job_dir = args.output_dir.expanduser().resolve() / job_name
    job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGGER.info(
        "Starting cross-validation | folds=%d | repeats=%d",
        args.n_folds,
        args.n_repeats,
    )
    LOGGER.info("Device: %s", device)
    LOGGER.info("Output directory: %s", job_dir)

    base_config = load_yaml_config(args.config)
    log_configuration(base_config)

    run_indices = [
        (fold, repetition)
        for fold in range(args.n_folds)
        for repetition in range(args.n_repeats)
    ]

    completed_epochs: dict[tuple[int, int], int] = {}

    progress = tqdm(
        run_indices,
        desc="Cross-validation",
        unit="run",
        disable=not args.verbose,
    )

    for fold, repetition in progress:
        run_dir = build_run_directory(
            job_dir=job_dir,
            fold=fold,
            repetition=repetition,
        )

        progress.set_postfix(
            {
                "fold": f"{fold + 1}/{args.n_folds}",
                "repeat": (f"{repetition + 1}/{args.n_repeats}"),
            },
            refresh=False,
        )

        LOGGER.info(
            "Starting run | fold=%d/%d | repeat=%d/%d",
            fold + 1,
            args.n_folds,
            repetition + 1,
            args.n_repeats,
        )

        training_args = build_training_arguments(
            args=args,
            device=device,
            run_dir=run_dir,
            fold=fold,
            repetition=repetition,
        )

        try:
            stop_epoch = train_single_run(
                project_name=args.project,
                job_name=job_name,
                known_args=training_args,
                verbose=args.verbose,
                log=args.log,
            )

        except Exception:
            LOGGER.exception(
                "Training failed | fold=%d | repeat=%d | run_dir=%s",
                fold,
                repetition,
                run_dir,
            )
            raise

        completed_epochs[(fold, repetition)] = stop_epoch

        LOGGER.info(
            "Completed run | fold=%d/%d | repeat=%d/%d | epochs=%d",
            fold + 1,
            args.n_folds,
            repetition + 1,
            args.n_repeats,
            stop_epoch,
        )

    LOGGER.info(
        "Completed epochs per (fold, repeat): %s",
        completed_epochs,
    )

    return job_dir

def test_cross_validation(
    job_dir: str | Path,
    device: str = "cpu",
    num_workers: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Select the best repeat of each fold and test it on the excluded fold.

    Parameters
    ----------
    job_dir : str | Path
        Cross-validation job directory.
    device : str, default="cpu"
        PyTorch device used for the test.
    num_workers : int | None, default=None
        DataLoader workers; the checkpoint value when None.
    verbose : bool, default=False
        Display progress bars.

    Returns
    -------
    pd.DataFrame
        Test metrics, one row per tested model (`test_metrics.csv`).

    Raises
    ------
    RuntimeError
        If no selected model could be tested.
    """
    job_dir = Path(job_dir)
    checkpoints = find_run_checkpoints(job_dir)

    # Validation results of every run.
    validation_results = collect_validation_results(job_dir)
    validation_results.to_csv(
        job_dir / VALIDATION_RESULTS_FILE,
        index=False,
    )
    mean_dataframe(validation_results).to_csv(
        job_dir / MEAN_VALIDATION_FILE,
        index=False,
    )

    # Best repeat per fold on the training reference metric.
    reference_metric = validation_results["ref_metric"].iloc[0]
    metric_mode = validation_results["metric_mode"].iloc[0]

    if reference_metric in validation_results.columns:
        selection = select_best_repeat(
            df=validation_results,
            reference_metric=reference_metric,
            metric_mode=metric_mode,
            n_best=1,
        )
    else:
        LOGGER.warning(
            "No validation metric `%s` in the checkpoints (trained without "
            "validation); every repeat is tested.",
            reference_metric,
        )
        selection = list(
            zip(
                validation_results["test"].astype(int),
                validation_results["repeat"].astype(int),
            )
        )

    LOGGER.info(
        "Selected (fold, repeat) on %s (%s): %s",
        reference_metric,
        metric_mode,
        selection,
    )

    best_in_root = copy_best_to_root(
        job_dir,
        selection,
        checkpoints,
    )

    # Test of the selected models.
    test_predictions = []
    test_metrics = []

    for test_fold, repeats in best_in_root.items():
        for repeat, model_path in repeats.items():
            try:
                predictions, metrics = test_model(
                    model_path=model_path,
                    device=device,
                    output_dir=job_dir / f"test_{test_fold}" / f"rep_{repeat}",
                    num_workers=num_workers,
                    verbose=verbose,
                )

            except Exception:
                LOGGER.exception(
                    "Testing failed | fold=%d | repeat=%d | model=%s",
                    test_fold,
                    repeat,
                    model_path,
                )
                continue

            test_predictions.append(predictions)
            test_metrics.append(metrics)

    if not test_metrics:
        raise RuntimeError(f"No selected model could be tested in {job_dir}.")

    pd.concat(
        test_predictions,
        ignore_index=True,
    ).to_csv(
        job_dir / TEST_RESULTS_FILE,
        index=False,
    )

    test_metrics_table = pd.DataFrame(test_metrics)
    test_metrics_table.to_csv(
        job_dir / TEST_METRICS_FILE,
        index=False,
    )

    for metric_name in SUMMARY_METRICS:
        LOGGER.info(
            "Test %s over folds: %.4f +/- %.4f",
            metric_name,
            test_metrics_table[metric_name].mean(),
            test_metrics_table[metric_name].std(),
        )

    return test_metrics_table


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(
    known_args: list[str] | None = None,
) -> Path:
    """
    Run repeated cross-validation training, selection and test.

    Parameters
    ----------
    known_args : list of str, optional
        Explicit launcher arguments. When None, use command-line
        arguments.

    Returns
    -------
    pathlib.Path
        Job directory holding runs and result tables.
    """
    args = parse_job_arguments(known_args=known_args)

    # `--clock` timings are logged at INFO level.
    configure_logging(verbose=args.verbose or args.clock)

    device = resolve_device(args)
    timer = timetracker(
        name="PROiFHESi cross-validation",
        verbose=args.clock,
    )

    timer.tic()
    job_dir = train_cross_validation(
        args,
        device=device,
    )
    timer.toc()

    LOGGER.info("Cross-validation training completed successfully.")

    try:
        test_cross_validation(
            job_dir,
            device=device,
            num_workers=args.num_workers,
            verbose=args.verbose,
        )
        timer.toc()

    finally:
        LOGGER.info("Cross-validation results saved in: %s.", job_dir)

    return job_dir


if __name__ == "__main__":
    main()
