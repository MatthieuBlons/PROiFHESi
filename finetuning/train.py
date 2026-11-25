"""
Training entry point for one PROiFHESi run (one test fold, one repeat).

The loop only talks to the model through the `BaseModel` interface
(`train_batch`, `validate_batch`, `finalize_validation_epoch`,
`create_checkpoint`), as in MILKYWAY.

Each epoch
----------
1. train on the training loader;
2. with `use_val`: validate, step the scheduler (after `lr_warm_up`
   epochs), update early stopping, which saves `job_dir/model.pt.tar` and
   `job_dir/best_model.pt.tar`;
   without `use_val`: step the scheduler and save `job_dir/model.pt.tar`.

Logging
-------
console     : `logging` (INFO with `verbose`): run setup, one line per
              epoch, early stopping and best epoch; tqdm progress bars
W&B         : training loss and learning rates every `log_steps` batches,
              epoch metrics (`log=True`, primary process only)
TensorBoard : epoch metrics in `job_dir/tensorboard` (`log=True`)

Usage
-----
python -m finetuning.train --config <config.yaml> [--option value ...]
"""

from __future__ import annotations

# General libraries
import logging
from pathlib import Path
from typing import Sequence

# Data libraries
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Project modules
from finetuning.arguments import get_arguments
from finetuning.base_model import (
    close_logging_backends,
    configure_logging,
    get_current_learning_rates,
    initialize_logging_backends,
    register_exit_handlers,
    write_tensorboard_metrics,
)
from finetuning.model import PROiFHESi

LOGGER = logging.getLogger(__name__)

CHECKPOINT_NAME = "model.pt.tar"


# -----------------------------------------------------------------------------
# Epoch steps
# -----------------------------------------------------------------------------


def train_one_epoch(
    model: PROiFHESi,
    dataloader: DataLoader,
    verbose: bool = False,
    log_wandb: bool = False,
    log_steps: int = 100,
) -> float:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : PROiFHESi
        Model being optimised.
    dataloader : torch.utils.data.DataLoader
        Training loader.
    verbose : bool, default=False
        Display a batch-level progress bar.
    log_wandb : bool, default=False
        Log the batch loss and learning rates to W&B.
    log_steps : int, default=100
        W&B logging frequency in batches; 0 disables batch logging.

    Returns
    -------
    float
        Mean training loss of the epoch.

    Raises
    ------
    RuntimeError
        If the loader yields no batch.
    """
    batch_losses = []
    criterion_name = model.args.criterion

    progress = tqdm(
        dataloader,
        desc="Training",
        unit="batch",
        leave=False,
        disable=not verbose,
    )

    for batch in progress:
        loss = model.train_batch(batch)
        batch_losses.append(loss)

        progress.set_postfix(
            {
                f"{criterion_name}_loss": f"{loss:.4f}",
            },
            refresh=False,
        )

        should_log = (
            log_wandb and log_steps > 0 and model.counter["batch"] % log_steps == 0
        )

        if should_log:
            log_values = {
                "train_batch_loss": loss,
            }

            for index, learning_rate in enumerate(get_current_learning_rates(model)):
                log_values[f"train_lr_group_{index}"] = learning_rate

            wandb.log(
                log_values,
                step=model.counter["batch"],
            )

    if not batch_losses:
        raise RuntimeError(
            "The training DataLoader produced no batch "
            "(the last incomplete batch is dropped: check `batch_size`)."
        )

    mean_loss = float(np.mean(batch_losses))
    model.mean_train_loss = mean_loss

    return mean_loss


def validate_one_epoch(
    model: PROiFHESi,
    dataloader: DataLoader,
    verbose: bool = False,
    log_wandb: bool = False,
    log_tensorboard: bool = False,
) -> dict[str, float]:
    """
    Evaluate the model over one validation epoch.

    Parameters
    ----------
    model : PROiFHESi
        Model being evaluated.
    dataloader : torch.utils.data.DataLoader
        Validation loader.
    verbose : bool, default=False
        Display a batch-level progress bar.
    log_wandb : bool, default=False
        Log epoch metrics to W&B (prefixed with `validation_`).
    log_tensorboard : bool, default=False
        Log epoch metrics to TensorBoard (under `validation/`).

    Returns
    -------
    dict[str, float]
        Validation metrics, including mean training and validation losses.

    Raises
    ------
    RuntimeError
        If the loader yields no batch.
    """
    batch_losses = []
    criterion_name = model.args.criterion

    progress = tqdm(
        dataloader,
        desc="Validation",
        unit="batch",
        leave=False,
        disable=not verbose,
    )

    for batch in progress:
        loss = model.validate_batch(batch)
        batch_losses.append(loss)

        progress.set_postfix(
            {
                f"{criterion_name}_loss": f"{loss:.4f}",
            },
            refresh=False,
        )

    if not batch_losses:
        raise RuntimeError("The validation DataLoader produced no batch.")

    model.mean_validation_loss = float(np.mean(batch_losses))

    validation_metrics = model.finalize_validation_epoch()

    if log_wandb:
        wandb.log(
            {
                **{
                    f"validation_{name}": value
                    for name, value in validation_metrics.items()
                },
                "epoch": model.counter["epoch"],
            },
            step=model.counter["batch"],
        )

    if log_tensorboard:
        write_tensorboard_metrics(
            writer=model.writer,
            metrics=validation_metrics,
            epoch=model.counter["epoch"],
        )

    return validation_metrics


def log_training_epoch(
    model: PROiFHESi,
    log_wandb: bool = False,
    log_tensorboard: bool = False,
) -> None:
    """
    Log the mean training loss of an epoch run without validation.

    With validation, `mean_train_loss` is already part of the validation
    metrics.
    """
    training_metrics = {
        "mean_train_loss": float(model.mean_train_loss),
    }

    if log_wandb:
        wandb.log(
            {
                **{f"train_{name}": value for name, value in training_metrics.items()},
                "epoch": model.counter["epoch"],
            },
            step=model.counter["batch"],
        )

    if log_tensorboard:
        write_tensorboard_metrics(
            writer=model.writer,
            metrics=training_metrics,
            epoch=model.counter["epoch"],
            prefix="train",
        )


# -----------------------------------------------------------------------------
# Scheduler, early stopping, checkpoints
# -----------------------------------------------------------------------------


def step_learning_rate_scheduler(
    model: PROiFHESi,
    validation_metrics: dict[str, float] | None,
) -> None:
    """
    Advance the scheduler once the `lr_warm_up` epochs are over.

    The plateau scheduler monitors `ref_metric` and needs validation.

    Raises
    ------
    RuntimeError
        If the plateau scheduler is used without validation metrics.
    KeyError
        If `ref_metric` is missing from the validation metrics.
    """
    if not model.schedulers:
        return

    if model.counter["epoch"] <= getattr(model.args, "lr_warm_up", 0):
        return

    if model.args.lr_scheduler == "plateau":
        if validation_metrics is None:
            raise RuntimeError(
                "`lr_scheduler=plateau` requires validation (`use_val=True`)."
            )

        if model.reference_metric not in validation_metrics:
            raise KeyError(
                f"Scheduler monitor `ref_metric={model.reference_metric}` is not "
                f"computed. Available metrics: {sorted(validation_metrics)}."
            )

        model.update_learning_rate(validation_metrics[model.reference_metric])

    else:
        model.update_learning_rate()


def update_early_stopping(
    model: PROiFHESi,
    validation_metrics: dict[str, float],
) -> None:
    """
    Update early stopping on `ref_metric` and save the checkpoints.

    Early stopping is not evaluated before `minimum_epochs`.
    """
    model.early_stopping(
        monitored_value=validation_metrics[model.reference_metric],
        checkpoint=model.create_checkpoint(),
    )


def save_checkpoint_without_validation(model: PROiFHESi) -> None:
    """
    Save the latest checkpoint when no validation set is used.
    """
    torch.save(
        model.create_checkpoint(),
        Path(model.args.job_dir) / CHECKPOINT_NAME,
    )


def format_epoch_summary(
    model: PROiFHESi,
    validation_metrics: dict[str, float] | None,
    best_epoch: int | None,
) -> dict[str, str]:
    """
    Build compact progress-bar / log information for the current epoch.
    """
    summary = {
        "train_loss": f"{model.mean_train_loss:.4f}",
    }

    if np.isfinite(model.mean_validation_loss):
        summary["val_loss"] = f"{model.mean_validation_loss:.4f}"

    learning_rates = get_current_learning_rates(model)

    if learning_rates:
        summary["lr"] = f"{learning_rates[0]:.3e}"

    if validation_metrics is not None and model.best_reference_value is not None:
        summary[f"best_{model.reference_metric}"] = (
            f"{model.best_reference_value:.4f}"
        )

    if best_epoch is not None:
        summary["best_epoch"] = str(best_epoch)

    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(
    project_name: str | None = None,
    job_name: str | None = None,
    known_args: Sequence[str] | None = None,
    verbose: bool = False,
    log: bool = False,
) -> int:
    """
    Train PROiFHESi on one (test fold, repeat).

    Parameters
    ----------
    project_name : str | None, default=None
        W&B project name.
    job_name : str | None, default=None
        W&B group name (one group per cross-validation job).
    known_args : Sequence[str] | None, default=None
        Explicit command-line arguments (read from `sys.argv` when None).
    verbose : bool, default=False
        INFO console logging and progress bars.
    log : bool, default=False
        Enable W&B and TensorBoard logging.

    Returns
    -------
    int
        Number of completed epochs.
    """
    configure_logging(verbose=verbose)

    args = get_arguments(
        known_args=known_args,
        train=True,
    )

    args.job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = PROiFHESi(
        args=args,
        with_data=True,
    )

    LOGGER.info(
        "Run %s | encoder=%s | lora=%s | criterion=%s | "
        "train=%d batches | validation=%s batches.",
        args.job_dir,
        args.encoder,
        args.lora,
        args.criterion,
        len(model.train_loader),
        len(model.val_loader) if model.val_loader is not None else "no",
    )

    initialize_logging_backends(
        model=model,
        args=args,
        enabled=log,
        project_name=project_name,
        job_name=job_name,
        dir_name=args.job_dir,
    )

    register_exit_handlers(model)

    n_folds = args.n_folds or 1
    n_reps = args.n_reps or 1

    epoch_progress = tqdm(
        total=args.epochs,
        desc=(
            f"Training "
            f"repeat={args.repeat + 1}/{n_reps}, "
            f"fold={args.test_fold + 1}/{n_folds}"
        ),
        unit="epoch",
        leave=True,
        disable=not verbose,
    )

    best_epoch = None

    try:
        while model.counter["epoch"] < args.epochs:
            train_one_epoch(
                model=model,
                dataloader=model.train_loader,
                verbose=verbose,
                log_wandb=log,
                log_steps=getattr(
                    args,
                    "log_steps",
                    100,
                ),
            )

            validation_metrics = None

            if args.use_val:
                previous_best = model.best_reference_value

                validation_metrics = validate_one_epoch(
                    model=model,
                    dataloader=model.val_loader,
                    verbose=verbose,
                    log_wandb=log,
                    log_tensorboard=log,
                )

                if model.best_reference_value != previous_best:
                    best_epoch = model.counter["epoch"]

                step_learning_rate_scheduler(
                    model=model,
                    validation_metrics=validation_metrics,
                )

                update_early_stopping(
                    model=model,
                    validation_metrics=validation_metrics,
                )

            else:
                model.advance_counter()

                log_training_epoch(
                    model=model,
                    log_wandb=log,
                    log_tensorboard=log,
                )

                step_learning_rate_scheduler(
                    model=model,
                    validation_metrics=None,
                )

                save_checkpoint_without_validation(model)

            epoch_summary = format_epoch_summary(
                model=model,
                validation_metrics=validation_metrics,
                best_epoch=best_epoch,
            )

            epoch_progress.set_postfix(
                epoch_summary,
                refresh=False,
            )
            epoch_progress.update()

            LOGGER.info(
                "Epoch %d/%d | %s",
                model.counter["epoch"],
                args.epochs,
                " | ".join(f"{key}={value}" for key, value in epoch_summary.items()),
            )

            if args.use_val and model.early_stopping.stop:
                LOGGER.info(
                    "Early stopping at epoch %d.",
                    model.counter["epoch"],
                )
                break

    finally:
        epoch_progress.close()
        close_logging_backends(model)

    if args.use_val and model.best_reference_value is not None:
        LOGGER.info(
            "Best %s=%.4f at epoch %s, saved to %s.",
            model.reference_metric,
            model.best_reference_value,
            best_epoch,
            model.early_stopping.best_checkpoint_path,
        )

    return int(model.counter["epoch"])


if __name__ == "__main__":
    main(verbose=True)
