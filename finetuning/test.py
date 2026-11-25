"""
Test one trained PROiFHESi model on its test fold.

The checkpoint `args` define the data (`img_dir`, `input_path`,
`target_path`) and the test fold (`test_fold`): tiles whose fold column
equals `test_fold` are predicted without augmentation, then compared with
their measured signature.

Outputs (in `output_dir`, default: the checkpoint directory)
-------------------------------------------------------------
test_predictions.csv : one row per tile, `path`, `test`, `repeat`, then
                       `<target>` (measured) and `<target>_pred` for every
                       target
test_metrics.csv     : one row, `test`, `repeat`, `n_tiles`, `test_loss`
                       and the regression metrics of
                       `PROiFHESi.compute_regression_metrics`
                       (`mean_pearson`, `mean_r2`, `mean_ccc`, `mean_cossim`,
                       `mean_euclidsim`, `pearson_<target>`, ...)

No Monte Carlo dropout and no ensembling: one checkpoint, one deterministic
prediction per tile. Selecting the best repeat of each fold is done by
`validation/cross_val.py`.

Usage
-----
python -m finetuning.test --model_path <job_dir>/best_model.pt.tar [--device cuda]
"""

from __future__ import annotations

# General libraries
import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Sequence

# Data libraries
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Project modules
from finetuning.base_model import configure_logging
from finetuning.dataloader import DatasetHandler
from finetuning.model import PROiFHESi
from finetuning.utils import load_model_from_path, to_numpy

LOGGER = logging.getLogger(__name__)

PREDICTION_SUFFIX = "_pred"
PREDICTIONS_FILE = "test_predictions.csv"
METRICS_FILE = "test_metrics.csv"


# -----------------------------------------------------------------------------
# Inference and metrics
# -----------------------------------------------------------------------------


def test(
    model: PROiFHESi,
    dataloader: DataLoader,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Predict every tile of a test loader.

    Parameters
    ----------
    model : PROiFHESi
        Trained model.
    dataloader : torch.utils.data.DataLoader
        Test loader yielding `image`, `target` and `path`.
    verbose : bool, default=False
        Display a progress bar.

    Returns
    -------
    dict
        `path` (list[str], N), `prediction` and `target` (np.ndarray,
        (N, output_dim), fp32), in the loader order.

    Raises
    ------
    RuntimeError
        If the loader yields no batch.
    """
    paths: list[str] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for batch in tqdm(
        dataloader,
        desc="Test",
        unit="batch",
        leave=False,
        disable=not verbose,
    ):
        batch_result = model.predict_batch(batch)

        paths.extend(batch_result["path"])
        predictions.append(batch_result["prediction"])
        targets.append(to_numpy(batch["target"]).astype(np.float32))

    if not predictions:
        raise RuntimeError("The test DataLoader produced no batch.")

    return {
        "path": paths,
        "prediction": np.concatenate(
            predictions,
            axis=0,
        ),
        "target": np.concatenate(
            targets,
            axis=0,
        ),
    }


def compute_test_metrics(
    model: PROiFHESi,
    result: dict[str, Any],
) -> dict[str, float]:
    """
    Compute the test loss and the regression metrics.

    Parameters
    ----------
    model : PROiFHESi
        Model providing `criterion`, `target_names` and
        `compute_regression_metrics`.
    result : dict
        Output of `test`.

    Returns
    -------
    dict[str, float]
        `n_tiles`, `test_loss` (criterion over all tiles) and the
        regression metrics (same names as the validation metrics).
    """
    predictions = torch.from_numpy(result["prediction"])
    targets = torch.from_numpy(result["target"])

    metrics = {
        "n_tiles": int(len(result["path"])),
        "test_loss": float(
            model.compute_loss(
                predictions,
                targets,
            )
        ),
    }
    metrics.update(
        model.compute_regression_metrics(
            predictions=predictions,
            targets=targets,
            target_names=model.target_names,
        )
    )

    return metrics


def results_to_dataframe(
    result: dict[str, Any],
    target_names: list[str] | None = None,
    test_fold: int | None = None,
    repeat: int | None = None,
) -> pd.DataFrame:
    """
    Convert a `test` result into a tile-level DataFrame.

    Parameters
    ----------
    result : dict
        Output of `test`.
    target_names : list[str] | None, default=None
        Target names; `target_0`, `target_1`, ... when None.
    test_fold, repeat : int | None, default=None
        Run identifiers added as `test` / `repeat` columns.

    Returns
    -------
    pd.DataFrame
        One row per tile: `path`, `test`, `repeat`, then `<target>` and
        `<target>_pred` for every target.

    Raises
    ------
    ValueError
        If the number of `target_names` differs from the number of outputs.
    """
    predictions = np.asarray(result["prediction"])
    targets = np.asarray(result["target"])
    n_targets = predictions.shape[1]

    if target_names is None:
        target_names = [f"target_{index}" for index in range(n_targets)]

    if len(target_names) != n_targets:
        raise ValueError(
            f"Got {len(target_names)} `target_names` for {n_targets} outputs."
        )

    columns: dict[str, Any] = {
        "path": result["path"],
        "test": test_fold,
        "repeat": repeat,
    }

    for index, target_name in enumerate(target_names):
        columns[target_name] = targets[:, index]
        columns[f"{target_name}{PREDICTION_SUFFIX}"] = predictions[:, index]

    return pd.DataFrame(columns)


# -----------------------------------------------------------------------------
# Single-model test
# -----------------------------------------------------------------------------


def test_model(
    model_path: str | Path,
    device: str = "cpu",
    output_dir: str | Path | None = None,
    eval_batch_size: int | None = None,
    num_workers: int | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Load a checkpoint, predict its test fold and write the results.

    Parameters
    ----------
    model_path : str | Path
        Checkpoint written during training (`best_model.pt.tar`, or
        `model.pt.tar` without validation).
    device : str, default="cpu"
        PyTorch device.
    output_dir : str | Path | None, default=None
        Output directory; the checkpoint directory when None.
    eval_batch_size, num_workers : int | None, default=None
        Override the checkpoint DataLoader settings.
    verbose : bool, default=False
        Display a progress bar.

    Returns
    -------
    predictions : pd.DataFrame
        Tile-level predictions (see `results_to_dataframe`).
    metrics : dict[str, float]
        Test metrics, with `test` and `repeat`.

    Raises
    ------
    FileNotFoundError
        If `model_path` does not exist.
    """
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"`model_path` not found: {model_path}.")

    output_dir = model_path.parent if output_dir is None else Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_model_from_path(
        model_path,
        device=device,
    )
    args = model.args

    if eval_batch_size is not None:
        args.eval_batch_size = eval_batch_size

    if num_workers is not None:
        args.num_workers = num_workers

    dataset_handler = DatasetHandler(
        args,
        transform=model.network.transform,
        with_target=True,
        predict=False,
    )
    test_loader = dataset_handler.get_loader(training=False)

    LOGGER.info(
        "Testing %s | test fold=%s | repeat=%s | %d tiles.",
        model_path,
        args.test_fold,
        args.repeat,
        len(test_loader.dataset),
    )

    result = test(
        model,
        test_loader,
        verbose=verbose,
    )

    metrics = {
        "test": args.test_fold,
        "repeat": args.repeat,
        **compute_test_metrics(
            model,
            result,
        ),
    }
    predictions = results_to_dataframe(
        result,
        target_names=model.target_names,
        test_fold=args.test_fold,
        repeat=args.repeat,
    )

    predictions.to_csv(
        output_dir / PREDICTIONS_FILE,
        index=False,
    )
    pd.DataFrame([metrics]).to_csv(
        output_dir / METRICS_FILE,
        index=False,
    )

    LOGGER.info(
        "Test | loss=%.4f | mean_pearson=%.4f | mean_r2=%.4f | mean_ccc=%.4f | "
        "saved to %s.",
        metrics["test_loss"],
        metrics["mean_pearson"],
        metrics["mean_r2"],
        metrics["mean_ccc"],
        output_dir,
    )

    return predictions, metrics


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------


def get_test_arguments(known_args: Sequence[str] | None = None) -> Namespace:
    """
    Parse the test command line. Data and model settings come from the
    checkpoint `args`.
    """
    parser = ArgumentParser(
        description="Test one trained PROiFHESi model on its test fold.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Trained checkpoint (best_model.pt.tar).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device, for example cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the checkpoint directory.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Test batch size. Defaults to the checkpoint value.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader workers. Defaults to the checkpoint value.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="INFO logging and progress bar.",
    )

    return parser.parse_args(known_args)


def main(known_args: Sequence[str] | None = None) -> dict[str, float]:
    """
    Command-line entry point; returns the test metrics.
    """
    args = get_test_arguments(known_args)
    configure_logging(verbose=args.verbose)

    _, metrics = test_model(
        model_path=args.model_path,
        device=args.device,
        output_dir=args.output_dir,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        verbose=args.verbose,
    )

    return metrics


if __name__ == "__main__":
    main()
