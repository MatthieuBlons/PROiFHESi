"""
Predict proteomic signatures of new H&E tiles with a trained PROiFHESi model.

Unlike `finetuning.test`, no target is read: tiles come from `img_dir` (all
`*.<img_format>` files, or the `img_path` column of `input_path`) and every
tile is predicted, whatever its fold. Target names come from the checkpoint.

Intermediate representations can be saved to follow how the tile
representation evolves toward the molecular signature:

    embedding (feature_dim)          backbone output (`save_embeddings`)
    -> head.network.0 (width_fe[0])  hidden blocks of the head
    -> ...                           (`hidden_layers`, `RepresentationHook`)
    -> prediction (output_dim)

Outputs (in `output_dir`, default: `<checkpoint dir>/prediction`)
------------------------------------------------------------------
<output_name>.csv                 : one row per tile, `path` then one column
                                    per target (predicted values, in the
                                    target space used for training, e.g.
                                    `log1p` densities)
<output_name>_representations.h5  : with `save_embeddings` and/or
                                    `hidden_layers`:
                                    `img_path` (N,),
                                    `embedding` (N, feature_dim, fp32),
                                    `hidden/<layer name>` (N, width, fp32)

For tile features at scale (WSI coordinates h5), prefer the registered
HISTONAUT encoder `PROiFHESi_H1` with `preprocessing/tile_feat_extraction.py`.

Usage
-----
python -m finetuning.predict --model_path <job_dir>/best_model.pt.tar \
    --img_dir <tiles> [--input_path <table.csv>] [--save_embeddings] \
    [--hidden_layers [<layer name> ...]]

`--hidden_layers` alone records every hidden block of the head; names of
any network submodule (`network.named_modules()`) can be given instead.
"""

from __future__ import annotations

# General libraries
import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

# Data libraries
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Project modules
from finetuning.base_model import configure_logging
from finetuning.dataloader import DatasetHandler
from finetuning.model import PROiFHESi
from finetuning.network import RepresentationHook
from finetuning.utils import load_model_from_path

LOGGER = logging.getLogger(__name__)

EMBEDDING_KEY = "embedding"
HIDDEN_GROUP = "hidden"
IMAGE_KEY = "img_path"
DEFAULT_OUTPUT_NAME = "prediction_table"


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------


def predict(
    model: PROiFHESi,
    dataloader: DataLoader,
    return_embeddings: bool = False,
    hidden_layers: Sequence[str] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Predict every tile of a loader.

    Parameters
    ----------
    model : PROiFHESi
        Trained model.
    dataloader : torch.utils.data.DataLoader
        Loader yielding `image` and `path`.
    return_embeddings : bool, default=False
        Also collect the backbone embeddings.
    hidden_layers : Sequence[str] | None, default=None
        Network submodules whose outputs are recorded with
        `RepresentationHook`, e.g. `model.network.hidden_layer_names()`.
    verbose : bool, default=False
        Display a progress bar.

    Returns
    -------
    dict
        In the loader order:
        `path` (list[str], N), `prediction` (np.ndarray, (N, output_dim)),
        with `return_embeddings`: `embedding` (np.ndarray, (N, feature_dim)),
        with `hidden_layers`: `hidden` ({layer name: np.ndarray, (N, ...)}).

    Raises
    ------
    RuntimeError
        If the loader yields no batch.
    """
    hidden_layers = list(hidden_layers or [])
    paths: list[str] = []
    predictions: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    hidden: dict[str, list[np.ndarray]] = {name: [] for name in hidden_layers}

    hook_context = (
        RepresentationHook(
            model.network,
            hidden_layers,
        )
        if hidden_layers
        else nullcontext()
    )

    with hook_context as hook:
        for batch in tqdm(
            dataloader,
            desc="Prediction",
            unit="batch",
            leave=False,
            disable=not verbose,
        ):
            batch_result = model.predict_batch(
                batch,
                return_embeddings=return_embeddings,
            )

            paths.extend(batch_result["path"])
            predictions.append(batch_result["prediction"])

            if return_embeddings:
                embeddings.append(batch_result["embedding"])

            for name in hidden_layers:
                hidden[name].append(hook.representations[name].numpy())

    if not predictions:
        raise RuntimeError("The prediction DataLoader produced no batch.")

    result: dict[str, Any] = {
        "path": paths,
        "prediction": np.concatenate(
            predictions,
            axis=0,
        ),
    }

    if return_embeddings:
        result["embedding"] = np.concatenate(
            embeddings,
            axis=0,
        )

    if hidden_layers:
        result["hidden"] = {
            name: np.concatenate(
                values,
                axis=0,
            )
            for name, values in hidden.items()
        }

    return result


def load_tile(
    image_path: str | Path,
    transform,
) -> Tensor:
    """
    Load one tile as the dataloader does (PIL RGB -> encoder transform).

    Returns
    -------
    torch.Tensor
        Tile with shape (1, 3, H, W).
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return transform(image).unsqueeze(0)


def predict_image(
    model: PROiFHESi,
    image_path: str | Path,
    return_embeddings: bool = False,
    hidden_layers: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Predict the signature of a single tile.

    Parameters
    ----------
    model : PROiFHESi
        Trained model.
    image_path : str | Path
        Tile image file.
    return_embeddings : bool, default=False
        Also return the backbone embedding.
    hidden_layers : Sequence[str] | None, default=None
        Network submodules whose outputs are recorded (see `predict`).

    Returns
    -------
    dict
        `path`, `prediction` (np.ndarray, (output_dim,)), with
        `return_embeddings`: `embedding` (np.ndarray, (feature_dim,)),
        with `hidden_layers`: `hidden` ({layer name: np.ndarray}).
    """
    hidden_layers = list(hidden_layers or [])
    batch = {
        "image": load_tile(
            image_path,
            model.network.transform,
        ),
        "path": [str(image_path)],
    }

    hook_context = (
        RepresentationHook(
            model.network,
            hidden_layers,
        )
        if hidden_layers
        else nullcontext()
    )

    with hook_context as hook:
        batch_result = model.predict_batch(
            batch,
            return_embeddings=return_embeddings,
        )

    # Unbatch: keep the single tile of every entry.
    result = {key: value[0] for key, value in batch_result.items()}

    if hidden_layers:
        result["hidden"] = {
            name: hook.representations[name][0].numpy() for name in hidden_layers
        }

    return result


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------


def results_to_dataframe(
    result: dict[str, Any],
    target_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert a `predict` result into a tile-level DataFrame.

    Parameters
    ----------
    result : dict
        Output of `predict`.
    target_names : list[str] | None, default=None
        Target names; `target_0`, `target_1`, ... when None.

    Returns
    -------
    pd.DataFrame
        One row per tile: `path`, then one column per target.

    Raises
    ------
    ValueError
        If the number of `target_names` differs from the number of outputs.
    """
    predictions = np.asarray(result["prediction"])
    n_targets = predictions.shape[1]

    if target_names is None:
        target_names = [f"target_{index}" for index in range(n_targets)]

    if len(target_names) != n_targets:
        raise ValueError(
            f"Got {len(target_names)} `target_names` for {n_targets} outputs."
        )

    predictions_table = pd.DataFrame(
        predictions,
        columns=list(target_names),
    )
    predictions_table.insert(
        0,
        "path",
        result["path"],
    )

    return predictions_table


def save_representations(
    result: dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    Write the embeddings and / or hidden representations of a `predict`
    result to h5.

    Datasets: `img_path` (N,), `embedding` (N, feature_dim, fp32) when
    present, and `hidden/<layer name>` (N, ..., fp32) for every recorded
    layer.

    Raises
    ------
    KeyError
        If `result` holds neither `embedding` nor `hidden`.
    """
    if "embedding" not in result and "hidden" not in result:
        raise KeyError(
            "No `embedding` nor `hidden` in the prediction result; call "
            "`predict(..., return_embeddings=True)` or set `hidden_layers`."
        )

    with h5py.File(output_path, "w") as file:
        file.create_dataset(
            IMAGE_KEY,
            data=np.asarray(result["path"], dtype=object),
            dtype=h5py.string_dtype(),
        )

        if "embedding" in result:
            file.create_dataset(
                EMBEDDING_KEY,
                data=result["embedding"].astype(np.float32),
            )

        if "hidden" in result:
            hidden_group = file.create_group(HIDDEN_GROUP)

            for name, values in result["hidden"].items():
                hidden_group.create_dataset(
                    name,
                    data=values.astype(np.float32),
                )


# -----------------------------------------------------------------------------
# Dataset prediction
# -----------------------------------------------------------------------------


def resolve_hidden_layers(
    model: PROiFHESi,
    hidden_layers: Sequence[str] | None,
) -> list[str]:
    """
    Resolve the layers to record.

    None records nothing; an empty sequence means every hidden block of the
    head (`LoRAFactory.hidden_layer_names`); otherwise names are kept.
    """
    if hidden_layers is None:
        return []

    if len(hidden_layers) > 0:
        return list(hidden_layers)

    head_layers = model.network.hidden_layer_names()

    if not head_layers:
        LOGGER.warning(
            "The prediction head has no hidden block (`width_fe` is empty); "
            "no hidden representation is recorded."
        )

    return head_layers


def predict_dataset(
    model_path: str | Path,
    img_dir: str | Path,
    input_path: str | Path | None = None,
    img_format: str | None = None,
    device: str = "cpu",
    output_dir: str | Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
    save_embeddings: bool = False,
    hidden_layers: Sequence[str] | None = None,
    eval_batch_size: int | None = None,
    num_workers: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Load a checkpoint, predict a tile set and write the results.

    Parameters
    ----------
    model_path : str | Path
        Trained checkpoint.
    img_dir : str | Path
        Directory of the tiles.
    input_path : str | Path | None, default=None
        Optional csv with an `img_path` column restricting the tiles;
        every `*.<img_format>` file of `img_dir` when None.
    img_format : str | None, default=None
        Tile extension; the checkpoint value when None.
    device : str, default="cpu"
        PyTorch device.
    output_dir : str | Path | None, default=None
        Output directory; `<checkpoint dir>/prediction` when None.
    output_name : str, default="prediction_table"
        Output file stem.
    save_embeddings : bool, default=False
        Save the backbone embeddings to `<output_name>_representations.h5`.
    hidden_layers : Sequence[str] | None, default=None
        Layers recorded to `<output_name>_representations.h5`: None for
        none, empty for every hidden block of the head, or submodule names.
    eval_batch_size, num_workers : int | None, default=None
        Override the checkpoint DataLoader settings.
    verbose : bool, default=False
        Display a progress bar.

    Returns
    -------
    pd.DataFrame
        Tile-level predictions (see `results_to_dataframe`).

    Raises
    ------
    FileNotFoundError
        If `model_path` or `img_dir` does not exist.
    """
    model_path = Path(model_path)
    img_dir = Path(img_dir)

    if not model_path.is_file():
        raise FileNotFoundError(f"`model_path` not found: {model_path}.")

    if not img_dir.is_dir():
        raise FileNotFoundError(f"`img_dir` not found: {img_dir}.")

    output_dir = (
        model_path.parent / "prediction" if output_dir is None else Path(output_dir)
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_model_from_path(
        model_path,
        device=device,
    )
    hidden_layers = resolve_hidden_layers(
        model,
        hidden_layers,
    )

    # Data arguments of the new tile set; the checkpoint `args` are kept.
    data_args = Namespace(**vars(model.args))
    data_args.img_dir = img_dir
    data_args.input_path = None if input_path is None else Path(input_path)
    data_args.target_path = None

    if img_format is not None:
        data_args.img_format = img_format

    if eval_batch_size is not None:
        data_args.eval_batch_size = eval_batch_size

    if num_workers is not None:
        data_args.num_workers = num_workers

    dataset_handler = DatasetHandler(
        data_args,
        transform=model.network.transform,
        with_target=False,
        predict=True,
    )
    dataloader = dataset_handler.get_loader(training=False)

    LOGGER.info(
        "Predicting %d tiles of %s with %s | embeddings=%s | hidden layers=%s.",
        len(dataloader.dataset),
        img_dir,
        model_path,
        save_embeddings,
        hidden_layers or "none",
    )

    result = predict(
        model,
        dataloader,
        return_embeddings=save_embeddings,
        hidden_layers=hidden_layers,
        verbose=verbose,
    )

    predictions_table = results_to_dataframe(
        result,
        target_names=model.target_names,
    )
    predictions_path = output_dir / f"{output_name}.csv"
    predictions_table.to_csv(
        predictions_path,
        index=False,
    )
    LOGGER.info("Predictions saved to %s.", predictions_path)

    if save_embeddings or hidden_layers:
        representations_path = output_dir / f"{output_name}_representations.h5"
        save_representations(
            result,
            representations_path,
        )
        LOGGER.info("Representations saved to %s.", representations_path)

    return predictions_table


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------


def get_predict_arguments(known_args: Sequence[str] | None = None) -> Namespace:
    """
    Parse the prediction command line. Model settings come from the
    checkpoint `args`.
    """
    parser = ArgumentParser(
        description="Predict proteomic signatures of H&E tiles with PROiFHESi.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Trained checkpoint (best_model.pt.tar).",
    )
    parser.add_argument(
        "--img_dir",
        type=Path,
        required=True,
        help="Directory of the tiles to predict.",
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=None,
        help="Optional csv with an `img_path` column restricting the tiles.",
    )
    parser.add_argument(
        "--img_format",
        type=str,
        default=None,
        help="Tile extension. Defaults to the checkpoint value.",
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
        help="Output directory. Defaults to <checkpoint dir>/prediction.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help="Output file stem.",
    )
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Save the fine-tuned encoder embeddings (h5).",
    )
    parser.add_argument(
        "--hidden_layers",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Record hidden representations (h5). Alone: every hidden block of "
            "the head; or network submodule names, e.g. head.network.0."
        ),
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Prediction batch size. Defaults to the checkpoint value.",
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


def main(known_args: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Command-line entry point; returns the prediction table.
    """
    args = get_predict_arguments(known_args)
    configure_logging(verbose=args.verbose)

    return predict_dataset(
        model_path=args.model_path,
        img_dir=args.img_dir,
        input_path=args.input_path,
        img_format=args.img_format,
        device=args.device,
        output_dir=args.output_dir,
        output_name=args.output_name,
        save_embeddings=args.save_embeddings,
        hidden_layers=args.hidden_layers,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
