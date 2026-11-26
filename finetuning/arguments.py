"""
Command-line and YAML configuration utilities for PROiFHESi training.

The configuration pipeline follows this precedence order:

    parser defaults < YAML configuration < explicit command-line arguments

After parsing, the namespace goes through a chain of `configure_*` and
`validate_*` functions. Each one takes `args`, returns `args`, and resolves
dependent defaults (encoder feature dimension, scheduler patience, reference
metric direction, ...) or checks consistency. Errors become `parser.error`.

Supported options
-----------------
task         : regression
criterion    : mse, rmse, mae, l1, huber (`huber_delta`),
               hierarchy (MSE + `hierarchy_alpha` * hierarchy penalty)
optimizer    : adamw, sgd
lr_scheduler : none, cosine (warm restarts), plateau (ReduceLROnPlateau)
ref_metric   : mean_val_loss, mean_pearson, mean_cossim, mean_euclidsim,
               mean_r2 (coefficient of determination),
               mean_ccc (concordance correlation coefficient),
               or per target: pearson_<target>, r2_<target>, ccc_<target>
"""

from __future__ import annotations

# General libraries
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    BooleanOptionalAction,
    Namespace,
)
from pathlib import Path
from typing import Any, Sequence

# Data libraries
import yaml

# HISTONAUT
from histonaut.encoder.factory import fetch_embeddings_dim, get_encoder_mapping

SUPPORTED_ENCODER = set(get_encoder_mapping().keys())

# Project modules
from finetuning.dataloader import SUPPORTED_TASKS


SUPPORTED_CRITERIA = {
    "mse",
    "rmse",
    "mae",
    "l1",
    "huber",
    "hierarchy",
}

SUPPORTED_OPTIMIZERS = {
    "adamw",
    "sgd",
}

SUPPORTED_SCHEDULERS = {
    "none",
    "cosine",
    "plateau",
}

# Validation metrics computed by `PROiFHESi._compute_metrics`, with their
# optimisation direction.
REFERENCE_METRICS = {
    "mean_val_loss": "min",
    "mean_pearson": "max",
    "mean_cossim": "max",
    "mean_euclidsim": "max",
    "mean_r2": "max",
    "mean_ccc": "max",
}

# Per-target metrics, named `<prefix><target>` (e.g. `pearson_CD8a+`).
PER_TARGET_METRICS = {
    "pearson_": "max",
    "r2_": "max",
    "ccc_": "max",
}

REQUIRED_ARGUMENTS = (
    "img_dir",
    "input_path",
    "target_path",
    "job_dir",
)

AUGMENTATION_PROBABILITIES = (
    "p_hflip",
    "p_vflip",
    "p_rotate90",
    "p_hed",
    "p_brightness_contrast",
    "p_blur",
    "p_sharpen",
    "p_noise",
    "p_compression",
)

PATH_ARGUMENTS = (
    "config",
    "img_dir",
    "input_path",
    "target_path",
    "job_dir",
    "model_path",
)


# -----------------------------------------------------------------------------
# Configuration I/O
# -----------------------------------------------------------------------------


def load_yaml_config(config_path: str | Path | None) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str | Path | None
        Path to the YAML file. None returns an empty configuration.

    Returns
    -------
    dict[str, Any]
        Mapping of argument names to values.

    Raises
    ------
    FileNotFoundError
        If `config_path` does not exist.
    TypeError
        If the YAML content is not a mapping.
    """
    if config_path is None:
        return {}

    config_path = Path(config_path).expanduser()

    if not config_path.is_file():
        raise FileNotFoundError(f"`config` file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise TypeError(
            f"`config` must contain a mapping of argument names to values, "
            f"got {type(config).__name__}."
        )

    return config


def validate_config_keys(
    parser: ArgumentParser,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Reject YAML keys that are not parser arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Complete parser (see `build_parser`).
    config : dict[str, Any]
        YAML configuration.

    Returns
    -------
    dict[str, Any]
        Unmodified configuration.

    Raises
    ------
    ValueError
        If the configuration contains unknown keys, or `config` itself.
    """
    valid_keys = {
        action.dest
        for action in parser._actions
        if action.dest not in {"help", "config"}
    }
    unknown_keys = set(config) - valid_keys

    if unknown_keys:
        raise ValueError(
            f"Unknown configuration arguments: {sorted(unknown_keys)}. "
            f"`config` cannot be set from a YAML file."
        )

    return config


def _make_yaml_serializable(value: Any) -> Any:
    """
    Recursively convert configuration values into YAML-safe objects.
    """
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {key: _make_yaml_serializable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_make_yaml_serializable(item) for item in value]

    return value


def write_config(args: Namespace) -> Namespace:
    """
    Write the fully resolved configuration to `job_dir/config.yaml`.

    Parameters
    ----------
    args : argparse.Namespace
        Resolved configuration.

    Returns
    -------
    argparse.Namespace
        Unmodified configuration.
    """
    job_dir = Path(args.job_dir)
    job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    config = {key: _make_yaml_serializable(value) for key, value in vars(args).items()}

    with (job_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            default_flow_style=False,
        )

    return args


# -----------------------------------------------------------------------------
# Argument configuration
# -----------------------------------------------------------------------------


def normalize_path_arguments(args: Namespace) -> Namespace:
    """
    Cast path arguments to expanded `pathlib.Path` objects.
    """
    for name in PATH_ARGUMENTS:
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(value).expanduser())

    return args


def configure_training_mode(
    args: Namespace,
    train: bool,
) -> Namespace:
    """
    Store the training mode and resolve early-stopping defaults.

    `patience=None` disables early stopping (patience = `epochs`).
    """
    args.train = train

    if args.patience is None:
        args.patience = args.epochs

    if args.minimum_epochs is None:
        args.minimum_epochs = 0

    return args


def configure_criterion(args: Namespace) -> Namespace:
    """
    Clear loss parameters that do not apply to the selected criterion.
    """
    if args.criterion != "huber":
        args.huber_delta = None

    if args.criterion != "hierarchy":
        args.hierarchy_alpha = None

    return args


def configure_data_loading(args: Namespace) -> Namespace:
    """
    Resolve evaluation batch size and validation-split parameters.

    `eval_batch_size` defaults to `batch_size`. `val_fraction` is cleared
    when no validation subset is used.
    """
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    if not args.use_val:
        args.val_fraction = None

    return args


def configure_augmentations(args: Namespace) -> Namespace:
    """
    Clear augmentation probabilities when augmentations are disabled.
    """
    if not args.augment:
        for name in AUGMENTATION_PROBABILITIES:
            setattr(args, name, None)

    return args


def configure_lr_scheduler(args: Namespace) -> Namespace:
    """
    Resolve scheduler-dependent parameters.

    `patience_lr` is only used by the plateau scheduler; it defaults to
    half of the early-stopping patience.
    """
    if args.lr_scheduler != "plateau":
        args.patience_lr = None

    elif args.patience_lr is None:
        args.patience_lr = max(
            1,
            args.patience // 2,
        )

    return args


def configure_lora(args: Namespace) -> Namespace:
    """
    Clear LoRA parameters when LoRA is disabled.
    """
    if not args.lora:
        args.lora_r = None
        args.lora_a = None
        args.lora_target = []

    return args


def configure_feature_dim(args: Namespace) -> Namespace:
    """
    Resolve `feature_dim` from the encoder when it is not given.

    Uses `histonaut.encoder.factory.fetch_embeddings_dim`.

    Raises
    ------
    ValueError
        If `feature_dim` is unknown for `encoder` and not given.
    """
    if args.feature_dim is not None:
        return args

    args.feature_dim = fetch_embeddings_dim(args.encoder)

    if args.feature_dim is None:
        raise ValueError(
            f"Unknown embedding dimension for `encoder={args.encoder}`. "
            f"Set `feature_dim` explicitly."
        )

    return args


def configure_hidden_layers(args: Namespace) -> Namespace:
    """
    Normalise `width_fe` to a list of hidden-layer widths.

    An empty list gives a single linear layer `feature_dim -> output_dim`.
    """
    args.width_fe = list(args.width_fe or [])

    return args


def configure_reference_metric(args: Namespace) -> Namespace:
    """
    Resolve the reference metric and its optimisation direction.

    Sets `metric_mode` ("min" / "max") when not given, and `sgn_metric`
    (-1 / +1) used by `PROiFHESi._keep_best_metrics` and early stopping.

    Raises
    ------
    ValueError
        If `ref_metric` is not supported.
    """
    default_mode = REFERENCE_METRICS.get(args.ref_metric)

    if default_mode is None:
        default_mode = next(
            (
                mode
                for prefix, mode in PER_TARGET_METRICS.items()
                if args.ref_metric.startswith(prefix)
            ),
            None,
        )

    if default_mode is None:
        raise ValueError(
            f"Unsupported `ref_metric={args.ref_metric}`. Expected one of "
            f"{sorted(REFERENCE_METRICS)} or "
            f"{[f'{prefix}<target>' for prefix in PER_TARGET_METRICS]}."
        )

    if args.metric_mode is None:
        args.metric_mode = default_mode

    args.sgn_metric = 1 if args.metric_mode == "max" else -1

    return args


# -----------------------------------------------------------------------------
# Argument validation
# -----------------------------------------------------------------------------


def validate_required_arguments(
    args: Namespace,
    train: bool,
) -> Namespace:
    """
    Check that required arguments are set after merging YAML and CLI.

    Raises
    ------
    ValueError
        If any required argument is missing.
    """
    required_arguments = list(REQUIRED_ARGUMENTS)

    if not train:
        required_arguments.append("model_path")

    missing_arguments = [
        name for name in required_arguments if getattr(args, name, None) is None
    ]

    if missing_arguments:
        raise ValueError(
            "Missing required arguments after resolving YAML and CLI values: "
            + ", ".join(f"`--{name}`" for name in missing_arguments)
        )

    return args


def validate_cross_validation(args: Namespace) -> Namespace:
    """
    Validate fold and repetition indices.

    Raises
    ------
    ValueError
        If `test_fold` / `repeat` are out of range, or set without
        `n_folds` / `n_reps`.
    """
    if args.n_folds is not None:
        if args.n_folds < 2:
            raise ValueError(f"`n_folds` must be at least 2, got {args.n_folds}.")

        if not 0 <= args.test_fold < args.n_folds:
            raise ValueError(
                f"`test_fold` must be in [0, {args.n_folds - 1}], "
                f"got {args.test_fold}."
            )

    elif args.test_fold != 0:
        raise ValueError("`test_fold` was provided without setting `n_folds`.")

    if args.n_reps is not None:
        if args.n_reps < 1:
            raise ValueError(f"`n_reps` must be at least 1, got {args.n_reps}.")

        if not 0 <= args.repeat < args.n_reps:
            raise ValueError(
                f"`repeat` must be in [0, {args.n_reps - 1}], got {args.repeat}."
            )

    elif args.repeat != 0:
        raise ValueError("`repeat` was provided without setting `n_reps`.")

    return args


def validate_training_parameters(args: Namespace) -> Namespace:
    """
    Check value ranges of optimisation, loss, data-loading and head
    parameters.

    Raises
    ------
    ValueError
        If a parameter is out of range.
    """
    positive_arguments = (
        "epochs",
        "batch_size",
        "eval_batch_size",
        "lr",
        "feature_dim",
        "output_dim",
        "img_size",
    )

    for name in positive_arguments:
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"`{name}` must be greater than 0, got {value}.")

    if args.huber_delta is not None and args.huber_delta <= 0:
        raise ValueError(
            f"`huber_delta` must be greater than 0, got {args.huber_delta}."
        )

    if args.hierarchy_alpha is not None and args.hierarchy_alpha < 0:
        raise ValueError(
            f"`hierarchy_alpha` cannot be negative, got {args.hierarchy_alpha}."
        )

    if args.val_fraction is not None and not 0.0 < args.val_fraction < 1.0:
        raise ValueError(
            f"`val_fraction` must be in (0, 1), got {args.val_fraction}."
        )

    for name in AUGMENTATION_PROBABILITIES:
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"`{name}` must be in [0, 1], got {value}.")

    if not 0.0 <= args.hed_sigma_alpha < 1.0:
        raise ValueError(
            f"`hed_sigma_alpha` must be in [0, 1), got {args.hed_sigma_alpha}."
        )

    if args.hed_sigma_beta < 0.0 or args.hed_od_threshold < 0.0:
        raise ValueError(
            "`hed_sigma_beta` and `hed_od_threshold` cannot be negative, got "
            f"{args.hed_sigma_beta} and {args.hed_od_threshold}."
        )

    if not 0.0 <= args.dropout < 1.0:
        raise ValueError(f"`dropout` must be in [0, 1), got {args.dropout}.")

    if any(width <= 0 for width in args.width_fe):
        raise ValueError(
            f"`width_fe` values must be positive, got {args.width_fe}."
        )

    if args.num_workers < 0:
        raise ValueError(
            f"`num_workers` cannot be negative, got {args.num_workers}."
        )

    if args.persistent_workers and args.num_workers == 0:
        raise ValueError("`persistent_workers` requires `num_workers` > 0.")

    if args.log_steps < 0:
        raise ValueError(
            f"`log_steps` cannot be negative, got {args.log_steps}."
        )

    if args.lora and args.lora_r <= 0:
        raise ValueError(f"`lora_r` must be greater than 0, got {args.lora_r}.")

    return args


# =============================================================================
# Parser construction
# =============================================================================


def build_parser(train: bool = True) -> ArgumentParser:
    """
    Build the complete command-line parser.

    Parameters
    ----------
    train : bool, default=True
        If False, adds `--model_path` (inference / test).

    Returns
    -------
    argparse.ArgumentParser
        Parser with every PROiFHESi argument.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "--write_config",
        action="store_true",
        help="Write the resolved configuration to job_dir/config.yaml.",
    )

    # -------------------------------------------------------------------------
    # Data paths
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--img_dir",
        type=Path,
        default=None,
        help="Directory containing the H&E tiles used as inputs.",
    )
    parser.add_argument(
        "--img_format",
        type=str,
        default="jpeg",
        help="File extension of the H&E tiles.",
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=None,
        help="CSV table with an `img_path` column and a `test` fold column.",
    )
    parser.add_argument(
        "--target_path",
        type=Path,
        default=None,
        help="HDF5 file containing the per-tile proteomic signatures.",
    )
    parser.add_argument(
        "--job_dir",
        type=Path,
        default=None,
        help="Directory used to store outputs.",
    )

    # -------------------------------------------------------------------------
    # Task
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--task",
        type=str,
        choices=sorted(SUPPORTED_TASKS),
        default="regression",
        help="Prediction task.",
    )

    # -------------------------------------------------------------------------
    # Runtime
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device, for example cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=29,
        help="Random seed used for splitting and sampling.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--pin_memory",
        action=BooleanOptionalAction,
        default=False,
        help="Use pinned CPU memory in DataLoaders.",
    )
    parser.add_argument(
        "--persistent_workers",
        action=BooleanOptionalAction,
        default=False,
        help="Keep DataLoader workers alive between epochs.",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=100,
        help="W&B training-loss logging frequency in batches (0 disables it).",
    )

    # -------------------------------------------------------------------------
    # Cross-validation
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--n_folds",
        type=int,
        default=None,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--test_fold",
        type=int,
        default=0,
        help="Index of the fold used as the test set.",
    )
    parser.add_argument(
        "--n_reps",
        type=int,
        default=None,
        help="Number of repetitions per fold.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Repetition index used to identify the run.",
    )

    # -------------------------------------------------------------------------
    # DataLoader and validation split
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Validation / test / prediction batch size. Defaults to batch_size.",
    )
    parser.add_argument(
        "--use_val",
        action=BooleanOptionalAction,
        default=True,
        help="Create a validation subset from the training folds.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Fraction of training tiles assigned to validation.",
    )

    # -------------------------------------------------------------------------
    # Image augmentations (training only)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--augment",
        action=BooleanOptionalAction,
        default=True,
        help="Apply image augmentations to training tiles.",
    )
    parser.add_argument(
        "--p_hflip",
        type=float,
        default=0.5,
        help="Probability of a horizontal flip.",
    )
    parser.add_argument(
        "--p_vflip",
        type=float,
        default=0.5,
        help="Probability of a vertical flip.",
    )
    parser.add_argument(
        "--p_rotate90",
        type=float,
        default=0.5,
        help="Probability of a random 90 degree rotation.",
    )
    parser.add_argument(
        "--p_hed",
        type=float,
        default=0.5,
        help="Probability of an HED stain jitter (0 disables it).",
    )
    parser.add_argument(
        "--hed_sigma_alpha",
        type=float,
        default=0.2,
        help="HED jitter: H and E amounts scaled by U(1 - s, 1 + s).",
    )
    parser.add_argument(
        "--hed_sigma_beta",
        type=float,
        default=0.05,
        help="HED jitter: H and E concentrations shifted by U(-s, s).",
    )
    parser.add_argument(
        "--hed_od_threshold",
        type=float,
        default=0.15,
        help="HED jitter: pixels with max optical density below it are left "
        "unchanged (background).",
    )
    parser.add_argument(
        "--p_brightness_contrast",
        type=float,
        default=0.5,
        help="Probability of a brightness-contrast jitter.",
    )
    parser.add_argument(
        "--p_blur",
        type=float,
        default=0.1,
        help="Probability of a Gaussian blur.",
    )
    parser.add_argument(
        "--p_noise",
        type=float,
        default=0.1,
        help="Probability of Gaussian noise (std about 3 to 7 grey levels).",
    )
    parser.add_argument(
        "--p_sharpen",
        type=float,
        default=0.1,
        help="Probability of a sharpening (mirror of the Gaussian blur).",
    )
    parser.add_argument(
        "--p_compression",
        type=float,
        default=0.1,
        help="Probability of a JPEG re-compression (quality 60 to 95).",
    )

    # -------------------------------------------------------------------------
    # Encoder
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--encoder",
        type=str,
        default="hoptimus1",
        choices=sorted(SUPPORTED_ENCODER),
        help="Patch encoder name (histonaut `patch_encoder_factory`).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Center-crop size when the encoder has no eval transforms.",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=None,
        help="Encoder output dimension. Inferred from `encoder` when not set.",
    )

    # -------------------------------------------------------------------------
    # LoRA
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--lora",
        action=BooleanOptionalAction,
        default=False,
        help="Fine-tune the encoder with LoRA (frozen encoder otherwise).",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_a",
        type=float,
        default=1.0,
        help="LoRA scaling factor alpha.",
    )
    parser.add_argument(
        "--lora_target",
        type=str,
        nargs="*",
        default=None,
        help="Encoder modules targeted by LoRA.",
    )

    # -------------------------------------------------------------------------
    # Regression head
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--width_fe",
        type=int,
        nargs="*",
        default=None,
        help="Hidden widths of the prediction MLP (empty: single linear layer).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout probability in the prediction head.",
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=17,
        help="Number of predicted targets (proteomic signature length).",
    )

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--criterion",
        type=str,
        choices=sorted(SUPPORTED_CRITERIA),
        default="mse",
        help="Training loss.",
    )
    parser.add_argument(
        "--huber_delta",
        type=float,
        default=1.0,
        help="Huber loss threshold (criterion=huber).",
    )
    parser.add_argument(
        "--hierarchy_alpha",
        type=float,
        default=1.0,
        help=(
            "Weight of the hierarchy penalty: loss = MSE + alpha * penalty "
            "(criterion=hierarchy)."
        ),
    )

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=sorted(SUPPORTED_OPTIMIZERS),
        default="adamw",
        help="Optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Optimizer weight decay.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum.",
    )
    parser.add_argument(
        "--gradient_clip_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm; use null in YAML to disable clipping.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs.",
    )

    # -------------------------------------------------------------------------
    # Learning-rate scheduler
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--lr_warm_up",
        type=int,
        default=0,
        help="Number of warm-up epochs before the scheduler is turned on.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=sorted(SUPPORTED_SCHEDULERS),
        default="cosine",
        help="Learning-rate scheduler.",
    )
    parser.add_argument(
        "--patience_lr",
        type=int,
        default=None,
        help="Plateau scheduler patience. Defaults to patience // 2.",
    )
    parser.add_argument(
        "--restart_period",
        type=int,
        default=1,
        help="Initial restart period (T_0) for cosine warm restarts.",
    )
    parser.add_argument(
        "--restart_multiplier",
        type=int,
        default=2,
        help="Cosine restart-period multiplier (T_mult).",
    )

    # -------------------------------------------------------------------------
    # Model selection and early stopping
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--ref_metric",
        type=str,
        default="mean_val_loss",
        help=(
            f"Validation metric used for best-model selection: "
            f"{sorted(REFERENCE_METRICS)}, or per target "
            f"{[f'{prefix}<target>' for prefix in PER_TARGET_METRICS]}."
        ),
    )
    parser.add_argument(
        "--metric_mode",
        type=str,
        choices=["min", "max"],
        default=None,
        help="Minimise or maximise `ref_metric`. Inferred when not set.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Non-improving epochs tolerated. None disables early stopping.",
    )
    parser.add_argument(
        "--minimum_epochs",
        type=int,
        default=None,
        help="Minimum number of epochs before early stopping is allowed.",
    )
    parser.add_argument(
        "--minimum_improvement",
        type=float,
        default=0.0,
        help="Minimum improvement of `ref_metric` required to reset patience.",
    )

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    if not train:
        parser.add_argument(
            "--model_path",
            type=Path,
            default=None,
            help="Path to a trained model checkpoint.",
        )
    else:
        parser.set_defaults(model_path=None)

    return parser


# =============================================================================
# Parsing pipeline
# =============================================================================


def get_arguments(
    known_args: Sequence[str] | None = None,
    train: bool = True,
    config: str | Path | None = None,
) -> Namespace:
    """
    Parse command-line and YAML arguments into a resolved namespace.

    Parameters
    ----------
    known_args : Sequence[str] | None
        Argument list to parse. None reads `sys.argv`.
    train : bool, default=True
        Training mode. If False, `--model_path` is added and required.
    config : str | Path | None
        Default YAML configuration, overridden by `--config`.

    Returns
    -------
    argparse.Namespace
        Resolved configuration, passed to `PROiFHESi` and stored in
        checkpoints.

    Notes
    -----
    Precedence: parser defaults < YAML configuration < explicit CLI values.
    """
    # Parse only the configuration path first.
    config_parser = ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=config,
    )
    config_args, _ = config_parser.parse_known_args(known_args)

    parser = build_parser(train=train)

    try:
        yaml_config = load_yaml_config(config_args.config)
        validate_config_keys(parser, yaml_config)

    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))

    # YAML values replace parser defaults; explicit CLI values replace YAML.
    parser.set_defaults(**yaml_config)
    args = parser.parse_args(known_args)
    args.config = config_args.config

    try:
        args = normalize_path_arguments(args)
        args = validate_required_arguments(args, train=train)
        args = configure_training_mode(args, train=train)
        args = configure_criterion(args)
        args = configure_data_loading(args)
        args = configure_augmentations(args)
        args = configure_lr_scheduler(args)
        args = configure_lora(args)
        args = configure_feature_dim(args)
        args = configure_hidden_layers(args)
        args = configure_reference_metric(args)
        args = validate_cross_validation(args)
        args = validate_training_parameters(args)

    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    if args.write_config:
        write_config(args)

    return args
