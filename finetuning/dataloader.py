"""
Data loading utilities for H&E tiles with (optional) per-tile molecular
signatures.

Supported inputs
----------------
img_dir
    Directory of H&E tiles, one image file per tile (`img_format`, jpeg by
    default).

input_path (optional)
    CSV table with one row per tile:
        - `img_path`: tile file name, relative to `img_dir`.
        - `test` (optional): cross-validation fold of the tile.
    Without it, every `*.<img_format>` file of `img_dir` is used
    (prediction on a new image folder).

target_path (optional)
    HDF5 file written by `preprocessing/dataset/*/make_lora_dataset.py`:
        - `signature`: float array (N, F), with attribute `feat` (the F
          target names, e.g. `cells`, `CD45+`, ..., `SMA+`).
        - `img_path`: byte strings (N,), tile file names matching
          `input_path`.

Supported tasks
---------------
regression
    Predict the molecular signature (F continuous targets) of a tile.

Supported augmentations (training only, see `build_augmentations`)
-----------------------
spatial : horizontal / vertical flips, 90 degree rotations
color   : HED stain jitter (optional, `HEDStainAugment`), brightness-contrast,
          Gaussian blur, sharpening, Gaussian noise, JPEG compression
"""

from __future__ import annotations

# General libraries
import logging
import random
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

# Data libraries
import albumentations as A
import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from skimage.color import hed_from_rgb, rgb_from_hed
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader, Dataset, Sampler, SubsetRandomSampler

LOGGER = logging.getLogger(__name__)

# Classification could be added as a pretext task.
SUPPORTED_TASKS = {
    "regression",
}

IMAGE_COLUMN = "img_path"
FOLD_COLUMN = "test"

SIGNATURE_KEY = "signature"
SIGNATURE_NAMES_ATTR = "feat"
SIGNATURE_IMAGE_KEY = "img_path"


# -----------------------------------------------------------------------------
# Augmentations
# -----------------------------------------------------------------------------


class HEDStainAugment(A.ImageOnlyTransform):
    """
    HED stain jitter (Tellez et al., IEEE TMI 2018; MedIA 2019).

    The tile is converted to optical density (OD), projected on the fixed
    Ruifrok & Johnston H / E / DAB stain vectors, the H and E concentrations
    are perturbed, and the tile is reconstructed:

        C'_s = alpha_s * C_s + beta_s,  s in {H, E}
        alpha_s ~ U(1 - sigma_alpha, 1 + sigma_alpha)
        beta_s  ~ U(-sigma_beta, sigma_beta)

    Parameters
    ----------
    sigma_alpha : float
        Half-width of the multiplicative (stain amount) jitter.
    sigma_beta : float
        Half-width of the additive jitter on concentrations. 0 disables it.
    od_threshold : float
        Pixels whose maximum OD over RGB channels is below this value are
        treated as background and left unchanged. 0 perturbs every pixel.
    p : float
        Probability of applying the transform.
    always_apply : bool | None
        Albumentations 1.4 argument, kept for API compatibility.

    Notes
    -----
    The DAB channel (residual on H&E) is never perturbed: jittering it gives
    brownish artefacts. Leaving the background unchanged and `sigma_beta=0`
    avoid tinting the glass. With `alpha = 1` and `beta = 0` the transform is
    the identity (up to uint8 rounding).
    """

    STAIN_TO_OD = rgb_from_hed.astype(np.float32)
    OD_TO_STAIN = hed_from_rgb.astype(np.float32)
    PERTURBED_STAINS = 2  # H and E

    def __init__(
        self,
        sigma_alpha: float = 0.05,
        sigma_beta: float = 0.0,
        od_threshold: float = 0.15,
        always_apply: bool | None = None,
        p: float = 0.5,
    ) -> None:
        super().__init__(
            always_apply=always_apply,
            p=p,
        )
        if not 0.0 <= sigma_alpha < 1.0:
            raise ValueError(
                f"`sigma_alpha` must be in [0, 1), got {sigma_alpha}."
            )
        if sigma_beta < 0.0:
            raise ValueError(f"`sigma_beta` cannot be negative, got {sigma_beta}.")
        if od_threshold < 0.0:
            raise ValueError(
                f"`od_threshold` cannot be negative, got {od_threshold}."
            )
        self.sigma_alpha = sigma_alpha
        self.sigma_beta = sigma_beta
        self.od_threshold = od_threshold

    def get_params(self) -> dict[str, Any]:
        alpha = [
            random.uniform(
                1.0 - self.sigma_alpha,
                1.0 + self.sigma_alpha,
            )
            for _ in range(self.PERTURBED_STAINS)
        ]
        beta = [
            random.uniform(
                -self.sigma_beta,
                self.sigma_beta,
            )
            for _ in range(self.PERTURBED_STAINS)
        ]
        return {
            "alpha": np.asarray(alpha, dtype=np.float32),
            "beta": np.asarray(beta, dtype=np.float32),
        }

    def apply(
        self,
        img: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(
                "`HEDStainAugment` expects uint8 RGB images (H, W, 3), "
                f"got {img.dtype} {img.shape}."
            )
        shape = img.shape
        pixels = img.reshape(-1, 3).astype(np.float32)
        optical_density = -np.log((pixels + 1.0) / 256.0)
        concentrations = optical_density @ self.OD_TO_STAIN

        tissue = optical_density.max(axis=1) > self.od_threshold
        n_stains = self.PERTURBED_STAINS
        concentrations[tissue, :n_stains] = (
            concentrations[tissue, :n_stains] * alpha + beta
        )

        optical_density = concentrations @ self.STAIN_TO_OD
        image = 256.0 * np.exp(-optical_density) - 1.0
        return np.clip(np.rint(image), 0, 255).astype(np.uint8).reshape(shape)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "sigma_alpha",
            "sigma_beta",
            "od_threshold",
        )


def build_augmentations(
    args: Namespace,
    training: bool,
) -> A.Compose | None:
    """
    Build the training augmentation pipeline (applied on uint8 RGB arrays).

    Parameters
    ----------
    args : argparse.Namespace
        Resolved configuration. Uses `augment` and the probabilities
        `p_hflip`, `p_vflip`, `p_rotate90`, `p_hed`, `p_brightness_contrast`,
        `p_blur`, `p_sharpen`, `p_noise`, `p_compression`. `p_hed` also reads
        `hed_sigma_alpha`, `hed_sigma_beta` and `hed_od_threshold`.
        `p_hed`, `p_sharpen` and `p_compression` default to 0 when missing
        (configurations older than these options).
    training : bool
        Augmentations are only built for training datasets.

    Returns
    -------
    albumentations.Compose | None
        Spatial then color augmentations, or None when disabled.

    Notes
    -----
    The HED stain jitter runs before the other color transforms, on the
    unaltered tile colors. Blur and sharpening mirror each other (scanner
    focus / post-processing). `GaussNoise(var_limit=(10, 50))` uses the
    albumentations 1.4 API (variance in grey levels, std about 3 to 7).
    JPEG compression runs last, as in an acquisition pipeline.
    """
    if not training or not args.augment:
        return None

    p_hed = getattr(args, "p_hed", 0.0) or 0.0
    stain_augmentations = []
    if p_hed > 0.0:
        stain_augmentations.append(
            HEDStainAugment(
                sigma_alpha=getattr(args, "hed_sigma_alpha", 0.05),
                sigma_beta=getattr(args, "hed_sigma_beta", 0.0),
                od_threshold=getattr(args, "hed_od_threshold", 0.15),
                p=p_hed,
            )
        )

    return A.Compose(
        [
            # Spatial
            A.HorizontalFlip(p=args.p_hflip),
            A.VerticalFlip(p=args.p_vflip),
            A.RandomRotate90(p=args.p_rotate90),
            # Color
            *stain_augmentations,
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=args.p_brightness_contrast,
            ),
            A.GaussianBlur(
                blur_limit=(7, 7),
                sigma_limit=(0.1, 1.5),
                p=args.p_blur,
            ),
            A.Sharpen(
                alpha=(0.1, 0.3),
                lightness=(1.0, 1.0),
                p=getattr(args, "p_sharpen", 0.0) or 0.0,
            ),
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=args.p_noise,
            ),
            A.ImageCompression(
                quality_range=(60, 95),
                compression_type="jpeg",
                p=getattr(args, "p_compression", 0.0) or 0.0,
            ),
        ]
    )


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class HETileDataset(Dataset):
    """
    Dataset of H&E tiles, with or without their molecular signature.

    Parameters
    ----------
    args : argparse.Namespace
        Runtime configuration.

        Required attributes:

        img_dir
            Directory containing the H&E tiles.

        img_format
            Tile file extension.

        img_size
            Center-crop size of the default transform.

        test_fold
            Fold reserved for testing when `input_path` has a `test` column.

        Optional attributes:

        input_path
            Tile table (see module docstring). None lists `img_dir`.

        target_path
            Signature HDF5 file. Required when `with_target=True`.

        output_dim
            Expected signature length, checked against the HDF5 file.

        augment, p_*
            Training augmentations (see `build_augmentations`).

    use_train : bool
        If True, retain tiles outside `test_fold`. If False, retain tiles
        of `test_fold`.
    transform : Callable | None, default=None
        PIL image -> tensor transform, usually the encoder `eval_transforms`.
        Defaults to `CenterCrop(img_size)` + `ToTensor()`.
    with_target : bool, default=True
        If True, load the signatures and drop tiles without one.
    predict : bool, default=False
        If True, ignore test-fold filtering and retain all tiles.
    augment : bool | None, default=None
        Apply training augmentations. Defaults to `use_train`; set False
        for a validation dataset built on the training folds.

    Raises
    ------
    FileNotFoundError
        If `img_dir`, `input_path` or `target_path` does not exist.
    KeyError
        If `input_path` has no `img_path` column.
    ValueError
        If `with_target=True` without `target_path`, or if `output_dim`
        does not match the number of signature targets.
    RuntimeError
        If no tile is retained.
    """

    def __init__(
        self,
        args: Namespace,
        use_train: bool,
        transform: Callable | None = None,
        with_target: bool = True,
        predict: bool = False,
        augment: bool | None = None,
    ) -> None:
        super().__init__()

        self.args = args
        self.img_dir = Path(args.img_dir)
        self.img_format = str(args.img_format).lstrip(".")
        self.use_train = bool(use_train)
        self.with_target = bool(with_target)
        self.predict = bool(predict)

        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"`img_dir` does not exist: {self.img_dir}")

        self.transform = transform or T.Compose(
            [
                T.CenterCrop(args.img_size),
                T.ToTensor(),
            ]
        )
        self.augmentations = build_augmentations(
            args,
            training=self.use_train if augment is None else bool(augment),
        )

        self.input_table = self._read_input_table()
        self.target_names: list[str] = []
        signature_lookup: dict[str, np.ndarray] | None = None

        if self.with_target:
            self.target_names, signature_lookup = self._read_signatures()

        self.files, self.targets = self._make_db(signature_lookup)

    def __len__(self) -> int:
        """Return the number of tiles in the dataset."""
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Load one tile and, when available, its signature.

        Parameters
        ----------
        idx : int
            Dataset index.

        Returns
        -------
        sample : dict
            image
                Float tensor (C, H, W) after augmentation and `transform`.
            path
                Tile file path, as a string.
            target
                Float tensor (F,), only when `with_target=True`.
        """
        path = self.files[idx]
        sample = {
            "image": self._load_image(path),
            "path": str(path),
        }

        if self.targets is not None:
            sample["target"] = torch.from_numpy(self.targets[idx])

        return sample

    @property
    def input_files(self) -> list[Path]:
        """Tile paths (alias of `files`, used by test / predict)."""
        return self.files

    @property
    def output_dim(self) -> int:
        """Number of signature targets (0 without targets)."""
        return len(self.target_names)

    def _load_image(self, path: Path) -> torch.Tensor:
        """
        Read a tile as RGB, augment it, then apply `transform`.
        """
        image = np.asarray(Image.open(path).convert("RGB"))

        if self.augmentations is not None:
            image = self.augmentations(image=image)["image"]

        return self.transform(Image.fromarray(image))

    def _read_input_table(self) -> pd.DataFrame | None:
        """
        Read the tile table, or return None when `input_path` is not set.
        """
        input_path = getattr(self.args, "input_path", None)

        if input_path is None:
            return None

        input_path = Path(input_path)

        if not input_path.is_file():
            raise FileNotFoundError(f"`input_path` does not exist: {input_path}")

        table = pd.read_csv(input_path)

        if IMAGE_COLUMN not in table.columns:
            raise KeyError(
                f"`input_path` must contain a `{IMAGE_COLUMN}` column, "
                f"got {list(table.columns)}."
            )

        return table

    def _read_signatures(self) -> tuple[list[str], dict[str, np.ndarray]]:
        """
        Read target names and signatures from the HDF5 target file.

        Returns
        -------
        target_names : list[str]
            Names of the F signature targets.
        signature_lookup : dict[str, numpy.ndarray]
            Mapping from tile file name to its float32 signature (F,).
        """
        target_path = getattr(self.args, "target_path", None)

        if target_path is None:
            raise ValueError("`target_path` is required when `with_target=True`.")

        target_path = Path(target_path)

        if not target_path.is_file():
            raise FileNotFoundError(f"`target_path` does not exist: {target_path}")

        with h5py.File(target_path, "r") as file:
            signatures = file[SIGNATURE_KEY][:].astype(np.float32)
            target_names = [
                _decode(name)
                for name in file[SIGNATURE_KEY].attrs[SIGNATURE_NAMES_ATTR]
            ]
            image_names = [_decode(name) for name in file[SIGNATURE_IMAGE_KEY][:]]

        expected_dim = getattr(self.args, "output_dim", None)

        if expected_dim is not None and expected_dim != len(target_names):
            raise ValueError(
                f"`output_dim={expected_dim}` does not match the "
                f"{len(target_names)} targets of `target_path`: {target_names}."
            )

        signature_lookup = dict(zip(image_names, signatures))

        return target_names, signature_lookup

    def _make_db(
        self,
        signature_lookup: dict[str, np.ndarray] | None,
    ) -> tuple[list[Path], np.ndarray | None]:
        """
        Select the tiles of the requested partition and match their targets.

        Parameters
        ----------
        signature_lookup : dict[str, numpy.ndarray] | None
            Tile name -> signature. None for an image-only dataset.

        Returns
        -------
        files : list[pathlib.Path]
            Retained tile paths.
        targets : numpy.ndarray | None
            Signatures (len(files), F) aligned with `files`, or None.

        Raises
        ------
        RuntimeError
            If no tile is retained.
        """
        available = {path.name for path in self.img_dir.glob(f"*.{self.img_format}")}

        if self.input_table is None:
            rows = [{IMAGE_COLUMN: name} for name in sorted(available)]
        else:
            rows = self.input_table.to_dict("records")

        files: list[Path] = []
        targets: list[np.ndarray] = []
        n_missing_image = 0
        n_missing_target = 0

        for row in rows:
            name = str(row[IMAGE_COLUMN])

            if not self._is_in_db(row):
                continue

            if name not in available:
                n_missing_image += 1
                continue

            if signature_lookup is not None:
                if name not in signature_lookup:
                    n_missing_target += 1
                    continue
                targets.append(signature_lookup[name])

            files.append(self.img_dir / name)

        if n_missing_image or n_missing_target:
            LOGGER.warning(
                "Skipped %d tiles without image in `img_dir` and %d tiles "
                "without signature in `target_path`.",
                n_missing_image,
                n_missing_target,
            )

        if not files:
            raise RuntimeError(
                f"No tile retained from `img_dir={self.img_dir}` "
                f"(use_train={self.use_train}, predict={self.predict})."
            )

        LOGGER.info(
            "Constructed %s dataset with %d tiles.",
            "training" if self.use_train else "evaluation",
            len(files),
        )

        return files, (np.stack(targets) if signature_lookup is not None else None)

    def _is_in_db(self, row: dict[str, Any]) -> bool:
        """
        Determine whether a tile belongs to this partition.

        In prediction mode, or without a fold column, every tile is
        retained. Otherwise `test_fold` goes to the evaluation dataset only.
        """
        if self.predict or FOLD_COLUMN not in row:
            return True

        is_test = row[FOLD_COLUMN] == self.args.test_fold

        return not is_test if self.use_train else is_test


def _decode(value: bytes | str) -> str:
    """Decode HDF5 byte strings."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


# -----------------------------------------------------------------------------
# Samplers
# -----------------------------------------------------------------------------


class SequentialSubsetSampler(Sampler[int]):
    """
    Yield a predefined subset of dataset indices in deterministic order.

    Unlike `SubsetRandomSampler`, indices are not shuffled, so validation
    predictions keep a stable order across epochs.

    Parameters
    ----------
    indices : Sequence[int]
        Dataset indices to yield.
    """

    def __init__(
        self,
        indices: Sequence[int],
    ) -> None:
        self.indices = [int(index) for index in indices]

    def __iter__(self) -> Iterator[int]:
        """Yield the stored indices in their original order."""
        return iter(self.indices)

    def __len__(self) -> int:
        """Return the number of stored indices."""
        return len(self.indices)


# -----------------------------------------------------------------------------
# Dataset handler
# -----------------------------------------------------------------------------


class DatasetHandler:
    """
    Build datasets, samplers and data loaders for training, test and
    prediction.

    Training mode (`predict=False`):

    - `dataset_train`: tiles outside `test_fold`, with augmentations.
    - `dataset_val`: the same tiles, without augmentation (if `use_val`).
      Train and validation indices come from one `ShuffleSplit` of these
      tiles (`val_fraction`, `seed`).
    - `dataset_test`: tiles of `test_fold`, without augmentation.

    Prediction mode (`predict=True`): one `dataset_predict` with every tile,
    without augmentation.

    Parameters
    ----------
    args : argparse.Namespace
        Data-loading configuration.

        Required attributes:

        batch_size
            Training batch size.

        eval_batch_size
            Batch size of the validation, test and prediction loaders.

        use_val
            Reserve a validation subset of the training folds.

        val_fraction
            Fraction of training tiles assigned to validation.

        seed
            Seed of the train / validation split and of the train shuffling.

        num_workers, pin_memory, persistent_workers
            DataLoader options.

        Plus the `HETileDataset` attributes.

    transform : Callable | None, default=None
        PIL image -> tensor transform, usually the encoder `eval_transforms`.
    with_target : bool, default=True
        Load the molecular signatures (False: image-only datasets).
    predict : bool, default=False
        Build a single prediction dataset with all tiles.
    """

    def __init__(
        self,
        args: Namespace,
        transform: Callable | None = None,
        with_target: bool = True,
        predict: bool = False,
    ) -> None:
        self.args = args
        self.transform = transform
        self.with_target = bool(with_target)
        self.predict = bool(predict)
        self.use_val = bool(args.use_val) and not self.predict
        self.seed = int(args.seed)
        self.num_workers = int(getattr(args, "num_workers", 0))

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.dataset_predict = None
        self.train_sampler = None
        self.val_sampler = None

        if self.predict:
            self.dataset_predict = self._get_dataset(
                use_train=True,
                augment=False,
            )

        else:
            self.dataset_train = self._get_dataset(
                use_train=True,
                augment=True,
            )
            if self.use_val:
                self.dataset_val = self._get_dataset(
                    use_train=True,
                    augment=False,
                )
            self.dataset_test = self._get_dataset(
                use_train=False,
                augment=False,
            )
            self.train_sampler, self.val_sampler = self._get_sampler()

    @property
    def target_names(self) -> list[str]:
        """Names of the signature targets (empty for image-only data)."""
        dataset = self.dataset_predict if self.predict else self.dataset_train
        return dataset.target_names

    def _get_dataset(
        self,
        use_train: bool,
        augment: bool,
    ) -> HETileDataset:
        """
        Construct one dataset partition.

        Parameters
        ----------
        use_train : bool
            Training folds when True, `test_fold` when False.
        augment : bool
            Apply training augmentations.

        Returns
        -------
        HETileDataset
            Configured dataset.
        """
        return HETileDataset(
            self.args,
            use_train=use_train,
            transform=self.transform,
            with_target=self.with_target,
            predict=self.predict,
            augment=augment,
        )

    def _get_sampler(
        self,
    ) -> tuple[SubsetRandomSampler, SequentialSubsetSampler | None]:
        """
        Split the training folds into train and validation samplers.

        Returns
        -------
        train_sampler : torch.utils.data.SubsetRandomSampler
            Shuffled training indices (seeded).
        val_sampler : SequentialSubsetSampler | None
            Ordered validation indices, or None when `use_val=False`.

        Raises
        ------
        ValueError
            If the training folds are too small to split.
        """
        n_tiles = len(self.dataset_train)
        all_indices = np.arange(n_tiles)

        if not self.use_val:
            train_sampler = SubsetRandomSampler(
                all_indices.tolist(),
                generator=self.generator,
            )
            return train_sampler, None

        if n_tiles < 2:
            raise ValueError(
                f"Cannot split {n_tiles} training tile(s) into train and "
                f"validation; set `use_val=False`."
            )

        splitter = ShuffleSplit(
            n_splits=1,
            test_size=self.args.val_fraction,
            random_state=self.seed,
        )
        train_indices, val_indices = next(splitter.split(all_indices))

        LOGGER.info(
            "Train / validation split: train=%d, validation=%d tiles.",
            len(train_indices),
            len(val_indices),
        )

        train_sampler = SubsetRandomSampler(
            train_indices.tolist(),
            generator=self.generator,
        )

        val_sampler = SequentialSubsetSampler(val_indices.tolist())

        return train_sampler, val_sampler

    def get_loader(
        self,
        training: bool,
    ) -> tuple[DataLoader, DataLoader | None] | DataLoader:
        """
        Construct training / validation, test, or prediction loaders.

        Parameters
        ----------
        training : bool
            Return `(train_loader, val_loader)` when True (`val_loader` is
            None without validation). Otherwise return the ordered test
            loader, or the prediction loader in prediction mode.

        Returns
        -------
        tuple[DataLoader, DataLoader | None] or DataLoader
            Requested loaders.

        Raises
        ------
        RuntimeError
            If training loaders are requested in prediction mode.
        """
        common_kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": bool(getattr(self.args, "pin_memory", False)),
            "persistent_workers": bool(
                getattr(
                    self.args,
                    "persistent_workers",
                    False,
                )
            )
            and self.num_workers > 0,
        }

        if training:
            if self.predict:
                raise RuntimeError(
                    "Training loaders are unavailable in prediction mode."
                )

            train_loader = DataLoader(
                dataset=self.dataset_train,
                batch_size=self.args.batch_size,
                sampler=self.train_sampler,
                drop_last=True,
                generator=self.generator,
                **common_kwargs,
            )

            val_loader = None

            if self.use_val:
                val_loader = DataLoader(
                    dataset=self.dataset_val,
                    batch_size=self.args.eval_batch_size,
                    sampler=self.val_sampler,
                    drop_last=False,
                    **common_kwargs,
                )

            return train_loader, val_loader

        dataset = self.dataset_predict if self.predict else self.dataset_test

        return DataLoader(
            dataset=dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            **common_kwargs,
        )
