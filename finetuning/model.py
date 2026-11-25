"""
PROiFHESi model: LoRA fine-tuning of a pathology foundation model to predict
the proteomic signature of H&E tiles.

`PROiFHESi` implements the `BaseModel` interface shared with MILKYWAY
(`train_batch`, `validate_batch`, `predict_batch`, `create_checkpoint`), so
the training loop only talks to the model through these methods.

Supported options
-----------------
task         : regression
criterion    : mse, rmse, mae, l1, huber (`huber_delta`),
               hierarchy (MSE + `hierarchy_alpha` * hierarchy penalty)
optimizer    : adamw, sgd
lr_scheduler : none, cosine (warm restarts), plateau (ReduceLROnPlateau)

Validation metrics
------------------
mean_val_loss, mean_train_loss
mean_pearson, mean_r2, mean_ccc     (averaged over targets)
mean_cossim, mean_euclidsim         (averaged over tiles)
pearson_<target>, r2_<target>, ccc_<target>

Mixed precision
---------------
The backbone keeps its native precision (bf16 / fp16 from HISTONAUT), LoRA
adapters and head are fp32. Forward passes run under `autocast` with the
backbone dtype; a `GradScaler` is only enabled for fp16.
"""

from __future__ import annotations

# General libraries
import logging
import math
from pathlib import Path
from typing import Any

# Data libraries
import numpy as np
import torch
from torch import Tensor, autocast
from torch.amp import GradScaler
from torch.nn import (
    HuberLoss,
    L1Loss,
    MSELoss,
    Module,
)
from torch.nn.functional import cosine_similarity
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)

# Project modules
from finetuning.base_model import BaseModel, EarlyStopping
from finetuning.dataloader import SUPPORTED_TASKS, DatasetHandler
from finetuning.metrics import (
    HierarchyLoss,
    RMSELoss,
    coefficient_of_determination,
    concordance_correlation,
    euclidean_similarity,
    INEQUALITY_CONSTRAINTS,
    pearson,
)
from finetuning.network import LoRAFactory

LOGGER = logging.getLogger(__name__)


class PROiFHESi(BaseModel):
    """
    LoRA fine-tuned tile encoder with a regression head predicting the
    proteomic signature of H&E tiles.

    Parameters
    ----------
    args : argparse.Namespace
        Output of `finetuning.arguments.get_arguments`.

        Important attributes include:

        encoder, lora, lora_r, lora_a, lora_target
            Backbone and LoRA configuration.

        feature_dim, width_fe, dropout, output_dim
            Regression head.

        criterion, huber_delta, hierarchy_alpha
            Training loss.

        optimizer, lr, weight_decay, momentum, gradient_clip_norm
            Optimisation.

        lr_scheduler, patience_lr, restart_period, restart_multiplier
            Learning-rate scheduler.

        ref_metric, metric_mode, patience, minimum_epochs,
        minimum_improvement, job_dir
            Model selection and early stopping.

    target_names : list[str] | None, default=None
        Names of the predicted targets. Read from the data when
        `with_data=True`; must be given otherwise for `criterion=hierarchy`
        and per-target metrics (e.g. when loading a checkpoint).
    with_data : bool, default=False
        Build the training / validation loaders and the early stopping.

    Raises
    ------
    ValueError
        If `args.task`, `args.criterion`, `args.optimizer` or
        `args.lr_scheduler` is not supported.
    """

    def __init__(
        self,
        args,
        target_names: list[str] | None = None,
        with_data: bool = False,
    ) -> None:
        super().__init__(args)

        self.task = str(args.task).lower()

        if self.task not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported `task={self.task}`. "
                f"Expected one of {sorted(SUPPORTED_TASKS)}."
            )

        self.reference_metric = args.ref_metric
        self.metric_mode = getattr(args, "metric_mode", "min")

        self.network = self.build_network()
        self.precision = self.network.precision
        self.trainable = [
            parameter
            for parameter in self.network.parameters()
            if parameter.requires_grad
        ]

        self.train_loader = None
        self.val_loader = None
        self.dataset_handler = None
        self.target_names = target_names

        if with_data:
            self.get_data()

        self.optimizer = self.build_optimizer()
        self.optimizers = [self.optimizer]
        self.schedulers = self.build_schedulers()
        self.criterion = self.build_criterion()
        self.gradient_scaler = GradScaler(
            device=self.device.type,
            enabled=self.precision["backbone"] == torch.float16,
        )

        self.best_metrics: dict[str, float] | None = None
        self.best_reference_value: float | None = None

        self.mean_train_loss = math.nan
        self.mean_validation_loss = math.nan

        self.validation_results = self._empty_validation_results()

        self.early_stopping = None

        if with_data:
            self.early_stopping = EarlyStopping(
                patience=getattr(args, "patience", None) or args.epochs,
                minimum_epochs=getattr(args, "minimum_epochs", 0) or 0,
                mode=self.metric_mode,
                checkpoint_path=Path(args.job_dir) / "model.pt.tar",
                minimum_improvement=getattr(
                    args,
                    "minimum_improvement",
                    0.0,
                ),
            )

    def build_network(self) -> Module:
        """
        Construct the backbone (+ LoRA) and the regression head on the
        device. See `finetuning.network.LoRAFactory` for trainable parameters
        and precisions.
        """
        network = LoRAFactory(self.args)
        network.log_trainable_parameters()

        return network.to(self.device)

    def get_data(self) -> None:
        """
        Construct the training and validation loaders and read the target
        names from the data.

        Raises
        ------
        ValueError
            If `target_names` given at construction differ from the data.
        """
        self.dataset_handler = DatasetHandler(
            self.args,
            transform=self.network.transform,
            with_target=True,
            predict=False,
        )

        (
            self.train_loader,
            self.val_loader,
        ) = self.dataset_handler.get_loader(training=True)

        data_target_names = list(self.dataset_handler.target_names)

        if self.target_names is not None and list(self.target_names) != (
            data_target_names
        ):
            raise ValueError(
                f"`target_names` {list(self.target_names)} differ from the "
                f"targets of `target_path` {data_target_names}."
            )

        self.target_names = data_target_names

    def build_optimizer(self) -> Optimizer:
        """
        Construct the optimizer over the trainable parameters (LoRA + head).
        """
        optimizer_name = self.args.optimizer.lower()
        weight_decay = getattr(
            self.args,
            "weight_decay",
            1e-5,
        )

        if optimizer_name == "adamw":
            return AdamW(
                self.trainable,
                lr=self.args.lr,
                weight_decay=weight_decay,
            )

        if optimizer_name == "sgd":
            return SGD(
                self.trainable,
                lr=self.args.lr,
                momentum=getattr(
                    self.args,
                    "momentum",
                    0.9,
                ),
                weight_decay=weight_decay,
            )

        raise ValueError(
            f"Unsupported `optimizer={self.args.optimizer}`. "
            "Expected one of ['adamw', 'sgd']."
        )

    def build_schedulers(self) -> list[Any]:
        """
        Construct zero or one learning-rate scheduler per optimizer.

        Warm-up (`lr_warm_up`) is handled by the training loop, which does
        not step the scheduler during the first epochs.
        """
        scheduler_name = str(
            getattr(
                self.args,
                "lr_scheduler",
                "none",
            )
        ).lower()

        if scheduler_name == "none":
            return []

        if scheduler_name == "plateau":
            return [
                ReduceLROnPlateau(
                    optimizer,
                    mode=self.metric_mode,
                    patience=self.args.patience_lr,
                    factor=0.5,
                )
                for optimizer in self.optimizers
            ]

        if scheduler_name == "cosine":
            return [
                CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=getattr(
                        self.args,
                        "restart_period",
                        1,
                    ),
                    T_mult=getattr(
                        self.args,
                        "restart_multiplier",
                        2,
                    ),
                )
                for optimizer in self.optimizers
            ]

        raise ValueError(
            f"Unsupported `lr_scheduler={scheduler_name}`. "
            "Expected one of ['none', 'cosine', 'plateau']."
        )

    def build_criterion(self) -> Module:
        """
        Construct the regression loss.

        Supported criteria
        ------------------
        mse, rmse, mae / l1, huber (`huber_delta`),
        hierarchy (MSE + `hierarchy_alpha` * hierarchy penalty).

        Raises
        ------
        ValueError
            If the criterion is unknown, or `criterion=hierarchy` without
            `target_names`.
        """
        criterion_name = self.args.criterion.lower()

        if criterion_name == "mse":
            return MSELoss().to(self.device)

        if criterion_name == "rmse":
            return RMSELoss().to(self.device)

        if criterion_name in {"mae", "l1"}:
            return L1Loss().to(self.device)

        if criterion_name == "huber":
            return HuberLoss(
                delta=getattr(
                    self.args,
                    "huber_delta",
                    1.0,
                )
            ).to(self.device)

        if criterion_name == "hierarchy":
            return self.build_hierarchy_criterion()

        raise ValueError(
            f"Unsupported `criterion={criterion_name}`. "
            "Expected one of ['hierarchy', 'huber', 'l1', 'mae', 'mse', 'rmse']."
        )

    def build_hierarchy_criterion(self) -> Module:
        """
        Construct `HierarchyLoss` on the constraints whose two markers are
        both predicted.

        Constraints involving a marker absent from `target_names` are
        dropped with a warning.
        """
        if self.target_names is None:
            raise ValueError(
                "`criterion=hierarchy` requires `target_names` "
                "(load data with `with_data=True` or pass them explicitly)."
            )

        marker_to_index = {
            target_name: index for index, target_name in enumerate(self.target_names)
        }
        constraints = [
            (marker_a, marker_b)
            for marker_a, marker_b in INEQUALITY_CONSTRAINTS
            if marker_a in marker_to_index and marker_b in marker_to_index
        ]

        n_dropped = len(INEQUALITY_CONSTRAINTS) - len(constraints)

        if n_dropped:
            LOGGER.warning(
                "Hierarchy loss: %d constraint(s) dropped (marker not in targets).",
                n_dropped,
            )

        if not constraints:
            raise ValueError(
                "`criterion=hierarchy`: no hierarchy constraint applies to "
                f"`target_names` {self.target_names}."
            )

        return HierarchyLoss(
            marker_to_idx=marker_to_index,
            constraints=constraints,
            mse_weight=1.0,
            hierarchy_weight=getattr(
                self.args,
                "hierarchy_alpha",
                1.0,
            ),
        ).to(self.device)

    def _autocast(self) -> autocast:
        """
        Autocast context in the backbone dtype (disabled for fp32).
        """
        return autocast(
            device_type=self.device.type,
            dtype=self.precision["backbone"],
            enabled=self.precision["backbone"] != torch.float32,
        )

    def move_batch_to_device(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Move images (backbone dtype) and targets (fp32) to the device.
        """
        prepared = dict(batch)
        prepared["image"] = batch["image"].to(
            self.device,
            dtype=self.precision["backbone"],
            non_blocking=True,
        )

        if "target" in batch:
            prepared["target"] = batch["target"].to(
                self.device,
                dtype=torch.float32,
                non_blocking=True,
            )

        return prepared

    def forward(self, images: Tensor) -> Tensor:
        """
        Predict signatures from a batch of images, shape (B, output_dim).
        """
        return self.network(images)

    def forward_without_gradients(
        self,
        images: Tensor,
        return_embeddings: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Inference forward pass in eval mode, returned in fp32.

        With `return_embeddings=True`, returns `(predictions, embeddings)`,
        the embeddings being the backbone output (B, feature_dim).
        """
        if self.network.training:
            self.network.eval()

        with torch.inference_mode(), self._autocast():
            outputs = self.network(
                images,
                return_embeddings=return_embeddings,
            )

        if return_embeddings:
            predictions, embeddings = outputs
            return predictions.float(), embeddings.float()

        return outputs.float()

    def compute_loss(
        self,
        outputs: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        Compute the criterion after checking output and target shapes.
        """
        if outputs.shape != targets.shape:
            raise ValueError(
                "Output and target shapes differ: "
                f"output={tuple(outputs.shape)}, "
                f"target={tuple(targets.shape)}."
            )

        return self.criterion(
            outputs,
            targets,
        )

    # -------------------------------------------------------------------------
    # BaseModel interface
    # -------------------------------------------------------------------------

    def train_batch(
        self,
        batch: dict[str, Any],
    ) -> float:
        """
        Perform one optimisation step (autocast, gradient scaling, clipping).
        """
        self.network.train()
        batch = self.move_batch_to_device(batch)

        self.zero_optimizer_gradients()

        with self._autocast():
            outputs = self.forward(batch["image"])
            loss = self.compute_loss(
                outputs.float(),
                batch["target"],
            )

        self.gradient_scaler.scale(loss).backward()

        gradient_clip_norm = getattr(
            self.args,
            "gradient_clip_norm",
            None,
        )

        if gradient_clip_norm is not None:
            self.gradient_scaler.unscale_(self.optimizer)
            clip_grad_norm_(
                self.trainable,
                max_norm=gradient_clip_norm,
            )

        self.gradient_scaler.step(self.optimizer)
        self.gradient_scaler.update()

        self.counter["batch"] += 1

        return loss.detach().item()

    def validate_batch(
        self,
        batch: dict[str, Any],
    ) -> float:
        """
        Evaluate one batch and store its predictions and targets.
        """
        batch = self.move_batch_to_device(batch)
        outputs = self.forward_without_gradients(batch["image"])

        loss = self.compute_loss(
            outputs,
            batch["target"],
        )

        self.validation_results["outputs"].append(outputs.cpu())
        self.validation_results["targets"].append(batch["target"].cpu())
        self.validation_results["paths"].extend(batch["path"])

        return float(loss.detach().cpu())

    def finalize_validation_epoch(self) -> dict[str, float]:
        """
        Compute validation metrics, keep the best ones and clear the stored
        predictions.

        Returns
        -------
        dict[str, float]
            Regression metrics with mean training and validation losses.
        """
        outputs = torch.cat(
            self.validation_results["outputs"],
            dim=0,
        )
        targets = torch.cat(
            self.validation_results["targets"],
            dim=0,
        )

        validation_metrics = self.compute_regression_metrics(
            predictions=outputs,
            targets=targets,
            target_names=self.target_names,
        )
        validation_metrics["mean_train_loss"] = float(self.mean_train_loss)
        validation_metrics["mean_val_loss"] = float(self.mean_validation_loss)

        self.update_best_metrics(validation_metrics)
        self.validation_results = self._empty_validation_results()

        self.advance_counter()

        return validation_metrics

    @staticmethod
    def _empty_validation_results() -> dict[str, list]:
        """Return an empty validation accumulator."""
        return {
            "outputs": [],
            "targets": [],
            "paths": [],
        }

    def predict_batch(
        self,
        batch: dict[str, Any],
        return_embeddings: bool = False,
    ) -> dict[str, Any]:
        """
        Predict the signatures of one batch.

        Parameters
        ----------
        batch : dict
            Batch with `image` and `path` (`target` is ignored).
        return_embeddings : bool, default=False
            Also return the backbone embeddings.

        Returns
        -------
        dict
            `path` (list[str]), `prediction` (np.ndarray, (B, output_dim))
            and, with `return_embeddings`, `embedding` (np.ndarray,
            (B, feature_dim)).
        """
        prepared_batch = self.move_batch_to_device(batch)

        if not return_embeddings:
            predictions = self.forward_without_gradients(prepared_batch["image"])
            return {
                "path": batch["path"],
                "prediction": predictions.cpu().numpy(),
            }

        predictions, embeddings = self.forward_without_gradients(
            prepared_batch["image"],
            return_embeddings=True,
        )

        return {
            "path": batch["path"],
            "prediction": predictions.cpu().numpy(),
            "embedding": embeddings.cpu().numpy(),
        }

    # -------------------------------------------------------------------------
    # Metrics and model selection
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_regression_metrics(
        predictions: Tensor,
        targets: Tensor,
        target_names: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Compute per-target and aggregated regression metrics.

        Parameters
        ----------
        predictions, targets : torch.Tensor
            Shape (N, D), fp32 on CPU.
        target_names : list[str] | None, default=None
            Names used for per-target metrics; indices when None.

        Returns
        -------
        dict[str, float]
            `pearson_<t>`, `r2_<t>`, `ccc_<t>` per target, and
            `mean_pearson`, `mean_r2`, `mean_ccc` (NaN-aware averages over
            targets), `mean_cossim`, `mean_euclidsim` (averages over tiles).
        """
        predictions = predictions.float()
        targets = targets.float()

        if target_names is None:
            target_names = [str(index) for index in range(targets.shape[1])]

        per_target = {
            "pearson": pearson(
                targets,
                predictions,
                dim=0,
            ).numpy(),
            "r2": coefficient_of_determination(
                targets,
                predictions,
                dim=0,
            ).numpy(),
            "ccc": concordance_correlation(
                targets,
                predictions,
                dim=0,
            ).numpy(),
        }

        metric_values: dict[str, float] = {}

        for metric_name, values in per_target.items():
            for target_name, value in zip(target_names, values):
                metric_values[f"{metric_name}_{target_name}"] = float(value)

            metric_values[f"mean_{metric_name}"] = (
                float(np.nanmean(values)) if np.isfinite(values).any() else math.nan
            )

        metric_values["mean_cossim"] = float(
            cosine_similarity(
                targets,
                predictions,
                dim=1,
            ).mean()
        )
        metric_values["mean_euclidsim"] = float(
            euclidean_similarity(
                targets,
                predictions,
            ).mean()
        )

        return metric_values

    def update_best_metrics(
        self,
        current_metrics: dict[str, float],
    ) -> None:
        """
        Retain the metrics of the best validation epoch (`ref_metric`).

        Raises
        ------
        KeyError
            If `ref_metric` is not among the computed metrics.
        """
        if self.reference_metric not in current_metrics:
            raise KeyError(
                f"`ref_metric={self.reference_metric}` is not computed. "
                f"Available metrics: {sorted(current_metrics)}."
            )

        current_value = current_metrics[self.reference_metric]

        if not np.isfinite(current_value):
            return

        if self.best_reference_value is None:
            improved = True
        elif self.metric_mode == "min":
            improved = current_value < self.best_reference_value
        else:
            improved = current_value > self.best_reference_value

        if improved:
            self.best_reference_value = float(current_value)
            self.best_metrics = dict(current_metrics)


    def create_checkpoint(self) -> dict[str, Any]:
        """
        Create a checkpoint for resuming training or inference.

        Notes
        -----
        `model_state_dict` holds the full network, frozen backbone included.
        """
        checkpoint = {
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dicts": [
                scheduler.state_dict() for scheduler in self.schedulers
            ],
            "gradient_scaler_state_dict": self.gradient_scaler.state_dict(),
            "counter": dict(self.counter),
            "args": self.args,
            "task": self.task,
            "best_metrics": self.best_metrics,
            "best_reference_value": self.best_reference_value,
            "target_names": self.target_names,
            "output_dim": self.args.output_dim,
        }

        if self.train_loader is not None:
            checkpoint["input_table"] = self.train_loader.dataset.input_table

        return checkpoint
