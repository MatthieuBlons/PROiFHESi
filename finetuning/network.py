"""
PROiFHESi network: HISTONAUT patch encoder (+ LoRA adapters) and a
regression head.

    image (B, 3, H, W) -> backbone -> embedding (B, feature_dim)
                       -> PredictionHead -> signature (B, output_dim)

The backbone is the HISTONAUT encoder returned by
`histonaut.encoder.factory.patch_encoder_factory`, not its inner `.model`,
so encoder-specific forward passes (e.g. Virchow `[CLS, mean patch]`,
Phikon CLS token) are kept and the embedding size matches `feature_dim`.

Trainable parameters
--------------------
lora=True  : LoRA adapters (`peft`, fp32) + head (fp32). The backbone
             weights are frozen by `peft` and keep their native precision.
lora=False : head only; the backbone is frozen explicitly.

On CPU the backbone is cast to fp32 (half precision is not supported).

Hidden representations
----------------------
`RepresentationHook` hooks any named submodule during inference, e.g.
the head hidden blocks (`LoRAFactory.hidden_layer_names`), to follow how
the tile representation evolves from the embedding to the signature:

    embedding (feature_dim) -> head.network.0 (width_fe[0]) -> ...
                            -> head.network.<last> (output_dim)
"""

from __future__ import annotations

# General libraries
import logging
from argparse import Namespace
from typing import Sequence

# Data libraries
import torch
from peft import LoraConfig, get_peft_model
from torch import Tensor
from torch.nn import Module, Sequential
from torch.utils.hooks import RemovableHandle
from torchinfo import ModelStatistics, summary

# HISTONAUT
from histonaut.encoder.factory import patch_encoder_factory

# Project modules
from finetuning.custom_layers import PredictionHead

LOGGER = logging.getLogger(__name__)

LORA_DROPOUT = 0.05
TRAINABLE_PRECISION = torch.float32
SUMMARY_PARTS = {
    "full",
    "backbone",
    "head",
}


class LoRAFactory(Module):
    """
    HISTONAUT patch encoder with optional LoRA adapters and a regression
    head.

    Parameters
    ----------
    args : argparse.Namespace
        Model configuration.

        Required attributes:

        encoder
            HISTONAUT patch encoder name (e.g. `hoptimus1`).

        lora, lora_r, lora_a, lora_target
            LoRA switch, rank, scaling alpha and targeted module names.

        feature_dim, width_fe, dropout, output_dim
            Head input size, hidden widths, dropout and number of targets.

        device
            Target device; the backbone is cast to fp32 on CPU.

    Attributes
    ----------
    backbone : torch.nn.Module
        HISTONAUT encoder, wrapped in a `peft.PeftModel` when `lora=True`.
    head : PredictionHead
        Regression head.
    transform : Callable
        Encoder `eval_transforms` (PIL image -> normalised tensor).
    precision : dict[str, torch.dtype]
        Dtype of `backbone`, `head` and, with LoRA, `lora`.
    lora : bool
        Whether LoRA adapters are used.
    """

    ARG_NETWORK = {
        "lora",
        "encoder",
        "lora_r",
        "lora_a",
        "lora_target",
        "feature_dim",
        "width_fe",
        "dropout",
        "output_dim",
    }

    def __init__(
        self,
        args: Namespace,
    ) -> None:
        super().__init__()

        self.network_args = self._fetch_network_args(args)

        self.lora = bool(self.network_args.lora)
        self.precision: dict[str, torch.dtype] = {}

        self.backbone, self.transform = self.build_backbone()

        if self.lora:
            self.apply_lora()
        else:
            self.freeze_backbone()

        self.head = self.build_head()

        if torch.device(args.device).type == "cpu":
            self.cast_backbone(precision=torch.float32)

    def _fetch_network_args(
        self,
        args: Namespace,
    ) -> Namespace:
        """
        Keep only the arguments required by the selected network.

        Parameters
        ----------
        args : argparse.Namespace
            Full runtime configuration.

        Returns
        -------
        argparse.Namespace
            Namespace restricted to `ARG_NETWORK`.

        Raises
        ------
        ValueError
            If one of the `ARG_NETWORK` names is missing from `args`.
        """

        missing = [name for name in self.ARG_NETWORK if not hasattr(args, name)]

        if missing:
            raise KeyError(f"Missing network arguments: " f"{sorted(missing)}.")

        return Namespace(**{name: getattr(args, name) for name in self.ARG_NETWORK})

    def forward(
        self,
        images: Tensor,
        return_embeddings: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Predict signatures from a batch of transformed tiles.

        Parameters
        ----------
        images : torch.Tensor
            Tiles with shape (B, 3, H, W), in the backbone dtype.
        return_embeddings : bool, default=False
            Also return the backbone (fine-tuned encoder) embeddings.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            Predicted signatures (B, output_dim), and the embeddings
            (B, feature_dim) when `return_embeddings=True`.
        """
        embeddings = self.backbone(images)
        predictions = self.head(embeddings)

        if return_embeddings:
            return predictions, embeddings

        return predictions

    def build_backbone(self) -> tuple[Module, object]:
        """
        Load the HISTONAUT encoder in its native precision.

        Returns
        -------
        backbone : torch.nn.Module
            HISTONAUT `BasePatchEncoder`.
        transform : Callable
            Its `eval_transforms`.
        """
        encoder = patch_encoder_factory(self.network_args.encoder)
        encoder = encoder.to(encoder.precision)
        self.precision["backbone"] = encoder.precision

        return encoder, encoder.eval_transforms

    def apply_lora(self) -> None:
        """
        Wrap the backbone with `peft` LoRA adapters kept in fp32.
        """
        lora_config = LoraConfig(
            r=self.network_args.lora_r,
            lora_alpha=self.network_args.lora_a,
            target_modules=self.network_args.lora_target,
            lora_dropout=LORA_DROPOUT,
        )
        self.backbone = get_peft_model(
            self.backbone,
            lora_config,
        )

        for name, module in self.backbone.named_modules():
            if "lora_" in name:
                module.to(dtype=TRAINABLE_PRECISION)

        self.precision["lora"] = TRAINABLE_PRECISION

    def freeze_backbone(self) -> None:
        """
        Disable gradients of every backbone parameter (no LoRA).
        """
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def build_head(self) -> PredictionHead:
        """
        Build the regression head in fp32.
        """
        head = PredictionHead(
            input_dim=self.network_args.feature_dim,
            hidden_dims=self.network_args.width_fe or [],
            output_dim=self.network_args.output_dim,
            dropout=self.network_args.dropout,
        )
        self.precision["head"] = TRAINABLE_PRECISION

        return head.to(TRAINABLE_PRECISION)

    def cast_backbone(
        self,
        precision: torch.dtype,
    ) -> None:
        """
        Cast the backbone (LoRA adapters included) to `precision`.
        """
        self.backbone.to(precision)
        self.precision["backbone"] = precision

    def log_trainable_parameters(self) -> None:
        """
        Log the number of trainable and total parameters.
        """
        n_trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        n_total = sum(parameter.numel() for parameter in self.parameters())

        LOGGER.info(
            "Trainable parameters (%s + head): %d / %d (%.4f%%).",
            "LoRA" if self.lora else "no LoRA",
            n_trainable,
            n_total,
            100 * n_trainable / max(n_total, 1),
        )

    def log_summary(
        self,
        part: str = "full",
        depth: int = 4,
    ) -> ModelStatistics:
        """
        Log a `torchinfo` summary of the network or one of its parts.

        Parameters
        ----------
        part : {"full", "backbone", "head"}, default="full"
            Module to summarise.
        depth : int, default=4
            Nesting depth displayed.

        Returns
        -------
        torchinfo.ModelStatistics
            Summary object.

        Raises
        ------
        ValueError
            If `part` is not supported.
        """
        modules = {
            "full": Sequential(
                self.backbone,
                self.head,
            ),
            "backbone": self.backbone,
            "head": self.head,
        }

        if part not in SUMMARY_PARTS:
            raise ValueError(
                f"Unsupported `part={part}`. Expected one of {sorted(SUMMARY_PARTS)}."
            )

        statistics = summary(
            modules[part],
            depth=depth,
            verbose=0,
        )
        LOGGER.info("%s", statistics)

        return statistics

    def hidden_layer_names(self) -> list[str]:
        """
        Names of the hidden blocks of the prediction head.

        One name per `LinearBlock` (`width_fe` entry): `head.network.<i>`,
        whose output is the post-activation tile representation of width
        `width_fe[i]`. The final linear layer (the prediction) is excluded.

        Returns
        -------
        list[str]
            Module names usable with `RepresentationHook`; empty when
            the head is a single linear layer.
        """
        n_hidden_blocks = len(self.head.network) - 1
        return [f"head.network.{index}" for index in range(n_hidden_blocks)]


# -----------------------------------------------------------------------------
# Hidden representations
# -----------------------------------------------------------------------------


class RepresentationHook:
    """
    Record the outputs of named submodules during forward passes.

    Forward hooks are registered on entering the context and removed on
    exit. After each forward pass, `representations[name]` holds the output
    of module `name` for the last batch (detached, fp32, on CPU).

    Parameters
    ----------
    network : torch.nn.Module
        Network to observe, usually `PROiFHESi.network`.
    layer_names : Sequence[str]
        Submodule names as listed by `network.named_modules()`, e.g. the
        head hidden blocks `head.network.0`, `head.network.1` (see
        `LoRAFactory.hidden_layer_names`), or any backbone module.

    Raises
    ------
    KeyError
        If a name is not a submodule of `network`.

    Notes
    -----
    Backbone modules output token tensors (B, n_tokens, dim): recording
    them over a whole dataset can take a lot of memory.

    Examples
    --------
    >>> layer_names = model.network.hidden_layer_names()
    >>> with RepresentationHook(model.network, layer_names) as recorder:
    ...     model.predict_batch(batch)
    ...     hidden = recorder.representations  # {name: (B, width)}
    """

    def __init__(
        self,
        network: Module,
        layer_names: Sequence[str],
    ) -> None:
        modules = dict(network.named_modules())
        missing = [name for name in layer_names if name not in modules]

        if missing:
            head_names = [name for name in modules if name.startswith("head")]
            raise KeyError(
                f"`layer_names` {missing} are not submodules of the network. "
                f"Head modules: {head_names}."
            )

        self.layer_names = list(layer_names)
        self.modules = {name: modules[name] for name in self.layer_names}
        self.representations: dict[str, Tensor] = {}
        self._handles: list[RemovableHandle] = []

    def _make_hook(self, name: str):
        """Build the forward hook storing the output of module `name`."""

        def hook(module: Module, inputs: tuple, output: Tensor) -> None:
            self.representations[name] = output.detach().float().cpu()

        return hook

    def __enter__(self) -> RepresentationHook:
        self.representations = {}
        self._handles = [
            module.register_forward_hook(self._make_hook(name))
            for name, module in self.modules.items()
        ]
        return self

    def __exit__(self, *exc_info) -> None:
        for handle in self._handles:
            handle.remove()

        self._handles = []
