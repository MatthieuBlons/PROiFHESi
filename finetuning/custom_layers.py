"""
Building blocks of the PROiFHESi regression head.

`LinearBlock` and `PredictionHead` are kept identical to MILKYWAY
(`MIL.custom_layers`) so that both projects share the same head definition.

Layers
------
LinearBlock    : Linear -> LayerNorm (optional) -> GELU -> Dropout
PredictionHead : stack of `LinearBlock` + final Linear, no output activation
"""

from __future__ import annotations

# Data libraries
from torch import Tensor
from torch.nn import (
    Dropout,
    GELU,
    Identity,
    LayerNorm,
    Linear,
    Module,
    Sequential,
)


class LinearBlock(Module):
    """
    Fully connected block: Linear -> LayerNorm (optional) -> GELU -> Dropout.

    Parameters
    ----------
    input_dim : int
        Size of the input features.

    output_dim : int
        Size of the output features.

    dropout : float
        Dropout probability applied after the activation.

    use_norm : bool, default=True
        If True, apply LayerNorm after the linear layer. Otherwise use an
        identity.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        self.block = Sequential(
            Linear(input_dim, output_dim),
            LayerNorm(output_dim) if use_norm else Identity(),
            GELU(),
            Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Apply the block to the last dimension of the input.

        Parameters
        ----------
        inputs : torch.Tensor
            Input features with shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Output features with shape (..., output_dim).
        """
        return self.block(inputs)


class PredictionHead(Module):
    """
    Multilayer perceptron mapping tile embeddings to the predicted signature.

    No final activation is applied: outputs are raw regression values, in
    the space of the targets (e.g. `log1p` densities).

    Parameters
    ----------
    input_dim : int
        Size of the tile embedding (`feature_dim` of the encoder).

    hidden_dims : list[int]
        Sizes of the hidden layers, each built as a `LinearBlock`. An empty
        list gives a single linear layer connected to the output.

    output_dim : int
        Number of predicted targets (signature length).

    dropout : float
        Dropout probability used in every hidden block.

    use_norm : bool, default=True
        If True, apply LayerNorm in every hidden block.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float,
        use_norm: bool = True,
    ) -> None:
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(
                LinearBlock(
                    input_dim=current_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                    use_norm=use_norm,
                )
            )
            current_dim = hidden_dim

        layers.append(
            Linear(
                current_dim,
                output_dim,
            )
        )

        self.network = Sequential(*layers)

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        """
        Map tile embeddings to the predicted signature.

        Parameters
        ----------
        features : torch.Tensor
            Tile embeddings with shape (B, input_dim).

        Returns
        -------
        torch.Tensor
            Raw predictions with shape (B, output_dim).
        """
        return self.network(features)
