import lab.torch as B
import torch
import torch.nn as nn

from .decoder import SetConv1dDecoder
from .discretisation import Discretisation1d
from .encoder import SetConv1dEncoder
from .unet import UNet
from .util import convert_batched_data

__all__ = ["RegConvCNP"]


class RegConvCNP(nn.Module):
    def __init__(
        self,
        sigma: float = 0.1,
        points_per_unit: float = 32,
        small: bool = False,
    ):
        super(RegConvCNP, self).__init__()

        # Construct CNN:
        self.conv = UNet(
            dimensionality=1,
            in_channels=2,  # Two channels for regression
            out_channels=2,  # Mean and variance for regression
            channels=(8, 16, 16, 32) if small else (8, 16, 16, 32, 32, 64),
        )

        # Construct discretisation:
        self.disc = Discretisation1d(
            points_per_unit=points_per_unit,
            multiple=2 ** self.conv.num_halving_layers,
            margin=0.1,
        )

        # Construct encoder and decoder:
        self.encoder = SetConv1dEncoder(self.disc)
        self.decoder = SetConv1dDecoder(self.disc)

        # Learnable observation noise for regression:
        self.log_sigma = nn.Parameter(
            B.log(torch.tensor(sigma, dtype=torch.float32)),
            requires_grad=True,
        )

    def forward(self, batch):
        # Ensure that inputs are of the right shape.
        batch = {k: convert_batched_data(v) for k, v in batch.items()}

        # Construct discretisation.
        with B.on_device(batch["x_context_class"]):
            x_grid = self.disc(
                batch["x_context_class"],
                batch["x_target_class"],
                batch["x_context_reg"],
                batch["x_target_reg"],
            )[None, :, None]

        # Run single encoder.
        z = self.encoder(
            batch["x_context_reg"],
            batch["y_context_reg"],
            x_grid,
        )

        # Run CNN.
        z = self.conv(z)

        # Run single decoder.
        z = self.decoder(x_grid, z, batch["x_target_reg"])

        # Return single parameter for classification
        return 0, (z[:, :, :1], B.exp(z[:, :, 1:]))
