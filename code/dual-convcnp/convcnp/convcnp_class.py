import lab.torch as B
import torch
import torch.nn as nn

from .decoder import SetConv1dDecoder
from .discretisation import Discretisation1d
from .encoder import SetConv1dEncoder
from .unet import UNet
from .util import convert_batched_data

__all__ = ["ClassConvCNP"]


class ClassConvCNP(nn.Module):
    def __init__(
        self,
        sigma: float = 0.1,
        points_per_unit: float = 32,
        small: bool = False,
    ):
        super(ClassConvCNP, self).__init__()

        # Construct CNN:
        self.conv = UNet(
            dimensionality=1,
            in_channels=2,  # Two channels for classification
            out_channels=1,  # One for class. prob.
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
            batch["x_context_class"],
            batch["y_context_class"],
            x_grid,
        )

        # Run CNN.
        z = self.conv(z)

        # Run single decoder.
        z = self.decoder(x_grid, z, batch["x_target_class"])

        # Return single parameter for classification
        return B.sigmoid(z)
