import lab.torch as B
import torch
import torch.nn as nn

from .decoder import SetConv1dDecoder
from .discretisation import Discretisation1d
from .encoder import SetConv1dEncoder
from .unet import UNet
from .util import convert_batched_data

__all__ = ["DualConvCNP"]


class MultiConvCNP(nn.Module):
    def __init__(
        self,
        sigma: float = 0.1,
        points_per_unit: float = 32,
        small: bool = False,
        output_structure: tuple = (1, 1),
    ):
        super(MultiConvCNP, self).__init__()

        # Construct CNN:
        self.conv = UNet(
            dimensionality=1,
            in_channels=int(2*(output_structure[0] + output_structure[1])),  # Two for each discrete and continuous output
            out_channels=int(output_structure[0] + 2*output_structure[1]),  # One for class. prob. and two for mean and variance
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
        batch = [{k: convert_batched_data(v) for k, v in output.items()} for output in batch]

        # Construct discretisation.
        with B.on_device(batch[0]["x_context"]):
            x_grid = self.disc(batch)[None, :, None]

            # Run encoders.
            z = [] # This needs to be converted to torch tensor, then B.concat operation not needed
            for output in batch:
                z.append(self.encoder(
                    output["x_context"],
                    output["y_context"],
                    x_grid,
                ))

        # Run CNN.
        z = B.concat(z_class, z_reg, axis=1)
        z = self.conv(z)
        z_class = z[:, :1, :]
        z_reg = z[:, 1:, :]

        # Run decoders.
        z_class = self.decoder(x_grid, z_class, batch["x_target_class"])
        z_reg = self.decoder(x_grid, z_reg, batch["x_target_reg"])

        # Return parameters for classification and regression.
        return z_class, (z_reg[:, :, :1], B.exp(z_reg[:, :, 1:]))
