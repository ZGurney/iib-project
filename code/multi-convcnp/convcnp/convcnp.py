import enum
import lab.torch as B
import torch
import torch.nn as nn

from .decoder import SetConv1dDecoder
from .discretisation import Discretisation1d
from .encoder import SetConv1dEncoder
from .unet import UNet
from .util import convert_batched_data

__all__ = ["MultiConvCNP"]


class MultiConvCNP(nn.Module):
    def __init__(
        self,
        sigma: float = 0.1,
        points_per_unit: float = 32,
        small: bool = False,
        num_class: int = 1,
        num_reg: int = 1,
    ):
        super(MultiConvCNP, self).__init__()

        self.num_class = num_class
        self.num_reg = num_reg

        # Construct CNN:
        self.conv = UNet(
            dimensionality=1,
            in_channels=int(2*(self.num_class + self.num_reg)),  # Two for each discrete and continuous output
            out_channels=int(self.num_class + 2*self.num_reg),  # One for class. prob. and two for mean and variance
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
        batch = batch.copy()
        for output in batch:
            output = {k: convert_batched_data(v) for k, v in output.items()}

        # Construct discretisation.
        with B.on_device(batch[0]["x_context"]):
            x_grid = self.disc(batch)[None, :, None]

        # Run encoders.
        for index, output in enumerate(batch):
            z_output = self.encoder(
                    output["x_context"],
                    output["y_context"],
                    x_grid,
                )
            if index == 0:
                z = z_output
            else:
                z = B.concat(z, z_output, axis=1)

        # Run CNN.
        z = self.conv(z)
        z_outputs = []
        for i in range(self.num_class):
            z_outputs.append(z[:, i:(i+1), :])
        for i in range(self.num_reg):
            z_outputs.append(z[:, (self.num_class+2*i):(self.num_class+2*(i+1)), :])
        assert B.shape(z_outputs)[0] == B.shape(batch)[0], "Number of outputs do not match"

        # Run decoders.
        for index, output in enumerate(batch):
            z_outputs[index] = self.decoder(
                x_grid, 
                z_outputs[index], 
                output["x_target"]
            )

        # Return parameters for classification and regression.
        return z_outputs
