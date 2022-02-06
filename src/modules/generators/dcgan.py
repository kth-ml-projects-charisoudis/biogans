import torch
from torch import nn

from modules.partial.decoding import ExpandingBlock
from utils.ifaces import BalancedFreezable
from utils.pytorch import get_total_params


class DCGanGenerator(nn.Module, BalancedFreezable):
    """
    DCGanGenerator Class:
    Implements the Deep Convolutional GAN generator architecture. Noise-to-Image unconditional generation.
    """

    def __init__(self, c_out: int = 2, z_dim: int = 100, norm_type: str = 'batch', c_hidden: int = 512):
        # Initialize utils.ifaces.BalancedFreezable
        BalancedFreezable.__init__(self)
        # Initialize torch.nn.Module
        nn.Module.__init__(self)
        # Input Layer
        layers = [
            ExpandingBlock(c_in=z_dim, c_out=c_hidden, activation='relu', use_norm=True, norm_type=norm_type,
                           kernel_size=(3, 5), stride=1, padding=0, output_padding=0),
        ]
        # Hidden Layers
        for i in range(3):
            layers.append(
                ExpandingBlock(c_in=c_hidden // 2 ** i, activation='relu', use_norm=True, norm_type=norm_type,
                               kernel_size=2, stride=2, padding=0, output_padding=0),
            )
        # Output Layer
        layers.append(
            ExpandingBlock(c_in=c_hidden // 8, c_out=c_out, activation='tanh', use_norm=False,
                           kernel_size=2, stride=2, padding=0, output_padding=0),
        )
        self.gen = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.gen(z.reshape([z.shape[0], -1, 1, 1]))


if __name__ == '__main__':
    _gen = DCGanGenerator(c_out=2, z_dim=100)
    _z = torch.randn(10, 100)
    _x_hat = _gen(_z)
    print(_x_hat.shape)
    get_total_params(_gen, True, True)
