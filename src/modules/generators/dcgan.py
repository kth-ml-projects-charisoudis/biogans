from typing import Tuple, Optional

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

    def __init__(self, c_out: int = 2, z_dim: int = 100, norm_type: str = 'batch', c_hidden: int = 512,
                 n_extra_layers: int = 0, red_portion: Optional[float] = None):
        """
        DCGanGenerator class constructor.
        :param int c_out: expected number of output channels from the DCGAN Generator
        :param int z_dim: input noise vector dimensionality
        :param str norm_type: one of 'batch', 'instance', 'pixel', 'layer'
        :param int c_hidden: number of hidden conv channels to be used as a reference
        :param (optional) red_portion: if set, Separable architecture will be employed
        """
        # Initialize utils.ifaces.BalancedFreezable
        BalancedFreezable.__init__(self)
        # Initialize torch.nn.Module
        nn.Module.__init__(self)
        # Input Layer
        layers = [
            ExpandingBlock(c_in=z_dim, c_out=c_hidden, activation='relu', use_norm=True, norm_type=norm_type,
                           kernel_size=(3, 5), stride=1, padding=0, output_padding=0, bias=False,
                           red_portion=red_portion),
        ]
        # Hidden Layers
        for i in range(3):
            layers.append(
                ExpandingBlock(c_in=c_hidden // 2 ** i, activation='relu', use_norm=True, norm_type=norm_type,
                               kernel_size=4, stride=2, padding=1, output_padding=0, bias=False,
                               red_portion=red_portion),
            )
        # Extra Hidden Layers
        for i in range(n_extra_layers):
            layers.append(
                ExpandingBlock(c_in=c_hidden // 8, c_out=c_hidden // 8, activation='relu', use_norm=True,
                               norm_type=norm_type, kernel_size=3, stride=1, padding=1, output_padding=0,
                               bias=False, red_portion=red_portion),
            )
        # Output Layer
        layers.append(
            ExpandingBlock(c_in=c_hidden // 8, c_out=c_out, activation='tanh', use_norm=False,
                           kernel_size=4, stride=2, padding=1, output_padding=0, bias=False,
                           red_portion=red_portion),
        )
        self.gen = nn.Sequential(*layers)
        self.z_dim = z_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 2, f'z must be 2-dimensional ({z.dim()}-d tensor provided)'
        return self.gen(z.reshape([z.shape[0], -1, 1, 1]))

    def get_random_z(self, batch_size: int = 1, device='cpu') -> torch.Tensor:
        return torch.randn(batch_size, self.z_dim, device=device)


class SeparableDCGanGenerator(DCGanGenerator):
    """
    SeparableDCGanGenerator Class:
    Implements the Separable architecture version of the DCGAN generator. Noise-to-Image unconditional generation, with
    the green channel being influenced by the red one, but NOT vice-versa.
    """

    def __init__(self, c_out: int = 2, n_extra_layers: int = 0):
        red_portion = 1.0 / float(c_out)
        super().__init__(c_out=c_out, z_dim=100, norm_type='batch', c_hidden=512, n_extra_layers=n_extra_layers,
                         red_portion=red_portion)
        self.c_out_red = int(c_out * red_portion)
        self.c_out_green = c_out - self.c_out_red

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = super().forward(z)
        return torch.split(out, [self.c_out_red, self.c_out_green], dim=1)


if __name__ == '__main__':
    _gen = DCGanGenerator(c_out=6 + 1, z_dim=100, n_extra_layers=2)
    # _z = torch.randn(10, 100)
    # print(_gen)
    # print(_gen(_z).shape)
    # get_total_params(_gen, True, True)

    _gen = SeparableDCGanGenerator(c_out=7, n_extra_layers=0)
    print(_gen)
    x_red, x_green = _gen(_gen.get_random_z(batch_size=1))
    print(x_red.shape, x_green.shape)
    get_total_params(_gen, True, True)
