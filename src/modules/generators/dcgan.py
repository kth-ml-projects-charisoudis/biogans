import sys
from typing import Tuple

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
        self.z_dim = z_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 2, f'z must be 2-dimensional ({z.dim()}-d tensor provided)'
        return self.gen(z.reshape([z.shape[0], -1, 1, 1]))

    def get_random_z(self, batch_size: int = 1, device='cpu') -> torch.Tensor:
        return torch.randn(batch_size, self.z_dim, device=device)


class SeparableDCGanGenerator(nn.Module, BalancedFreezable):
    """
    SeparableDCGanGenerator Class:
    Implements the Separable architecture version of the DCGAN generator. Noise-to-Image unconditional generation, with
    the green channel being influenced by the red one, but NOT vice-versa.
    """

    def __init__(self, c_out: int = 2, z_dim: int = 100, norm_type: str = 'batch', c_hidden: int = 512):
        # Initialize utils.ifaces.BalancedFreezable
        BalancedFreezable.__init__(self)
        # Initialize torch.nn.Module
        nn.Module.__init__(self)
        z_dim = z_dim // 2
        c_hidden = c_hidden // 2
        self.z_dim = z_dim
        for c in ['red', 'green']:
            # Input Layer
            layers = [
                ExpandingBlock(c_in=z_dim, c_out=c_hidden, activation='relu', use_norm=True, norm_type=norm_type,
                               kernel_size=(3, 5), stride=1, padding=0, output_padding=0),
            ]
            # Hidden Layers
            layer_c_out = None
            for i in range(3):
                c_in = c_hidden // 2 ** i
                layer_c_out = c_in // 2
                if c == 'green':
                    c_in *= 2  # red channels will be concatenated before entering green layer
                layers.append(
                    ExpandingBlock(c_in=c_in, c_out=layer_c_out, activation='relu', use_norm=True, norm_type=norm_type,
                                   kernel_size=2, stride=2, padding=0, output_padding=0),
                )
            # Output Layer
            if c == 'green':
                layer_c_out *= 2
            layers.append(
                ExpandingBlock(c_in=layer_c_out, c_out=1 if c == 'red' else (c_out - 1), activation='tanh',
                               use_norm=False, kernel_size=2, stride=2, padding=0, output_padding=0),
            )
            setattr(self, f'gen_{c}', nn.ModuleList(layers))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the separable DCGAN Generator.
        :param torch.Tensor z: size (B, z_dim)
        :return: a tuple containing the green and red channel outputs
        """
        assert z.dim() == 2, f'z must be 2-dimensional ({z.dim()}-d tensor provided)'
        # Split noise
        x_red, x_green = torch.split(z.reshape([z.shape[0], -1, 1, 1]), self.z_dim, dim=1)
        # Feed them into the two towers
        for layer_red, layer_green in zip(self.gen_red, self.gen_green):
            x_red = layer_red(x_red)
            x_green = layer_green(x_green)
            if layer_red.c_out != 1:  # not output layer
                # Concatenate red layer's output to green before feeding into next green tower's layer
                print(f'concatenating red {x_red.shape} --> green {x_green.shape}', file=sys.stderr)
                x_green = torch.cat((x_red, x_green), dim=1)
        return x_red, x_green


if __name__ == '__main__':
    _gen = DCGanGenerator(c_out=2, z_dim=100)
    _z = torch.randn(10, 100)
    print(_gen(_z).shape)
    get_total_params(_gen, True, True)

    _gen = SeparableDCGanGenerator(c_out=2, z_dim=100)
    x_red, x_green = _gen(_z)
    print(x_red.shape, x_green.shape)
    get_total_params(_gen, True, True)
