from collections import OrderedDict
from typing import OrderedDict as OrderedDictT

import torch
from torch import nn

from modules.generators.dcgan import DCGanSubGenerator
from modules.partial.decoding import ExpandingBlockStarShaped
from utils.ifaces import BalancedFreezable


class DCGanSubGeneratorStarShaped(DCGanSubGenerator):
    def peak(self) -> torch.Tensor:
        out = self.GEN_FORWARDS[self.index][self.rank]
        self.index += 1
        # print(f'[SG{self.rank}][peak] out_shape={out.shape} | new_index={self.index}')
        return out

    def get_random_z(self, *args, **kwargs):
        return self.gen.get_random_z(*args, **kwargs)


class DCGanGeneratorStarShaped(nn.Module, BalancedFreezable, object):
    """
    DCGanGeneratorStarShaped Class:
    Implements the DCGAN generator architecture with "Star-Shaped" convolutions. They all fall in the Noise-to-Image
    unconditional generation category.
    """
    SUB_GENS = None

    def __init__(self, c_out: int = 2, z_dim: int = 100, n_classes: int = 6, c_hidden: int = 512,
                 red_portion: float = 0.5):
        """
        DCGanGeneratorStarShaped class constructor.
        :param int c_out: expected number of output channels from the DCGAN Generator
        :param int z_dim: input noise vector dimensionality
        :param int n_classes: number of dataset classes
        :param int c_hidden: number of hidden conv channels to be used as a reference
        :param (optional) red_portion: if set, Separable architecture will be employed
        """
        # Initialize utils.ifaces.BalancedFreezable
        BalancedFreezable.__init__(self)
        # Initialize torch.nn.Module
        nn.Module.__init__(self)
        # Input Layer
        self.gen_layers = nn.ModuleList()
        self.gen_layers.append(
            ExpandingBlockStarShaped(c_in=z_dim, c_out=c_hidden, use_norm=True, activation='relu', kernel_size=(3, 5),
                                     stride=1, padding=0, n_classes=n_classes, bias=False, red_portion=red_portion),
        )
        # Hidden Layers
        for i in range(3):
            self.gen_layers.append(
                ExpandingBlockStarShaped(c_in=c_hidden // 2 ** i, use_norm=True, activation='relu', kernel_size=4,
                                         stride=2, padding=1, n_classes=n_classes, bias=False, red_portion=red_portion),
            )
        # Output Layer
        self.gen_layers.append(
            ExpandingBlockStarShaped(c_in=c_hidden // 8, c_out=c_out, use_norm=False, activation='tanh', kernel_size=4,
                                     stride=2, padding=1, n_classes=n_classes, bias=False, red_portion=red_portion),
        )
        self.z_dim = z_dim
        self.red_portion = red_portion
        self.n_classes = n_classes
        self.z_dim_red = int(self.z_dim * red_portion)
        self.z_dim_green = self.z_dim - self.z_dim_red

    def forward(self, z: torch.Tensor, class_idx: int or None = None) -> torch.Tensor:
        """
        :param Tensor z:
        :param (optional) class_idx:
        :return: a tensor of shape (n_classes, B, 2, W, H) if class_idx is None else (B, 2, W, H)
        """
        # assert z.dim() == 2, f'z must be 2-dimensional ({z.dim()}-d tensor provided)'
        z = z.reshape([z.shape[0], -1, 1, 1])
        y_red = z[:, :self.z_dim_red, :, :]
        y_green = z[:, self.z_dim_red:, :, :]
        if class_idx is None:
            y_green = [y_green] * self.n_classes
            for layer in self.gen_layers:
                y_red, y_green = layer(y_red, y_green, class_idx=None)
            output = torch.stack([
                torch.cat([y_red, y_green[i]], dim=1)
                for i in range(self.n_classes)
            ], dim=0)
        else:
            for layer in self.gen_layers:
                y_red, y_green = layer(y_red, y_green, class_idx=class_idx)
            output = torch.cat([y_red, y_green], dim=1)
        return output

    def get_random_z(self, batch_size: int = 1, device='cpu') -> torch.Tensor:
        return torch.randn(batch_size, self.z_dim, device=device)

    def load_aosokin_state_dict(self, state_dict: OrderedDictT[str, torch.Tensor], class_idx: int = 0):
        self_keys = [k for k in self.state_dict().keys() if not k.endswith('num_batches_tracked')]
        sd_keys = list(state_dict.keys())
        class_state_dict = OrderedDict({k_new: state_dict[k] for k, k_new in zip(sd_keys, self_keys)})
        self.load_state_dict(class_state_dict)

    def get_gens(self):
        if self.__class__.SUB_GENS is None:
            self.__class__.SUB_GENS = [DCGanSubGeneratorStarShaped(self, rank=i) for i in range(self.n_classes)]
        else:
            [sub_gen.reset() for sub_gen in self.__class__.SUB_GENS]
        return self.__class__.SUB_GENS


if __name__ == '__main__':
    gen = DCGanGeneratorStarShaped(c_out=2, z_dim=100, n_classes=6, c_hidden=512, red_portion=0.5)
    # print(gen)
    x = torch.rand(2, 100)
    y = gen(x)
    print(y.shape)
