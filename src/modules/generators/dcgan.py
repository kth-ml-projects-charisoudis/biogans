from collections import OrderedDict
from typing import Optional, OrderedDict as OrderedDictT

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
        # assert z.dim() == 2, f'z must be 2-dimensional ({z.dim()}-d tensor provided)'
        return self.gen(z.reshape([z.shape[0], -1, 1, 1]))

    def get_random_z(self, batch_size: int = 1, device='cpu') -> torch.Tensor:
        return torch.randn(batch_size, self.z_dim, device=device)

    def load_aosokin_state_dict(self, state_dict: OrderedDictT[str, torch.Tensor], class_idx: int = 0):
        self_keys = [k for k in self.state_dict().keys() if not k.endswith('num_batches_tracked')]
        sd_keys = [k for k in state_dict.keys() if k.startswith(f'main.{class_idx}')]
        class_state_dict = OrderedDict({k_new: state_dict[k] for k, k_new in zip(sd_keys, self_keys)})
        self.load_state_dict(class_state_dict)


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


class DCGanGeneratorInd6Class(nn.Module, BalancedFreezable):
    def __init__(self, c_out: int = 2, z_dim: int = 100, norm_type: str = 'batch', c_hidden: int = 512,
                 n_extra_layers: int = 0, red_portion: Optional[float] = None):
        # Initialize utils.ifaces.BalancedFreezable
        BalancedFreezable.__init__(self)
        # Initialize torch.nn.Module
        nn.Module.__init__(self)
        # Initialize all generators
        self.gens = nn.ModuleList([
            DCGanGenerator(c_out=c_out, z_dim=z_dim, norm_type=norm_type, c_hidden=c_hidden,
                           n_extra_layers=n_extra_layers, red_portion=red_portion)
            for _ in range(6)
        ])
        self.z_dim = z_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = [None] * 6
        for i_class in range(6):
            out[i_class] = self.gens[i_class](z.clone())
        return torch.stack(out, dim=0)

    def get_random_z(self, batch_size: int = 1, device='cpu') -> torch.Tensor:
        return self.gens[0].get_random_z(batch_size, device)

    def load_aosokin_state_dict(self, state_dict: OrderedDictT[str, torch.Tensor], logger=None):
        for class_idx in range(6):
            if logger:
                logger.debug(f'    - loading class={class_idx}')
            self.gens[class_idx].load_aosokin_state_dict(state_dict, class_idx)


if __name__ == '__main__':
    # _gen = DCGanGenerator(c_out=6 + 1, z_dim=100, n_extra_layers=2)
    # _z = torch.randn(10, 100)
    # print(_gen)
    # print(_gen(_z).shape)
    # get_total_params(_gen, True, True)

    _gen = SeparableDCGanGenerator(c_out=2, n_extra_layers=0)
    print(_gen)
    _out = _gen(_gen.get_random_z(batch_size=1))
    print(_out.shape)
    x_red, x_green = torch.split(_out, [_gen.c_out_red, _gen.c_out_green], dim=1)
    print(x_red.shape, x_green.shape)
    get_total_params(_gen, True, True)

    # with open('/home/achariso/PycharmProjects/kth-ml-course-projects/biogans/.gdrive_personal/Models/model_name' +
    #           '=BioGanInd1class_alp14/Configurations/wgan-gp-independent-sep.yaml') as fp:
    #     config = yaml.load(fp, Loader=yaml.FullLoader)
    # _gen = DCGanGenerator(**config['gen'])
    # chkpt_path = '/aosokin_wgan_id_sep.pth'
    # _gen.load_aosokin_state_dict(state_dict=torch.load(chkpt_path, map_location='cpu'))
