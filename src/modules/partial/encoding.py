from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from utils.pytorch import get_total_params
from utils.string import to_human_readable


class Conv2dSeparable(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int = 1, padding: int = 0, bias=True,
                 red_portion=0.5):
        """
        ConvTranspose2dSeparable class constructor.
        :param int c_in: see nn.ConvTranspose2d
        :param int c_out: see nn.ConvTranspose2d
        :param int kernel_size: see nn.ConvTranspose2d
        :param int stride: see nn.ConvTranspose2d
        :param int padding: see nn.ConvTranspose2d
        :param bool bias: see nn.ConvTranspose2d
        :param float red_portion: portion of red channels (e.g. 0.5 for 2-channel outputs, or 0.166 for 7-channel outs)
        """
        super(Conv2dSeparable, self).__init__()
        self.c_in_red = int(c_in * red_portion)
        self.c_out_red = int(c_out * red_portion)
        self.conv_red = nn.Conv2d(self.c_in_red, self.c_out_red, kernel_size, stride, padding, bias=bias)
        self.c_out_green = c_out - self.c_out_red
        self.conv_green = nn.Conv2d(c_in, self.c_out_green, kernel_size, stride, padding, bias=bias)

    def forward(self, x: torch.Tensor):
        x_red = x[:, :self.c_in_red, :, :]
        y_red = self.conv_red(x_red)
        y_green = self.conv_green(x)
        return torch.cat((y_red, y_green), dim=1)


class ContractingBlock(nn.Module):
    """
    ContractingBlock Class
    Performs a convolution followed by a max pool operation and an optional instance norm.
    """

    def __init__(self, c_in: int, use_norm: bool = False, kernel_size: int = 3, activation: Optional[str] = 'lrelu',
                 c_out: int = None, stride: int = 2, padding: int = 1, padding_mode: str = 'reflect',
                 norm_type: str = 'instance', use_dropout: bool = False, bias: bool = True,
                 red_portion: Optional[float] = None):
        """
        ContractingBlock class constructor.
        :param (int) c_in: the number of channels to expect from a given input
        :param (bool) use_norm: indicates if InstanceNormalization2d is applied or not after Conv2d layer
        :param (int) kernel_size: filter (kernel) size
        :param (optional) activation: type of activation function used (supported: 'relu', 'lrelu') or None to bypass
                                      output activation function
        :param (optional) c_out: set to `None` to output `2*c_in` channels
        :param (int) stride: stride as integer (same for W and H)
        :param (int) padding: padding as integer (same for W and H)
        :param (int) padding_mode: see torch.nn.Conv2d of more info on this argument
        :param (str) norm_type: available types are 'batch', 'instance', 'pixel', 'layer'
        :param (bool) use_dropout: set to True to add a nn.Dropout2d (aka Spatial Dropout) layer before activation layer
        :param (bool) bias: set to False to disable bias in conv layers
        :param (optional) red_portion: if set, Separable architecture will be employed
        """
        super(ContractingBlock, self).__init__()
        c_out = c_in * 2 if not c_out else c_out
        # noinspection PyTypeChecker
        if red_portion is None:
            _layers = [nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride,
                                 padding_mode=padding_mode, bias=bias), ]
        else:
            _layers = [Conv2dSeparable(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride,
                                       bias=bias, red_portion=red_portion), ]
        if use_norm:
            from modules.partial.normalization import PixelNorm2d, LayerNorm2d
            normalizations_switcher = {
                'batch': nn.BatchNorm2d(c_out),
                'instance': nn.InstanceNorm2d(c_out),
                'pixel': PixelNorm2d(),
                'layer': LayerNorm2d(c_out),
            }
            _layers.append(normalizations_switcher[norm_type])
        if use_dropout:
            _layers.append(nn.Dropout2d(p=0.2))
        if activation is not None:
            activations_switcher = {
                'relu': nn.ReLU(inplace=True),
                'lrelu': nn.LeakyReLU(0.2, inplace=True),
                'tanh': nn.Tanh(),
                'sigmoid': nn.Sigmoid(),
            }
            _layers.append(activations_switcher[activation])
        self.contracting_block = nn.Sequential(*_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of ContractingBlock:
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        :param x: image tensor of shape (N, C_in, H, W)
        :return: transformed image tensor of shape (N, C_in*2, H/2, W/2)
        """
        return self.contracting_block(x)


class UNETContractingBlock(nn.Module):
    """
    UNETContractingBlock Class:
    Performs two convolutions followed by a max pool operation.
    Attention: Unlike UNET paper, we add padding=1 to Conv2d layers to make a "symmetric" version of UNET.
    """

    def __init__(self, c_in: int, use_bn: bool = True, use_dropout: bool = False, kernel_size: int = 3,
                 activation: str = 'lrelu'):
        """
        UNETContractingBlock class constructor.
        :param (int) c_in: number of input channels
        :param (bool) use_bn: indicates if Batch Normalization is applied or not after Conv2d layer
        :param (bool) use_dropout: indicates if Dropout is applied or not after Conv2d layer
        :param (int) kernel_size: filter (kernel) size
        :param (str) activation: type of activation function used (supported: 'relu', 'lrelu')
        """
        super(UNETContractingBlock, self).__init__()
        self.unet_contracting_block = nn.Sequential(
            # 1st convolutional layer
            ContractingBlock(c_in=c_in, c_out=c_in * 2, kernel_size=kernel_size, stride=1, padding=1,
                             use_norm=use_bn, norm_type='batch', use_dropout=use_dropout, activation=activation),
            # 2nd convolutional layer
            ContractingBlock(c_in=c_in * 2, c_out=c_in * 2, kernel_size=kernel_size, stride=1, padding=1,
                             use_norm=use_bn, norm_type='batch', use_dropout=use_dropout, activation=activation),
            # Downsampling (using MaxPool) layer (preparing for next block)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of UNETContractingBlock:
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        :param (torch.Tensor) x: image tensor of shape (N, C_in, H, W)
        :return: transformed image tensor of shape (N, C_in*2, H/2, W/2)
        """
        return self.unet_contracting_block(x)


class MLPBlock(nn.Module):
    """
    MLPBlock Class
    This is a Multi-Layer Perceptron block composed of (Linear-Relu)*2+Linear.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, activation: str = 'relu', n_blocks: int = 3):
        """
        MLPBlock class constructor.
        :param (int) in_dim: number of input neurons
        :param (int) hidden_dim: number of neurons in hidden layers
        :param (int) out_dim: number of neurons in output layer
        :param (str) activation: type of activation function used (supported: 'relu', 'lrelu')
        :param (int) n_blocks: number of (FC+ReLU) blocks
        """
        super(MLPBlock, self).__init__()
        _layers = []
        for bi in range(n_blocks - 1):
            _layers.append(nn.Linear(in_features=in_dim if bi == 0 else hidden_dim, out_features=hidden_dim))
            _layers.append(nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2))
        _layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp_block = nn.Sequential(*_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of MLPBlock:
        Given a tensor, completes a MLP block and returns the transformed tensor.
        :param (torch.Tensor) x: tensor of shape (N, in_dim)
        :return: transformed tensor of shape (N, out_dim)
        """
        return self.mlp_block(x)


class NoiseMappingNetwork(MLPBlock):
    """
    NoiseMappingNetwork Class:
    This class implements a layer of the Noise Mapping network proposed in the original StyleGAN paper.
    """

    def __init__(self, z_dim: int, hidden_dim: int, w_dim: int, n_blocks: int = 3):
        """
        NoiseMappingNetwork class constructor.
        :param (int) z_dim: the dimension of the noise vector
        :param (int) hidden_dim: the inner dimension
        :param (int) w_dim: the dimension of the w-vector (i.e. the vector in the less-entangled learned vector space)
        :param (int) n_blocks: number of (FC+ReLU) blocks
        """
        super(NoiseMappingNetwork, self).__init__(in_dim=z_dim, hidden_dim=hidden_dim, out_dim=w_dim, n_blocks=n_blocks)


if __name__ == '__main__':
    _nmn = NoiseMappingNetwork(z_dim=512, hidden_dim=512, w_dim=512, n_blocks=4)
    print(_nmn)
    print(to_human_readable(get_total_params(_nmn)))
