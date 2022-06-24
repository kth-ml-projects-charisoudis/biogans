import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from modules.partial.conv_variants import Conv2dSeparable, ConvTranspose2dSeparable, ConvTranspose2dStarShaped


class ExpandingBlock(nn.Module):
    """
    ExpandingBlock Class:
    Performs a convolutional transpose operation in order to up-sample, with an optional norm and non-linearity
    activation functions.
    """

    def __init__(self, c_in: int, use_norm: bool = True, kernel_size: int = 3, activation: Optional[str] = 'relu',
                 output_padding: int = 1, stride: int = 2, padding: int = 1, c_out: Optional[int] = None,
                 norm_type: str = 'instance', use_dropout: bool = False, use_skip: bool = False, bias: bool = True,
                 red_portion: Optional[float] = None):
        """
        ExpandingBlock class constructor.
        :param int c_in: number of input channels
        :param bool use_norm: indicates if InstanceNormalization2d is applied or not after Conv2d layer
        :param int kernel_size: filter (kernel) size of torch.nn.Conv2d
        :param str activation: type of activation function used (supported: 'relu', 'lrelu')
        :param int output_padding: output_padding of torch.nn.ConvTranspose2d
        :param int stride: stride of torch.nn.ConvTranspose2d (defaults to 2 for our needs)
        :param int padding: padding of torch.nn.ConvTranspose2d
        :param str c_out: number of output channels
        :param str norm_type: available types are 'batch', 'instance', 'pixel', 'layer'
        :param bool use_dropout: set to True to add a `nn.Dropout()` layer with probability of 0.2
        :param bool use_skip: set to True to enable UNET-like behaviour
        :param bool bias: see nn.Conv{Transpose}2d
        :param (optional) red_portion: if set, Separable architecture is employed instead of the original one
        """
        super(ExpandingBlock, self).__init__()
        c_out = c_in // 2 if c_out is None else c_out

        # Upscaling layer using transposed convolution
        # noinspection PyTypeChecker
        if c_out != c_in:
            if red_portion is None:
                self.upscale = nn.ConvTranspose2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                                  output_padding=output_padding, bias=bias)
            else:
                self.upscale = ConvTranspose2dSeparable(c_in, c_out, kernel_size, stride=stride, padding=padding,
                                                        bias=bias, red_portion=red_portion)
        else:
            if red_portion is None:
                self.upscale = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                         bias=bias)
            else:
                self.upscale = Conv2dSeparable(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=bias,
                                               red_portion=red_portion)
        _layers = []
        if use_skip:
            # noinspection PyTypeChecker
            _layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))

        if use_norm:
            from modules.partial.normalization import PixelNorm2d, LayerNorm2d
            switcher = {
                'batch': nn.BatchNorm2d(c_out),
                'instance': nn.InstanceNorm2d(c_out),
                'pixel': PixelNorm2d(),
                'layer': LayerNorm2d(c_out),
            }
            _layers.append(switcher[norm_type])
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
        self.expanding_block = nn.Sequential(*_layers)

        self.c_out = c_out
        self.use_skip = use_skip

    def forward(self, x: Tensor, skip_conn_at_x: Optional[Tensor] = None) -> Tensor:
        """
        Function for completing a forward pass of ExpandingBlock:
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        :param (Tensor) x: image tensor of shape (N, C, H, W)
        :param (Tensor) skip_conn_at_x: the image tensor from the contracting path (from the opposing block of x)
                                        for the skip connection
        :return: transformed image tensor of shape (N, C/2, H*2, W*2)
        """
        # Check if skip connection is present
        assert self.use_skip is False or skip_conn_at_x is not None, 'use_skip was set, but skip_conn_at_x is None!'
        # Upscale current input
        x = self.upscale(x)
        # Append skip connection (if one exists)
        if self.use_skip:
            # Specify cat()'s dim to be 1 (aka channels), since we want a channel-wise concatenation of the two tensors
            x = torch.cat([x, ExpandingBlock.crop_skip_connection(skip_conn_at_x, x.shape)], dim=1)
        return self.expanding_block(x)

    @staticmethod
    def crop_skip_connection(skip_con: Tensor, shape: torch.Size) -> Tensor:
        """
        Function for cropping an image tensor: Given an image tensor and the new shape,
        crops to the center pixels. Crops (H, W) dims of input tensor.
        :param skip_con: image tensor of shape (N, C, H, W)
        :param shape: a torch.Size object with the shape you want skip_con to have: (N, C, H_hat, W_hat)
        :return: torch.Tensor of shape (N, C, H_hat, W_hat)
        """
        new_w = shape[-1]
        start_index = math.ceil((skip_con.shape[-1] - new_w) / 2.0)
        cropped_skip_con = skip_con[:, :, start_index:start_index + new_w, start_index:start_index + new_w]
        return cropped_skip_con


class FeatureMapLayer(nn.Module):
    """
    FeatureMapLayer Class:
    The final layer of a Generator; maps each the output to the desired number of output channels
    Values:
        c_in: the number of channels to expect from a given input
        c_out: the number of channels to expect for a given output
    """

    def __init__(self, c_in: int, c_out: int):
        """
        FeatureMapLayer class constructor.
        :param c_in: number of output channels
        """
        super(FeatureMapLayer, self).__init__()
        # noinspection PyTypeChecker
        self.feature_map_block = nn.Conv2d(c_in, c_out, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of FeatureMapLayer:
        Given an image tensor, returns it mapped to the desired number of channels.
        :param x: image tensor of shape (N, C_in, H, W)
        :return: transformed image tensor of shape (N, C_out, H, W)
        """
        return self.feature_map_block(x)


class ChannelsProjectLayer(nn.Module):
    """
    ChannelsProjectLayer Class:
    Layer to project C_in channels of input tensor to C_out channels in output tensor
    Values:
        c_in: the number of channels to expect from a given input
        c_out: the number of channels to expect for a given output
    """

    def __init__(self, c_in: int, c_out: int, use_spectral_norm: bool = False, padding: int = 0,
                 kernel_size: int or tuple = 1, stride: int or tuple = 1, bias: bool = True,
                 red_portion: Optional[float] = None):
        """
        ChannelsProjectLayer class constructor.
        :param int c_in: number of input channels
        :param int c_out: number of output channels
        :param bool use_spectral_norm: set to True to add a spectral normalization layer after the Conv2d
        :param int padding: nn.Conv2d's padding argument
        :param int kernel_size: filter (kernel) size of torch.nn.Conv2d
        :param int stride: stride of torch.nn.ConvTranspose2d (defaults to 2 for our needs)
        :param bool bias: use bias in conv layers
        :param (optional) red_portion: if set, Separable architecture is employed instead of the original one
        """
        super(ChannelsProjectLayer, self).__init__()
        # noinspection PyTypeChecker
        if red_portion is None:
            self.feature_map_block = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                               bias=bias)
            if use_spectral_norm:
                self.feature_map_block = nn.utils.spectral_norm(self.feature_map_block)
        else:
            self.feature_map_block = Conv2dSeparable(c_in, c_out, kernel_size=kernel_size, stride=stride,
                                                     padding=padding, bias=bias, red_portion=red_portion)
            self.feature_map_block.conv_red = nn.utils.spectral_norm(self.feature_map_block.conv_red)
            self.feature_map_block.conv_green = nn.utils.spectral_norm(self.feature_map_block.conv_green)

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of ChannelsProjectLayer:
        Given an image tensor, returns it mapped to the desired number of channels.
        :param x: image tensor of shape (N, C_in, H, W)
        :return: transformed image tensor of shape (N, C_out, H, W)
        """
        return self.feature_map_block(x)


class ExpandingBlockStarShaped(nn.Module):
    """
    ExpandingBlockStarShaped Class:
    Performs a convolutional transpose operation in order to up-sample, with an optional norm and non-linearity
    activation functions, using the "Star-Shaped" convolutions.
    """

    def __init__(self, c_in: int, use_norm: bool = True, kernel_size: int = 3, activation: Optional[str] = 'relu',
                 n_classes: int = 6, stride: int = 2, padding: int = 1, c_out: Optional[int] = None,
                 bias: bool = True, red_portion: float = 0.5):
        """
        ExpandingBlock class constructor.
        :param int c_in: number of input channels
        :param bool use_norm: indicates if InstanceNormalization2d is applied or not after Conv2d layer
        :param int kernel_size: filter (kernel) size of torch.nn.Conv2d
        :param str activation: type of activation function used (supported: 'relu', 'lrelu')
        :param int n_classes: number of classes for Star-Shaped convolutions
        :param int stride: stride of torch.nn.ConvTranspose2d (defaults to 2 for our needs)
        :param int padding: padding of torch.nn.ConvTranspose2d
        :param str c_out: number of output channels
        :param bool bias: see nn.Conv{Transpose}2d
        :param (optional) red_portion: if set, Separable architecture is employed instead of the original one
        """
        super(ExpandingBlockStarShaped, self).__init__()
        c_out = c_in // 2 if c_out is None else c_out

        # Upscaling layer using transposed convolution
        self.upscale = ConvTranspose2dStarShaped(c_in, c_out, kernel_size, stride=stride, padding=padding,
                                                 bias=bias, red_portion=red_portion, n_classes=n_classes)
        self.use_norm = use_norm
        if use_norm:
            self.bn_red = nn.BatchNorm2d(int(c_out * red_portion))
            self.bn_green = nn.BatchNorm2d(c_out - int(c_out * red_portion))

        self.use_activation = activation is not None
        if self.use_activation:
            activations_switcher = {
                'relu': nn.ReLU(inplace=True),
                'lrelu': nn.LeakyReLU(0.2, inplace=True),
                'tanh': nn.Tanh(),
                'sigmoid': nn.Sigmoid(),
            }
            self.activation = activations_switcher[activation]
        self.c_out = c_out
        self.n_classes = n_classes

    def forward(self, x_red: Tensor, x_green: Tensor, class_idx: int or None = None) -> Tensor:
        """
        Function for completing a forward pass of ExpandingBlockStarShaped:
        :param Tensor x_red: green input(s) of shape (B, 1, W, H)
        :param Tensor x_green: image tensor of shape (n_classes, B, 1, H, W) if class_idx is None, else (B, 1, W, H)
        :param (optional) class_idx:
        :return: transformed image tensors of shape (N, cout_red, H*2, W*2) and (n_classes, N, cout_green, H*2, W*2)
        """
        # Upscale current input
        x_red, x_green = self.upscale(x_red, x_green, class_idx)
        # Norm
        if self.use_norm:
            x_red = self.bn_red(x_red)
            if class_idx is None:
                x_green = [self.bn_green(x_green_i) for x_green_i in x_green]
        # Activation
        if self.activation:
            x_red = self.bn_red(x_red)
            if class_idx is None:
                x_green = [self.bn_green(x_green_i) for x_green_i in x_green]

        # Append skip connection (if one exists)
        if self.use_activation:
            x_red = self.activation(x_red)
            if class_idx is None:
                x_green = [self.activation(x_green_i) for x_green_i in x_green]
        return x_red, x_green
