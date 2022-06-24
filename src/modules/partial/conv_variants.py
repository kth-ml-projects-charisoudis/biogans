from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


#################################
# Separable Conv Layers
################################

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


class ConvTranspose2dSeparable(nn.Module):
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
        super(ConvTranspose2dSeparable, self).__init__()
        self.c_in_red = int(c_in * red_portion)
        self.c_out_red = int(c_out * red_portion)
        self.conv_red = nn.ConvTranspose2d(self.c_in_red, self.c_out_red, kernel_size, stride, padding, bias=bias)
        self.c_out_green = c_out - self.c_out_red
        self.conv_green = nn.ConvTranspose2d(c_in, self.c_out_green, kernel_size, stride, padding, bias=bias)

    def forward(self, x: torch.Tensor):
        x_red = x[:, :self.c_in_red, :, :]
        y_red = self.conv_red(x_red)
        y_green = self.conv_green(x)
        return torch.cat((y_red, y_green), dim=1)


#################################
# Star-Shaped Conv Layers
################################

class ConvTranspose2dStarShaped(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int = 1, padding: int = 0, bias=True,
                 red_portion=0.5, n_classes: int = 6):
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
        super(ConvTranspose2dStarShaped, self).__init__()
        self.c_in_red = int(c_in * red_portion)
        self.c_out_red = int(c_out * red_portion)
        self.conv_red = nn.ConvTranspose2d(self.c_in_red, self.c_out_red, kernel_size, stride, padding, bias=bias)
        self.c_out_green = c_out - self.c_out_red

        self.conv_green = nn.ModuleList()
        for i_c in range(n_classes):
            self.conv_green.append(
                nn.ConvTranspose2d(c_in, self.c_out_green, kernel_size, stride, padding, bias=bias)
            )
        self.n_classes = n_classes

    def forward(self, x_red: Tensor, x_green: Tensor, class_idx: int or None = None) -> Tuple[Tensor, Tensor or list]:
        """
        :param Tensor x_red: green input(s) of shape (B, 1, W, H)
        :param Tensor x_green: image tensor of shape (n_classes, B, 1, H, W) if class_idx is None, else (B, 1, W, H)
        :param (optional) class_idx:
        """
        y_red = self.conv_red(x_red)
        if class_idx is None:
            y_green = [None] * self.n_classes
            for class_idx in range(self.n_classes):
                x_green_i = torch.cat([x_red, x_green[class_idx]], dim=1)
                y_green[class_idx] = self.conv_green[class_idx](x_green_i)
        else:
            x_green_i = torch.cat([x_red, x_green], dim=1)
            y_green = self.conv_green[class_idx](x_green_i)
        return y_red, y_green
