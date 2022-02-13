from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from modules.partial.decoding import ChannelsProjectLayer
from modules.partial.encoding import ContractingBlock
from utils.command_line_logger import CommandLineLogger
from utils.ifaces import BalancedFreezable, Verbosable
from utils.pytorch import get_total_params
from utils.string import to_human_readable


class DCGanDiscriminator(nn.Module, BalancedFreezable, Verbosable):
    """
    DCGanDiscriminator Class:
    This class implements the DCGAN discriminator network as was proposed in the DCGAN paper and modified in "GANs for
    Biological Image Synthesis"
    """

    def __init__(self, c_in: int, c_hidden: int = 64, n_contracting_blocks: int = 4, use_spectral_norm: bool = False,
                 logger: Optional[CommandLineLogger] = None, adv_criterion: Optional[str] = None,
                 output_kernel_size: Optional[Tuple] = None):
        """
        DCGanDiscriminator class constructor.
        :param (int) c_in: number of input channels
        :param (int) c_hidden: the initial number of discriminator convolutional filters (channels)
        :param (int) n_contracting_blocks: number of contracting blocks
        :param (bool) use_spectral_norm: set to True to use Spectral Normalization in the ChannelsProject (last) layer
        :param (optional) logger: CommandLineLogger instance to be used when verbose is enabled
        :param (optional) adv_criterion: str description of desired default adversarial criterion (e.g. 'MSE', 'BCE',
                                         'BCEWithLogits', etc.). If None, then it must be set in the respective function
                                         call
        :param (optional) output_kernel_size: if not set defaults to (3,5); e.g. a (2,4) kernel results to a PatchGAN
                                              like discriminator with 2x2 output size
        """
        if output_kernel_size is None:
            output_kernel_size = (3, 5)

        # Initialize utils.ifaces.BalancedFreezable
        BalancedFreezable.__init__(self)

        # Initialize torch.nn.Module
        nn.Module.__init__(self)
        self.disc = nn.Sequential(
            # Encoding (aka contracting) blocks
            ContractingBlock(c_in=c_in, c_out=c_hidden, kernel_size=4, stride=2, padding=1),
            *[
                ContractingBlock(c_hidden * 2 ** i, kernel_size=4, stride=2, padding=1, use_norm=True,
                                 norm_type='batch')
                for i in range(0, n_contracting_blocks - 1)
            ],
            ChannelsProjectLayer(c_hidden * 2 ** (n_contracting_blocks - 1), 1, use_spectral_norm=use_spectral_norm,
                                 kernel_size=output_kernel_size, stride=1, padding=0)
        )

        # Save args
        self.n_contracting_blocks = n_contracting_blocks
        self.logger = CommandLineLogger(name=self.__class__.__name__) if logger is None else logger
        self.adv_criterion = getattr(nn, f'{adv_criterion}Loss')() if adv_criterion is not None else None
        self.verbose_enabled = False

    @property
    def nparams_hr(self):
        return to_human_readable(get_total_params(self))

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Function for completing a forward pass of PatchGANDiscriminator:
        Given an image tensor, returns a 2D matrix of realness probabilities for each image's "patches".
        :param (torch.Tensor) x: image tensor of shape (N, C_in, H, W)
        :param (torch.Tensor) y: image tensor of shape (N, C_y, H, W) containing the condition images (e.g. for pix2pix)
        :return: transformed image tensor of shape (N, 1, P_h, P_w)
        """
        if y is not None:
            x = torch.cat([x, y], dim=1)  # channel-wise concatenation
        if self.verbose_enabled:
            self.logger.debug(f'_: {x.shape}')
        return self.disc(x)

    # noinspection DuplicatedCode
    def get_loss(self, real: Tensor, fake: Tensor, condition: Optional[Tensor] = None,
                 criterion: Optional[nn.modules.Module] = None, real_unassoc: Optional[Tensor] = None) -> Tensor:
        """
        Compute adversarial loss.
        :param (torch.Tensor) real: image tensor of shape (N, C, H, W) from real dataset
        :param (torch.Tensor) fake: image tensor of shape (N, C, H, W) produced by generator (i.e. fake images)
        :param (optional) condition: condition image tensor of shape (N, C_in/2, H, W) that is stacked before input to
                                     PatchGAN discriminator (optional)
        :param (optional) criterion: loss function (such as nn.BCELoss, nn.MSELoss and others)
        :param (torch.Tensor) real_unassoc: (to use only for Associated/Unassociated discriminator (e.g. PixelDTGan))
        :return: torch.Tensor containing loss value(s)
        """
        # Setup criterion
        criterion = self.adv_criterion if criterion is None else criterion
        # Proceed with loss calculation
        predictions_on_real = self(real, condition)
        predictions_on_fake = self(fake, condition)
        print(predictions_on_fake.shape)
        # print('DISC OUTPUT SHAPE: ' + str(predictions_on_fake.shape))
        if type(criterion) == torch.nn.modules.loss.BCELoss:
            predictions_on_real = nn.Sigmoid()(predictions_on_real)
            predictions_on_fake = nn.Sigmoid()(predictions_on_fake)
        loss_on_real = criterion(predictions_on_real, torch.ones_like(predictions_on_real))
        loss_on_fake = criterion(predictions_on_fake, torch.zeros_like(predictions_on_fake))
        losses = [loss_on_real, loss_on_fake]
        if real_unassoc is not None:
            predictions_on_real_unassoc = self(real_unassoc[0:condition.shape[0], :, :, :], condition)
            if type(criterion) == torch.nn.modules.loss.BCELoss:
                predictions_on_real_unassoc = nn.Sigmoid()(predictions_on_real_unassoc)
            loss_on_real_unassoc = criterion(predictions_on_real_unassoc, torch.zeros_like(predictions_on_real_unassoc))
            losses.append(loss_on_real_unassoc)
        return torch.mean(torch.stack(losses))

    def get_layer_attr_names(self) -> List[str]:
        return ['patch_gan_discriminator', ]


if __name__ == '__main__':
    _disc = DCGanDiscriminator(c_in=2, n_contracting_blocks=4, use_spectral_norm=True, adv_criterion='MSE',
                               output_kernel_size=(3, 5))
    print(_disc)
    print(_disc.nparams_hr)

    _h_in, _w_in = 48, 80
    _real = torch.randn(1, 2, _h_in, _w_in)
    _fake = torch.randn(1, 2, _h_in, _w_in)
    _loss = _disc.get_loss(real=_real, fake=_fake)
    print(_loss)
