from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from modules.partial.decoding import ChannelsProjectLayer
from modules.partial.encoding import ContractingBlock
from utils import pytorch
from utils.command_line_logger import CommandLineLogger
from utils.ifaces import BalancedFreezable, Verbosable
from utils.pytorch import get_total_params, get_gradient_penalty, get_gradient_penalties
from utils.string import to_human_readable


class DCGanDiscriminator(nn.Module, BalancedFreezable, Verbosable):
    """
    DCGanDiscriminator Class:
    This class implements the DCGAN discriminator network as was proposed in the DCGAN paper and modified in "GANs for
    Biological Image Synthesis"
    """

    def __init__(self, c_in: int, c_hidden: int = 64, n_contracting_blocks: int = 4, use_spectral_norm: bool = False,
                 logger: Optional[CommandLineLogger] = None, adv_criterion: Optional[str] = None,
                 output_kernel_size: Optional[Tuple] = None, red_portion: Optional[float] = None,
                 gp_lambda: bool = None):
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
        :param (optional) red_portion: if set, Separable architecture will be employed
        """
        if output_kernel_size is None:
            output_kernel_size = (3, 5)

        # Initialize utils.ifaces.BalancedFreezable
        BalancedFreezable.__init__(self)

        # Initialize torch.nn.Module
        nn.Module.__init__(self)
        self.disc = nn.Sequential(
            # Encoding (aka contracting) blocks
            ContractingBlock(c_in=c_in, c_out=c_hidden, kernel_size=4, stride=2, padding=1, bias=False,
                             activation='lrelu', red_portion=red_portion),
            *[
                ContractingBlock(c_hidden * 2 ** i, kernel_size=4, stride=2, padding=1, use_norm=True, bias=False,
                                 norm_type='batch', activation='lrelu', red_portion=red_portion)
                for i in range(0, n_contracting_blocks - 1)
            ],
            ChannelsProjectLayer(c_hidden * 2 ** (n_contracting_blocks - 1), 1, use_spectral_norm=use_spectral_norm,
                                 kernel_size=output_kernel_size, stride=1, padding=0, bias=False,
                                 red_portion=red_portion)
        )

        # Save args
        self.n_contracting_blocks = n_contracting_blocks
        self.logger = CommandLineLogger(name=self.__class__.__name__) if logger is None else logger
        self.adv_criterion = None
        if adv_criterion is not None:
            if hasattr(nn, f'{adv_criterion}Loss'):
                self.adv_criterion = getattr(nn, f'{adv_criterion}Loss')()
            elif hasattr(pytorch, f'{adv_criterion}Loss'):
                self.adv_criterion = getattr(pytorch, f'{adv_criterion}Loss')()
            else:
                raise RuntimeError(f'adv_criterion="{adv_criterion}" could be found (tried torch.nn and utils.pytorch)')
        self.gp_lambda = gp_lambda
        self.verbose_enabled = False

    @property
    def nparams_hr(self):
        return to_human_readable(get_total_params(self))

    def forward(self, x: Tensor) -> Tensor:
        """
        Function for completing a forward pass of PatchGANDiscriminator:
        Given an image tensor, returns a 2D matrix of realness probabilities for each image's "patches".
        :param (torch.Tensor) x: image tensor of shape (N, C_in, H, W)
        :return: transformed image tensor of shape (N, 1, P_h, P_w)
        """
        if self.verbose_enabled:
            self.logger.debug(f'_: {x.shape}')
        return self.disc(x).view(-1)

    # noinspection DuplicatedCode
    def get_loss_both(self, real: Tensor, fake: Tensor, criterion: Optional[nn.modules.Module] = None) -> Tensor:
        """
        Compute adversarial loss using both real and fake images.
        :param (torch.Tensor) real: image tensor of shape (N, C, H, W) from real dataset
        :param (torch.Tensor) fake: image tensor of shape (N, C, H, W) produced by generator (i.e. fake images)
        :param (optional) criterion: loss function (such as nn.BCELoss, nn.MSELoss and others)
        :return: torch.Tensor containing loss value(s)
        """
        # scores_a = self(real)
        # scores_b = self(fake)
        # gradient_penalties = get_gradient_penalty(self, real.data, fake.data)
        # mean_dim = 0 if scores_a.dim() in [1, 4] else 1
        # gradient_penalty = gradient_penalties.mean(mean_dim)
        # return scores_a.mean(mean_dim) - scores_b.mean(mean_dim) + self.gp_lambda * gradient_penalty

        loss_on_real = self.get_loss(real, is_real=True, criterion=criterion)
        loss_on_fake = self.get_loss(fake, is_real=False, criterion=criterion)
        total_loss = loss_on_real + loss_on_fake
        if self.gp_lambda is None:
            return total_loss
        # Calculate gradient penalty and append to loss
        gradient_penalty = get_gradient_penalty(disc=self, real=real, fake=fake).mean()
        return total_loss + self.gp_lambda * gradient_penalty

    # noinspection DuplicatedCode
    def get_loss(self, x: Tensor, is_real: bool, criterion: Optional[nn.modules.Module] = None) -> Tensor:
        """
        Compute adversarial loss using both real and fake images.
        :param (torch.Tensor) x: image tensor of shape (N, C, H, W) from either dataset
        :param (bool) is_real: set to True to compare predictions with 1s, else with 0s
        :param (optional) criterion: loss function (such as nn.BCELoss, nn.MSELoss and others)
        :return: torch.Tensor containing loss value(s)
        """
        # Setup criterion
        criterion = self.adv_criterion if criterion is None else criterion
        assert criterion is not None
        # Proceed with loss calculation
        predictions = self(x)
        # print('DISC OUTPUT SHAPE: ' + str(predictions.shape))
        if type(criterion) == nn.modules.loss.BCELoss:
            predictions = nn.Sigmoid()(predictions)
        if type(criterion) == pytorch.WassersteinLoss:
            reference = -1.0 * torch.ones_like(predictions) if is_real else 1.0 * torch.ones_like(predictions)
        else:
            reference = torch.ones_like(predictions) if is_real else torch.zeros_like(predictions)
        return criterion(predictions, reference)
        # scores = self(x)
        # mean_dim = 0 if scores.dim() in [1, 4] else 1
        # return scores.mean(mean_dim) if is_real else -scores.mean(mean_dim)

    def get_layer_attr_names(self) -> List[str]:
        return ['patch_gan_discriminator', ]


class DCGanDiscriminatorInd6Class(nn.Module, BalancedFreezable, Verbosable):
    def __init__(self, **disc_kwargs):
        # Initialize utils.ifaces.BalancedFreezable
        BalancedFreezable.__init__(self)
        # Initialize torch.nn.Module
        nn.Module.__init__(self)
        # Initialize Discriminators
        self.disc = nn.ModuleList([
            DCGanDiscriminator(**disc_kwargs)
            for _ in range(6)
        ])
        self.adv_criterion = self.disc[0].adv_criterion
        self.gp_lambda = self.disc[0].gp_lambda
        self.verbose_enabled = False

    @property
    def nparams_hr(self):
        return to_human_readable(get_total_params(self))

    def forward(self, x: Tensor) -> Tensor:
        if self.verbose_enabled:
            self.logger.debug(f'_: {x.shape}')
        out = [None] * 6
        for class_idx in range(6):
            out[class_idx] = self.disc[class_idx](x[class_idx])
        return torch.stack(out, dim=0)

    # # noinspection DuplicatedCode
    # def get_loss_both(self, real: Tensor, fake: Tensor) -> Tensor:
    #     assert type(self.adv_criterion) == pytorch.WassersteinLoss
    #     scores_r = self(real)
    #     scores_f = self(fake)
    #     gradient_penalties = get_gradient_penalty(disc=self, real=real.data, fake=fake.data)
    #     while gradient_penalties.dim() < scores_r.dim():
    #         gradient_penalties = gradient_penalties.unsqueeze(-1)
    #
    #     mean_dim = 1
    #     scores_r = scores_r.mean(mean_dim)
    #     scores_f = scores_f.mean(mean_dim)
    #     return torch.stack([
    #         scores_r[class_idx] - scores_f[class_idx] + self.gp_lambda * gradient_penalties[class_idx]
    #         for class_idx in range(6)
    #     ], dim=0).mean()
    #
    # # noinspection DuplicatedCode
    # def get_loss(self, x: Tensor, is_real: bool) -> Tensor:
    #     assert type(self.adv_criterion) == pytorch.WassersteinLoss
    #     scores_r = self(x)
    #     return scores_r.mean() if is_real else scores_r.mean()

    # noinspection DuplicatedCode
    def get_loss_both(self, real: Tensor, fake: Tensor, criterion: Optional[nn.modules.Module] = None) -> Tensor:
        """
        Compute adversarial loss using both real and fake images.
        :param (torch.Tensor) real: image tensor of shape (N, C, H, W) from real dataset
        :param (torch.Tensor) fake: image tensor of shape (N, C, H, W) produced by generator (i.e. fake images)
        :param (optional) criterion: loss function (such as nn.BCELoss, nn.MSELoss and others)
        :return: torch.Tensor containing loss value(s)
        """
        # scores_a = self(real)
        # scores_b = self(fake)
        # gradient_penalties = get_gradient_penalty(self, real.data, fake.data)
        # mean_dim = 0 if scores_a.dim() in [1, 4] else 1
        # gradient_penalty = gradient_penalties.mean(mean_dim)
        # return scores_a.mean(mean_dim) - scores_b.mean(mean_dim) + self.gp_lambda * gradient_penalty

        if type(criterion) == pytorch.WassersteinLoss:
            scores_real = self(real)
            scores_fake = self(fake)
            gradient_penalties = get_gradient_penalties(self, real.data, fake.data)
            mean_dim = 0 if scores_real.dim() == 1 else 1
            gradient_penalty = gradient_penalties.mean(mean_dim)
            return scores_real.mean(mean_dim) - scores_fake.mean(mean_dim) + self.gp_lambda * gradient_penalty

        loss_on_real = self.get_loss(real, is_real=True, criterion=criterion)
        loss_on_fake = self.get_loss(fake, is_real=False, criterion=criterion)
        total_loss = loss_on_real + loss_on_fake
        if self.gp_lambda is None:
            return total_loss
        # Calculate gradient penalty and append to loss
        gradient_penalties = get_gradient_penalties(self, real.data, fake.data)
        mean_dim = 0 if real.dim() == 4 else 1
        gradient_penalty = gradient_penalties.mean(mean_dim)
        return total_loss + self.gp_lambda * gradient_penalty

    # noinspection DuplicatedCode
    def get_loss(self, x: Tensor, is_real: bool, criterion: Optional[nn.modules.Module] = None) -> Tensor:
        """
        Compute adversarial loss using both real and fake images.
        :param (torch.Tensor) x: image tensor of shape (N, C, H, W) from either dataset
        :param (bool) is_real: set to True to compare predictions with 1s, else with 0s
        :param (optional) criterion: loss function (such as nn.BCELoss, nn.MSELoss and others)
        :return: torch.Tensor containing loss value(s)
        """
        # Setup criterion
        criterion = self.adv_criterion if criterion is None else criterion
        assert criterion is not None
        # Proceed with loss calculation
        predictions = self(x)
        # print('DISC OUTPUT SHAPE: ' + str(predictions.shape))
        if type(criterion) == nn.modules.loss.BCELoss:
            predictions = nn.Sigmoid()(predictions)
        if type(criterion) == pytorch.WassersteinLoss:
            reference = -1.0 if is_real else 1.0
        else:
            reference = torch.ones_like(predictions) if is_real else torch.zeros_like(predictions)
        return criterion(predictions, 1.0 if is_real else -1.0)
        # scores = self(x)
        # mean_dim = 0 if scores.dim() in [1, 4] else 1
        # return scores.mean(mean_dim) if is_real else -scores.mean(mean_dim)

    def get_layer_attr_names(self) -> List[str]:
        return [f'dcgan_disc_{i}' for i in range(6)]


if __name__ == '__main__':
    _disc = DCGanDiscriminator(c_in=2, n_contracting_blocks=4, use_spectral_norm=True, adv_criterion='MSE',
                               output_kernel_size=(3, 5))
    # print(_disc)
    # print(_disc.nparams_hr)
    out = _disc(torch.rand(6, 2, 2, 48, 80))
    print(out.shape, out.dim())
