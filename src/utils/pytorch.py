import math
import os
import random
from typing import Any, Optional, Tuple

import PIL
import humanize
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Function, Variable
from torch.nn import Module
from torchvision.transforms import transforms

from utils.command_line_logger import CommandLineLogger
from utils.ifaces import Verbosable
from utils.string import to_human_readable


def corr(x: torch.Tensor, eps: float = 1e-08) -> torch.Tensor:
    """
    Compute the correlation matrix of given input vector $x$ using the unbiased sample correlation estimator.
    (credits to mauricett, https://github.com/pytorch/pytorch/issues/19037#issuecomment-739002393)
    :param x: the input vector of shape (N, n_vars)
    :param eps: constant for numerical stability when normalizing $x$
    :return: a torch.Tensor of shape (n_vars, n_vars)
    """
    n_samples, n_vars = tuple(x.shape)
    x -= torch.mean(x, dim=0)
    x /= torch.std(x, dim=0) + eps
    return 1 / (n_samples - 1) * x.transpose(-1, -2) @ x


def cov(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the covariance matrix of given input vector $x$ using the unbiased sample covariance estimator.
    (credits to mauricett, https://github.com/pytorch/pytorch/issues/19037#issuecomment-739002393)
    :param x: the input vector of shape (N, n_vars)
    :return: a torch.Tensor of shape (n_vars, n_vars)
    """
    n_samples, n_vars = tuple(x.shape)
    x -= torch.mean(x, dim=0)
    return 1 / (n_samples - 1) * x.transpose(-1, -2) @ x


def enable_verbose(model: nn.Module, logger: Optional[CommandLineLogger] = None) -> None:
    """
    Register verbose hooks on model's forward pass to output the shape of each layer's output tensor.
    :param (nn.Module) model: an torch.nn.Module instance
    :param (optional) logger: set internal logger
    """
    if logger is None:
        logger = model.logger if hasattr(model, 'logger') else \
            CommandLineLogger(log_level=os.getenv('LOG_LEVEL', 'debug'), name='ModelVerbose')
    # Recursively set for submodules
    if isinstance(model, Verbosable):
        for attr_name in model.get_layer_attr_names():
            enable_verbose(getattr(model, attr_name), logger=logger)
    else:
        # Register a hook for each layer in root module
        for _name, _layer in model.named_children():
            _layer.register_forward_hook(lambda _l, _, _out: logger.debug(f"{_name}: {_out.shape}"))
    # Flag model
    model.verbose_enabled = True


def get_gpu_memory_gb(gpu_index: int = 0) -> int:
    """
    Returns total memory capacity in GB of GPU indexed at :attr:`gpu_index`.
    :param (int) gpu_index: index of GPU device (defaults to 0)
    :return: an integer object with the number of GB of total GPU memory (attention: not the free/available but the
             total memory is returned)
    """
    total_memory = torch.cuda.get_device_properties(gpu_index).total_memory  # bytes
    total_memory_gb_str = humanize.filesize.naturalsize(total_memory, binary=True, gnu=True)  # e.g. "23.7G"
    total_memory_gb = float(total_memory_gb_str[0:-1])  # e.g. float(23.7)
    return math.ceil(total_memory_gb)


def get_gradient(disc: nn.Module, real: torch.Tensor, fake: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
    """
    Return the gradient of the critic's scores w.r.t. mixes of real and fake images.
    Attention: Returns the gradient of c(x) w.r.t. to x. If for example x is a (3,128,128) image, then grad_x(c(x)) will
               be a (3,128,128) tensor containing how much the c(x) changes if x[i][j][k] changes.
    :param (nn.Module) disc: the critic model
    :param (torch.Tensor) real: a batch of real images
    :param (torch.Tensor) fake: a batch of fake images
    :param (torch.Tensor) epsilon:
    :return: (torch.Tensor) the gradient of the critic's scores, with respect to the mixed image
    """
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_images = torch.autograd.Variable(mixed_images, requires_grad=True)

    # Calculate the critic's scores on the mixed images
    mixed_scores = disc(mixed_images)

    # Take the gradient of the scores with respect to the images
    return torch.autograd.grad(
        # Note: We need to take the gradient of outputs with respect to inputs.
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        inputs=mixed_images,
        outputs=mixed_scores,
        # These other parameters have to do with how the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]


def get_gradient_penalty_from_gradient(gradient: torch.Tensor) -> torch.Tensor:
    """
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, we calculate the magnitude of each image's gradient and penalize the mean
    quadratic distance of each magnitude to 1.
    :param (torch.Tensor) gradient: the gradient of the critic's scores, with respect to the mixed image
    :return: a scalar torch.Tensor object containing the gradient penalty
    """
    # Flatten the gradients so that each row captures one image
    if gradient.dim() == 5:
        needs_unfold = (gradient.shape[0], gradient.shape[1])
        gradient = gradient.view(gradient.shape[0] * gradient.shape[1], -1)
    else:
        needs_unfold = None
        gradient = gradient.view(gradient.shape[0], -1)
    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    # Penalize the mean squared distance of the gradient norms from 1
    gradient_penalties = (gradient_norm - 1) ** 2
    if needs_unfold is not None:
        gradient_penalties = gradient_penalties.view(*needs_unfold)
        return torch.mean(gradient_penalties, dim=1)
    return torch.mean(gradient_penalties, dim=0)


def get_gradient_penalties(disc: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """
    Return the gradient of the critic's scores w.r.t. mixes of real and fake images.
    Attention: Returns the gradient of c(x) w.r.t. to x. If for example x is a (3,128,128) image, then grad_x(c(x)) will
               be a (3,128,128) tensor containing how much the c(x) changes if x[i][j][k] changes.
    :param (nn.Module) disc: the critic model
    :param (torch.Tensor) real: a batch of real images
    :param (torch.Tensor) fake: a batch of fake images
    :return: (torch.Tensor) the gradient of the critic's scores, with respect to the mixed image
    """
    # equalize batch sizes
    if real.dim() == 4:
        batch_size = min(real.size(0), real.size(0))
        real = real[:batch_size]
        fake = fake[:batch_size]
        # get noisy inputs
        eps = torch.rand(batch_size)
        while eps.dim() < real.dim():
            eps = eps.unsqueeze(-1)
    elif real.dim() == 5:
        assert (real.size(0) == fake.size(0))
        batch_size = min(real.size(1), fake.size(1))
        real = real[:, :batch_size, :, :, :]
        fake = fake[:, :batch_size, :, :, :]
        # get noisy inputs
        eps = torch.rand(real.size(0), batch_size)
        while eps.dim() < real.dim():
            eps = eps.unsqueeze(-1)
    else:
        raise RuntimeError("Unknown dimension of image data: {0}".format(real.dim()))

    eps = eps.cuda() if real.is_cuda else eps
    mixed_images = eps * real + (1 - eps) * fake
    mixed_images = Variable(mixed_images, requires_grad=True)

    # push through network
    mixed_scores = disc(mixed_images)

    # compute the gradients
    grads = torch.ones(mixed_scores.size())
    grads = grads.cuda() if real.is_cuda else grads
    gradients = torch.autograd.grad(outputs=mixed_scores, inputs=mixed_images, grad_outputs=grads, create_graph=True,
                                    only_inputs=True)
    if real.dim() == 4:
        gradient_input = gradients[0].view(batch_size, -1)
    else:
        gradient_input = gradients[0].view(real.size(0) * batch_size, -1)

    # compute the penalties
    gradient_penalties = (gradient_input.norm(2, dim=1) - 1) ** 2

    if real is not None and real.dim() != 4:
        gradient_penalties = gradient_penalties.view(real.size(0), batch_size)

    return gradient_penalties


def get_gradient_penalty(disc: nn.Module, real: torch.Tensor, fake: torch.Tensor,
                         epsilon: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Get the gradient penalty regularization term, given the discriminator (critic) model, a set of real and fake images
    and the parameter :attr:`epsilon` to mix the two set of images (approximating an average input to critic).
    :param (nn.Module) disc: the critic model
    :param (torch.Tensor) real: a batch of real images
    :param (torch.Tensor) fake: a batch of fake images
    :param (torch.Tensor) epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    :return: (torch.Tensor) the gradient of the critic's scores, with respect to the mixed image
    """
    # if epsilon is None:
    #     # if no epsilon param provided, use a random mixing weight for each pair of real/fake images in current batch
    #     epsilon = torch.rand(real.size(0), real.size(1)) if real.dim() == 5 else torch.rand(real.size(0))
    #     while epsilon.dim() < real.dim():
    #         epsilon = epsilon.unsqueeze(-1)
    #     epsilon = epsilon.to(real.device)
    # return get_gradient_penalty_from_gradient(
    #     get_gradient(disc=disc, real=real, fake=fake, epsilon=epsilon)
    # )
    gradient_penalties = get_gradient_penalties(disc, real, fake)
    mean_dim = 0 if real.dim() == 1 else 1
    return gradient_penalties.mean(mean_dim)


class WassersteinLoss(nn.modules.Module):
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, predictions: torch.Tensor, weights: torch.Tensor or float) -> torch.Tensor:
        # mean_dim = 0 if predictions.dim() == 1 else 1
        return (weights * predictions).mean()


def get_total_params(model: Module, print_table: bool = False, sort_desc: bool = False) -> int or None:
    """
    Get total number of parameters from given nn.Module.
    :param model: model to count parameters for
    :param print_table: if True prints counts for every sub-module and returns None, else returns total count only
    :param sort_desc: if True sorts array in DESC order wrt to parameter count before printing
    :return: total number of parameters if $print$ is set to False, else prints counts and returns nothing
    """
    total_count_orig = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = 0
    count_dict = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count = parameter.numel()
        count_dict.append({
            'name': name,
            'count': count,
            'count_hr': to_human_readable(count),
            'prc': '%.2f %%' % (count / total_count_orig)
        })
        total_count += count

    assert total_count_orig == total_count, "Should be equal..."

    if print_table is True:
        from prettytable import PrettyTable

        if sort_desc:
            count_dict = sorted(count_dict, key=lambda k: k['count'], reverse=True)

        table = PrettyTable()
        table.field_names = ["Module", "Count", "Count Human", "Percentage"]
        [table.add_row(count_dict[i].values()) for i in range(len(count_dict))]

        print(table.get_string(fields=["Module", "Count Human", "Percentage"]))
        print(f"Total Trainable Params: {to_human_readable(total_count)}")
        return None

    return total_count


def invert_transforms(ts: transforms.Compose) -> transforms.Compose:
    """
    Inverts Normalize and ToTensor transforms in a transformation sequence defined via torchvision.transforms.Compose.
    :param ts: transforms.Compose object with the list of transforms to be inverted
    :return: a torchvision.transforms.Compose object with the last transforms inverted
    """
    if type(ts) == transforms.ToTensor:
        return transforms.Compose([ToTensorOrPass(), ])
    inverse_transforms_list = []
    for i in reversed(range(len(ts.transforms))):
        t = ts.transforms[i]
        if type(t) == transforms.Normalize:
            inverse_transforms_list.append(UnNormalize(mean=t.mean, std=t.std, inplace=t.inplace))
        elif type(t) == transforms.ToTensor:
            # inverse_transforms_list.append(transforms.ToPILImage())
            # We reached the ToTensor() transform; no point in continuing inverting upwards where the image was still a
            # PIL image object
            break
        elif type(t) == LinToTensorNormalized:
            tt = t.normalize
            inverse_transforms_list.append(UnNormalize(mean=tt.mean, std=tt.std, inplace=tt.inplace))
    return transforms.Compose(inverse_transforms_list)


def matrix_sqrt(mat: torch.Tensor) -> torch.Tensor:
    """
    Compute the square root of the given matrix.
    :param mat: the input matrix as a torch.Tensor object
    :return: a torch.Tensor fo the same shape as the input
    """
    return MatrixSquareRoot.apply(mat)


class MatrixSquareRoot(Function):
    """
    MatrixSquareRoot Class:
    This class is used to compute square root of a positive definite matrix given as torch.Tensor object.
    NOTE: matrix square root is NOT differentiable for matrices with zero eigenvalues.
    Source: https://github.com/steveli/pytorch-sqrtm
    """

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        return MatrixSquareRoot.backward(ctx, grad_inputs[0])

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx: Any, mat: torch.Tensor, **kwargs):
        mat_np = mat.detach().cpu().numpy().astype(np.float_)
        mat_sqrt = scipy.linalg.sqrtm(mat_np).real
        mat_sqrt = torch.from_numpy(mat_sqrt).to(mat)
        ctx.save_for_backward(mat_sqrt)
        return mat_sqrt

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        grad_input = None
        if ctx.needs_input_grad[0]:
            mat_sqrt, = ctx.saved_tensors
            mat_sqrt = mat_sqrt.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_mat_sqrt = scipy.linalg.solve_sylvester(mat_sqrt, mat_sqrt, gm)
            grad_input = torch.from_numpy(grad_mat_sqrt).to(grad_output)
        return grad_input


class RandomVerticalFlip(object):
    """
    Randomly vertically flips the given PIL.Image with a probability of 0.5
    Source: https://github.com/aosokin/biogans
    """

    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


def pad_channels(output, nc, value=-1.0):
    if output.size(1) < nc:
        padded_channels = value * torch.ones(output.size(0), nc - output.size(1), output.size(2), output.size(3))
        if isinstance(output, torch.autograd.Variable):
            padded_channels = torch.autograd.Variable(padded_channels, requires_grad=False)
        if output.is_cuda:
            padded_channels = padded_channels.cuda()
        output = torch.cat((output, padded_channels), 1)
    return output


class ReceptiveFieldCalculator:
    """
    Class to calculate the receptive field of a CNN.
    Source: https://rubikscode.net/2020/05/18/receptive-field-arithmetic-for-convolutional-neural-networks/
    """

    @staticmethod
    def calculate(architecture, input_image_size, print_results: bool = False) -> int:
        input_layer = ('input_layer', input_image_size, 1, 1, 0.5)
        if print_results:
            ReceptiveFieldCalculator.print_layer_info(input_layer)

        current_layer = []
        for key in architecture:
            current_layer = ReceptiveFieldCalculator.calculate_layer_info(architecture[key], input_layer, key)
            if print_results:
                ReceptiveFieldCalculator.print_layer_info(current_layer)
            input_layer = current_layer

        return current_layer[2] if len(current_layer) > 2 else -1

    @staticmethod
    def print_layer_info(layer):
        print(f'------')
        print(f'{layer[0]}: n = {layer[1]}; r = {layer[2]}; j = {layer[3]}; start = {layer[4]}')
        print(f'------')

    @staticmethod
    def calculate_layer_info(current_layer, input_layer, layer_name) -> Tuple[int, int, int, int, int]:
        n_in = input_layer[1]
        j_in = input_layer[2]
        r_in = input_layer[3]
        start_in = input_layer[4]

        k = current_layer[0]
        s = current_layer[1]
        p = current_layer[2]

        n_out = math.floor((n_in - k + 2 * p) / s) + 1
        padding = (n_out - 1) * s - n_in + k
        # noinspection PyUnusedLocal
        p_right = math.ceil(padding / 2)
        p_left = math.floor(padding / 2)

        j_out = j_in * s
        r_out = r_in + (k - 1) * j_in
        start_out = start_in + ((k - 1) / 2 - p_left) * j_in
        return layer_name, n_out, j_out, r_out, start_out


class ToTensorOrPass(transforms.ToTensor):
    """
    ToTensorOrPass Class:
    Completes ToTensor() transform class by skipping if input is already a tensor.
    """

    def __init__(self, renormalize: bool = True):
        """
        ToTensorOrPass class constructor.
        :param (bool) renormalize: set to True to renormalize input tensors to [0,1], otherwise if Tensor object
                                   encountered in input it will return it intact
        """
        super().__init__()
        self.renormalize = renormalize

    def __call__(self, pic_or_tensor):
        # case: PIL image
        if type(pic_or_tensor) is not torch.Tensor:
            return super(ToTensorOrPass, self).__call__(pic_or_tensor)

        # case: torch.Tensor
        if self.renormalize:
            tensor_min = torch.min(pic_or_tensor)
            tensor_max = torch.max(pic_or_tensor)
            return (pic_or_tensor - tensor_min) / (tensor_max - tensor_min)

        return pic_or_tensor


class UnNormalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False):
        super(UnNormalize, self).__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.inverse = transforms.Compose([
            transforms.Normalize(
                np.zeros_like(np.asarray(mean)),
                np.reciprocal(
                    np.multiply(std, np.ones_like(np.asarray(std)))
                ),
                inplace=inplace
            ),
            transforms.Normalize(
                -1 * np.asarray(mean),
                np.ones_like(np.asarray(std)),
                inplace=inplace
            )
        ])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.inverse(tensor)


class ToNumpy:
    """
    Source: https://github.com/aosokin/biogans/blob/cb72bb0457be335fad6c27a16bb1761b937a6d06/code/custom_dataloaders.py
    """

    # noinspection PyMethodMayBeStatic
    def __call__(self, img: PIL.Image):
        # convert to numpy array
        img = np.array(img.getdata()).reshape((img.size[1], img.size[0], 3))
        # permute dimensions
        img = np.transpose(img.copy(), (2, 0, 1))
        return img


class LinToTensorNormalized:
    """
    Convert to torch tensor manually (torchvision.transforms.ToTensor is buggy).
    Source: https://github.com/aosokin/biogans/blob/cb72bb0457be335fad6c27a16bb1761b937a6d06/code/custom_dataloaders.py
    """

    def __init__(self):
        super().__init__()
        self.normalize = transforms.Normalize(0.5, 0.5)
        self.mask = None

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        if self.mask is not None:
            for c in range(1, img.shape[0]):  # skip red
                img[c, :, :] *= self.mask
        img = torch.from_numpy(img.astype(np.float32)) / 255.0
        img_norm = []
        for i_c in range(img.size(0)):
            img_norm.append(self.normalize(img[i_c].unsqueeze(0)))
        img = torch.cat(img_norm, 0).contiguous()
        return img
