# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Perceptual Path Length (PPL) from the paper "A Style-Based Generator Architecture for Generative Adversarial Networks".
Matches the original implementation by Karras et al. at
https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py
"""

import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm

from modules.classifiers.vgg16 import LPIPSLoss
from utils.ifaces import FilesystemFolder


def slerp(a: torch.Tensor, b: torch.Tensor, t) -> torch.Tensor:
    """
    Spherical interpolation of a batch of vectors.
    :param (torch.Tensor) a:
    :param (torch.Tensor) b:
    :param t:
    :return: a torch.Tensor object containing len(t) spherical interpolations between a and b.
    """
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * torch.acos(d)
    c = b - d * a
    c = c / c.norm(dim=-1, keepdim=True)
    d = a * torch.cos(p) + c * torch.sin(p)
    d = d / d.norm(dim=-1, keepdim=True)
    return d


class PPL(nn.Module):
    """
    PPL Class:
    Source: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/perceptual_path_length.py
    """

    LPIPSLoss: LPIPSLoss = None

    def __init__(self, model_fs_folder_or_root: FilesystemFolder, device: str, n_samples: int, batch_size: int = 8,
                 epsilon: float = 1e-4, space: str = 'w', sampling: str = 'full'):
        """
        :param (FilesystemFolder) model_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` instance for cloud or
                                                           locally saved models using the same API
        :param (str) device: the device type on which to run the Inception model (defaults to 'cpu')
        :param (int) n_samples: the total number of samples used to compute the metric (defaults to 512; the higher this
                                number gets, the more accurate the metric is)
        :param (float) epsilon: step dt
        :param (str) space: one of 'z', 'w'
        :param (str) sampling: one of 'full', 'end'
        """
        # Check arguments
        assert space in ['z', 'w']
        assert sampling in ['full', 'end']
        # Init nn.Module
        super().__init__()
        # Init LPIPS loss calculator
        if self.__class__.LPIPSLoss is None:
            self.__class__.LPIPSLoss = LPIPSLoss(model_gfolder_or_groot=model_fs_folder_or_root, chkpt_step='397923af',
                                                 use_dropout=False, requires_grad=False).to(device)
        # Save model attributes
        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.device = device
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.tqdm = tqdm

    def sampler(self, c: torch.Tensor, gen) -> torch.Tensor:
        """
        :param (Tensor) c: dataset one-hot labels as torch.Tensor object
        :param gen: the generator module
        :return: a torch.Tensor of shape (B, 1)
        """
        # Generate random latents and interpolation t-values.
        t = torch.rand([c.shape[0]], device=c.device) * (1 if self.sampling == 'full' else 0)
        z0, z1 = torch.randn([c.shape[0] * 2, gen.z_dim], device=c.device).chunk(2)

        # Interpolate in W or Z.
        # TODO: This version only works on unlabelled data. If labels exist, they are ignored, whereas in StyleGAN2 they
        # TODO: are also used to further disentangle the noise space.
        zt0 = slerp(z0, z1, t.unsqueeze(1))
        zt1 = slerp(z0, z1, t.unsqueeze(1) + self.epsilon)

        # Generate images.
        img_t0 = gen(zt0)
        img_t1 = gen(zt1)
        #   - add 3rd channel
        img_t0 = torch.concat((img_t0, torch.zeros(img_t0.shape[0], 1, 48, 80).cuda()), dim=1)
        img_t1 = torch.concat((img_t0, torch.zeros(img_t0.shape[0], 1, 48, 80).cuda()), dim=1)

        # Compute individual losses and then sum up
        lpips_t0 = self.__class__.LPIPSLoss(img_t0)
        lpips_t1 = self.__class__.LPIPSLoss(img_t1)
        return (lpips_t0 - lpips_t1).square().sum(dim=0) / self.epsilon ** 2

    # noinspection PyUnusedLocal
    def forward(self, dataset: Dataset, gen: nn.Module, target_index: Optional[int] = None,
                condition_indices: Optional[Union[int, tuple]] = None, z_dim: Optional[int] = None,
                show_progress: bool = True, **kwargs) -> Tensor:
        """
        Compute the Perception Path Length between random $self.n_samples$ images from the given dataset and equal
        number of images generated by the given generator network.
        :param dataset: a torch.utils.data.Dataset object to access real images. Attention: no transforms should be
                        applied when __getitem__ is called since the transforms are different on Inception v3
        :param gen: the Generator network
        :param target_index: index of target (real) output from the arguments that returns dataset::__getitem__() method
        :param condition_indices: indices of images that will be passed to the Generator in order to generate fake
                                  images (for image-to-image translation tasks). If set to None, the generator is fed
                                  with random noise.
        :param z_dim: if $condition_indices$ is None, then this is necessary to produce random noise to feed into the
                      DCGAN-like generator
        :param (bool) show_progress: set to True to display progress using `tqdm` lib
        :return: a scalar torch.Tensor object containing the computed PPL value
        """
        # Create the dataloader instance
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        if self.device in ['cuda:0', 'cuda'] and torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Sampling loop.
        dist = []
        cur_samples, break_after = 0, False
        for real_samples in self.tqdm(dataloader, total=int(math.ceil(self.n_samples / self.batch_size)),
                                      disable=not show_progress, desc='PPL'):
            if cur_samples >= self.n_samples:
                break_after = True
            cur_batch_size = real_samples.shape[0]

            # No labels --> Replace with zeros
            labels = torch.zeros(cur_batch_size, 0).to(self.device)

            x = self.sampler(labels, gen=gen)
            dist.append(x)

            cur_samples += cur_batch_size
            if break_after:
                break

        # Compute PPL
        dist = torch.cat(dist)[:cur_samples].detach().cpu().numpy()
        lo = np.percentile(dist, 1, interpolation='lower')
        hi = np.percentile(dist, 99, interpolation='higher')
        ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()
        return torch.tensor(ppl)
