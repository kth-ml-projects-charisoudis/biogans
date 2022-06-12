from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
# noinspection PyProtectedMember
from torch.utils.data import Dataset

from utils.ifaces import FilesystemFolder
from utils.metrics.fid import FID


class C2ST(FID):
    """
    C2ST Class:
    This class is used to compute the Classifier 2-Sample Test (C2ST) metric between real and generated images as
    proposed in the paper "Revisiting Classifier Two-Sample Tests".
    Inspired by: https://github.com/mkirchler/deep-2-sample-test/blob/master/deeptest/c2st.py
    """

    def __init__(self, model_fs_folder_or_root: Union[FilesystemFolder, str], device: torch.device or str = 'cpu',
                 n_samples: int = 1024, batch_size: int = 8):
        """
        C2ST class constructor.
        :param (FilesystemFolder or str) model_fs_folder_or_root: absolute path to model checkpoints directory or
                                                                  FilesystemFolder instance for cloud-synced models
        :param (str) device: the device type on which to run the Inception model (defaults to 'cpu')
        :param (int) n_samples: the total number of samples used to compute the metric (defaults to 512; the higher this
                          number gets, the more accurate the metric is)
        :param (int) batch_size: the number of samples to precess at each loop
        """
        super(C2ST, self).__init__(model_fs_folder_or_root=model_fs_folder_or_root, device=device,
                                   n_samples=n_samples, batch_size=batch_size)

    # noinspection PyUnusedLocal
    def forward(self, dataset: Dataset, gen: nn.Module, target_index: Optional[int] = None,
                condition_indices: Optional[Union[int, tuple]] = None, z_dim: Optional[int] = None,
                show_progress: bool = True, use_fid_embeddings: bool = False, **kwargs) -> Tensor:
        """
        Compute the C2ST score between random $self.n_samples$ images from the given dataset and same number of images
        generated by the given generator network.
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
        :param (bool) use_fid_embeddings: set to True to avoid re-computing ImageNET embeddings and use the ones
                                          computed during calculation of the FID metric
        :return: the C2ST value as a torch.Tensor object
        """
        # Extract ImageNET embeddings
        if not use_fid_embeddings or FID.LastRealEmbeddings is None or FID.LastFakeEmbeddings is None:
            real_embeddings, fake_embeddings = self.get_embeddings(dataset, gen=gen, target_index=target_index,
                                                                   z_dim=z_dim, condition_indices=condition_indices,
                                                                   show_progress=show_progress, desc="C2ST")
        else:
            real_embeddings = FID.LastRealEmbeddings
            fake_embeddings = FID.LastFakeEmbeddings
        # Initialize manifolds
        c2st = 0.0
        return c2st
