__all__ = ['fid', 'f1', 'ssim', 'is_', 'GanEvaluator', 'GanEvaluator6Class']

from typing import Optional, Union, Dict

import torch
from torch import nn
# noinspection PyProtectedMember
from torch.utils.data import Dataset, Subset

from modules.ifaces import IGanGModule
from utils.ifaces import FilesystemFolder
from utils.metrics.c2st import C2ST
from utils.metrics.f1 import F1
from utils.metrics.fid import FID
from utils.metrics.is_ import IS
from utils.metrics.ppl import PPL
from utils.metrics.ssim import SSIM


class GanEvaluator(object):
    """
    GanEvaluator Class:
    This class is used to evaluate the image generation performance of a GAN.
    """

    def __init__(self, model_fs_folder_or_root: FilesystemFolder, gen_dataset: Dataset, n_samples: int = 1e4,
                 batch_size: int = 32, ssim_c_img: int = 3, device: torch.device or str = 'cpu',
                 target_index: Optional[int] = None, condition_indices: Optional[Union[int, tuple]] = None,
                 z_dim: Optional[int] = None, f1_k: int = 3, gan_instance=None, c2st_epochs: int = 100,
                 is_final: bool = False):
        """
        GanEvaluator class constructor.
        :param (FilesystemFolder) model_fs_folder_or_root: absolute path to model checkpoints directory or
                                                           FilesystemFolder instance for cloud-synced models
        :param (Dataset) gen_dataset: the dataset used to train the generator as a `torch.utils.data.Dataset` object
        :param (int) n_samples: the number of samples (images) used to evaluate GAN performance (the bigger the number
                                of samples, the more robust the evaluation metric)
        :param (int) batch_size: the batch size to access the dataset's images
        :param (int) ssim_c_img: the number of images' channels (3 for RGB, 1 for grayscale); needed by the SSIM metric
        :param (str) device: the device on which to run the calculations (supported: "cuda", "cuda:<GPU_INDEX>", "cpu")
        :param (int or None) target_index: index of target (real) output from the arguments that returns
                                           dataset::__getitem__() method
        :param (int) z_dim: if :attr:`condition_indices` is `None`, then this is necessary to produce random noise to
                            feed into the DCGAN-like generator
        :param (int or tuple or None) condition_indices: indices of images that will be passed to the Generator in
                                                            order to generate fake images (for image-to-image
                                                            translation tasks). If set to None, the generator is fed
                                                            with random noise.
        :param (int) f1_k: `k` param of precision/recall metric (default is 3)
        :param (bool) is_final: set this to True to also include final_calculators in the calculators dict
        """
        gen_dataset_underlying = gen_dataset.dataset if isinstance(gen_dataset, Subset) else gen_dataset
        if hasattr(gen_dataset_underlying, 'transforms'):
            self.gen_transforms = gen_dataset_underlying.transforms
        else:
            raise NotImplementedError('gen_dataset should expose image transforms (to invert them before entering '
                                      'ImageNET classifier to extract embeddings')
        # Save the dataset used by the generator in instance
        self.dataset = gen_dataset
        # Define metric calculators
        self.calculators = {
            'fid': FID(model_fs_folder_or_root=model_fs_folder_or_root, n_samples=n_samples,
                       batch_size=batch_size, device=device),
            'is': IS(model_fs_folder_or_root=model_fs_folder_or_root, n_samples=n_samples,
                     batch_size=batch_size, device=device),
            'f1': F1(model_fs_folder_or_root=model_fs_folder_or_root, n_samples=n_samples,
                     batch_size=batch_size, device=device),
            'ssim': SSIM(n_samples=n_samples, batch_size=batch_size, c_img=ssim_c_img, device=device),
            'ppl': PPL(model_fs_folder_or_root=model_fs_folder_or_root, n_samples=n_samples,
                       batch_size=batch_size, device=device)
        }
        self.final_calculators = {  # run only on final evaluation
            'c2st': C2ST(model_fs_folder_or_root=model_fs_folder_or_root, n_samples=n_samples,
                         batch_size=batch_size, device=device, train_epochs=c2st_epochs, gan_instance=gan_instance),
        }
        if is_final:
            self.calculators.update(self.final_calculators)
        # Save args
        self.target_index = target_index
        self.condition_indices = condition_indices
        self.z_dim = z_dim
        self.f1_k = f1_k
        self._gan_instance = gan_instance

    @property
    def gan_instance(self) -> IGanGModule:
        return self._gan_instance

    @gan_instance.setter
    def gan_instance(self, gan_instance: IGanGModule) -> None:
        self._gan_instance = gan_instance
        self.final_calculators['c2st'].gan = gan_instance

    def evaluate(self, gen: nn.Module, metric_name: Optional[str] = None, show_progress: bool = True,
                 dataset=None, print_dict: bool = False) -> Dict[str, float]:
        """
        Evaluate the generator's current state and return a `dict` with metric names as keys and evaluation results as
        values.
        :param (nn.Module) gen: the generator network as a `torch.nn.Module` object
        :param (optional) metric_name: the name of the evaluation metric to be applied
        :param (bool) show_progress: set to True to have the progress of evaluation metrics displayed (using `tqdm` lib)
        :param (optional) dataset: override `self.dataset` for this evaluation
        :param (bool) print_dict: set to True to print the metrics dict (e.g. in multiple classes to log intermediate
                                  results)
        :return: if :attr:`metric` is `None` then a `dict` of all available metrics is returned, only the given metric
                 is returned otherwise
        """
        # Fix z_dim
        if self.z_dim == -1:
            self.z_dim = gen.z_dim
        # Set generator in evaluation mode
        gen = gen.eval()
        metrics_dict = {}
        if dataset is None:
            dataset = self.dataset
        for metric_name in (self.calculators.keys() if not metric_name or 'all' == metric_name else (metric_name,)):
            # Evaluate model
            metric = self.calculators[metric_name](dataset, gen=gen, target_index=self.target_index,
                                                   condition_indices=self.condition_indices, z_dim=self.z_dim,
                                                   skip_asserts=True, show_progress=show_progress, k=self.f1_k,
                                                   use_fid_embeddings=True)
            # Unpack metrics
            if 'f1' == metric_name:
                metrics_dict['f1'], metrics_dict['precision'], metrics_dict['recall'] = \
                    tuple(map(lambda _m: _m.item(), metric))
            else:
                metrics_dict[metric_name] = metric.item()
        gen.train()
        # Return metrics dict
        if print_dict:
            print(metrics_dict)
        return metrics_dict


class GanEvaluator6Class(GanEvaluator):
    def __init__(self, *args, **kwargs):
        GanEvaluator.__init__(self, *args, **kwargs)

    def evaluate(self, gen: nn.Module, metric_name: Optional[str] = None, show_progress: bool = True,
                 dataset=None, print_dict: bool = False) -> Dict[str, float]:
        assert hasattr(gen, 'gens') and hasattr(self.dataset, 'datasets')
        dict_list = [
            super(GanEvaluator6Class, self).evaluate(gen_c, metric_name, show_progress, dataset_c, print_dict)
            for gen_c, dataset_c in zip(gen.gens, self.dataset.datasets.values())
        ]
        return {
            k: sum(d[k] for d in dict_list) / len(dict_list)
            for k in dict_list[0].keys()
        }
