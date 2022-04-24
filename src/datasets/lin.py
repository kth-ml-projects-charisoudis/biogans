import json
import math
import os
import pathlib
from json import JSONDecodeError
from typing import Optional, Tuple, Union

import click
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from matplotlib import pyplot as plt
from torch import Tensor
# noinspection PyProtectedMember
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
from tqdm.autonotebook import tqdm

from utils.command_line_logger import CommandLineLogger
from utils.filesystems.gdrive import GDriveDataset
from utils.filesystems.local import LocalCapsule, LocalFilesystem, LocalFolder
from utils.ifaces import FilesystemFolder
from utils.pytorch import LinToTensorNormalized, ToNumpy
from utils.string import to_human_readable


class LINDataset(Dataset, GDriveDataset):
    """
    LINDataset Class:
    This class is used to define the way LIN dataset is accessed for the purpose of biological image synthesis GANs
    (biogans).
    """

    # Dataset name is the name of the folder in Google Drive under which dataset's "Img.zip" file exists
    DatasetName = 'LIN_48x80'

    # Pre-define which classes will be used
    Classes = [
        'Alp14',
        'Arp3',
        'Cki2',
        'Mkh1',
        'Sid2',
        'Tea1',
    ]

    def __init__(self, dataset_fs_folder_or_root: FilesystemFolder, image_transforms: Optional[Compose] = None,
                 train_not_test: bool = True, which_classes: str = 'all', logger: Optional[CommandLineLogger] = None,
                 return_path: bool = False):
        """
        LINDataset class constructor.
        :param (FilesystemFolder) dataset_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` object to download / use
                                                             dataset from local or remote (Google Drive) filesystem
        :param (optional) image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param bool train_not_test: set to True for the training set, False for the test set
        :param str which_classes: one of 'all', '6class', or a single class (e.g. 'Rho1')
        :raises FileNotFoundError: either when the dataset is not present in local filesystem or when the
                                   `polarity_factor_info.json` is not present inside dataset's (local) root
        """
        # Instantiate `torch.utils.data.Dataset` class
        Dataset.__init__(self)
        # Instantiate `utils.filesystems.gdrive.GDriveDataset` class
        dataset_fs_folder = dataset_fs_folder_or_root if dataset_fs_folder_or_root.name == self.DatasetName else \
            dataset_fs_folder_or_root.subfolder_by_name(folder_name=self.DatasetName, recursive=False)
        GDriveDataset.__init__(self, dataset_fs_folder=dataset_fs_folder,
                               zip_filename='LIN_Normalized_WT_size-48-80.zip')
        self.root = dataset_fs_folder.local_root
        # Initialize instance properties
        self.logger = logger if logger is not None else \
            CommandLineLogger(log_level=os.getenv('LOG_LEVEL', 'info'), name=self.__class__.__name__)
        # Check that the dataset is present at the local filesystem
        if not self.is_fetched_and_unzipped():
            if click.confirm(f'Dataset is not fetched and unzipped. Would you like to fetch now?', default=True):
                self.fetch_and_unzip(in_parallel=False, show_progress=True)
            else:
                raise FileNotFoundError(f'Dataset not found in local filesystem (tried {self.root})')
        # Load images info file
        self.train_img_dir_path = self.dataset_gfolder \
            .subfolder_by_name('LIN_Normalized_WT_size-48-80_train').local_root
        self.test_img_dir_path = self.dataset_gfolder.subfolder_by_name('LIN_Normalized_WT_size-48-80_test').local_root
        self.img_dir_path = getattr(self, f'{"train" if train_not_test else "test"}_img_dir_path')
        assert os.path.exists(self.img_dir_path) and os.path.isdir(self.img_dir_path)
        if which_classes in ['all', '6class']:
            self.root = self.img_dir_path
            self.polarity_factors_info_path = os.path.join(self.root,
                                                           f'polarity_factors_info_{which_classes}.json')
        else:
            self.root = os.path.join(self.img_dir_path, which_classes)
            self.polarity_factors_info_path = os.path.join(self.root, f'polarity_factor_info.json')
        if not os.path.exists(self.polarity_factors_info_path):
            if click.confirm(f'Dataset is not scraped. Would you like to scrape now?', default=True):
                LINScraper.run(forward_pass=True, backward_pass=True)
            else:
                raise FileNotFoundError(f'Dataset is not scraped. polarity_factors_info.json could not be found.')
        with open(self.polarity_factors_info_path, 'r') as json_fp:
            self.polarity_factors_info = json.load(json_fp)
        # Assign dataset metadata
        self.img_count = self.polarity_factors_info['cell_images_count']
        self.logger.debug(f'Found {to_human_readable(self.img_count)} image pairs in the dataset ' +
                          f'(train={train_not_test} | which={which_classes})')
        # Save transforms
        self._transforms = None
        self.transforms = image_transforms
        self.return_path = return_path

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, t: Optional[Compose] = None) -> None:
        if t is None:  # load default transforms
            self._transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, ] * 3, [0.5, ] * 3)
            ])
        else:
            self._transforms = t

    def __getitem__(self, index: int) -> Union[Tuple[Tensor, str], Tensor]:
        """
        Implements abstract Dataset::__getitem__() method.
        :param (int) index: integer with the current image index that we want to read from disk
        :return: an image from the dataset as a torch.Tensor object
        """
        # Fetch images
        cell_img_path = self.polarity_factors_info['cell_images'][index]
        cell_img_path_abs = os.path.join(self.root, cell_img_path)
        try:
            cell_img = Image.open(cell_img_path_abs)
        except UnidentifiedImageError:
            self.logger.critical(f'Image opening failed (path: {cell_img_path_abs})')
            return self.__getitem__(index + 1)
        # Apply transforms
        if self._transforms is not None:
            cell_img_tensor = self._transforms(cell_img)
            cell_img.close()
            cell_img = cell_img_tensor[:2, :, :]  # 3rd channel (blue) is 0
        if self.return_path:
            return cell_img, os.path.join(self.polarity_factors_info['path'], cell_img_path)
        return cell_img

    def __len__(self) -> int:
        """
        Implements abstract Dataset::__len__() method. This method returns the total "length" of the dataset which is
        the total number of  images contained in each pile (or the min of them if they differ).
        :return: integer
        """
        return self.img_count


class LINDataset6Class(Dataset):
    def __init__(self, **ds_kwargs):
        Dataset.__init__(self)
        self.logger = CommandLineLogger(log_level=os.getenv('LOG_LEVEL', 'info'), name=self.__class__.__name__)
        self.datasets = {}
        self.dataset_lengths = {}
        ds_kwargs['logger'] = self.logger
        for class_name in LINDataset.Classes:
            ds_kwargs['which_classes'] = class_name
            self.datasets[class_name] = LINDataset(**ds_kwargs)
            self.dataset_lengths[class_name] = len(self.datasets[class_name])

    def __getitem__(self, index: int) -> torch.Tensor:
        out = []
        for class_name in LINDataset.Classes:
            out.append(self.datasets[class_name][index % self.dataset_lengths[class_name]])
        return torch.stack(out, dim=0)

    def __len__(self) -> int:
        return max(self.dataset_lengths.values())

    @property
    def transforms(self):
        return self.datasets['Alp14'].transforms


class LINNearestDataset(LINDataset):
    """
    LINNearestDataset Class:
    Implements nearest-neighbors version of LIN dataset. Each "image" contains 1 red channel and 6 greens (one for each
    of the 6 classes).
    """

    def __init__(self, dataset_fs_folder_or_root: FilesystemFolder, image_transforms: Optional[Compose] = None,
                 which_classes: str = 'all', logger: Optional[CommandLineLogger] = None, reds_only: bool = False):
        """
        LINDataset class constructor.
        :param (FilesystemFolder) dataset_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` object to download / use
                                                             dataset from local or remote (Google Drive) filesystem
        :param (optional) image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param str which_classes: one of 'all', '6class', or a single class (e.g. 'Rho1')
        :param bool reds_only: set to True to retrieve only nearest-neighbors' red channels
        :raises FileNotFoundError: either when the dataset is not present in local filesystem or when the
                                   `polarity_factor_info.json` is not present inside dataset's (local) root
        """
        LINDataset.__init__(self, dataset_fs_folder_or_root=dataset_fs_folder_or_root, which_classes=which_classes,
                            image_transforms=image_transforms, train_not_test=True, return_path=False, logger=logger)
        # Nearest-Neighbors info
        self.nearest_neighbors_info_path = os.path.join(self.root, f'nearest_neighbors_info.json')
        self.nearest_neighbors_info = None
        if os.path.exists(self.nearest_neighbors_info_path):
            try:
                with open(self.nearest_neighbors_info_path, 'r') as json_fp:
                    self.nearest_neighbors_info = json.load(json_fp)
            except JSONDecodeError as e:
                self.logger.error(str(e))
                os.remove(self.nearest_neighbors_info_path)
        self.nearest_neighbors_path = os.path.join(self.root, f'nearest_neighbors.pth')
        self.nearest_neighbors_all = None
        self.nearest_neighbors = None
        if not os.path.exists(self.nearest_neighbors_path):
            self.logger.critical(f'nearest_neighbors_path={self.nearest_neighbors_path}: NOT FOUND')
            raise FileNotFoundError(self.nearest_neighbors_path)
        self.reds_only = reds_only

    def load_nearest_neighbors(self):
        self.nearest_neighbors_all = torch.load(self.nearest_neighbors_path)
        self.nearest_neighbors = self.nearest_neighbors_all['reds' if self.reds_only else 'red_greens']

    def __getitem__(self, index: int) -> Tensor:
        """
        Implements abstract Dataset::__getitem__() method.
        :param (int) index: integer with the current image index that we want to read from disk
        :return: an image from the dataset as a torch.Tensor object
        """
        if self.nearest_neighbors is None:
            self.load_nearest_neighbors()
        return self.nearest_neighbors[index]

    def __len__(self) -> int:
        """
        Implements abstract Dataset::__len__() method. This method returns the total "length" of the dataset which is
        the total number of  images contained in each pile (or the min of them if they differ).
        :return: integer
        """
        if self.nearest_neighbors is None:
            self.load_nearest_neighbors()
        return self.nearest_neighbors.shape[0]


class LINDataloader(DataLoader):
    """
    LINDataloader Class:
    This class is used to load and access LIN dataset using PyTorch's DataLoader interface.
    """

    def __init__(self, dataset_fs_folder_or_root: FilesystemFolder, image_transforms: Optional[Compose] = None,
                 train_not_test: bool = True, which_classes: str = 'all', logger: Optional[CommandLineLogger] = None,
                 return_path: bool = False, **dl_kwargs):
        """
        LINDataloader class constructor.
        :param (FilesystemFolder) dataset_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` object to download / use
                                                             dataset from local or remote (Google Drive) filesystem
        :param (optional) image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param bool train_not_test: set to True for the training set, False for the test set
        :param str which_classes: one of 'all', '6class', or a single class (e.g. 'Rho1')
        :param batch_size: the number of images per (mini) batch
        :param (bool) pin_memory: set to True to have data transferred in GPU from the Pinned RAM (this is thoroughly
                                  explained here: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc)
        :param bool return_path: set to True to return img_batch, path_batch on every next() call
        :raises FileNotFoundError: either when the dataset is not present in local filesystem or when the
                                   `polarity_factor_info.json` is not present inside dataset's (local) root
        """
        # Instantiate dataset
        dataset = LINDataset(dataset_fs_folder_or_root=dataset_fs_folder_or_root, image_transforms=image_transforms,
                             train_not_test=train_not_test, which_classes=which_classes, return_path=return_path,
                             logger=logger)
        # Instantiate dataloader
        DataLoader.__init__(self, dataset=dataset, **dl_kwargs)
        self.transforms = dataset.transforms


class LINDataloader6Class(DataLoader):
    def __init__(self, dataset_fs_folder_or_root: FilesystemFolder, image_transforms: Optional[Compose] = None,
                 train_not_test: bool = True, logger: Optional[CommandLineLogger] = None, **dl_kwargs):
        # Instantiate dataset
        ds = LINDataset6Class(dataset_fs_folder_or_root=dataset_fs_folder_or_root, image_transforms=image_transforms,
                              train_not_test=train_not_test, return_path=False, logger=logger)
        # Instantiate dataloader
        dl_kwargs['collate_fn'] = lambda batch: torch.stack(batch, dim=1)
        DataLoader.__init__(self, dataset=ds, **dl_kwargs)
        self.transforms = ds.datasets['Alp14'].transforms


class LINNearestDataloader(DataLoader):
    """
    LINNearestDataloader Class:
    This class is used to load and access LINNearestDataset class using PyTorch's DataLoader interface.
    """

    def __init__(self, dataset_fs_folder_or_root: FilesystemFolder, image_transforms: Optional[Compose] = None,
                 which_classes: str = 'all', logger: Optional[CommandLineLogger] = None, reds_only: bool = False,
                 **dl_kwargs):
        """
        LINDataloader class constructor.
        :param (FilesystemFolder) dataset_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` object to download / use
                                                             dataset from local or remote (Google Drive) filesystem
        :param (optional) image_transforms: a list of torchvision.transforms.* sequential image transforms
        :param str which_classes: one of 'all', '6class', or a single class (e.g. 'Rho1')
        :param batch_size: the number of images per (mini) batch
        :param bool pin_memory: set to True to have data transferred in GPU from the Pinned RAM (this is thoroughly
                                  explained here: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc)
        :param bool reds_only: set to True to retrieve only nearest-neighbors' red channels
        :raises FileNotFoundError: either when the dataset is not present in local filesystem or when the
                                   `polarity_factor_info.json` is not present inside dataset's (local) root
        """
        # Instantiate dataset
        dataset = LINNearestDataset(dataset_fs_folder_or_root=dataset_fs_folder_or_root, reds_only=reds_only,
                                    image_transforms=image_transforms, which_classes=which_classes, logger=logger)
        # Instantiate dataloader
        DataLoader.__init__(self, dataset=dataset, **dl_kwargs)


class LINScraper:
    """
    LINScraper Class:
    This class is used to scrape LIN dataset's images for the purpose of replicating biogans experiments.
    """

    def __init__(self, root: str = '/data/Datasets/LIN_48x80', train_not_test: bool = True, which_classes: str = 'all'):
        """
        LINScraper class constructor.
        :param (str) root: LIN dataset's root directory path
        :param bool train_not_test: set to True to scrape training set's images; False for test set's ones
        :param str which_classes: one of 'all', '6class'
        """
        self.logger = CommandLineLogger(log_level=os.getenv('LOG_LEVEL', 'info'))
        self.train_img_dir_path = f'{root}/LIN_Normalized_WT_size-48-80_train'
        assert os.path.exists(self.train_img_dir_path)
        self.test_img_dir_path = f'{root}/LIN_Normalized_WT_size-48-80_test'
        assert os.path.exists(self.test_img_dir_path)
        self.img_dir_path = getattr(self, f'{"train" if train_not_test else "test"}_img_dir_path')
        assert os.path.exists(self.img_dir_path) and os.path.isdir(self.img_dir_path)
        if which_classes == 'all':
            self.polarity_dirs = next(os.walk(self.img_dir_path))[1]
        elif which_classes == '6class':
            self.polarity_dirs = LINDataset.Classes
        self.which_classes = which_classes

    def forward(self) -> None:
        """
        Method for completing a forward pass in scraping LIN dataset's images:
        Visits every item directory, process its images and saves image information to a JSON file named
        `polarity_factor_info.json`.
        """
        for polarity_dir in tqdm(self.polarity_dirs):
            polarity_dir_path = f'{self.img_dir_path}/{polarity_dir}'
            if not os.path.exists(f'{polarity_dir_path}/polarity_factor_info.json'):
                images = os.listdir(polarity_dir_path)
                images = sorted(
                    [_i for _i in images if _i.endswith('.png')],
                    key=lambda _i: int(_i.replace('cell', '').replace('.png', ''))
                )
                images_info = {
                    'polarity': polarity_dir,
                    'path': f'/{polarity_dir}',
                    'cell_images': images,
                    'cell_images_count': len(images),
                }
                with open(f'{polarity_dir_path}/polarity_factor_info.json', 'w') as json_fp:
                    json.dump(images_info, json_fp, indent=4)
            self.logger.debug(f'{polarity_dir_path}: [DONE]')

    def backward(self) -> None:
        """
        Method for completing a backward pass in scraping LookBook images:
        Similar to DeepFashion scraper's backward pass, recursively visits all directories under image root merging
        information saved in JSON files found inside children directories.
        """
        # Initialize aggregator
        polarity_factors_info = {
            'polarity': 'all',
            'path': '',
            'cell_images': [],
            'cell_images_count': 0,
        }
        # Start merging
        for polarity_dir in tqdm(self.polarity_dirs):
            id_dir_path = f'{self.img_dir_path}/{polarity_dir}'
            polarity_factor_info_path = f'{id_dir_path}/polarity_factor_info.json'
            assert os.path.exists(polarity_factor_info_path), f'images_info_path={polarity_factor_info_path}: NOT FOUND'
            with open(polarity_factor_info_path) as json_fp:
                polarity_factor_info = json.load(json_fp)
            # Prefix images
            file_prefix = polarity_factor_info['path'].lstrip('/')
            for _i, _name in enumerate(polarity_factor_info['cell_images']):
                polarity_factor_info['cell_images'][_i] = f'{file_prefix}/{_name}'
            # Merge item in aggregated items info
            polarity_factors_info['cell_images'] += polarity_factor_info['cell_images']
            polarity_factors_info['cell_images_count'] += polarity_factor_info['cell_images_count']
        # Save JSON file
        with open(f'{self.img_dir_path}/polarity_factors_info_{self.which_classes}.json', 'w') as json_fp:
            json.dump(polarity_factors_info, json_fp, indent=4)

    @staticmethod
    def run(forward_pass: bool = True, backward_pass: bool = True) -> None:
        """
        Entry point of class.
        :param forward_pass: set to True to run scraper's forward pass (create polarity_factor_info.json files in
                             polarity factors dirs)
        :param backward_pass: set to True to run scraper's backward pass (recursively merge polarity factors JSON files)
                              Note: if :attr:`forward_pass` is set to True, then :attr:`backward_pass` will be True.
        """
        for _which_classes in ['all', '6class']:
            for _train_not_test in [True, False]:
                scraper = LINScraper(train_not_test=_train_not_test, which_classes=_which_classes)
                scraper.logger.info(
                    f'which_classes={_which_classes} | SCRAPE DIR = {os.path.basename(scraper.img_dir_path)}')
                if forward_pass:
                    # Forward pass
                    scraper.logger.info('[forward] STARTING')
                    scraper.forward()
                    scraper.logger.info('[forward] DONE')
                    backward_pass = True
                # Backward pass
                if backward_pass:
                    scraper.logger.info('[backward] STARTING')
                    scraper.backward()
                    scraper.logger.info('[backward] DONE')
                scraper.logger.info(f'[DONE] which_classes={_which_classes}')
                scraper.logger.info(f'')
        scraper.logger.info('DONE')


class LINNearestNeighborsScraper:
    """
    LINNearestNeighborsScraper Class:
    Searches k-NNs for every training image in the 6 classes.
    """

    def __init__(self, local_gdrive_root: str, which_search: str = 'train'):
        """
        LINNearestNeighborsScraper class constructor.
        :param str local_gdrive_root: local Google Drive mount point
        :param str which_search: one of 'train', 'test'
        """
        self.logger = CommandLineLogger(log_level=os.getenv('LOG_LEVEL', 'info'))
        # Via locally-mounted Google Drive (when running from inside Google Colaboratory)
        capsule = LocalCapsule(local_gdrive_root)
        fs = LocalFilesystem(ccapsule=capsule)
        dataset_gfolder = LocalFolder.root(capsule_or_fs=fs).subfolder_by_name('Datasets')
        self.nn_dl_bs = 50
        self.searched_classes = LINDataset.Classes
        self.query_dataloaders = {k: LINDataloader(dataset_gfolder, train_not_test=True, logger=self.logger,
                                                   which_classes=k, batch_size=20, pin_memory=True, return_path=True,
                                                   image_transforms=ToNumpy())
                                  for k in self.searched_classes}
        # searched_classes = [k for k in LINDataset.Classes if k.startswith('A')]
        self.nn_dataloaders = {k: LINDataloader(dataset_gfolder, train_not_test=which_search == 'train',
                                                logger=self.logger, which_classes=k, batch_size=self.nn_dl_bs,
                                                return_path=True, pin_memory=True, num_workers=4)
                               for k in self.searched_classes}
        self.which_search = which_search
        self.dataset_gfolder: FilesystemFolder = dataset_gfolder
        self.dataset_lin_gfolder: FilesystemFolder = dataset_gfolder.subfolder_by_name('LIN_48x80')
        self.dl_transforms = transforms.Compose([
            ToNumpy(),
            LinToTensorNormalized()
        ])

    @staticmethod
    def _compute_L2_dists(query_img: Tensor, nn_imgs: Tensor, channel_index: int = 0) -> Tensor:
        """
        Modified version of the one presented in paper's repo.
        Source of original: https://github.com/aosokin/biogans
        :param Tensor query_img: query image, of shape (C, H, W)
        :param Tensor nn_imgs: to-search images mini batch, of shape (B, C, H, W)
        :param channel_index: the index of the image channel based on which comparison will take place
        :return: a torch.Tensor object containing the L2 distances between query image and all images, of shape (B,)
        """
        assert query_img.shape == nn_imgs.shape[1:]
        query_img = query_img[channel_index].view(1, -1)
        nn_imgs = nn_imgs[:, channel_index, :, :].view(nn_imgs.shape[0], -1)
        return torch.sum((nn_imgs - query_img) ** 2, dim=1)

    @staticmethod
    def _find_neighbors_in_batch(img: Tensor, nn_imgs: Tensor, img_paths: list, k: int = 5) -> Tuple[list, Tensor]:
        """
        Modified version of the one presented in paper's repo.
        Source of original: https://github.com/aosokin/biogans
        :param Tensor img: query image, of shape (C, H, W)
        :param Tensor nn_imgs: to-search images mini batch, of shape (B, C, H, W)
        :param list img_paths: image paths corresponding to the mini batch images
        :param int k: number of nearest neighbors to keep
        :return: a tuple containing the subset of image paths, the corresponding L2 distances
        """
        # compute the L2 distances
        dists = LINNearestNeighborsScraper._compute_L2_dists(query_img=img, nn_imgs=nn_imgs)
        # sort in the order of increasing distance
        dists_sorted, indices = torch.sort(dists, dim=0, descending=False)
        indices = indices.cpu()
        dists_sorted = dists_sorted.cpu()
        # return the nearest neighbors
        return [img_paths[i] for i in indices[:k]], dists_sorted[:k]

    @staticmethod
    def _merge_lists(based_on: np.ndarray, based_on_vs: Optional[np.ndarray], list1: list, list1_vs: Optional[list],
                     list2: Optional[list] = None,
                     list2_vs: Optional[list] = None) -> Tuple[list, list, Optional[list]] or Tuple[list, list]:
        """
        :param based_on: compare this (contd below)
        :param based_on_vs: with this
        :param list1: to select each element of list1 vs list1_vs
        :param list1_vs:
        :param list2: and select each element of list1 vs list2_vs
        :param list2_vs:
        :return: a tuple containing either a list and a None or two lists
        """
        if based_on_vs is None:
            if list2 is None:
                return based_on, list1
            return based_on, list1, list2
        assert based_on.shape == based_on_vs.shape, f'{based_on.shape} vd {based_on_vs.shape}'
        assert len(based_on) == len(list1)
        assert list1_vs is None or len(list1) == len(list1_vs), f'{len(list1)} vs {len(list1_vs)}'
        assert list2 is None or len(list1) == len(list2)
        assert list2_vs is None or len(list2) == len(list2_vs)
        list1_final = []
        list2_final = []
        based_on_concat = np.concatenate((based_on, based_on_vs))
        bo_s, bo_sis = np.sort(based_on_concat), np.argsort(based_on_concat)
        based_on_final = bo_s[:len(based_on)]
        bo_sis_bool = bo_sis >= len(based_on)
        for bo_si, bo_si_bool in zip(bo_sis, bo_sis_bool):
            if bo_si_bool:
                bo_si -= len(based_on)
                assert list1_vs is not None
                list1_final.append(list1_vs[bo_si])
                if list2 is not None:
                    list2_final.append(list2_vs[bo_si])
            else:
                list1_final.append(list1[bo_si])
                if list2 is not None:
                    list2_final.append(list2[bo_si])
        if len(list2_final) == 0:
            return based_on_final, list1_final[:len(list1)]
        return based_on_final, list1_final[:len(list1)], list2_final[:len(list2)]

    def forward(self, k: int = 5) -> None:
        """
        For every image in the dataset find its k-NNs based on the red channel. For all the images in a polarity factor
        directory, create a `nearest_neighbors_info.json` and save it inside the directory.
        :param int k: number of nearest neighbors to keep (defaults to 5 - incl. self)
        """
        for query_dir, query_dataloader in self.query_dataloaders.items():
            nearest_neighbors_info_path = os.path.join(query_dataloader.dataset.root,
                                                       f'nearest_neighbors_info_{self.which_search}.json')

            # Check if already calculated
            if os.path.exists(nearest_neighbors_info_path):
                _path = pathlib.Path(nearest_neighbors_info_path)
                self.logger.info(f'[forward][{self.which_search}/{query_dir}] File exists at ' +
                                 f'{_path.relative_to(_path.parent.parent.parent)}')
                continue

            # Start a fresh computation
            nearest_neighbors_info = {
                "_path": query_dir,
                "_searched_classes": self.searched_classes,
                "_searched_dataset": self.which_search,
                "_k": k
            }
            pbar = tqdm(query_dataloader)
            for query_img_batch, query_img_path_batch in pbar:
                query_img_batch = query_img_batch.cuda()
                for bi in range(query_img_batch.shape[0]):
                    query_img = query_img_batch[bi].squeeze()
                    query_img_path = query_img_path_batch[bi]

                    nearest_neighbors_info[query_img_path] = {}
                    nearest_neighbors_info[query_img_path]['_one_per_class'] = {
                        "dists": [],
                        "img_paths": [],
                    }

                    nn_dists_final_final = None
                    nn_img_paths_final_final = None
                    for nn_dir, nn_dataloader in self.nn_dataloaders.items():
                        if nn_dir == query_dir:
                            continue
                        nn_dists_final = None
                        nn_img_paths_final = None
                        for nn_bi, (nn_imgs, nn_img_paths) in enumerate(nn_dataloader):
                            pbar.set_description(f'{query_img_path.strip("/")} | ' +
                                                 f'{self.which_search}/{os.path.basename(nn_dataloader.dataset.root)}' +
                                                 f': {nn_bi}/{len(nn_dataloader)}')
                            if nn_bi == len(nn_dataloader) - 1 and nn_imgs.shape[0] != self.nn_dl_bs:
                                # FIX: repeat data when fewer than the batch size
                                times = math.ceil(float(self.nn_dl_bs) / float(nn_imgs.shape[0]))
                                nn_imgs = torch.cat(times * [nn_imgs, ])[:self.nn_dl_bs]
                                nn_img_paths = (times * nn_img_paths)[:self.nn_dl_bs]
                            nnb_img_paths, nnb_dists = LINNearestNeighborsScraper._find_neighbors_in_batch(
                                img=query_img,
                                nn_imgs=nn_imgs.cuda(),
                                img_paths=nn_img_paths,
                                k=k
                            )
                            nn_dists_final, nn_img_paths_final = LINNearestNeighborsScraper._merge_lists(
                                based_on=nnb_dists.cpu().numpy(),
                                based_on_vs=nn_dists_final,
                                list1=nnb_img_paths,
                                list1_vs=nn_img_paths_final
                            )
                            nn_dists_final_final, nn_img_paths_final_final = LINNearestNeighborsScraper._merge_lists(
                                based_on=nnb_dists.cpu().numpy(),
                                based_on_vs=nn_dists_final_final,
                                list1=nnb_img_paths,
                                list1_vs=nn_img_paths_final_final
                            )
                        nearest_neighbors_info[query_img_path][f'{nn_dir}'] = {
                            "dists": nn_dists_final.tolist(),
                            "img_paths": nn_img_paths_final,
                        }
                        # noinspection PyUnresolvedReferences
                        nearest_neighbors_info[query_img_path]['_one_per_class']['dists'].append(
                            nn_dists_final.tolist()[0])
                        # noinspection PyUnresolvedReferences
                        nearest_neighbors_info[query_img_path]['_one_per_class']['img_paths'].append(
                            nn_img_paths_final[0])
                    nearest_neighbors_info[query_img_path][f'_all'] = {
                        "dists": nn_dists_final_final.tolist(),
                        "img_paths": nn_img_paths_final_final,
                    }
            # Save json in each directory
            with open(nearest_neighbors_info_path, 'w') as json_fp:
                json.dump(nearest_neighbors_info, json_fp, indent=4, sort_keys=True)
            self.logger.info(f'[forward][DONE] {query_dir} (saved at: {nearest_neighbors_info_path})')

    def backward(self) -> None:
        """
        Backward Pass: Visits every training directory and merges the train and test nearest_neighbors_info_*.json
                       files. Then merges all nearest_neighbors_info.json files of sub-directories to parent.
        """
        for query_dir, query_dataloader in self.query_dataloaders.items():
            query_dataset: LINDataset = query_dataloader.dataset
            nearest_neighbors_info_train_path = os.path.join(query_dataset.root,
                                                             f'nearest_neighbors_info_train.json')
            nearest_neighbors_info_test_path = os.path.join(query_dataset.root,
                                                            f'nearest_neighbors_info_test.json')
            try:
                assert os.path.exists(nearest_neighbors_info_train_path) \
                       and os.path.exists(nearest_neighbors_info_test_path)
                nearest_neighbors_info_path = os.path.join(query_dataset.root, f'nearest_neighbors_info.json')
                # os.remove(nearest_neighbors_info_path)
                if os.path.exists(nearest_neighbors_info_path):
                    _path = pathlib.Path(nearest_neighbors_info_path)
                    self.logger.info(f'[backward][{self.which_search}/{query_dir}] File exists at ' +
                                     f'{_path.relative_to(_path.parent.parent.parent)}')
                    continue
                    # os.rename(nearest_neighbors_info_path, f'{nearest_neighbors_info_path}.bak')
                # Open both files and update image paths
                nni_train = query_dataset.nearest_neighbors_info_train
                nni_test = query_dataset.nearest_neighbors_info_test
                assert nni_train is not None and nni_test is not None
            except AssertionError as e:
                self.logger.critical(f'[backward][{self.which_search}/{query_dir}] AssertionError {str(e)}')
                continue
            path = pathlib.Path(query_dataset.root)
            train_dir_name = str(path.relative_to(path.parent.parent).parent)
            test_dir_name = train_dir_name.replace('_train', '_test')
            self.logger.info(f'[backward][{self.which_search}/{query_dir}] Start merging...')
            #   - prefix image paths
            for img_path, img_data in nni_train.items():
                if img_path.startswith('_'):
                    continue
                for nn_dir in img_data.keys():
                    nni_train[img_path][nn_dir]['img_paths'] = [f'{train_dir_name}{p}'
                                                                for p in nni_train[img_path][nn_dir]['img_paths']]
            for img_path, img_data in nni_test.items():
                if img_path.startswith('_'):
                    continue
                for nn_dir in img_data.keys():
                    nni_test[img_path][nn_dir]['img_paths'] = [f'{test_dir_name}{p}'
                                                               for p in nni_test[img_path][nn_dir]['img_paths']]
            #   - combine json files image-by-image
            if nni_train['_k'] != nni_test['_k']:
                self.logger.debug(f'[backward][{self.which_search}/{query_dir}] different k\'s detected ' +
                                  f'({nni_train["_k"]} vs {nni_test["_k"]})')
                k = min(nni_train['_k'], nni_test['_k'])
            else:
                k = nni_train['_k']
            nni = {
                "_k": k,
                "_path": query_dir,
                "_searched_classes": self.searched_classes,
                "_searched_dataset": "train_test"
            }
            pbar = tqdm([k for k in nni_train.keys() if not k.startswith('_')])
            for query_img_path in pbar:
                nni[query_img_path] = {}
                for nn_dir in nni_train[query_img_path].keys():
                    train_info = nni_train[query_img_path][nn_dir]
                    test_info = nni_test[query_img_path][nn_dir]

                    if nn_dir != '_one_per_class':
                        dists_final, img_paths_final = LINNearestNeighborsScraper._merge_lists(
                            based_on=np.array(train_info['dists']),
                            based_on_vs=np.array(test_info['dists']),
                            list1=train_info['img_paths'],
                            list1_vs=test_info['img_paths']
                        )
                    else:
                        dists_final = []
                        img_paths_final = []
                        for _dtr, _dte, _iptr, _ipte in zip(train_info['dists'], test_info['dists'],
                                                            train_info['img_paths'], test_info['img_paths']):
                            if _dtr <= _dte:
                                dists_final.append(_dtr)
                                img_paths_final.append(_iptr)
                            else:
                                dists_final.append(_dte)
                                img_paths_final.append(_ipte)
                        dists_final = np.array(dists_final)
                    nni[query_img_path][nn_dir] = {
                        "dists": dists_final.tolist(),
                        "img_paths": img_paths_final,
                    }
            # Save json in each directory
            with open(nearest_neighbors_info_path, 'w') as json_fp:
                json.dump(nni, json_fp, indent=4, sort_keys=True)
            self.logger.info(f'[backward][{self.which_search}/{query_dir}] DONE ' +
                             f'(saved at: {nearest_neighbors_info_path})')

    def backward_root(self):
        """
        Merge all *_info files of subdirectories into
        :return:
        """
        # TODO
        pass

    def generate_multi_channel_images(self):
        dataset_root = self.dataset_lin_gfolder.local_root

        def path2class_idx(__path: str) -> int:
            for __class_idx, __class_name in enumerate(self.searched_classes):
                if f'{__class_name}/' in __path:
                    return __class_idx
            raise RuntimeError(f'No class found for path: {__path} (searched: {self.searched_classes})')

        for query_dir, query_dataloader in self.query_dataloaders.items():
            query_dataset: LINDataset = query_dataloader.dataset
            nearest_neighbors_info_path = os.path.join(query_dataset.root, f'nearest_neighbors_info.json')
            assert os.path.exists(nearest_neighbors_info_path), f'searched at: {nearest_neighbors_info_path}'
            nearest_neighbors_pth_path = os.path.join(query_dataset.root, f'nearest_neighbors.pth')
            if os.path.exists(nearest_neighbors_pth_path):
                os.rename(nearest_neighbors_pth_path, f'{nearest_neighbors_pth_path}.bak')
            if os.path.exists(nearest_neighbors_pth_path):
                _path = pathlib.Path(nearest_neighbors_pth_path)
                self.logger.info(f'[generate][{query_dir}] File exists at ' +
                                 f'{_path.relative_to(_path.parent.parent.parent)}')
                continue
            # Read (composite) json file
            with open(nearest_neighbors_info_path, 'r') as json_fp:
                nn_info = json.load(json_fp)
            nn_pkl_items = []
            nn_reds_pkl_items = []
            pkl_path_to_index = {}
            PKL_INDEX = 0
            pbar = tqdm(query_dataloader)
            for query_img_batch, query_img_path_batch in pbar:
                for bi in range(query_img_batch.shape[0]):
                    query_img = query_img_batch[bi].squeeze()
                    query_img_path = query_img_path_batch[bi]
                    query_class_idx = path2class_idx(query_img_path)
                    pbar.set_description(query_img_path)
                    #   - init data holder for red + greens
                    nn_pkl_item = torch.zeros(1 + len(self.searched_classes), 48, 80)
                    red_binary_mask = query_img[0, :, :] > 0
                    self.dl_transforms.transforms[-1].mask = red_binary_mask.clone().numpy()
                    nn_pkl_item[0, :, :] = query_img[0, :, :] * red_binary_mask
                    nn_pkl_item[1 + query_class_idx, :, :] = query_img[1, :, :] * red_binary_mask
                    #   - init data holder for reds
                    nn_reds_pkl_item = torch.zeros(len(self.searched_classes), 48, 80)
                    nn_reds_pkl_item[query_class_idx, :, :] = query_img[0, :, :]
                    #   - load nn info
                    nn_info_img = nn_info[query_img_path]
                    nn_img_paths = nn_info_img['_one_per_class']['img_paths']
                    #   - load nn images
                    for nn_img_path_i, nn_img_path in enumerate(nn_img_paths):
                        nn_img_path_abs = os.path.join(dataset_root, nn_img_path)
                        nn_img = self.dl_transforms(Image.open(nn_img_path_abs))
                        #   - add red channel to reds
                        nn_class_idx = path2class_idx(nn_img_path)
                        assert torch.allclose(nn_reds_pkl_item[nn_class_idx],
                                              torch.zeros_like(nn_reds_pkl_item[nn_class_idx])), \
                            f'{query_class_idx} vs. {nn_class_idx}'
                        nn_reds_pkl_item[nn_class_idx, :, :] = nn_img[0, :, :]
                        #   - add green channel to greens
                        assert torch.allclose(nn_pkl_item[1 + nn_class_idx],
                                              torch.zeros_like(nn_pkl_item[1 + nn_class_idx])), \
                            f'{query_class_idx} vs. {nn_class_idx}'
                        nn_pkl_item[1 + nn_class_idx, :, :] = nn_img[1, :, :] * red_binary_mask
                    #   - check assignments
                    for _i in range(len(self.searched_classes)):
                        assert not torch.allclose(nn_reds_pkl_item[_i], torch.zeros_like(nn_reds_pkl_item[_i])), f'{_i}'
                        assert not torch.allclose(nn_pkl_item[1 + _i], torch.zeros_like(nn_pkl_item[1 + _i])), f'{_i}'
                        for _j in range(len(self.searched_classes)):
                            if _i == _j:
                                continue
                            assert not torch.allclose(nn_reds_pkl_item[_i], nn_reds_pkl_item[_j]), f'{_i} vs. {_j} reds'
                            assert not torch.allclose(nn_pkl_item[1 + _i], nn_pkl_item[1 + _j]), f'{_i} vs. {_j} greens'
                    #   - append to parent list
                    nn_pkl_items.append(nn_pkl_item)
                    nn_reds_pkl_items.append(nn_reds_pkl_item)
                    pkl_path_to_index[query_img_path] = PKL_INDEX
                    PKL_INDEX += 1
            # Save pkl files
            nn_pkl_items_tensor = torch.stack(nn_pkl_items)
            nn_reds_pkl_items_tensor = torch.stack(nn_reds_pkl_items)
            torch.save({
                'red_greens': nn_pkl_items_tensor,
                'reds': nn_reds_pkl_items_tensor,
                'path2index': pkl_path_to_index,
            }, nearest_neighbors_pth_path)
            self.logger.info(f'[generate][{query_dir}] DONE ' +
                             f'(saved at: {nearest_neighbors_pth_path})')

    def clear(self):
        fs_folder = self.dataset_gfolder \
            .subfolder_by_name(f'LIN_48x80') \
            .subfolder_by_name(f'LIN_Normalized_WT_size-48-80_{self.which_search}')
        for d, _, fs in os.walk(fs_folder.local_root):
            if d.endswith(self.which_search):
                continue
            for f in fs:
                if f.startswith('nearest'):
                    os.remove(os.path.join(d, f))
                    self.logger.info(f'[clear] File REMOVed at {os.path.join(d, f)}')

    @staticmethod
    def run(local_gdrive_root: str, k: int = 5, forward_pass: bool = True, backward_pass: bool = True,
            generate_images: bool = False) -> None:
        """
        Entry point of class.
        :param str local_gdrive_root: absolute path to local Google Drive mount point
        :param int k: number of NNs
        :param bool forward_pass: set to True to run scraper's forward pass (create polarity_factor_info.json files in
                             polarity factors dirs)
        :param bool backward_pass: set to True to run scraper's backward pass (recursively merge polarity factors JSON
                                   files). Note: if :attr:`forward_pass` is set to True, then :attr:`backward_pass`
                                   will be True.
        :param bool generate_images: set to True to generate multi-channel images consisting of each training images
                                     inside any of the 6 classes red channel + the original + 4 other green channels.
        """
        nn_scraper_train = LINNearestNeighborsScraper(local_gdrive_root, which_search='train')
        nn_scraper_test = LINNearestNeighborsScraper(local_gdrive_root, which_search='test')
        if forward_pass:
            for _which_search in ['train', 'test']:
                nn_scraper = nn_scraper_train if _which_search == 'train' else nn_scraper_test
                # if click.confirm(f'Clear previous NN scrapes for "{_which_search}"?', default=True):
                #     nn_scraper.clear()
                nn_scraper.logger.info(f'[forward] STARTING which_search={_which_search}')
                nn_scraper.forward(k=k)
                nn_scraper.logger.info(f'[forward] DONE')
            backward_pass = True
        if backward_pass:
            nn_scraper_train = LINNearestNeighborsScraper(local_gdrive_root, which_search='train')
            nn_scraper_train.logger.info('[backward] STARTING')
            nn_scraper_train.backward()
            nn_scraper_train.backward_root()
            nn_scraper_train.logger.info('[backward] DONE')
        if generate_images:
            nn_scraper_train.logger.info('[generate_images] STARTING')
            nn_scraper_train.generate_multi_channel_images()
            nn_scraper_train.logger.info('[generate_images] DONE')
        nn_scraper_train.logger.info('DONE')


if __name__ == '__main__':
    _local_gdrive_root = '/home/achariso/PycharmProjects/kth-ml-course-projects/biogans/.gdrive_personal'
    # if click.confirm('Do you want to (re)scrape the dataset now?', default=True):
    #     # Scape images to create info files
    #     LINScraper.run(forward_pass=True, backward_pass=True)
    #     # Scrape nearest neighbors of each image in the training set
    #     LINNearestNeighborsScraper.run(_local_gdrive_root, k=5, forward_pass=False, backward_pass=False,
    #                                    generate_images=True)

    # Via locally-mounted Google Drive (when running from inside Google Colaboratory)
    _capsule = LocalCapsule(_local_gdrive_root)
    _fs = LocalFilesystem(ccapsule=_capsule)
    _groot = LocalFolder.root(capsule_or_fs=_fs).subfolder_by_name('Datasets')

    # _lin_train = LINDataset(dataset_fs_folder_or_root=_groot, train_not_test=True, which_classes='6class')
    # # _lin_test = LINDataset(dataset_fs_folder_or_root=_groot, train_not_test=False, which_classes='6class',
    # #                        logger=_lin_train.logger)
    # _lin_alp14_train = LINDataset(dataset_fs_folder_or_root=_groot, train_not_test=True, which_classes='Cki2')
    # s = _lin_alp14_train[0]
    # plt.imshow(s[0, :].reshape(48, 80), cmap="Reds")
    # plt.show()
    # s3 = torch.concat((s, -torch.ones((1, 48, 80))), dim=0)
    # s3 = invert_transforms(_lin_alp14_train.transforms)(s3)
    # plt.imshow(s3.numpy().transpose(1, 2, 0))
    # plt.show()
    # plt.imshow(s[0].numpy().reshape(48, 80), cmap="Reds")
    # plt.show()
    # plt.imshow(s[1].numpy().reshape(48, 80), cmap="Greens")
    # plt.show()
    # # _lin_alp14_test = LINDataset(dataset_fs_folder_or_root=_groot, train_not_test=False, which_classes='Alp14',
    # #                              logger=_lin_train.logger)
    # # # _lin.fetch_and_unzip()
    # # _lin_nn_alp14 = LINNearestDataset(dataset_fs_folder_or_root=_groot, which_classes='Alp14')
    # # plt.imshow(_lin_nn_alp14[0][1:].reshape(48 * 6, 80), cmap="Greens")
    # # plt.show()
    # # _lin_nn_alp14 = LINNearestDataset(dataset_fs_folder_or_root=_groot, which_classes='Alp14', reds_only=True)
    # # plt.imshow(_lin_nn_alp14[0].reshape(48 * 6, 80), cmap="Reds")
    # # plt.show()
    # print(_lin_alp14_train[0].shape)

    lin6dl = LINDataloader6Class(_groot, train_not_test=True, batch_size=10)
    batch = next(iter(lin6dl))
    print(batch.shape)
    plt.imshow(batch[:, 1, 0, :, :].reshape(48 * 6, 80), cmap="Reds")
    plt.show()
    plt.imshow(batch[:, 1, 1, :, :].reshape(48 * 6, 80), cmap="Greens")
    plt.show()
    # lin6ds = LINDataset6Class(dataset_fs_folder_or_root=_groot, train_not_test=True)
    # print(lin6ds[0].shape)
    # plt.imshow(lin6ds[4000][:, 0, :, :].reshape(48 * 6, 80), cmap="Reds")
    # plt.show()
