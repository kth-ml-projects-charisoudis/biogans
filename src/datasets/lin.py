import json
import os
from typing import Optional, Tuple, Union

import click
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
# noinspection PyProtectedMember
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms, Compose
from tqdm.autonotebook import tqdm

from utils.command_line_logger import CommandLineLogger
from utils.filesystems.gdrive import GDriveDataset
from utils.filesystems.local import LocalCapsule, LocalFilesystem, LocalFolder
from utils.ifaces import FilesystemFolder
from utils.string import to_human_readable


class LINDataset(Dataset, GDriveDataset):
    """
    LINDataset Class:
    This class is used to define the way LookBook dataset is accessed for the purpose of pixel-wise domain transfer
    (PixelDT).
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

    # Default normalization parameters for ICRB (converts tensors' ranges to [-1,1]
    NormalizeMean = 0.5
    NormalizeStd = 0.5

    def __init__(self, dataset_fs_folder_or_root: FilesystemFolder, image_transforms: Optional[Compose] = None,
                 train_not_test: bool = True, which_classes: str = 'all', logger: Optional[CommandLineLogger] = None,
                 return_path: bool = True):
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
        self.train_img_dir_path = self.subfolders[1].local_root
        self.test_img_dir_path = self.subfolders[0].local_root
        # Check that the dataset is present at the local filesystem
        if not self.is_fetched_and_unzipped():
            if click.confirm(f'Dataset is not fetched and unzipped. Would you like to fetch now?', default=True):
                self.fetch_and_unzip(in_parallel=False, show_progress=True)
            else:
                raise FileNotFoundError(f'Dataset not found in local filesystem (tried {self.root})')
        # Load images info file
        self.img_dir_path = getattr(self, f'{"train" if train_not_test else "test"}_img_dir_path')
        assert os.path.exists(self.img_dir_path) and os.path.isdir(self.img_dir_path)
        if which_classes in ['all', '6class']:
            self.root = self.img_dir_path
            self.polarity_factors_info_path = os.path.join(self.root,
                                                           f'polarity_factors_info_{which_classes}.json')
        else:
            self.root = os.path.join(self.img_dir_path, which_classes)
            self.polarity_factors_info_path = os.path.join(self.root, f'polarity_factor_info.json')
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
        self._transforms = t if t else transforms.ToTensor()

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
        if self.transforms:
            cell_img = self.transforms(cell_img)
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

    # @staticmethod
    # def


class LINDataloader(DataLoader):
    """
    LINDataloader Class:
    This class is used to load and access LIN dataset using PyTorch's DataLoader interface.
    """

    def __init__(self, dataset_fs_folder_or_root: FilesystemFolder, image_transforms: Optional[Compose] = None,
                 train_not_test: bool = True, which_classes: str = 'all', logger: Optional[CommandLineLogger] = None,
                 **dl_kwargs):
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
        :raises FileNotFoundError: either when the dataset is not present in local filesystem or when the
                                   `polarity_factor_info.json` is not present inside dataset's (local) root
        """
        # Instantiate dataset
        dataset = LINDataset(dataset_fs_folder_or_root=dataset_fs_folder_or_root, image_transforms=image_transforms,
                             train_not_test=train_not_test, which_classes=which_classes, logger=logger)
        # Instantiate dataloader
        super(LINDataloader, self).__init__(dataset=dataset, **dl_kwargs)


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

    def forward(self, ) -> None:
        """
        Method for completing a forward pass in scraping LIN dataset's images:
        Visits every item directory, process its images and saves image information to a JSON file named
        `polarity_factor_info.json`.
        """
        for polarity_dir in tqdm(self.polarity_dirs):
            polarity_dir_path = f'{self.img_dir_path}/{polarity_dir}'
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
            with open(f'{polarity_dir_path}/polarity_factor_info_{self.which_classes}.json', 'w') as json_fp:
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
    def _compute_L2_dists(query_img, nn_imgs, channel_index: int = 0):
        assert (query_img.shape == nn_imgs.shape[1:])
        query_img = query_img[channel_index].view(1, -1)
        nn_imgs = nn_imgs[:, channel_index, :, :].view(nn_imgs.shape[0], -1)
        return torch.sum((nn_imgs - query_img) ** 2, dim=1)

    @staticmethod
    def _find_neighbors_in_batch(img, nn_imgs, img_paths, k: int = 5):
        # compute the L2 distances
        dists = LINScraper._compute_L2_dists(img, nn_imgs)
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
        assert based_on.shape == based_on_vs.shape
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

    @staticmethod
    def nearest_neighbors(dataset_gfolder: FilesystemFolder, k: int = 5):
        """
        :param dataset_gfolder:
        :param k:
        :return:
        """
        logger = CommandLineLogger(log_level=os.getenv('LOG_LEVEL', 'info'), name=LINDataset.__name__)
        query_dataloaders = {k: LINDataloader(dataset_gfolder, train_not_test=True, logger=logger, which_classes=k,
                                              batch_size=1, pin_memory=True)
                             for k in LINDataset.Classes}
        # searched_classes = [k for k in LINDataset.Classes if k.startswith('A')]
        searched_classes = LINDataset.Classes
        nn_dataloaders = {k: LINDataloader(dataset_gfolder, train_not_test=True, logger=logger, which_classes=k,
                                           batch_size=200, pin_memory=True, num_workers=4)
                          for k in searched_classes}

        for query_dir, query_dataloader in query_dataloaders.items():

            nearest_neighbors_info_path = os.path.join(query_dataloader.dataset.root, 'nearest_neighbors_info.json')
            nearest_neighbors_info = {
                "_path": query_dir,
                "_searched_classes": searched_classes,
                "_k": k
            }

            pbar = tqdm(query_dataloader)
            for query_img, query_img_path in pbar:
                query_img = query_img.cuda().squeeze()
                query_img_path = query_img_path[0]

                nn_dists_final = None
                nn_img_paths_final = None
                for nn_dir, nn_dataloader in nn_dataloaders.items():
                    for nn_bi, (nn_imgs, nn_img_paths) in enumerate(nn_dataloader):
                        pbar.set_description(f'{query_img_path.strip("/")} | ' +
                                             f'{os.path.basename(nn_dataloader.dataset.root)}: ' +
                                             f'{nn_bi}/{len(nn_dataloader)}')
                        nnb_img_paths, nnb_dists = LINScraper._find_neighbors_in_batch(
                            img=query_img,
                            nn_imgs=nn_imgs.cuda(),
                            img_paths=nn_img_paths,
                            k=k
                        )
                        nn_dists_final, nn_img_paths_final = LINScraper._merge_lists(
                            based_on=nnb_dists.cpu().numpy(),
                            based_on_vs=nn_dists_final,
                            list1=nnb_img_paths,
                            list1_vs=nn_img_paths_final
                        )
                nearest_neighbors_info[query_img_path] = {
                    "dists": nn_dists_final.tolist(),
                    "img_paths": nn_img_paths_final,
                }
            # Save json in each directory
            with open(nearest_neighbors_info_path, 'w') as json_fp:
                json.dump(nearest_neighbors_info, json_fp, indent=4)
            logger.info(f'[DONE] {query_dir} (saved at: {nearest_neighbors_info_path})')

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


if __name__ == '__main__':
    # if click.confirm('Do you want to (re)scrape the dataset now?', default=True):
    #     LINScraper.run(forward_pass=True, backward_pass=True)
    # Via locally-mounted Google Drive (when running from inside Google Colaboratory)
    _local_gdrive_root = '/home/achariso/PycharmProjects/kth-ml-course-projects/biogans/.gdrive_personal'
    _capsule = LocalCapsule(_local_gdrive_root)
    _fs = LocalFilesystem(ccapsule=_capsule)
    _groot = LocalFolder.root(capsule_or_fs=_fs).subfolder_by_name('Datasets')

    # Scrape nearest neighbors of each image in the training set
    LINScraper.nearest_neighbors(_groot)
    exit(0)

    _lin_train = LINDataset(dataset_fs_folder_or_root=_groot, train_not_test=True, which_classes='6class')
    _lin_test = LINDataset(dataset_fs_folder_or_root=_groot, train_not_test=False, which_classes='6class',
                           logger=_lin_train.logger)
    _lin_alp14_train = LINDataset(dataset_fs_folder_or_root=_groot, train_not_test=True, which_classes='Alp14',
                                  logger=_lin_train.logger)
    _lin_alp14_test = LINDataset(dataset_fs_folder_or_root=_groot, train_not_test=False, which_classes='Alp14',
                                 logger=_lin_train.logger)
    # _lin.fetch_and_unzip()
