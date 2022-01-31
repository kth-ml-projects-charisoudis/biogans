import os
from typing import Optional, Tuple

import click
from PIL import Image, UnidentifiedImageError
from torch import Tensor
# noinspection PyProtectedMember
from torch.utils.data import Dataset
from torchvision.transforms import transforms, Compose

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

    # Default normalization parameters for ICRB (converts tensors' ranges to [-1,1]
    NormalizeMean = 0.5
    NormalizeStd = 0.5

    def __init__(self, dataset_fs_folder_or_root: FilesystemFolder, image_transforms: Optional[Compose] = None):
        """
        LINDataset class constructor.
        :param (FilesystemFolder) dataset_fs_folder_or_root: a `utils.ifaces.FilesystemFolder` object to download / use
                                                             dataset from local or remote (Google Drive) filesystem
        :param (optional) image_transforms: a list of torchvision.transforms.* sequential image transforms
        :raises FileNotFoundError: either when the dataset is not present in local filesystem or when the
                                   `items_dt_info.json` is not present inside dataset's (local) root
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
        self.logger = CommandLineLogger(log_level=os.getenv('LOG_LEVEL', 'info'), name=self.__class__.__name__)
        self.train_img_dir_path = self.subfolders[1].local_root
        self.test_img_dir_path = self.subfolders[0].local_root
        # Check that the dataset is present at the local filesystem
        if not self.is_fetched_and_unzipped():
            if click.confirm(f'Dataset is not fetched and unzipped. Would you like to fetch now?', default=True):
                self.fetch_and_unzip(in_parallel=False, show_progress=True)
            else:
                raise FileNotFoundError(f'Dataset not found in local filesystem (tried {self.root})')
        # Load item info
        # TODO
        self.train_img_count = 0
        self.test_img_count = 0
        self.logger.debug(f'Found {to_human_readable(self.train_img_count)} image pairs in the TRAINING dataset')
        self.logger.debug(f'Found {to_human_readable(self.test_img_count)} image pairs in the TEST dataset')
        # Save transforms
        self._transforms = image_transforms

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, t: Optional[Compose] = None) -> None:
        self._transforms = t if t else transforms.ToTensor()

    def index_to_paths(self, index: int) -> Tuple[str, str]:
        """
        Given an image-pair index it returns the file paths of the pair's images.
        :param (int) index: image-pair's index
        :return: a tuple containing (image_1_path, image_2_path), where lll file paths are absolute
        """
        image_1_path, image_2_path = self.dt_image_pairs[index]
        return f'{self.img_dir_path}/{image_1_path}', f'{self.img_dir_path}/{image_2_path}'

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Implements abstract Dataset::__getitem__() method.
        :param (int) index: integer with the current image index that we want to read from disk
        :return: a tuple containing the images from domain A (human) and B (product), each as a torch.Tensor object
        """
        paths_tuple = self.index_to_paths(index)
        # Fetch images
        try:
            img_s = Image.open(paths_tuple[0])
        except UnidentifiedImageError:
            self.logger.critical(f'Image opening failed (path: {paths_tuple[0]})')
            return self.__getitem__(index + 1)
        img_t_path = paths_tuple[1]
        try:
            img_t = Image.open(img_t_path)
        except UnidentifiedImageError:
            self.logger.critical(f'Image opening failed (path: {img_t_path})')
            return self.__getitem__(index + 1)
        # Apply transforms
        if self.transforms:
            img_s = self.transforms(img_s)
            img_t = self.transforms(img_t)
        return img_s, img_t

    def __len__(self) -> int:
        """
        Implements abstract Dataset::__len__() method. This method returns the total "length" of the dataset which is
        the total number of  images contained in each pile (or the min of them if they differ).
        :return: integer
        """
        return 0


if __name__ == '__main__':
    # Via locally-mounted Google Drive (when running from inside Google Colaboratory)
    _local_gdrive_root = '/home/achariso/PycharmProjects/gans-thesis/.gdrive'
    _capsule = LocalCapsule(_local_gdrive_root)
    _fs = LocalFilesystem(ccapsule=_capsule)
    _groot = LocalFolder.root(capsule_or_fs=_fs).subfolder_by_name('Datasets')

    _lin = LINDataset(dataset_fs_folder_or_root=_groot)
    print(_lin.fetch_and_unzip())
