import os
import os.path
import random

import numpy as np
import torch

from utils.ifaces import Reproducible


def count_dirs(path: str, recursive: bool = False) -> int:
    """
    Get the number of directories under the given path.
    :param path: the root path to start searching for directories
    :param recursive: if True goes into every directory and counts sub-directories recursively
    :return: the total number of directories (and sub-directories if $recursive$ is set) under given $path$
    """
    return sum(len(dirs) for _, dirs, _ in os.walk(path)) if recursive else len(next(os.walk(path))[1])


def count_files(path: str, recursive: bool = False) -> int:
    """
    Get the number of files under the given path.
    :param path: the root path to start searching for files
    :param recursive: if True goes into every directory and counts files in sub-directories in a recursive manner
    :return: the total number of files in $path$ (and sub-directories of $path$ if $recursive$ is set)
    """
    return sum(len(files) for _, _, files in os.walk(path)) if recursive else len(next(os.walk(path))[2])


class ManualSeedReproducible(Reproducible):

    @staticmethod
    def manual_seed(seed: int) -> int:
        if Reproducible.is_seeded():
            return Reproducible._seed
        # Set seeder value
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        Reproducible._seed = seed
        return seed


def unzip_file(zip_filepath: str) -> bool:
    """
    Unzips a zip file at given :attr:`zip_filepath` using `unzip` lib & shell command.
    :param zip_filepath: the absolute path to the .zip file
    :return: a `bool` object set to True if shell command return 0, False otherwise
    """
    return True if 0 == os.system(f'unzip -q "{zip_filepath}" -d ' +
                                  f'"{zip_filepath.replace("/" + os.path.basename(zip_filepath), "")}"') \
        else False


def unnanify(y: np.ndarray) -> np.ndarray:
    """
    Remove NaNs from np array.
    (source: https://stackoverflow.com/a/6520696/13634700)
    :param (np.ndarray) y: input 1d array
    :return: an 1d array as np.ndarray object
    """
    nans, x = np.isnan(y), lambda z: z.nonzero()[0]
    y_out = y.copy()
    y_out[nans] = np.interp(x(nans), x(~nans), y_out[~nans])
    return y_out
