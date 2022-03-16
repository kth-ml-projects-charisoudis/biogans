import os
import sys
from typing import Union, Optional, Sized, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
# noinspection PyProtectedMember
from torch.utils.data import random_split, Dataset


def get_alpha_curve(num_iters: int, alpha_multiplier: float = 10.0,
                    return_x: bool = False) -> Tuple[np.ndarray, np.ndarray] or np.ndarray:
    """
    Return the sigmoid curve fro StyleGAN's alpha parameter.
    :param (int) num_iters: total number of iterations (equals the number of points in curve)
    :param (float) alpha_multiplier: parameter which controls the sharpness of the curve (1=linear, 1000=delta at half
                                     the interval - defaults to 10 that a yields a fairly smooth transition)
    :param (bool) return_x: set to True to have the method also return the x-values
    :return: either a tuple with x,y as np.ndarray objects  or y as np.ndarray object
    """
    if num_iters < 2:
        return np.arange(2)
    x = np.arange(num_iters)
    c = num_iters // 2
    a = alpha_multiplier / num_iters
    y = 1. / (1 + np.exp(-a * (x - c)))
    y = y / (y[-1] - y[0])
    if return_x:
        return x, y + (1.0 - y[-1]) + 1e-14
    return y + (1.0 - y[-1]) + 1e-14


def get_optimizer(*models, optim_type: str = 'Adam',
                  scheduler_type: Optional[str] = None, scheduler_kwargs: Optional[dict] = None,
                  **optim_args) -> Tuple[Optimizer, Optional[CyclicLR or ReduceLROnPlateau]]:
    """
    Get Adam optimizer for jointly training ${models} argument.
    :param models: one or more models to apply optimizer on
    :param (float) optim_type: type of optimizer to use (all of PyTorch's optimizers are supported)
    :param (str|None) scheduler_type: if set, then it is the type of LR Scheduler user
    :param (optional) scheduler_kwargs: scheduler kwargs as a dict object (NOT **KWARGS-LIKE PASSING)
    :param (dict) optim_args: optimizer class's arguments
    :return: a tuple containing an instance of `torch.optim.Adam` optimizer, the LR scheduler instance or None
    """
    # Initialize optimizer
    joint_params = []
    for model in models:
        joint_params += list(model.parameters())
    optim_class = getattr(torch.optim, optim_type)
    optim = optim_class(joint_params, **optim_args)
    # If no LR scheduler requested, return None as the 2nd parameter
    if scheduler_type is None:
        return optim, None
    # Initiate the LR Scheduler and return it as well
    lr_scheduler = get_optimizer_lr_scheduler(optimizer=optim, schedule_type=scheduler_type, **scheduler_kwargs)
    return optim, lr_scheduler


def get_optimizer_lr_scheduler(optimizer: Optimizer, schedule_type: str, **kwargs) \
        -> CyclicLR or ReduceLROnPlateau:
    """
    Set optimiser's learning rate scheduler based on $schedule_type$ string.
    :param optimizer: instance of torch.optim.Optimizer subclass
    :param schedule_type: learning-rate schedule type (supported: 'on_plateau', 'cyclic',)
    :param kwargs: scheduler-specific keyword arguments
    """
    switcher = {
        'on_plateau': ReduceLROnPlateau,
        'cyclic': CyclicLR,
    }
    return switcher[schedule_type](optimizer=optimizer, **kwargs)


def load_model_chkpt(model: nn.Module, model_name: str, step: Union[str, int] = 'latest',
                     model_opt: Optional[Optimizer] = None, dict_key: Optional[str] = None,
                     chkpts_root: Optional[str] = None, state_dict: Optional[dict] = None,
                     gdmc=None) -> Tuple[dict, Optional[int], Optional[int]]:
    """
    Load model (and model's optimizer) checkpoint. The checkpoint is searched in given checkpoints root (absolute path)
    and if one found it is loaded. The function also returns the checkpoint step as well as
    :param (nn.Module) model: the model as a torch.nn.Module instance
    :param (str) model_name: name of model which is also model checkpoint's file name prefix
    :param (str or int) step: step of checkpoint to load or 'latest' to load latest model checkpoint
    :param (optional) dict_key: name of the key of state dict regarding the model's state, or None if only one model'
                                state is saved in the state dict
    :param (optional) model_opt: model's optimizer instance
    :param (optional) chkpts_root: absolute path to model checkpoints directory
    :param (optional) state_dict: state dict (used to avoid duplicate calls)
    :param (optional) gdmc: utils.gdrive.GDriveModelCheckpoints object to interact with GoogleDrive API in order to
                            fetch model checkpoint
    :return: a tuple of the form (<state_dict>, <checkpoint_step>, <checkpoint_batch_size>)
    """
    # Check if running inside Colab or Kaggle (auto prefixing)
    chkpt_path = None
    if 'gdrive' == chkpts_root and gdmc:
        result, chkpt_path = gdmc.download_model_checkpoint(model_name=model_name, step=step, in_parallel=False)
    elif 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ:
        chkpts_root = f'/content/drive/MyDrive/Model Checkpoints'
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        chkpts_root = f'/kaggle/working/Model Checkpoints'
    elif not chkpts_root:
        chkpts_root: str = '/home/achariso/PycharmProjects/kth-ml-course-projects/biogans/.checkpoints'
    assert os.path.exists(chkpts_root) and os.path.isdir(chkpts_root), 'Checkpoints dir not existent or not readable'
    assert model_opt is None or dict_key is not None, 'model_opt and dict_key cannot be None simultaneously'

    chkpt_info_parts = ['None']
    if not state_dict:
        if not chkpt_path:
            # Find correct checkpoint path
            _, _, chkpt_files = next(os.walk(chkpts_root))
            chkpt_files = sorted([_f for _f in chkpt_files if _f.lower().startswith(model_name.lower())], reverse=True)
            assert len(chkpt_files) > 0, 'No model checkpoints found in given checkpoints dir'
            chkpt_file = chkpt_files[0]
            chkpt_info_parts = chkpt_file.replace(model_name, '').lstrip('_').replace('.pth', '').split('_')
            chkpt_path = os.path.join(chkpts_root, chkpt_file)

        # Load checkpoint
        state_dict = torch.load(chkpt_path, map_location='cpu')

    assert dict_key is None or dict_key in state_dict.keys(), f'dict_key={str(dict_key)} not found in state_dict.keys()'
    model.load_state_dict(state_dict[dict_key] if dict_key else state_dict)
    if model_opt:
        assert f'{dict_key}_opt' in state_dict.keys(), '$dict_key$_opt not found in state_dict.keys()'
        model_opt.load_state_dict(state_dict[f'{dict_key}_opt'])

    # Find epoch/current step
    chkpt_step = None
    chkpt_batch_size = None
    if len(chkpt_info_parts) == 2:
        chkpt_step = int(chkpt_info_parts[0])
        chkpt_batch_size = int(chkpt_info_parts[1])
    return state_dict, chkpt_step, chkpt_batch_size


def set_optimizer_lr(optimizer: Optimizer, new_lr: float) -> None:
    """
    Set optimiser's learning rate to match $new_lr$.
    :param optimizer: instance of torch.optim.Optimizer subclass
    :param new_lr: the new learning rate as a float
    """
    for group in optimizer.param_groups:
        group['lr'] = new_lr


def train_test_split(dataset: Union[Dataset, Sized], splits: list, seed: int = 42) \
        -> Tuple[Dataset or Sized, Dataset or Sized]:
    """
    Split :attr:`dataset` to training set and test set based on the percentages from :attr:`splits`.
    :param dataset: the dataset upon which to perform the split
    :param splits: the percentages of the split, (training_set, test_set) (e.g (90, 10) or (0.9, 0.1), both acceptable)
    :param seed: the manual seed parameter of the split
    :return: a tuple containing the two subsets as torch.utils.data.Dataset objects
    """
    # Get splits
    dataset_len = len(dataset)
    splits = np.array(splits, dtype=np.float32)
    if splits[0] > 1:
        splits *= 0.01
    splits *= dataset_len
    split_lengths = np.floor(splits).astype(np.int32)
    split_lengths[0] += dataset_len - split_lengths.sum()
    # Perform the split
    train_set, test_set = random_split(dataset, lengths=split_lengths, generator=torch.Generator().manual_seed(seed))
    return train_set, test_set


def weights_init_naive(module: nn.Module) -> None:
    """
    Apply naive weight initialization in given nn.Module. Should be called like network.apply(weights_init_naive).
    This naive approach simply sets all biases to 0 and all weights to the output of normal distribution with mean of 0
    and a std of 5e-2.
    :param module: input module
    """
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        torch.nn.init.normal_(module.weight, 0., 5.0e-2)
    if isinstance(module, nn.BatchNorm2d):
        torch.nn.init.normal_(module.weight, 0., 5.0e-2)
        torch.nn.init.constant_(module.bias, 0.)
