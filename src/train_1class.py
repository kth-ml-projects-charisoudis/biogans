# Get GoogleDrive root folder
import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.lin import LINDataloader
from modules.biogan import OneClassBioGan
from utils.filesystems.local import LocalFilesystem, LocalFolder, LocalCapsule
from utils.metrics import GanEvaluator

local_gdrive_root = os.environ.get('GDRIVE_ROOT')
log_level = os.environ.get('LOG_LEVEL')

# Via locally-mounted Google Drive (when running from inside Google Colaboratory)
fs = LocalFilesystem(LocalCapsule(local_gdrive_root))
groot = LocalFolder.root(capsule_or_fs=fs)

# Define folder roots
models_groot = groot.subfolder_by_name('Models')
datasets_groot = groot.subfolder_by_name('Datasets')

##########################################
###         Parse CLI Arguments        ###
##########################################
parser = argparse.ArgumentParser(description='Trains GAN model in PyTorch.')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                    help='execution device (\'cpu\', or \'cuda\')')
parser.add_argument('--chkpt_step', type=str, default='latest',
                    help='model checkpoint to be loaded (\'latest\' or str or int)')
parser.add_argument('--seed', type=int, default=42,
                    help='random generators seed value (default: 42)')
parser.add_argument('-use_refresh_token', action='store_true',
                    help='if set will use client_secrets.json to connect to Google Drive, else will ask for auth code')
parser.add_argument('--which_classes', type=str, default='Alp14',
                    help='Training Classes (e.g. for training 1-class GAN models)')
args = parser.parse_args()

###################################
###  Hyper-parameters settings  ###
###################################
#   - training
n_epochs = 1000
batch_size = 48
#   - evaluation
metrics_n_samples = 1000
metrics_batch_size = 32
f1_k = 3
#   - visualizations / checkpoints steps
display_step = 300
checkpoint_step = 600
metrics_step = 1800  # evaluate model every 3 checkpoints

###################################
###   Dataset Initialization    ###
###################################
#   - the dataloader used to access the training dataset of cross-scale/pose image pairs at every epoch
#     > len(dataloader) = <number of batches>
#     > len(dataloader.dataset) = <number of total dataset items>
dataloader = LINDataloader(dataset_fs_folder_or_root=datasets_groot, train_not_test=True,
                           batch_size=batch_size, pin_memory=True, shuffle=True,
                           which_classes=args.which_classes)
dataset = dataloader.dataset
#   - apply rudimentary tests
assert issubclass(dataloader.__class__, DataLoader)
assert len(dataloader) == len(dataset) // batch_size + (1 if len(dataset) % batch_size else 0)
_x = next(iter(dataloader))
assert tuple(_x.shape) == (batch_size, 2, 48, 80)

###################################
###    Models Initialization    ###
###################################
#   - initialize evaluator instance (used to run GAN evaluation metrics: FID, IS, PRECISION, RECALL, F1 and SSIM)
evaluator = GanEvaluator(model_fs_folder_or_root=models_groot, gen_dataset=dataset, device=args.device,
                         z_dim=-1, n_samples=metrics_n_samples, batch_size=metrics_batch_size, f1_k=f1_k,
                         ssim_c_img=2)
#   - initialize model
chkpt_step = args.chkpt_step
try:
    if chkpt_step == 'latest':
        _chkpt_step = chkpt_step
    elif isinstance(chkpt_step, str) and chkpt_step.isdigit():
        _chkpt_step = int(chkpt_step)
    else:
        _chkpt_step = None
except NameError:
    _chkpt_step = None
OneClassBioGan.WhichClass = args.which_classes
biogan = OneClassBioGan(model_fs_folder_or_root=models_groot, config_id='default', dataset_len=len(dataset),
                        chkpt_epoch=_chkpt_step, evaluator=evaluator, device=args.device, log_level=log_level)
biogan.logger.debug(f'Using device: {str(args.device)}')
biogan.logger.debug(f'Model initialized. Number of params = {biogan.nparams_hr}')
# #   - load dataloader state (from model checkpoint)
# if 'dataloader' in biogan.other_state_dicts.keys():
#     dataloader.set_state(biogan.other_state_dicts['dataloader'])
#     biogan.logger.debug(f'Loaded dataloader state! Current pem_index={dataloader.get_state()["perm_index"]}')

torch.cuda.empty_cache()

###################################
###       Training Loop         ###
###################################
#   - start training loop from last checkpoint's epoch and step
gcapture_ready = True
async_results = None
biogan.logger.info(f'[training loop] STARTING (epoch={biogan.epoch}, step={biogan.initial_step})')
biogan.initial_step = 1
biogan._counter = 0
for epoch in range(biogan.epoch, n_epochs):
    image_1: Tensor
    image_2: Tensor
    pose_2: Tensor

    # noinspection PyProtectedMember
    d = {
        'step': biogan.step,
        'initial_step': biogan.initial_step,
        'epoch': biogan.epoch,
        '_counter': biogan._counter,
        'epoch_inc': biogan.epoch_inc,
    }
    # initial_step = biogan.initial_step % len(dataloader)
    biogan.logger.debug('[START OF EPOCH] ' + str(d))
    for x in tqdm(dataloader):
        # Transfer image batches to GPU
        x = x.to(args.device)

        # Perform a forward + backward pass + weight update on the Generator & Discriminator models
        disc_loss, gen_loss = biogan(x)

        # Metrics & Checkpoint Code
        if biogan.step % checkpoint_step == 0:
            # Check if another upload is pending
            if not gcapture_ready and async_results:
                # Wait for previous upload to finish
                biogan.logger.warning('Waiting for previous gcapture() to finish...')
                [r.wait() for r in async_results]
                biogan.logger.warning('DONE! Starting new capture now.')
            # Capture current model state, including metrics and visualizations
            async_results = biogan.gcapture(checkpoint=True, metrics=biogan.step % metrics_step == 0,
                                            visualizations=True,
                                            dataloader=dataloader, in_parallel=True, show_progress=True,
                                            delete_after=False)
        # Visualization code
        elif biogan.step % display_step == 0:
            visualization_img = biogan.visualize()
            visualization_img.show()

        # Check if a pending checkpoint upload has finished
        if async_results:
            gcapture_ready = all([r.ready() for r in async_results])
            if gcapture_ready:
                biogan.logger.info(f'gcapture() finished')
                if biogan.latest_checkpoint_had_metrics:
                    biogan.logger.info(str(biogan.latest_metrics))
                async_results = None

    # biogan.visualize()
    # plt.show()

    # noinspection PyProtectedMember
    d = {
        'step': biogan.step,
        'initial_step': biogan.initial_step,
        'epoch': biogan.epoch,
        '_counter': biogan._counter,
        'epoch_inc': biogan.epoch_inc,
    }
    biogan.logger.debug('[END OF EPOCH] ' + str(d))

# Check if a pending checkpoint exists
if async_results:
    ([r.wait() for r in async_results])
    biogan.logger.info(f'last gcapture() finished')
    if biogan.latest_checkpoint_had_metrics:
        biogan.logger.info(str(biogan.latest_metrics))
    async_results = None

# Training finished!
biogan.logger.info('[training loop] DONE')
