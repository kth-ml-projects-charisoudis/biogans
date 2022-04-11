import os.path
import pathlib
from typing import Optional, Sequence, Tuple, Union

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import wget
from PIL.Image import Image
from torch import nn, Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Compose

from modules.discriminators.dcgan import DCGanDiscriminator
from modules.generators.dcgan import DCGanGenerator
from modules.ifaces import IGanGModule
from utils.ifaces import FilesystemFolder
from utils.metrics import GanEvaluator
from utils.plot import create_img_grid, plot_grid
from utils.train import get_optimizer, set_optimizer_lr, weights_init_naive, get_optimizer_lr_scheduler

PROJECT_DIR_PATH = pathlib.Path(__file__).parent.parent.parent.resolve()


class OneClassBioGan(nn.Module, IGanGModule):
    """
    OneClassBioGan Class:
    This class is used to access and use the entire 1-class BioGAN model (implemented according to the paper "GANs for
    Biological Image Synthesis") as a `nn.Module` instance but with additional functionality provided from inheriting
    `utils.gdrive.GDriveModel` (through the `utils.ifaces.IGanGModule` interface). Inheriting GDriveModel enables easy
    download / upload of model checkpoints to Google Drive using GoogleDrive API's python client and PyDrive.
    """

    # This is the latest model configuration that lead to SOTA results
    DefaultConfiguration = {
        'gen': {
            'z_dim': 100,
            'norm_type': 'batch',
            'c_hidden': 512,
            'n_extra_layers': 0,
            'red_portion': None
        },
        'gen_opt': {
            'optim_type': 'Adam',
            'optim_kwargs': {
                'lr': 0.0001,
                'betas': (0.0, 0.9),
            },
            'scheduler_type': None
        },
        'disc': {
            'c_hidden': 64,
            'n_contracting_blocks': 4,
            'use_spectral_norm': False,
            'adv_criterion': 'Wasserstein',
            'output_kernel_size': (3, 5),
            'gp_lambda': 10.0,
        },
        'disc_opt': {
            'optim_type': 'Adam',
            'optim_kwargs': {
                'lr': 0.0001,
                'betas': (0.0, 0.9),
            },
            'scheduler_type': None
        },
        'n_disc_iters': 5
    }

    PROTEIN_CLASS = 'Alp14'
    PROTEIN_CLASS_INDICES = {
        'Alp14': 0,
        'Arp3': 1,
        'Cki2': 2,
        'Mkh1': 3,
        'Sid2': 4,
        'Tea1': 5,
    }
    IS_SEPARABLE = False

    @classmethod
    def version(cls) -> str:
        sep_suffix = '_sep' if cls.IS_SEPARABLE else ''
        return f'{cls.PROTEIN_CLASS}{sep_suffix}'

    def __init__(self, model_fs_folder_or_root: FilesystemFolder, config_id: Optional[str] = None,
                 chkpt_epoch: Optional[int or str] = None, chkpt_step: Optional[int or str] = None,
                 device: torch.device or str = 'cpu', gen_transforms: Optional[Compose] = None, log_level: str = 'info',
                 dataset_len: Optional[int] = None, reproducible_indices: Sequence = (0, -1),
                 evaluator: Optional[GanEvaluator] = None, **evaluator_kwargs):
        """
        PGPG class constructor.
        :param (FilesystemFolder) model_fs_folder_or_root: a `utils.gdrive.GDriveFolder` object to download /
                                                           upload model checkpoints and metrics from / to local or
                                                           remote (Google Drive) filesystem
        :param (str or None) config_id: if not `None` then the model configuration matching the given identifier will be
                                        used to initialize the model
        :param (int or str or None) chkpt_epoch: if not `None` then the model checkpoint at the given :attr:`step` will
                                                 be loaded via `nn.Module().load_state_dict()`
        :param (int or str or None) chkpt_step: if not `None` then the model checkpoint at the given :attr:`step` and at
                                         the given :attr:`batch_size` will be loaded via `nn.Module().load_state_dict()`
                                         call. If you have aosokin chkpt path, pass it here as "aosokin:<path>".
        :param (str) device: the device used for training (supported: "cuda", "cuda:<GPU_INDEX>", "cpu")
        :param (Compose) gen_transforms: the image transforms of the dataset the generator is trained on (used in
                                         visualization)
        :param (optional) dataset_len: number of images in the dataset used to train the generator or None to fetched
                                       from the :attr:`evaluator` dataset property (used for epoch tracking)
        :param (Sequence) reproducible_indices: dataset indices to be fetched and visualized each time
                                                `PixelDTGan::visualize(reproducible=True)` is called
        :param (optional) evaluator: GanEvaluator instance of None to not evaluate models when taking snapshots
        :param evaluator_kwargs: if :attr:`evaluator` is `None` these arguments must be present to initialize a new
                                 `utils.metrics.GanEvaluator` instance
        """
        self.device = device
        self.__class__.IS_SEPARABLE = config_id is not None and 'sep' in config_id
        # Initialize interface
        IGanGModule.__init__(self, model_fs_folder_or_root, config_id, device=device, log_level=log_level,
                             dataset_len=dataset_len, reproducible_indices=reproducible_indices,
                             evaluator=evaluator, **evaluator_kwargs)
        # Instantiate torch.nn.Module class
        nn.Module.__init__(self)

        # Define BioGAN model
        #   - generator
        gen_conf = self._configuration['gen']
        gen_conf['c_out'] = 2
        self.gen = DCGanGenerator(**gen_conf)
        #   - discriminator
        disc_conf = self._configuration['disc']
        disc_conf['c_in'] = 2
        self.disc = DCGanDiscriminator(**disc_conf)

        # Define optimizers
        gen_opt_conf = self._configuration['gen_opt']
        self.gen_opt, _ = get_optimizer(self.gen, **gen_opt_conf)
        disc_opt_conf = self._configuration['disc_opt']
        self.disc_opt, _ = get_optimizer(self.disc, **disc_opt_conf)

        # Move models to {C,G,T}PU
        self.gen.to(device)
        self.disc.to(device)

        # Load checkpoint from Google Drive
        self.other_state_dicts = {}
        if chkpt_step is not None and chkpt_step.startswith('aosokin') or \
                chkpt_epoch is not None and chkpt_epoch.startswith('aosokin'):
            # load aosokin checkpoint in generator
            _, aosokin_path = chkpt_step.split(':') if chkpt_step is not None else chkpt_epoch.split(':')
            if os.path.basename(aosokin_path) == 'auto':
                aosokin_path = aosokin_path.replace(
                    '/auto',
                    f'/size-48-80_6class_{config_id.replace("wgan-gp", "wgangp")}-adam/netG_iter_50000.pth'
                )
                if not os.path.exists(aosokin_path):
                    # try downloading file
                    p = pathlib.Path(aosokin_path)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    wget.download(
                        f'http://www.di.ens.fr/sierra/research/biogans/models/{p.relative_to(p.parent.parent)}',
                        out=str(p.parent.absolute()))
            self.protein_class_index = self.__class__.PROTEIN_CLASS_INDICES[self.__class__.PROTEIN_CLASS]
            self.logger.debug(f'Loading AOSOKIN checkpoint (class={self.protein_class_index}): {aosokin_path}')
            self.gen.load_aosokin_state_dict(state_dict=torch.load(aosokin_path, map_location='cpu'),
                                             class_idx=self.protein_class_index)
            # TODO: what should be loaded at the Discriminator after this?
            chkpt_epoch = 3200
        elif chkpt_epoch is not None:
            try:
                chkpt_filepath = self.fetch_checkpoint(epoch_or_id=chkpt_epoch, step=chkpt_step)
                self.logger.debug(f'Loading checkpoint file: {chkpt_filepath}')
                _state_dict = torch.load(chkpt_filepath, map_location='cpu')
                self.load_state_dict(_state_dict)
                if 'gforward' in _state_dict.keys():
                    self.load_gforward_state(_state_dict['gforward'])
            except FileNotFoundError as e:
                self.logger.critical(str(e))
                chkpt_epoch = None
        if not chkpt_epoch:
            # Initialize weights with small values
            self.gen = self.gen.apply(weights_init_naive)
            self.disc = self.disc.apply(weights_init_naive)
            chkpt_epoch = 0

        # Define LR schedulers (after optimizer checkpoints have been loaded)
        if gen_opt_conf['scheduler_type']:
            if gen_opt_conf['scheduler_type'] == 'cyclic':
                self.gen_opt_lr_scheduler = get_optimizer_lr_scheduler(
                    self.gen_opt, schedule_type=str(gen_opt_conf['scheduler_type']), base_lr=0.1 * gen_opt_conf['lr'],
                    max_lr=gen_opt_conf['lr'], step_size_up=2 * dataset_len if evaluator else 1000, mode='exp_range',
                    gamma=0.9, cycle_momentum=False,
                    last_epoch=chkpt_epoch if 'initial_lr' in self.gen_opt.param_groups[0].keys() else -1)
            else:
                self.gen_opt_lr_scheduler = get_optimizer_lr_scheduler(self.gen_opt,
                                                                       schedule_type=gen_opt_conf['scheduler_type'])
        else:
            self.gen_opt_lr_scheduler = None
        if disc_opt_conf['scheduler_type']:
            self.disc_opt_lr_scheduler = get_optimizer_lr_scheduler(self.disc_opt,
                                                                    schedule_type=disc_opt_conf['scheduler_type'])
        else:
            self.disc_opt_lr_scheduler = None

        self.gen_opt_lr_scheduler = None
        self.disc_opt_lr_scheduler = None

        # Save transforms for visualizer
        if gen_transforms is not None:
            self.gen_transforms = gen_transforms

        # Initialize params
        self.g_out = None
        self.x = None
        self.device = device
        self.n_disc_iters = self._configuration.get('n_disc_iters', 1)

    def load_configuration(self, configuration: dict) -> None:
        IGanGModule.load_configuration(self, configuration)

    #
    # ------------
    # nn.Module
    # -----------
    #

    ##################################
    ## Checkpointing
    #################################

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """
        This method overrides parent method of `nn.Module` and is used to apply checkpoint dict to model.
        :param state_dict: see `nn.Module.load_state_dict()`
        :param strict: see `nn.Module.load_state_dict()`
        :return: see `nn.Module.load_state_dict()`
        """
        # Check if checkpoint is for different config
        if 'config_id' in state_dict.keys() and state_dict['config_id'] != self.config_id:
            self.logger.critical(f'Config IDs mismatch (self: "{self.config_id}", state_dict: '
                                 f'"{state_dict["config_id"]}"). NOT applying checkpoint.')
            if not click.confirm('Override config_id in checkpoint and attempt to load it?', default=False):
                return
        # Load model checkpoints
        # noinspection PyTypeChecker
        self.gen.load_state_dict(state_dict['gen'])
        self.gen_opt.load_state_dict(state_dict['gen_opt'])
        self.disc.load_state_dict(state_dict['disc'])
        self.disc_opt.load_state_dict(state_dict['disc_opt'])
        self._nparams = state_dict['nparams']
        # Update the latest metrics with checkpoint's metrics
        if 'metrics' in state_dict.keys():
            self.latest_metrics = state_dict['metrics']
        self.logger.debug(f'State dict loaded. Keys: {tuple(state_dict.keys())}')
        for _k in [_k for _k in state_dict.keys() if _k not in ('gen', 'gen_opt', 'disc', 'disc_opt', 'configuration')]:
            self.other_state_dicts[_k] = state_dict[_k]

        # def reset_bn(_m):
        #     if isinstance(_m, nn.BatchNorm2d):
        #         _m.reset_running_stats()
        #
        # # Reset BN stats
        # self.gen.apply(reset_bn)
        # self.disc.apply(reset_bn)
        # self.gen_opt.state = collections.defaultdict(dict)
        # self.disc_opt.state = collections.defaultdict(dict)

    def state_dict(self, *args, **kwargs) -> dict:
        """
        In this method we define the state dict, i.e. the model checkpoint that will be saved to the .pth file.
        :param args: see `nn.Module.state_dict()`
        :param kwargs: see `nn.Module.state_dict()`
        :return: see `nn.Module.state_dict()`
        """
        mean_gen_loss = np.mean(self.gen_losses)
        self.gen_losses.clear()
        mean_disc_loss = np.mean(self.disc_losses)
        self.disc_losses.clear()
        return {
            'gen': self.gen.state_dict(),
            'gen_loss': mean_gen_loss,
            'gen_opt': self.gen_opt.state_dict(),
            'disc': self.disc.state_dict(),
            'disc_loss': mean_disc_loss,
            'disc_opt': self.disc_opt.state_dict(),
            'nparams': self.nparams,
            'nparams_hr': self.nparams_hr,
            'config_id': self.config_id,
            'configuration': self._configuration,
        }

    ##################################
    ## Training
    #################################

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        This method implements the forward pass through Inception v3 network.
        :param (Tensor) x: the batch of input images as a `torch.Tensor` object
        :return: a tuple of `torch.Tensor` objects containing (disc_loss, gen_loss)
        """
        # Update gdrive model state
        batch_size = x.shape[0]
        self.gforward(batch_size)

        ##########################################
        ########   Update Discriminator   ########
        ##########################################
        with self.gen.frozen():
            for di in range(self.n_disc_iters):
                self.disc_opt.zero_grad()  # Zero out discriminator gradient (before backprop)
                z = self.gen.get_random_z(batch_size=batch_size, device=x.device)
                g_out = self.gen(z)
                disc_loss = self.disc.get_loss_both(real=x, fake=g_out.detach())
                disc_loss.backward(retain_graph=True)  # Update discriminator gradients
                self.disc_opt.step()  # Update discriminator weights
                # Update LR (if needed)
                if self.disc_opt_lr_scheduler:
                    if isinstance(self.disc_opt_lr_scheduler, ReduceLROnPlateau):
                        self.disc_opt_lr_scheduler.step(metrics=disc_loss)
                    else:
                        self.disc_opt_lr_scheduler.step()

        ##########################################
        ########     Update Generator     ########
        ##########################################
        with self.disc.frozen():
            self.gen_opt.zero_grad()
            z = self.gen.get_random_z(batch_size=batch_size, device=x.device)
            g_out = self.gen(z)
            gen_loss = self.disc.get_loss(x=g_out, is_real=True)
            gen_loss.backward()  # Update generator gradients
            self.gen_opt.step()  # Update generator weights
            # Update LR (if needed)
            if self.gen_opt_lr_scheduler:
                if isinstance(self.gen_opt_lr_scheduler, ReduceLROnPlateau):
                    self.gen_opt_lr_scheduler.step(metrics=gen_loss)
                else:
                    self.gen_opt_lr_scheduler.step()

        # Save for visualization
        self.g_out = g_out[::len(g_out) - 1].detach().cpu()
        self.x = x[::len(x) - 1].detach().cpu()
        self.gen_losses.append(gen_loss.item())
        self.disc_losses.append(disc_loss.item())

        return disc_loss, gen_loss

    def update_lr(self, gen_new_lr: Optional[float] = None, disc_new_lr: Optional[float] = None) -> None:
        """
        Updates learning-rate of model optimizers, for the non-None give arguments.
        :param (float|None) gen_new_lr: new LR for generator's optimizer (or None to leave as is)
        :param (float|None) disc_new_lr: new LR for real/fake discriminator's optimizer (or None to leave as is)
        """
        if gen_new_lr:
            set_optimizer_lr(self.gen_opt, new_lr=gen_new_lr)
        if disc_new_lr:
            set_optimizer_lr(self.disc_opt, new_lr=disc_new_lr)

    #
    # --------------
    # Visualizable
    # -------------
    #

    # noinspection SpellCheckingInspection
    def visualize_indices(self, indices: Union[int, tuple, Sequence]) -> Image:
        # # Fetch images
        # assert hasattr(self, 'evaluator') and hasattr(self.evaluator, 'dataset'), 'Could not find dataset from model'
        # images = []
        # with self.gen.frozen():
        #     for index in indices:
        #         x = self.evaluator.dataset[index]
        #         g_out = self.gen(x.unsqueeze(0)).squeeze(0)
        #         images.extend([x, g_out])
        # # Convert to grid of images
        # ncols = 2
        # nrows = int(len(images) / ncols)
        # grid = create_img_grid(images=torch.stack(images), nrows=nrows, ncols=ncols,
        #                        gen_transforms=self.gen_transforms)
        # # Plot
        # return plot_grid(grid=grid, figsize=(ncols, nrows),
        #                  footnote_l=f'epoch={str(self.epoch).zfill(3)} | step={str(self.step).zfill(10)} | '
        #                             f'indices={indices}',
        #                  footnote_r=f'gen_loss={"{0:0.3f}".format(round(np.mean(self.gen_losses).item(), 3))}, '
        #                             f'disc_loss={"{0:0.3f}".format(round(np.mean(self.disc_losses).item(), 3))}')
        raise NotImplementedError('Cannot really implement reproducibility in Noise-to-Image context.')

    def visualize(self, reproducible: bool = False, dl=None) -> Image:
        # Get first & last sample from saved images in self
        if self.x is None or self.g_out is None:
            assert dl is not None
            self.x = next(iter(dl)).detach().cpu()
            with torch.no_grad():
                self.g_out = self.gen(self.gen.get_random_z(batch_size=self.x.shape[0], device=self.device)).cpu()
        x_0 = self.x[0]
        g_out_0 = self.g_out[0]
        g_out__1 = self.g_out[-1]

        # Concat images to a 2x5 grid (each row is a separate generation, the columns contain real and generated images
        # side-by-side)
        grid = create_img_grid(images=torch.stack([
            x_0, g_out_0, g_out__1
        ]), gen_transforms=self.gen_transforms)

        # Plot
        return plot_grid(grid=grid, figsize=(3, 2),
                         footnote_l=f'epoch={str(self.epoch).zfill(3)} | step={str(self.step).zfill(10)}',
                         footnote_r=f'gen_loss={"{0:0.3f}".format(round(np.mean(self.gen_losses).item(), 3))}, '
                                    f'disc_loss={"{0:0.3f}".format(round(np.mean(self.disc_losses).item(), 3))}')


if __name__ == '__main__':
    from datasets.lin import LINDataloader
    from utils.filesystems.local import LocalFolder, LocalFilesystem, LocalCapsule

    # Get GoogleDrive root folder
    _local_gdrive_root = '/home/achariso/PycharmProjects/kth-ml-course-projects/biogans/.gdrive_personal'
    _log_level = 'debug'

    # Via locally-mounted Google Drive (when running from inside Google Colaboratory)
    _fs = LocalFilesystem(LocalCapsule(_local_gdrive_root))
    _groot = LocalFolder.root(capsule_or_fs=_fs)

    # Define folder roots
    _models_groot = _groot.subfolder_by_name('Models')
    _datasets_groot = _groot.subfolder_by_name('Datasets')

    exec_device = 'cpu'
    CLASS = 'Alp14'  # 'Alp14',  'Arp3', 'Cki2', 'Tea1'

    ###################################
    ###   Dataset Initialization    ###
    ###################################
    #   - the dataloader used to access the training dataset of cross-scale/pose image pairs at every epoch
    #     > len(dataloader) = <number of batches>
    #     > len(dataloader.dataset) = <number of total dataset items>
    dataloader = LINDataloader(dataset_fs_folder_or_root=_datasets_groot, train_not_test=True,
                               batch_size=2, which_classes=CLASS)
    dataset = dataloader.dataset

    ###################################
    ###    Models Initialization    ###
    ###################################
    #   - initialize evaluator instance (used to run GAN evaluation metrics: FID, IS, PRECISION, RECALL, F1 and SSIM)
    evaluator = GanEvaluator(model_fs_folder_or_root=_models_groot, gen_dataset=dataset, target_index=1,
                             device=exec_device, condition_indices=(0, 2), n_samples=2, batch_size=2, f1_k=2)
    #   - initialize model
    _chkpt_step = f'aosokin:{PROJECT_DIR_PATH}/aosokin_checkpoints/auto'
    # chkpt_step = None
    # try:
    #     if chkpt_step == 'latest':
    #         _chkpt_step = chkpt_step
    #     elif isinstance(chkpt_step, str) and chkpt_step.isdigit():
    #         _chkpt_step = int(chkpt_step)
    #     else:
    #         _chkpt_step = None
    # except NameError:
    #     _chkpt_step = None
    OneClassBioGan.PROTEIN_CLASS = CLASS
    biogan = OneClassBioGan(model_fs_folder_or_root=_models_groot, config_id='wgan-gp-independent-sep',
                            dataset_len=len(dataset), chkpt_epoch=_chkpt_step, evaluator=evaluator, device=exec_device,
                            log_level='debug', gen_transforms=dataloader.transforms)
    # print(biogan.gen)
    biogan.logger.debug(f'Using device: {str(exec_device)}')
    biogan.logger.debug(f'Model initialized. Number of params = {biogan.nparams_hr}')

    # Test visualization
    _disc_loss, _gen_loss = biogan(next(iter(dataloader)))
    print('disc_loss', _disc_loss, 'gen_loss', _gen_loss)
    biogan.visualize()
    plt.show()
