import os.path
import pathlib
from typing import Optional, Sequence, Tuple, Union

import click
import numpy as np
import torch
from PIL.Image import Image
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Compose

from modules.discriminators.dcgan import DCGanDiscriminatorInd6Class
from modules.generators.dcgan_star_shaped import DCGanGeneratorStarShaped
from modules.ifaces import IGanGModule
from utils.ifaces import FilesystemFolder
from utils.metrics import GanEvaluator, GanEvaluator6Class
from utils.plot import plot_grid, create_img_grid_6class, create_img_grid_6class_joint
from utils.pytorch import invert_transforms, ToTensorOrPass
from utils.train import get_optimizer, weights_init_naive

PROJECT_DIR_PATH = pathlib.Path(__file__).parent.parent.parent.resolve()


class BioGanStarShaped(nn.Module, IGanGModule):
    """
    BioGanStarShaped Class:
    Implements the "Star-Shaped" Biogan model.
    """

    def __init__(self, model_fs_folder_or_root: FilesystemFolder, config_id: Optional[str] = None,
                 chkpt_epoch: Optional[int or str] = None, chkpt_step: Optional[int or str] = None,
                 device: torch.device or str = 'cpu', gen_transforms: Optional[Compose] = None, log_level: str = 'info',
                 dataset_len: Optional[int] = None, reproducible_indices: Sequence = (0, -1),
                 evaluator: Optional[GanEvaluator] = None, **evaluator_kwargs):

        self.device = device

        # Initialize interface
        IGanGModule.__init__(self, model_fs_folder_or_root, config_id, device=device, log_level=log_level,
                             dataset_len=dataset_len, reproducible_indices=reproducible_indices,
                             evaluator=evaluator, **evaluator_kwargs)
        # Instantiate torch.nn.Module class
        nn.Module.__init__(self)

        # Define BioGAN model
        #   - generators
        gen_conf = self._configuration['gen']
        gen_conf['c_out'] = 2
        gen = DCGanGeneratorStarShaped(**gen_conf)
        #   - discriminators
        disc_conf = self._configuration['disc']
        disc_conf['c_in'] = 2
        disc = DCGanDiscriminatorInd6Class(**disc_conf)

        # Move models to {C,G,T}PU
        self.gen = gen.to(device)
        self.disc = disc.to(device)

        # Define optimizers
        gen_opt_conf = self._configuration['gen_opt']
        self.gen_opt, _ = get_optimizer(self.gen, **gen_opt_conf)
        disc_opt_conf = self._configuration['disc_opt']
        self.disc_opt, _ = get_optimizer(self.disc, **disc_opt_conf)

        # Load checkpoint from Google Drive
        self.other_state_dicts = {}
        # if chkpt_step is not None and type(chkpt_step) == str and chkpt_step.startswith('aosokin') or \
        #         (chkpt_epoch is not None and type(chkpt_step) == str and chkpt_epoch.startswith('aosokin')):
        #     # load aosokin checkpoint in generator
        #     _, aosokin_path = chkpt_step.split(':') if chkpt_step is not None else chkpt_epoch.split(':')
        #     if os.path.basename(aosokin_path) == 'auto':
        #         aosokin_path = aosokin_path.replace(
        #             '/auto',
        #             f'/size-48-80_6class_{config_id.replace("wgan-gp", "wgangp")}-adam/netG_iter_50000.pth'
        #         )
        #         if not os.path.exists(aosokin_path):
        #             # try downloading file
        #             p = pathlib.Path(aosokin_path)
        #             p.parent.mkdir(parents=True, exist_ok=True)
        #             wget.download(
        #                 f'http://www.di.ens.fr/sierra/research/biogans/models/{p.relative_to(p.parent.parent)}',
        #                 out=str(p.parent.absolute()))
        #     self.logger.debug(f'Loading AOSOKIN checkpoint: {aosokin_path}')
        #     self.gen.load_aosokin_state_dict(torch.load(aosokin_path, map_location='cpu'))
        #     # TODO: what should be loaded at the Discriminator after this?
        #     chkpt_epoch = 3200
        if chkpt_epoch is not None:
            try:
                # chkpt_filepath = self.fetch_checkpoint(epoch_or_id=chkpt_epoch, step=chkpt_step)
                chkpt_filepath = '/home/achariso/PycharmProjects/kth-ml-course-projects/biogans/.gdrive_personal/Models/model_name=bioganjoint6class_sep/Checkpoints/epoch=1703/0000253800.pth'
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

        # No LR schedulers
        self.gen_opt_lr_scheduler = None
        self.disc_opt_lr_scheduler = None

        # Save transforms for visualizer
        if gen_transforms is not None:
            self.gen_transforms = gen_transforms
            self.inv_transforms = invert_transforms(self.gen_transforms)
        else:
            self.inv_transforms = ToTensorOrPass()

            # Initialize params
        self.g_out = None
        self.x = None
        self.device = device
        self.n_disc_iters = self._configuration.get('n_disc_iters', 1)
        self.n_disc_iters_i = 0

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

    def generate_fixed_noise(self, batch_size: int = 64):
        # Fixed noise is stored in "<MODEL_GROOT>/Metrics/fixed_noise.pth"
        fixed_noise_gfile = self.metrics_gfolder.file_by_name('fixed_noise.pth')
        if fixed_noise_gfile is not None:
            assert fixed_noise_gfile.download(in_parallel=False, show_progress=True), 'Error fetching fixed noise file'
            fixed_noise = torch.load(fixed_noise_gfile.path)
        else:
            fixed_noise = self.gen.get_random_z(batch_size)
            torch.save(fixed_noise.cpu(), os.path.join(self.metrics_gfolder.local_root, 'fixed_noise.pth'))
            self.metrics_gfolder.upload_file(local_filename='fixed_noise.pth', in_parallel=False, show_progress=True)
        return Variable(fixed_noise, requires_grad=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Update gdrive model state
        batch_size = x.shape[1]
        self.gforward(batch_size)

        ##########################################
        ########   Update Discriminator   ########
        ##########################################
        self.disc_opt.zero_grad()  # Zero out discriminator gradient (before backprop)
        z = self.gen.get_random_z(batch_size=batch_size, device=x.device)
        g_out = self.gen(z)
        disc_loss = self.disc.get_loss_both(real=x.clone(), fake=g_out.detach()).mean()
        disc_loss.backward()  # Update discriminator gradients
        self.disc_opt.step()  # Update discriminator weights
        # Update LR (if needed)
        if self.disc_opt_lr_scheduler:
            if isinstance(self.disc_opt_lr_scheduler, ReduceLROnPlateau):
                self.disc_opt_lr_scheduler.step(metrics=disc_loss)
            else:
                self.disc_opt_lr_scheduler.step()

        self.n_disc_iters_i += 1
        if self.n_disc_iters_i == self.n_disc_iters:
            self.n_disc_iters_i = 0

            ##########################################
            ########     Update Generator     ########
            ##########################################
            with self.disc.frozen():
                self.gen_opt.zero_grad()
                z = self.gen.get_random_z(batch_size=batch_size, device=x.device)
                g_out = self.gen(z)
                gen_loss = self.disc.get_loss(x=g_out, is_real=True).mean()
                gen_loss.backward()  # Update generator gradients
                self.gen_opt.step()  # Update generator weights
                # Update LR (if needed)
                if self.gen_opt_lr_scheduler:
                    if isinstance(self.gen_opt_lr_scheduler, ReduceLROnPlateau):
                        self.gen_opt_lr_scheduler.step(metrics=gen_loss)
                    else:
                        self.gen_opt_lr_scheduler.step()

            # Save for visualization
            self.g_out = g_out[:, ::g_out.shape[1] - 1].detach().cpu()
            self.x = x[:, ::g_out.shape[1] - 1].detach().cpu()
            self.gen_losses.append(gen_loss.item())
            self.disc_losses.append(disc_loss.item())
        else:
            gen_loss = None
        return disc_loss, gen_loss

    #
    # --------------
    # Visualizable
    # -------------
    #

    # noinspection SpellCheckingInspection
    def visualize_indices(self, indices: Union[int, tuple, Sequence]) -> Image:
        raise NotImplementedError('Cannot really implement reproducibility in Noise-to-Image context.')

    def visualize2(self, reproducible: bool = False, dl=None, save_path=None) -> Image:
        # Get first & last sample from saved images in self
        if self.x is None or self.g_out is None:
            assert dl is not None
            self.x = next(iter(dl)).detach().cpu()
            with torch.no_grad():
                self.g_out = self.gen(self.gen.get_random_z(batch_size=self.x.shape[0], device=self.device)).cpu()
        x_0 = self.x[:, 0, :, :, :]
        g_out_0 = self.g_out[:, 0]
        g_out__1 = self.g_out[:, -1]

        # 1st row
        x_0 = [self.inv_transforms(x_0[class_idx]) for class_idx in range(6)]
        g_out_0 = [self.inv_transforms(g_out_0[class_idx]) for class_idx in range(6)]
        g_out__1 = [self.inv_transforms(g_out__1[class_idx]) for class_idx in range(6)]

        # Concat images to a 3x6 grid (each row is a separate generation, the columns contain real and generated images
        # side-by-side)
        grid = create_img_grid_6class(x_0, g_out_0, g_out__1)

        # Plot
        return plot_grid(grid=grid, figsize=(14, 2),
                         footnote_l=f'epoch={str(self.epoch).zfill(3)} | step={str(self.step).zfill(10)}',
                         footnote_r=f'gen_loss={"{0:0.3f}".format(round(np.mean(self.gen_losses).item(), 3))}, '
                                    f'disc_loss={"{0:0.3f}".format(round(np.mean(self.disc_losses).item(), 3))}',
                         save_path=save_path)

    def visualize(self, reproducible: bool = False, dl=None, save_path=None) -> Image:
        # Get first & last sample from saved images in self
        if self.x is None or self.g_out is None:
            assert dl is not None
            self.x = next(iter(dl)).detach().cpu()
            with torch.no_grad():
                self.g_out = self.gen(self.gen.get_random_z(batch_size=self.x.shape[0], device=self.device)).cpu()

        x_0 = self.inv_transforms(self.x[0])
        g_out_0 = self.inv_transforms(self.g_out[0])
        x__1 = self.inv_transforms(self.x[-1])
        g_out__1 = self.inv_transforms(self.g_out[-1])

        # Concat images to a 3x7 grid (first row has real sample, next 2 are 2 separate generations)
        grid = create_img_grid_6class_joint(x_0, g_out_0, g_out__1)

        # Plot
        return plot_grid(grid=grid, figsize=(10, 2),
                         footnote_l=f'epoch={str(self.epoch).zfill(3)} | step={str(self.step).zfill(10)}',
                         footnote_r=f'gen_loss={"{0:0.3f}".format(round(np.mean(self.gen_losses).item(), 3))}, '
                                    f'disc_loss={"{0:0.3f}".format(round(np.mean(self.disc_losses).item(), 3))}',
                         save_path=save_path)


if __name__ == '__main__':
    from datasets.lin import LINDataloader6Class
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
    # CLASS = 'Alp14'  # 'Alp14',  'Arp3', 'Cki2', 'Tea1'

    ###################################
    ###   Dataset Initialization    ###
    ###################################
    #   - the dataloader used to access the training dataset of cross-scale/pose image pairs at every epoch
    #     > len(dataloader) = <number of batches>
    #     > len(dataloader.dataset) = <number of total dataset items>
    dataloader = LINDataloader6Class(dataset_fs_folder_or_root=_datasets_groot, train_not_test=True,
                                     batch_size=6)
    dataset = dataloader.dataset

    ###################################
    ###    Models Initialization    ###
    ###################################
    #   - initialize evaluator instance (used to run GAN evaluation metrics: FID, IS, PRECISION, RECALL, F1 and SSIM)
    evaluator = GanEvaluator6Class(model_fs_folder_or_root=_models_groot, gen_dataset=dataset, z_dim=-1,
                                   device=exec_device, n_samples=16, batch_size=8, f1_k=2, ssim_c_img=2)
    #   - initialize model
    # _chkpt_step = f'aosokin:{PROJECT_DIR_PATH}/aosokin_checkpoints/auto'
    _chkpt_step = 'latest'
    # try:
    #     if chkpt_step == 'latest':
    #         _chkpt_step = chkpt_step
    #     elif isinstance(chkpt_step, str) and chkpt_step.isdigit():
    #         _chkpt_step = int(chkpt_step)
    #     else:
    #         _chkpt_step = None
    # except NameError:
    #     _chkpt_step = None
    biogan = BioGanStarShaped(model_fs_folder_or_root=_models_groot, config_id='wgan-gp-star-shaped',
                              dataset_len=len(dataset), chkpt_epoch=1703, chkpt_step=253800, evaluator=evaluator,
                              device=exec_device, log_level='debug', gen_transforms=dataloader.transforms)
    # print(biogan.gen)
    biogan.logger.debug(f'Using device: {str(exec_device)}')
    biogan.logger.debug(f'Model initialized. Number of params = {biogan.nparams_hr}')

    # Test visualization
    _x = next(iter(dataloader))
    print(_x.shape)
    _disc_loss, _gen_loss = biogan(_x)
    print('disc_loss', _disc_loss, 'gen_loss', _gen_loss)
    visualization_img = biogan.visualize(dl=dataloader, save_path='sample.pdf')
    plt.imshow(visualization_img)
    plt.show()
    # biogan.gcapture(metrics=False, visualizations=False, in_parallel=False, show_progress=True)
    # print(biogan.evaluate('fid'))
