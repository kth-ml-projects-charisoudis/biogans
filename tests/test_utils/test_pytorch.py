import os
import unittest

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from tqdm import tqdm

from modules.biogan_ind import BioGanInd1class as OneClassBioGan
from utils.pytorch import get_total_params, invert_transforms, UnNormalize, get_gpu_memory_gb


class TestPytorchUtils(unittest.TestCase):

    def test_get_total_params(self) -> None:
        conf = {
            'c_in': 3,
            'c_out': 10,
            'kernel': 3,
            'bias': True
        }
        test_module = torch.nn.Conv2d(in_channels=conf['c_in'], out_channels=conf['c_out'], kernel_size=conf['kernel'],
                                      bias=conf['bias'])
        self.assertEqual(conf['c_out'] * (conf['c_in'] * conf['kernel'] ** 2 + (1 if conf['bias'] else 0)),
                         get_total_params(test_module))

    def test_unnormalize(self) -> None:
        # Check UnNormalize transform
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        shape = 299
        t = transforms.Compose([
            transforms.Normalize(mean=mean, std=std),
            UnNormalize(mean=mean, std=std),
        ])
        x = torch.randn(3, 100, 100)
        x_hat = t(x)
        self.assertTrue(torch.allclose(x, x_hat, atol=1e-6))

        # Check invert_transforms(): types
        ts = transforms.Compose([
            transforms.Resize(shape),
            transforms.CenterCrop(shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        ts_i = invert_transforms(ts)
        self.assertEqual(1, len(ts_i.transforms))
        self.assertEqual(UnNormalize, type(ts_i.transforms[0]))

        # Evaluate on a real image
        test_img_path = '/home/achariso/Pictures/me.jpg'
        if os.path.exists(test_img_path):
            x = Image.open(test_img_path)
            x_tensor = ts(x)
            self.assertEqual(torch.Tensor, type(x_tensor))
            self.assertEqual((3, shape, shape), tuple(x_tensor.shape))
            x_tensor_hat = UnNormalize(mean=mean, std=std)(x_tensor)
            self.assertEqual(torch.Tensor, type(x_hat))
            self.assertEqual(tuple(x_tensor.shape), tuple(x_tensor_hat.shape))
            self.assertGreaterEqual(x_tensor_hat.min(), 0)  # test if is normalized in [0, 1] as if used ToTensor()
            self.assertLessEqual(x_tensor_hat.max(), 1)  # test if is normalized in [0, 1] as if used ToTensor()

    def test_get_gpu_memory_gb(self):
        self.assertEqual('2Gb', f'{get_gpu_memory_gb(gpu_index=0)}Gb')

    def test_torch_save_load(self):
        from datasets.lin import LINDataloader
        from utils.filesystems.local import LocalFolder, LocalFilesystem, LocalCapsule

        # Get GoogleDrive root folder
        _local_gdrive_root = os.environ.get('GDRIVE_ROOT')
        _log_level = 'debug'

        # Via locally-mounted Google Drive (when running from inside Google Colaboratory)
        _fs = LocalFilesystem(LocalCapsule(_local_gdrive_root))
        _groot = LocalFolder.root(capsule_or_fs=_fs)

        # Define folder roots
        _models_groot = _groot.subfolder_by_name('Models')
        _datasets_groot = _groot.subfolder_by_name('Datasets')

        exec_device = 'cpu'

        ###################################
        ###   Dataset Initialization    ###
        ###################################
        #   - the dataloader used to access the training dataset of cross-scale/pose image pairs at every epoch
        #     > len(dataloader) = <number of batches>
        #     > len(dataloader.dataset) = <number of total dataset items>
        dataloader = LINDataloader(dataset_fs_folder_or_root=_datasets_groot, train_not_test=False,
                                   batch_size=2, which_classes='Alp14')
        dataset = dataloader.dataset

        ###################################
        ###    Models Initialization    ###
        ###################################
        OneClassBioGan.WhichClass = 'Alp14'
        biogan = OneClassBioGan(model_fs_folder_or_root=_models_groot, config_id='default', dataset_len=len(dataset),
                                chkpt_epoch=79, chkpt_step=5400, evaluator=None, device='cuda', log_level='debug')
        biogan.logger.debug(f'Using device: {str(exec_device)}')
        biogan.logger.debug(f'Model initialized. Number of params = {biogan.nparams_hr}')
        z = biogan.gen.get_random_z(1, 'cuda')
        plt.imshow(torch.cat((biogan.gen(z).detach().cpu().squeeze().permute(1, 2, 0), torch.zeros(48, 80, 1)),
                             dim=2).numpy())
        plt.show()

        # backup params
        params = []
        for p in biogan.parameters():
            params.append(p.clone().detach().cpu())

        plt.imshow(biogan.gen(z).detach().cpu().numpy().squeeze()[0], cmap="Reds")
        plt.show()

        # save checkpoint
        torch.save(biogan.state_dict(), 'test.pth', _use_new_zipfile_serialization=False)

        plt.imshow(biogan.gen(z).detach().cpu().numpy().squeeze()[0], cmap="Reds")
        plt.show()

        # perform forward passes of an entire epoch
        pbar = tqdm(dataloader)
        for x in pbar:
            d_loss, g_loss = biogan(x.to(biogan.device))
            pbar.set_description(f'[d:{d_loss:.4f}|g:{g_loss:.4f}]')

        plt.imshow(biogan.gen(z).detach().cpu().numpy().squeeze()[0], cmap="Reds")
        plt.show()

        # load checkpoint
        biogan.load_state_dict(torch.load('test.pth'))

        plt.imshow(biogan.gen(z).detach().cpu().numpy().squeeze()[0], cmap="Reds")
        plt.show()

        # backup params
        for pi, p2 in enumerate(biogan.parameters()):
            self.assertTrue(torch.equal(p2.clone().detach().cpu(), params[pi]), msg=f'pi={pi}')
