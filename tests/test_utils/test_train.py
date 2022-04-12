import os
import unittest
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from datasets.lin import LINDataloader
from modules.biogan import BioGanInd1class
from utils.filesystems.local import LocalFolder, LocalCapsule, LocalFile
from utils.pytorch import get_total_params
from utils.train import get_optimizer, train_test_split, \
    get_optimizer_lr_scheduler


class TestTrainUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.chkpts_root: str = '/home/achariso/PycharmProjects/kth-ml-course-projects/biogans/.checkpoints'
        self.cur_step = 1024
        self.batch_size = 10
        self.conv_weight_value = 1.5

        self.chkpt_path = f'{self.chkpts_root}/pgpg_{self.cur_step}_{self.batch_size}.pth'
        self.files_to_remove = [self.chkpt_path, ]

        # Set GoogleDrive root folder
        local_gdrive_root = '/home/achariso/PycharmProjects/kth-ml-course-projects/biogans/.gdrive_personal'
        self.models_groot = LocalFolder.root(capsule_or_fs=LocalCapsule(local_gdrive_root)).subfolder_by_name('Models')
        datasets_groot = LocalFolder.root(capsule_or_fs=LocalCapsule(local_gdrive_root)).subfolder_by_name('Datasets')
        self.dataloader = LINDataloader(dataset_fs_folder_or_root=datasets_groot, train_not_test=True,
                                        batch_size=8, which_classes='Alp14')
        torch.cuda.empty_cache()

    def test_get_adam_optimizer(self) -> None:
        # Training of a single model
        model = nn.Conv2d(3, 10, 3, bias=False)
        optimizer, _ = get_optimizer(model)
        self.assertEqual(type(optimizer), torch.optim.Adam)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(len(optimizer.param_groups[0]['params']), 1)
        self.assertEqual(len(optimizer.param_groups[0]['params'][0].view(-1)), get_total_params(model))

        # Joint training of two models
        model1 = nn.Conv2d(3, 10, 3, bias=False)
        model2 = nn.Conv2d(3, 5, 2, bias=False)
        optimizer, _ = get_optimizer(model1, model2, lr=1e-2)
        self.assertEqual(torch.optim.Adam, type(optimizer))
        self.assertEqual(1, len(optimizer.param_groups))
        self.assertEqual(2, len(optimizer.param_groups[0]['params']))
        self.assertEqual(get_total_params(model1), len(optimizer.param_groups[0]['params'][0].view(-1)))
        self.assertEqual(get_total_params(model2), len(optimizer.param_groups[0]['params'][1].view(-1)))
        self.assertEqual(1e-2, optimizer.param_groups[0]['lr'])
        model1_wb = nn.Conv2d(3, 10, 3)
        model2_wb = nn.Conv2d(3, 5, 2)
        optimizer, _ = get_optimizer(model1_wb, model2_wb, lr=1e-5)
        self.assertEqual(1, len(optimizer.param_groups))
        self.assertEqual(4, len(optimizer.param_groups[0]['params']))
        self.assertEqual(get_total_params(model1), len(optimizer.param_groups[0]['params'][0].view(-1)))
        self.assertEqual(10, len(optimizer.param_groups[0]['params'][1].view(-1)))  # 1st model's bias params
        self.assertEqual(get_total_params(model2), len(optimizer.param_groups[0]['params'][2].view(-1)))
        self.assertEqual(5, len(optimizer.param_groups[0]['params'][3].view(-1)))  # 2nd model's bias params

    # noinspection PyUnresolvedReferences
    def test_get_optimizer_lr_scheduler(self):
        # Test CyclicLR scheduler
        # Every step_up steps the LR reaches maximum while every 2*steps_up, its minimum. It's like a sinusoid.
        test_model = nn.Conv2d(3, 3, 3)
        test_model_opt, _ = get_optimizer(test_model)
        base_lr = 1e-4
        max_lr = 1e-3
        steps_up = 100
        test_lr_scheduler = get_optimizer_lr_scheduler(optimizer=test_model_opt, schedule_type='cyclic',
                                                       cycle_momentum='momentum' in test_model_opt.defaults,
                                                       base_lr=base_lr, max_lr=max_lr, step_size_up=steps_up)
        self.assertEqual(1e-4, test_lr_scheduler.base_lrs[0])
        self.assertEqual(1e-3, test_lr_scheduler.max_lrs[0])
        for i in range(10 * steps_up * 2):
            test_model_opt.step()
            test_lr_scheduler.step()
            _lr = test_model_opt.param_groups[0]['lr']
            if (i + 1) % (2 * steps_up) == 0:
                self.assertLessEqual(base_lr - _lr, 1e-8, msg=f'step={i}, base_lr={base_lr}, _lr={_lr}')
            elif (i + 1) % steps_up == 0:
                self.assertLessEqual(max_lr - _lr, 1e-8, msg=f'step={i}, max_lr={max_lr}, _lr={_lr}')

        # Test ReduceLROnPlateau scheduler
        # Reduces the LR after some steps with same value (within certain tolerance)
        test_model_2 = nn.Conv2d(3, 3, 3)
        test_model_opt_2, _ = get_optimizer(test_model_2)
        test_lr_scheduler_2 = get_optimizer_lr_scheduler(test_model_opt_2, schedule_type='on_plateau')
        self.assertEqual(ReduceLROnPlateau, type(test_lr_scheduler_2))
        torch.set_default_dtype(torch.float32)

    @staticmethod
    def _get_params(model):
        gen_weights = []
        for name, param in model.gen.named_parameters():
            if 'weight' in name or 'bias' in name:
                gen_weights.append(param.detach().cpu().clone().numpy().flatten())
        gen_weights = np.concatenate(gen_weights).flatten()
        gen_opt_params = []
        for param in model.gen_opt.param_groups[0]['params']:
            gen_opt_params.append(param.detach().cpu().clone().numpy().flatten())
        gen_opt_params = np.concatenate(gen_opt_params).flatten()
        disc_weights = []
        for name, param in model.disc.named_parameters():
            if 'weight' in name or 'bias' in name:
                disc_weights.append(param.detach().cpu().clone().numpy().flatten())
        disc_weights = np.concatenate(disc_weights).flatten()
        disc_opt_params = []
        for param in model.disc_opt.param_groups[0]['params']:
            disc_opt_params.append(param.detach().cpu().clone().numpy().flatten())
        disc_opt_params = np.concatenate(disc_opt_params).flatten()
        return gen_weights, disc_weights, gen_opt_params, disc_opt_params

    def test_load_model_chkpt(self) -> None:
        biogan = BioGanInd1class(model_fs_folder_or_root=self.models_groot, config_id='default',
                                chkpt_epoch=88, chkpt_step=6000, device='cuda',
                                dataset_len=len(self.dataloader.dataset), log_level='debug')

        # Fetch params before saving model
        gen_weights0, disc_weights0, gen_opt_params0, disc_opt_params0 = self.__class__._get_params(biogan)

        for epoch in range(4):
            pbar = tqdm(self.dataloader)
            for batch_i, x in enumerate(pbar):
                x = x.to(biogan.device)
                disc_loss, gen_loss = biogan(x)
                pbar.set_description(f'd:{disc_loss:.3f}|g:{gen_loss:.3f}')
                if batch_i % 50 == 0:
                    biogan.visualize(reproducible=True)
                    plt.show()

        # Fetch params before saving model
        gen_weights, disc_weights, gen_opt_params, disc_opt_params = self.__class__._get_params(biogan)
        print(np.linalg.norm(gen_weights - gen_weights0))
        print(np.linalg.norm(disc_weights - disc_weights0))
        print(np.linalg.norm(gen_opt_params - gen_opt_params0))
        print(np.linalg.norm(disc_opt_params - disc_opt_params0))

        # Save checkpoint
        chkpt: LocalFile
        chkpt, = biogan.gcapture(checkpoint=True, metrics=False, visualizations=False, in_parallel=False,
                                 show_progress=True)

        # Load checkpoint
        biogan2 = BioGanInd1class(model_fs_folder_or_root=self.models_groot, config_id='default',
                                 chkpt_epoch=88, chkpt_step=biogan.step, device=biogan.device,
                                 dataset_len=len(self.dataloader.dataset), log_level='debug',
                                 gen_transforms=self.dataloader.transforms)
        biogan2.logger = biogan.logger

        biogan.update_lr(0.0, 0.0)
        biogan2(next(iter(self.dataloader)).to(biogan2.device))
        biogan2(next(iter(self.dataloader)).to(biogan2.device))

        # Fetch params after saving model
        gen_weights2, disc_weights2, gen_opt_params2, disc_opt_params2 = self.__class__._get_params(biogan2)

        self.assertTrue(np.allclose(gen_weights, gen_weights2), str(np.linalg.norm(gen_weights - gen_weights2)))
        self.assertTrue(np.allclose(disc_weights, disc_weights2))
        self.assertTrue(np.allclose(gen_opt_params, gen_opt_params2))
        self.assertTrue(np.allclose(disc_opt_params, disc_opt_params2))

        # # Signature of load_model_chkpt:  load_model_chkpt(model, model_name, dict_key, model_opt, chkpts_root)
        # self.assertRaises(AssertionError, load_model_chkpt, biogan, 'biogan', 'gen', gen_opt)
        # self.assertRaises(AssertionError, load_model_chkpt, gen, 'pgpg', 'gen1', gen_opt)
        # try:
        #     _, chkpt_step, chkpt_batch_size = load_model_chkpt(gen, 'pgpg', 'gen', gen_opt)
        #     total_images_in_checkpoint = chkpt_step * chkpt_batch_size
        # except AssertionError:
        #     self.fail("load_model_chkpt() raised AssertionError")
        # self.assertEqual(self.cur_step * self.batch_size, total_images_in_checkpoint)
        #
        # # Check loading of state dict to a new model
        # gen2 = PGPGGenerator(c_in=6, c_out=3, w_in=128, h_in=128).cpu()
        # load_model_chkpt(gen2, 'pgpg', 'gen')
        #
        # def _check_weights(_m):
        #     if isinstance(_m, nn.Conv2d) or isinstance(_m, nn.ConvTranspose2d):
        #         self.assertTrue(torch.all(_m.weight.data.eq(self.conv_weight_value * torch.ones_like(_m.weight.data))))
        #
        # gen2.apply(_check_weights)
        #
        # # Try loading inception checkpoint
        # inception = inception_v3(pretrained=False, init_weights=False)
        # try:
        #     load_model_chkpt(inception, 'inception_v3')
        # except AssertionError:
        #     self.fail("load_model_chkpt(inception, 'inception_v3') raised AssertionError")

    def test_save_model_chkpt(self) -> None:
        gdmc = GDriveModelCheckpoints.instance()
        self.assertIsNotNone(gdmc)

        conv1 = nn.Conv2d(3, 3, 3)
        conv2 = nn.Conv2d(3, 3, 3)

        use_threads_enum = [False, True]
        use_gdmc_enum = [False, True]
        for i, use_threads in enumerate(use_threads_enum):
            for j, use_gdmc in enumerate(use_gdmc_enum):
                target_chkpt_filepath = f'{self.chkpts_root}/test_{str(self.cur_step + i + j).zfill(10)}_' + \
                                        f'{self.batch_size}.pth'
                self.files_to_remove.append(target_chkpt_filepath)

                thread_or_result = save_model_chkpt({'conv1': conv1}, {'conv2': conv2},
                                                    model_chkpts_root=self.chkpts_root, chkpt_file_prefix='test',
                                                    step=self.cur_step + i + j, batch_size=self.batch_size,
                                                    metrics_dict=None, dataloader=None,
                                                    use_threads=use_threads, gdmc=gdmc if use_gdmc else None)
                # Test return type
                if use_threads:
                    self.assertEqual(Thread, type(thread_or_result))
                    thread_or_result.join()
                else:
                    self.assertEqual(bool, type(thread_or_result))
                    self.assertTrue(thread_or_result)
                # Test that local file created
                self.assertTrue(os.path.exists(target_chkpt_filepath))
                # Test that file has be uploaded to drive
                gdmc_file = None
                if use_gdmc:
                    # Test file name
                    test_chkpt_gdrive = gdmc.get_model_chkpt_data(model_name='test', step='latest')
                    self.assertEqual(dict, type(test_chkpt_gdrive))
                    self.assertEqual(os.path.basename(target_chkpt_filepath), test_chkpt_gdrive['title'])
                    # Download file locally to read its contents
                    gdmc_file = gdmc.gdrive.CreateFile({'id': test_chkpt_gdrive['id']})
                    target_chkpt_filepath = f'{self.chkpts_root}/test_gdrive.pth'
                    gdmc_file.GetContentFile(filename=target_chkpt_filepath)
                # Check file contents
                target_state_dict = torch.load(target_chkpt_filepath)
                self.assertListEqual(['conv1', 'conv2'], list(target_state_dict.keys()),
                                     msg=f'target_state_dict.keys()={str(target_state_dict.keys())}')
                self.assertListEqual(['bias', 'weight'], sorted(list(target_state_dict['conv1'].keys())),
                                     msg=f'conv1.keys()={str(target_state_dict["conv1"].keys())}')
                self.assertListEqual(['bias', 'weight'], sorted(list(target_state_dict['conv2'].keys())),
                                     msg=f'conv2.keys()={str(target_state_dict["conv2"].keys())}')
                # Remove file for next iteration
                os.remove(target_chkpt_filepath)
                # Remove file from Google Drive
                if use_gdmc and gdmc_file:
                    gdmc_file.Delete()

    # noinspection PyUnresolvedReferences
    def test_train_test_split(self) -> None:
        test_dataset = ICRBDataset()
        test_splits = [50, 50]
        training_set, test_set = train_test_split(test_dataset, splits=test_splits)
        self.assertLessEqual(len(training_set) - len(test_set), 1)
        self.assertEqual(len(test_dataset), len(training_set) + len(test_set))
        self.assertEqual(type(test_dataset), type(training_set.dataset))
        self.assertEqual(type(test_dataset), type(test_set.dataset))

    def tearDown(self) -> None:
        for file in self.files_to_remove:
            if os.path.exists(file):
                os.remove(file)
