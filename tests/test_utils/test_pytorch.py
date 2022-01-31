import os
import unittest

import torch
from PIL import Image
from torchvision.transforms import transforms

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
