import os
import shutil
import unittest

from datasets.samplers import ResumableRandomSampler
from utils.command_line_logger import CommandLineLogger


class TestDataUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.pwd = os.getcwd()
        self.test_dir = f'{self.pwd}/.TestDataUtils_test_dir'
        os.mkdir(f'{self.test_dir}')
        self.logger = CommandLineLogger(log_level='debug', name=str(ResumableRandomSampler.__class__))

    def test_ResumableRandomSampler(self) -> None:
        sized_source = range(0, 1000)
        sampler = ResumableRandomSampler(data_source=sized_source, shuffle=False, logger=self.logger)
        self.assertListEqual([i for i in sized_source], [s_i for s_i in sampler])

        sized_source = range(0, 10)
        test_indices_with_seed_42 = [2, 6, 1, 8, 4, 5, 0, 9, 3, 7]
        sampler2 = ResumableRandomSampler(data_source=sized_source, shuffle=True, seed=42, logger=self.logger)
        self.assertListEqual(test_indices_with_seed_42, [s_i for s_i in sampler2])

        test_indices = [i for i in range(0, 10)] + [i for i in range(0, 10)]
        sampler3 = ResumableRandomSampler(data_source=range(0, 10), shuffle=False, logger=self.logger)
        self.assertListEqual(test_indices, [next(iter(sampler3)) for _ in range(20)])

        sized_source = range(0, 10)
        test_indices_with_seed_42 = [2, 6, 1, 8, 4, 5, 0, 9, 3, 7, 2, 6, 1, 8, 4, 5, 0, 9, 3, 7]
        sampler4 = ResumableRandomSampler(data_source=sized_source, shuffle=True, seed=42, logger=self.logger)
        sampler_indices = [next(iter(sampler4)) for _ in range(20)]
        self.assertEqual(len(test_indices_with_seed_42), len(sampler_indices))
        self.assertListEqual(test_indices_with_seed_42[:10], sampler_indices[:10])

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)
