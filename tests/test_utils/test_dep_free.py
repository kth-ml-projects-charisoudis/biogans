from unittest import TestCase

from utils.dep_free import closest_pow


class Test(TestCase):
    def test_closest_pow(self):
        self.assertEqual(2 ** 1, closest_pow(2, of=2))
        self.assertEqual(3 ** 1, closest_pow(2, of=3))
        self.assertEqual(4 ** 0, closest_pow(2, of=4))
        self.assertEqual(4 ** 1, closest_pow(3, of=4))

    def test_closest_pow_of2(self):
        # 1096 --> 1024
        self.assertEqual(2 ** 10, closest_pow(1096, of=2))
        # 724 (2^9.499) --> 512 (2^9)
        self.assertEqual(2 ** 9, closest_pow(724, of=2))
        # 725 (2^9.501) --> 512 (2^10)
        self.assertEqual(2 ** 10, closest_pow(725, of=2))
