import sys
import unittest
from FCN.data.transforms import build_transform, build_untransform
from FCN.data.build import build_dataset
from FCN.config import cfg


class TestDataSet(unittest.TestCase):
    def test_dataset(self):
        train_trans=build_transform(cfg, is_train=True)
        val_trains=build_transform(cfg, is_train=False)
        train_dataset=build_dataset(cfg, train_trans, is_train=True)
        val_dataset=build_dataset(cfg, val_trains, is_train=False)
        from IPython import embed;
        embed()


if __name__ == '__main__':
    unittest.main()

