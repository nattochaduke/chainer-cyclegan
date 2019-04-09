import os.path as osp
import subprocess

import chainer
import chainercv

from .base import UnpairedDirectoriesDataset


ROOT_DIR = chainer.dataset.get_dataset_directory('wkentaro/chainer-cyclegan')


class DayNightDataset(UnpairedDirectoriesDataset):

    def __init__(self, directory_a='./data/DarkFace_Unlabeled', directory_b='./data/train2014', split='train'):

        super(DayNightDataset, self).__init__(directory_a, directory_b, split)

    def __len__(self):
        return 720 # As coco is very large, we shorted the epochs
