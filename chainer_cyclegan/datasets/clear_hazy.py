import os.path as osp
import subprocess

import chainer
import chainercv

from .base import UnpairedDirectoriesDataset


ROOT_DIR = chainer.dataset.get_dataset_directory('wkentaro/chainer-cyclegan')


class ClearHazyDataset(UnpairedDirectoriesDataset):

    def __init__(self, directory_a='./data/Outdoor_Hazy', directory_b='./data/train2014', split='train'):

        super(ClearHazyDataset, self).__init__(directory_a, directory_b, split)

    def __len__(self):
        return 2160 # As coco is very large, we shorted the epochs
