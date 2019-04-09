import argparse
import os
import os.path as osp
import sys

os.environ['MPLBACKEND'] = 'Agg'  # NOQA

import chainer
#os.chdir('chainer_cyclegan/')
from chainer_cyclegan.datasets import DayNightDataset
from chainer_cyclegan.datasets import CycleGANTransform
from examples.horse2zebra.train import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('load_size', type=int, default=0)
    parser.add_argument('fine_size', type=int, default=384)
    parser.add_argument('--batchsize', '-B', type=int, default=1,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--preprocessor', choices=['none', 'he', 'sqrt_he'])
    parser.add_argument('--color', type=float, default=0)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--jpeg', type=float, default=0.5)
    args = parser.parse_args()

    dataset_train = chainer.datasets.TransformDataset(
            DayNightDataset(directory_a='./data/DarkFace_Unlabeled', directory_b='./data/train2014', split='train'),
            CycleGANTransform(load_size=(args.load_size, args.load_size), fine_size=(args.fine_size, args.fine_size)))
    dataset_test = chainer.datasets.TransformDataset(
            DayNightDataset(directory_a='./data/DarkFace_Unlabeled', directory_b='./data/train2014', split='test'),
            CycleGANTransform(load_size=(args.load_size, args.load_size), fine_size=(args.fine_size, args.fine_size),
                              train=False))

    device = 3
    batchsize = 2

    train(dataset_train, dataset_test, device, batchsize)