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
from chainerui.utils import save_args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--load_size', type=int, default=0)
    parser.add_argument('--fine_size', type=int, default=384)
    parser.add_argument('--batchsize', '-B', type=int, default=2,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--niter', type=int, default=100)
    parser.add_argument('--skip', action='store_true')
    parser.add_argument('--intermeidates', type=int, default=9)
    args = parser.parse_args()

    dataset_train = chainer.datasets.TransformDataset(
            DayNightDataset(directory_a='./data/DarkFace_Unlabeled', directory_b='./data/train2014', split='train'),
            CycleGANTransform(load_size=(args.load_size, args.load_size), fine_size=(args.fine_size, args.fine_size)))
    dataset_test = chainer.datasets.TransformDataset(
            DayNightDataset(directory_a='./data/DarkFace_Unlabeled', directory_b='./data/train2014', split='test'),
            CycleGANTransform(load_size=(args.load_size, args.load_size), fine_size=(args.fine_size, args.fine_size),
                              train=False))

    train(dataset_train, dataset_test, args.device, args.batchsize,
          args.skip, args.intermediates, niter=args.niter, args=args)
if __name__ == '__main__':
    main()
