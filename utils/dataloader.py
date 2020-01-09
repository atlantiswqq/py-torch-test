# _*_ coding:utf-8 _*_
# Author:Atlantis
# Date:2020-01-09

import torchvision.datasets as dst
import torchvision.transforms as transforms


class DataLoader(object):
    def __init__(self, dst_folder):
        self.dst_folder = dst_folder

    def data_loader(self):
        print("load MNIST data...")
        train_dst = dst.MNIST(
            root=self.dst_folder,
            train=True,
            transform=transforms.ToTensor(),
            download=True
        )
        test_dst = dst.MNIST(
            root=self.dst_folder,
            train=False,
            transform=transforms.ToTensor()
        )
        return train_dst, test_dst
