# -*- coding:utf-8 -*-

"""
author: Hailong Zhang
date: 2024.05.27
"""
from Config import args_setting

import re
import os
from PIL import Image
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import Dataset


class Speckle_Origin(Dataset):
    def __init__(self, split='train', transform=False, args=args_setting):
        self.root = os.path.join(args.dir_path, 'input')
        self.transform = transform
        self.split = split
        self.args = args
        assert self.split in ['train', 'val', 'test']
        self.speckle_path = os.path.join(self.root, 'speckle-256.npy')
        self.origin_path1 = os.path.join(self.root, 'origin1-128.npy')
        self.origin_path2 = os.path.join(self.root, 'origin2-128.npy')
        self.speckle_data = np.load(self.speckle_path)
        self.origin_data1 = np.load(self.origin_path1)
        self.origin_data2 = np.load(self.origin_path2)
        assert len(self.speckle_data) == len(self.origin_data1) == len(self.origin_data2)
        if self.split == 'train':
            end_index = args.train_num
            self.speckle_data = self.speckle_data[:end_index]
            self.origin_data1 = self.origin_data1[:end_index]
            self.origin_data2 = self.origin_data2[:end_index]
        elif self.split == 'val':
            start_index = args.train_num
            end_index = args.train_num + args.val_num
            self.speckle_data = self.speckle_data[start_index:end_index]
            self.origin_data1 = self.origin_data1[start_index:end_index]
            self.origin_data2 = self.origin_data2[start_index:end_index]
        elif self.split == 'test':
            start_index = args.train_num+args.val_num
            self.speckle_data = self.speckle_data[start_index:]
            self.origin_data1 = self.origin_data1[start_index:]
            self.origin_data2 = self.origin_data2[start_index:]

    def __len__(self):
        return len(self.speckle_data)

    def __getitem__(self, idx):
        speckle = self.speckle_data[idx]
        origin1 = self.origin_data1[idx]
        origin2 = self.origin_data2[idx]
        speckle_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        origin_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        if self.transform is not False:
            speckle = speckle_transform(speckle)
            origin1 = origin_transform(origin1)
            origin2 = origin_transform(origin2)

        return speckle, origin1, origin2
