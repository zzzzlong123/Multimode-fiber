# -*- coding:utf-8 -*-

"""
author: Hailong Zhang
date: 2024.05.27
"""
from Config import args_setting
from Dataset import Speckle_Origin
from TrainVal import train, val
from Models import MLPMixer_Conv

import os
import tqdm
import time
import GPUtil
from math import inf
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler


def main():
    args = args_setting()
    torch.cuda.manual_seed_all(args.seed)
    output_root = os.path.join(args.dir_path, 'output')
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = Speckle_Origin(split='train', transform=True, args=args)
    val_dataset = Speckle_Origin(split='val', transform=True, args=args)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=True)
    train_dataset_size, val_dataset_size = len(train_dataset), len(val_dataset)
    print('Number of sequence in train dataset/ val dataset: {}/ {}.'
          .format(train_dataset_size, val_dataset_size))
    model = MLPMixer_Conv(in_channels=4, hidden_dim=args.hidden_dim, num_classes=2,
                          patch_size=args.patch_size, speckle_size=128, layer_num=args.layer_num,
                          token_dim=args.token_dim, channel_dim=args.channel_dim, p=args.p).to(device)
    with torch.no_grad():
        init_img = torch.ones((args.batch_size, 4, 128, 128), device=device)
        writer.add_graph(model, init_img)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model_save_path = os.path.join(output_root, 'reconstraction.pth')
    for epoch in tqdm.trange(args.num_epoch):
        train(model, optimizer, criterion, train_loader, device, train_dataset_size, epoch, args)
        val(model, criterion, val_loader, device, val_dataset_size, model_save_path, args)


if __name__ == '__main__':
    main()
