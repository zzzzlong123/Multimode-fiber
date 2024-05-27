# -*- coding:utf-8 -*-

"""
author: Hailong Zhang
date: 2024.05.27
"""
from Evaluator import ssim, psnr, mse

import os

import torch


def train(model, optimizer, criterion, train_loader, device, train_dataset_size, epoch, args):
    model.train()
    ssim_sum, mse_sum, psnr_sum = 0, 0, 0
    for batch_idx, (speckles, origins1, origins2) in enumerate(train_loader):
        optimizer.zero_grad()
        speckles, origins1, origins2 = speckles.to(device), origins1.to(device), origins2.to(device)
        batch_size, _, height, width = speckles.shape
        speckles_resize = speckles.view(batch_size, 4, height // 2, width // 2)
        preds = model(speckles_resize)
        preds1 = preds[:, 0, :, :].unsqueeze(1)
        preds2 = preds[:, 1, :, :].unsqueeze(1)
        train_loss = (criterion(preds1, origins1) + criterion(preds2, origins2)) / 2
        train_loss.backward()
        optimizer.step()
        ssim_sum += (ssim(img1=preds1, img2=origins1).item() +
                     ssim(img1=preds2, img2=origins2).item()) / 2
        psnr_sum += (psnr(img1=preds1, img2=origins1).item() +
                     psnr(img1=preds2, img2=origins2).item()) / 2
        mse_sum += (mse(img1=preds1, img2=origins1).item() +
                    mse(img1=preds2, img2=origins2).item()) / 2
    ssim_average = ssim_sum / int(train_dataset_size / args.batch_size)
    mse_average = mse_sum / int(train_dataset_size / args.batch_size)
    psnr_average = psnr_sum / int(train_dataset_size / args.batch_size)
    print('\ntrain results: \nSSIM_average: ', ssim_average, " MSE_average: ", mse_average, " PSNR_average: ",
          psnr_average)

    if epoch == 0:
        print("model is on GPU!" if next(model.parameters()).is_cuda else "model is on CPU!")
        print("speckles is on GPU!" if speckles.is_cuda else "speckles is on CPU!")
        print("origins1 is on GPU!" if origins1.is_cuda else "origins is on CPU!")
        print("origins2 is on GPU!" if origins2.is_cuda else "origins is on CPU!")
        print("train_loss is on GPU!" if train_loss.is_cuda else "train_loss is on CPU!")


def val(model, criterion, val_loader, device, val_dataset_size, model_save_path, args):
    model.eval()
    ssim_sum, mse_sum, psnr_sum = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (speckles, origins1, origins2) in enumerate(val_loader):
            speckles, origins1, origins2 = speckles.to(device), origins1.to(device), origins2.to(device)
            batch_size, _, height, width = speckles.shape
            speckles_resize = speckles.view(batch_size, 4, height // 2, width // 2)
            preds = model(speckles_resize)
            preds1 = preds[:, 0, :, :].unsqueeze(1)
            preds2 = preds[:, 1, :, :].unsqueeze(1)
            ssim_sum += (ssim(img1=preds1, img2=origins1).item() +
                         ssim(img1=preds2, img2=origins2).item()) / 2
            psnr_sum += (psnr(img1=preds1, img2=origins1).item() +
                         psnr(img1=preds2, img2=origins2).item()) / 2
            mse_sum += (mse(img1=preds1, img2=origins1).item() +
                        mse(img1=preds2, img2=origins2).item()) / 2
        ssim_average = ssim_sum / int(val_dataset_size / args.batch_size)
        mse_average = mse_sum / int(val_dataset_size / args.batch_size)
        psnr_average = psnr_sum / int(val_dataset_size / args.batch_size)
        print('val results: \nSSIM_average: ', ssim_average, " MSE_average: ", mse_average, " PSNR_average: ",
              psnr_average)
        torch.save(model.state_dict(), model_save_path)
