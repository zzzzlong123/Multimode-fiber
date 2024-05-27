# -*- coding:utf-8 -*-

"""
author: Hailong Zhang
date: 2024.05.27
"""
import torch
import torch.nn.functional as F
from piqa import SSIM, PSNR


ssim_cuda = SSIM(n_channels=1).cuda()
psnr_cuda = PSNR().cuda()


def normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def ssim(img1, img2, mean = 0.5, std = 0.5):
    mean = torch.tensor(mean).view(-1, 1, 1).cuda()
    std = torch.tensor(std).view(-1, 1, 1).cuda()

    origin_img = img1 * std + mean
    target_img = img2 * std + mean
    origin_img = torch.clamp(origin_img, 0, 1)
    target_img = torch.clamp(target_img, 0, 1)
    ssim_value = ssim_cuda(origin_img, target_img)
    return ssim_value


def psnr(img1, img2, mean = 0.5, std = 0.5):
    mean = torch.tensor(mean).view(-1, 1, 1).cuda()
    std = torch.tensor(std).view(-1, 1, 1).cuda()

    origin_img = img1 * std + mean
    target_img = img2 * std + mean
    origin_img = torch.clamp(origin_img, 0, 1)
    target_img = torch.clamp(target_img, 0, 1)
    psnr_value = psnr_cuda(origin_img, target_img)
    return psnr_value


def mse(img1, img2):
    return F.mse_loss(img1, img2)
