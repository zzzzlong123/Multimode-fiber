# -*- coding:utf-8 -*-

"""
author: Hailong Zhang
date: 2024.05.27
"""
from Config import args_setting
from Models import MLPMixer_Conv
from Dataset import Speckle_Origin
from Evaluator import ssim, psnr, mse

import os
import shutil

import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

args = args_setting()
output_root = os.path.join(args.dir_path, 'output')
if os.path.exists(os.path.join(output_root, 'prediction')):
    shutil.rmtree(os.path.join(output_root, 'prediction'))
os.makedirs(os.path.join(output_root, 'prediction'))
test_dataset = Speckle_Origin(split='test', transform=True, args=args)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=None, num_workers=0, drop_last=False)
test_dataset_size = len(test_dataset)
print('Number of sequence in test dataset: {}.'.format(test_dataset_size))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MLPMixer_Conv(in_channels=4, hidden_dim=args.hidden_dim, num_classes=2,
                      patch_size=args.patch_size, speckle_size=128, layer_num=args.layer_num,
                      token_dim=args.token_dim, channel_dim=args.channel_dim, p=args.p).to(device)

criterion = nn.MSELoss().to(device)
model.eval()
ssim_sum, mse_sum, psnr_sum = 0, 0, 0
model_save_path = os.path.join(output_root, 'reconstraction.pth')
model.load_state_dict(torch.load(model_save_path))

with torch.no_grad():
    for batch_idx, (speckles, origins1, origins2) in enumerate(test_loader):
        speckles, origins1, origins2 = speckles.to(device), origins1.to(device), origins2.to(device)
        batch_size, _, height, width = speckles.shape
        speckles_resize = speckles.view(batch_size, 4, height // 2, width // 2)
        preds = model(speckles_resize)
        preds1 = preds[:, 0, :, :].unsqueeze(1)
        preds2 = preds[:, 1, :, :].unsqueeze(1)
        loss = (criterion(preds1, origins1) + criterion(preds2, origins2)) / 2
        print('MSE of every batch is {}'.format(loss.item()))

        ssim_sum += (ssim(img1=preds1, img2=origins1).item() +
                     ssim(img1=preds2, img2=origins2).item()) / 2
        psnr_sum += (psnr(img1=preds1, img2=origins1).item() +
                     psnr(img1=preds2, img2=origins2).item()) / 2
        mse_sum += (mse(img1=preds1, img2=origins1).item() +
                    mse(img1=preds2, img2=origins2).item()) / 2

    ssim_average = ssim_sum / (batch_idx+1)
    mse_average = mse_sum / (batch_idx+1)
    psnr_average = psnr_sum / (batch_idx+1)
    print('prediction results: \nSSIM_average: ', ssim_average, " MSE_average: ", mse_average, " PSNR_average: ",
          psnr_average)

print('==> Finished')
