# -*- coding:utf-8 -*-

"""
author: Hailong Zhang
date: 2024.05.27
"""
from Config import args_setting

import torch
import torch.nn as nn

args = args_setting()


class FeedForward(nn.Module):
    def __init__(self, hidden_dim=256, other_dim=256, p=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, other_dim)
        self.linear2 = nn.Linear(other_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, hidden_dim=512, num_patch=256, token_dim=256, channel_dim=2048, p=0.1):
        super().__init__()
        self.feedforward_token = FeedForward(num_patch, token_dim, p)
        self.feedforward_channel = FeedForward(hidden_dim, channel_dim, p)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x1 = self.layer_norm(x)
        x1 = x1.permute(0, 2, 1)
        x1 = self.feedforward_token(x1)
        x1 = x1.permute(0, 2, 1)
        x = x + x1
        x1 = self.layer_norm(x)
        x1 = self.feedforward_channel(x1)
        x = x + x1
        return x


class MLPMixer_Conv(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=512, num_classes=2, patch_size=16, speckle_size=128,
                 layer_num=8, token_dim=256, channel_dim=2048, p=0.1):
        super(MLPMixer_Conv, self).__init__()
        assert speckle_size % patch_size == 0
        self.num_classes = num_classes
        self.hw_patch = speckle_size // patch_size
        self.num_patch = (speckle_size // patch_size) ** 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size, padding=0, bias=True)
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(layer_num):
            self.mixer_blocks.append(MixerBlock(hidden_dim, self.num_patch, token_dim, channel_dim, p))
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.tconv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1)
        self.tconv2 = nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, kernel_size=4, stride=2, padding=1)
        self.tconv3 = nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=4, stride=2, padding=1)
        self.tconv_out = nn.ConvTranspose2d(hidden_dim // 8, num_classes, kernel_size=4, stride=2, padding=1)
        self.ins_norm1 = nn.InstanceNorm2d(hidden_dim//2)
        self.ins_norm2 = nn.InstanceNorm2d(hidden_dim//4)
        self.ins_norm3 = nn.InstanceNorm2d(hidden_dim // 8)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = x.view(x.size(0), self.hw_patch, self.hw_patch, x.size(-1)).permute(0, 3, 1, 2)
        x = self.tconv1(x)
        x = self.ins_norm1(x)
        x = self.relu(x)
        x = self.tconv2(x)
        x = self.ins_norm2(x)
        x = self.relu(x)
        x = self.tconv3(x)
        x = self.ins_norm3(x)
        x = self.relu(x)
        x = self.tconv_out(x)
        x = self.tanh(x)
        return x
