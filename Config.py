# -*- coding:utf-8 -*-

"""
author: Hailong Zhang
date: 2024.05.27
"""
import argparse


def args_setting():
    parser = argparse.ArgumentParser(description='Parallel')
    parser.add_argument('--dir_path', type=str, default='G:/')
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=600)
    parser.add_argument('--train_num', type=int, default=14000)
    parser.add_argument('--val_num', type=int, default=2000)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--p', type=float, default=0.1)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--layer_num', type=int, default=8)
    parser.add_argument('--token_dim', type=int, default=256)
    parser.add_argument('--channel_dim', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()
    return args
