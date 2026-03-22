#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import json
import cv2
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import ImageFolder

class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path

def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)

class OpenCVResize:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __call__(self, image: Image.Image) -> Image.Image:
        # PIL → NumPy (RGB)
        img_np = np.array(image)

        # RGB → BGR（OpenCV形式）
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # OpenCVでリサイズ
        resized_bgr = cv2.resize(img_bgr, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        # BGR → RGB
        resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

        # NumPy → PIL
        return Image.fromarray(resized_rgb)


# numpyをjson形式に対応させるための関数
def numpy_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return json.JSONEncoder().default(obj)

def get_autoencoder_256_512(out_channels=384):
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
        # decoder
        nn.Upsample(size=3, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=8, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        #nn.Upsample(size=120, mode='bilinear'), #image_size 512
        #nn.Upsample(size=88, mode='bilinear'), #image_size 384
        #nn.Upsample(size=56, mode='bilinear'), #image_size 256
        #nn.Upsample(size=248, mode='bilinear'), #image_size 1024
        nn.Upsample(size=(56,120), mode='bilinear'),#image_size 256×512
        #nn.Upsample(size=(296,411), mode='bilinear'),#image_size raw
        #nn.Upsample(size=(68,96), mode='bilinear'),#image_size 304×416

        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3,
                  stride=1, padding=1)
    )

class AutoEncoder(nn.Module):
    def __init__(self, out_channels=384):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 8)
        )
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(64, 64, 4, 1, 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, 4, 1, 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, 4, 1, 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, 4, 1, 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, 4, 1, 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, 4, 1, 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, 1, 1)
        )

    def forward(self, x, target_size=None):
        encoded = self.encoder(x)
        decoded = self.decoder_conv(encoded)

        if target_size is not None:
            decoded = F.interpolate(decoded, size=target_size, mode='bilinear', align_corners=False)

        return decoded


def get_autoencoder(out_channels=384):
    return AutoEncoder(out_channels)

def get_pdn_small_ver2(out_channels=384, padding=False, dropout_rate=0.2):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=3, padding=1),
        nn.Upsample(size=(64, 128), mode='bilinear', align_corners=False)  # 明示的にサイズを合わせる
    )

def get_pdn_small(out_channels=384, padding=False, dropout_rate=0.2):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )

def get_pdn_medium(out_channels=384, padding=False, dropout_rate=0.2):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )

# numpyをjson形式に対応させるための関数
def numpy_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return json.JSONEncoder().default(obj)

from pathlib import Path
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
import os

# 例: すでに定義済みだと仮定
# from your_module import ImageFolderWithoutTarget

def make_train_dataset(dataset_path, train_transform):
    dataset_path = Path(dataset_path)

    # 候補フォルダ
    dirs = [
        dataset_path / 'train',                 # 例: オリジナル画像
        dataset_path / 'train' / 'annotated',   # 例: アノテーション付き画像
    ]

    datasets = []
    for d in dirs:
        if d.is_dir():
            ds = ImageFolderWithoutTarget(
                str(d),
                transform=train_transform
            )
            datasets.append(ds)

    if not datasets:
        raise FileNotFoundError(
            f"学習用のフォルダが見つかりませんでした。確認した候補: {', '.join(map(str, dirs))}"
        )

    # 1つだけ存在する場合はそれを返し、2つ以上なら結合して返す
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

# 使い方
# full_train_set = make_train_dataset(dataset_path, train_transform)
