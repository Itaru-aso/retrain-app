#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
from predict import predict

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                    teacher_mean, teacher_std, st_para, ae_para, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    # 使用したいGPUの番号を指定
    gpu_id = 1
    on_gpu = torch.cuda.is_available()
    # デバイスを作成
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.to(device)
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, st_para=st_para, ae_para=ae_para)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)

    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader):

    mean_outputs = []
    on_gpu = torch.cuda.is_available()

    # 使用したいGPUの番号を指定
    gpu_id = 1
    # デバイスを作成
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.to(device)
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.to(device)
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

