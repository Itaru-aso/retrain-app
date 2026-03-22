#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import itertools
import os
import random
import json
from tqdm import tqdm
from utils.common import get_autoencoder_256_512, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader, make_train_dataset
from omegaconf import DictConfig

# numpyをjson形式に対応させるための関数
def numpy_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return json.JSONEncoder().default(obj)

# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384

# data loading
default_transform = transforms.Compose([
    #transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #transforms.Normalize(mean=[0.518, 0.518, 0.518], std=[0.178, 0.178, 0.178])
])

pre_transform = transforms.Compose([
    transforms.RandomAffine(degrees=5, translate=(0.06, 0))
])

transform_ae = transforms.Compose([
    transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.2),
        transforms.ColorJitter(contrast=0.2),
        transforms.ColorJitter(saturation=0.2),
    ]),
    transforms.RandomAffine(
        degrees=5,
        translate=(0.06, 0)
    )
])


def train_transform(image):
    return default_transform(pre_transform(image)), default_transform(transform_ae(image))

def val_transform(image):
        return default_transform(image), default_transform(image)

def train_color(cfg: DictConfig):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_path = os.path.join(cfg.dataset_path, cfg.target_color, cfg.mode)
    train_output_dir = os.path.join(cfg.model_dir, cfg.target_color, cfg.mode)

    if os.path.isdir(train_output_dir):
        pass
    else:
        os.makedirs(train_output_dir)

    imagenet_train_path = cfg.imagenet_train_path

    pretrain_penalty = True
    if imagenet_train_path == 'none':
        pretrain_penalty = False

    full_train_set = make_train_dataset(dataset_path, train_transform)

    # mvtec dataset paper recommend 10% validation set
    train_size = int(0.9 * len(full_train_set))
    validation_size = len(full_train_set) - train_size
    rng = torch.Generator().manual_seed(seed)
    train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                        [train_size,
                                                        validation_size],
                                                        rng)


    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                            num_workers=0, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    if pretrain_penalty:
        # load pretraining data for penalty
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * 512, 2 * 256)),
            transforms.RandomCrop((512, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                0.225])
            #transforms.Normalize(mean=[0.518, 0.518, 0.518], std=[0.178, 0.178, 0.178])
        ])
        penalty_set = ImageFolderWithoutTarget(imagenet_train_path,
                                            transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True,
                                    num_workers=0, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    teacher = get_pdn_small(out_channels)
    student = get_pdn_small(2 * out_channels)

    #weights_path = r"D:\0032011\shisui_project\AI\EfficientAD\training\output\pretraining\monochro\teacher_small_color_final_state.pth"
    weights_path = os.path.join(cfg.model_dir, "pretraining", 'teacher_small_color_final_state.pth')
    state_dict = torch.load(weights_path, map_location='cpu')
    teacher.load_state_dict(state_dict)
    autoencoder = get_autoencoder_256_512(out_channels)

    # teacher frozen
    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    train_steps = 50000
    #train_steps = 5
    best_val_loss = float('inf')
    val_patience_counter = 0

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    optimizer = torch.optim.Adam(itertools.chain(student.parameters(),
                                                autoencoder.parameters()),
                                lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * train_steps), gamma=0.1)
    tqdm_obj = tqdm(range(train_steps))
    for iteration, (image_st, image_ae), image_penalty in zip(
            tqdm_obj, train_loader_infinite, penalty_loader_infinite):
        if on_gpu:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()
            if image_penalty is not None:
                image_penalty = image_penalty.cuda()
        with torch.no_grad():
            teacher_output_st = teacher(image_st)
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
        student_output_st = student(image_st)[:, :out_channels]
        distance_st = (teacher_output_st - student_output_st) ** 2
        d_hard = torch.quantile(distance_st, q=0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])

        if image_penalty is not None:
            student_output_penalty = student(image_penalty)[:, :out_channels]
            loss_penalty = torch.mean(student_output_penalty**2)
            loss_st = loss_hard + loss_penalty
        else:
            loss_st = loss_hard

        ae_output = autoencoder(image_ae)
        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)
            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
        student_output_ae = student(image_ae)[:, out_channels:]
        distance_ae = (teacher_output_ae - ae_output)**2
        distance_stae = (ae_output - student_output_ae)**2
        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)
        loss_total = loss_st + loss_ae + loss_stae

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            tqdm_obj.set_description(
                "Current loss: {:.4f}  ".format(loss_total.item()))

        if iteration % 2500 == 0:
            val_loss = compute_validation_loss(validation_loader, teacher, student, autoencoder,teacher_mean, teacher_std)
            print(f"Validation Loss: {val_loss:.4f}")

            torch.save(teacher.state_dict(), os.path.join(train_output_dir, 'teacher_state_temp.pth'))
            torch.save(student.state_dict(), os.path.join(train_output_dir, 'student_state_temp.pth'))
            torch.save(autoencoder.state_dict(), os.path.join(train_output_dir, 'autoencoder_state_temp.pth'))

            q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
                validation_loader=validation_loader, teacher=teacher,
                student=student, autoencoder=autoencoder,
                teacher_mean=teacher_mean, teacher_std=teacher_std,
                desc='Intermediate map normalization')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                val_patience_counter = 0
                torch.save(teacher.state_dict(), os.path.join(train_output_dir, 'teacher_state_best.pth'))
                torch.save(student.state_dict(), os.path.join(train_output_dir, 'student_state_best.pth'))
                torch.save(autoencoder.state_dict(), os.path.join(train_output_dir, 'autoencoder_state_best.pth'))

                q_st_start_l = [q_st_start.to('cpu').detach().numpy().copy()]
                q_st_end_l = [q_st_end.to('cpu').detach().numpy().copy()]
                q_ae_start_l = [q_ae_start.to('cpu').detach().numpy().copy()]
                q_ae_end_l = [q_ae_end.to('cpu').detach().numpy().copy()]
                teacher_mean_l = [teacher_mean.to('cpu').detach().numpy().copy()]
                teacher_std_l = [teacher_std.to('cpu').detach().numpy().copy()]

                para_json = {
                    'q_st_start':q_st_start_l, 'q_st_end':q_st_end_l, 'q_ae_start':q_ae_start_l, 'q_ae_end':q_ae_end_l,
                    'teacher_mean':teacher_mean_l, 'teacher_std':teacher_std_l
                            }

                para_file = open(os.path.join(train_output_dir, 'para.json'), 'w')
                json.dump(para_json, para_file ,indent=4, default=numpy_encoder)

            else:
                val_patience_counter += 1
                if val_patience_counter >= cfg.early_stop_patience:
                    print("Early stopping triggered based on validation loss.")
                    break

            # teacher frozen
            teacher.eval()
            student.train()
            autoencoder.train()


@torch.no_grad()
def compute_validation_loss(validation_loader, teacher, student, autoencoder,
                            teacher_mean, teacher_std, desc='Validation loss'):
    """
    Validationデータに対する平均ロスを返す。
    - 学習時の3項 (loss_hard, loss_ae, loss_stae) を使用
    - 画像ペナルティ (ImageNet) はValidationでは使用しない
    """
    teacher.eval()
    student.eval()
    autoencoder.eval()

    total_loss = 0.0
    num_batches = 0

    for image_st, image_ae in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()

        # ----- Student-Teacher（hardサンプル） -----
        t_out_st = teacher(image_st)
        t_out_st = (t_out_st - teacher_mean) / teacher_std
        s_out_st = student(image_st)[:, :out_channels]
        dist_st = (t_out_st - s_out_st) ** 2
        # ピクセル次元での0.999分位を閾値としてhard部分のみ平均
        d_hard = torch.quantile(dist_st, q=0.999)
        loss_hard = torch.mean(dist_st[dist_st >= d_hard])
        loss_st = loss_hard  # Validationではペナルティ無し

        # ----- Autoencoder整合性 -----
        ae_out = autoencoder(image_ae)
        t_out_ae = teacher(image_ae)
        t_out_ae = (t_out_ae - teacher_mean) / teacher_std
        s_out_ae = student(image_ae)[:, out_channels:]
        distance_ae = (t_out_ae - ae_out) ** 2
        distance_stae = (ae_out - s_out_ae) ** 2
        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)

        loss_total = loss_st + loss_ae + loss_stae
        total_loss += loss_total.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                        student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.9 * map_st + 0.2 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                    teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
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
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("./conf/config.yaml")
    train_color(cfg)
