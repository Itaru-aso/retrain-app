#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import itertools
import os
import random
import multiprocessing
import json
from tqdm import tqdm
from utils.common import get_autoencoder_256_512, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast
from filelock import FileLock
from omegaconf import OmegaConf
from functools import partial

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# numpyをjson形式に対応させるための関数
def numpy_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return json.JSONEncoder().default(obj)

def train_transform(image):

    pre_transform = transforms.Compose([
        #OpenCVResize(width=512, height=256),
        #transforms.Resize((304, 416)),
        transforms.RandomApply([
            #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAffine(degrees=2, translate=(0.03, 0)),
            #transforms.GaussianBlur(kernel_size=1)
        ], p=0.5),
    ])

    default_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return default_transform(pre_transform(image)), default_transform(pre_transform(image))

def train(cfg: DictConfig):

    dataset_path = os.path.join(cfg.dataset_path, cfg.target_color, cfg.mode)
    train_output_dir = os.path.join(cfg.model_dir, cfg.target_color, cfg.mode)

    out_channels = cfg.out_channels
    seed = cfg.seed
    # constants
    random.seed(0)

    image_size_height = cfg.image_size.height
    image_size_width = cfg.image_size.width

    mode = cfg.mode

    on_gpu = torch.cuda.is_available()
    num_workers = os.cpu_count() // 2  # 例：8コアなら4

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 使用したいGPUの番号を指定
    gpu_id = cfg.gpu_id
    # デバイスを作成
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    if os.path.isdir(train_output_dir):
        pass
    else:
        os.makedirs(train_output_dir)

    pretrain_penalty = True
    # load data
    full_train_set_1 = ImageFolderWithoutTarget(
        os.path.join(dataset_path, 'train'),
        transform=transforms.Lambda(train_transform)) #, 'original'を追加する必要あり
    """
    full_train_set_2 = ImageFolderWithoutTarget(
        os.path.join(dataset_path, 'train', 'annotated'),
        transform=transforms.Lambda(train_transform))

    #full_train_set = ConcatDataset([full_train_set_1, full_train_set_2])
    """
    full_train_set = full_train_set_1
    print(f'{cfg.mode}_Full train set size: {len(full_train_set)}')

    # 訓練、検証、テストに分割
    train_size = int(0.8 * len(full_train_set))
    validation_size = len(full_train_set) - train_size
    rng = torch.Generator().manual_seed(seed)
    train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                        [train_size,
                                                        validation_size],
                                                        rng)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=cfg.batch_size)

    train_loader_infinite = InfiniteDataloader(train_loader)

    if pretrain_penalty:
        # load pretraining data for penalty
        if mode == "color":
            penalty_transform = transforms.Compose([
                transforms.Resize((2 * image_size_width, 2 * image_size_height)),
                #transforms.RandomGrayscale(0.3),
                #transforms.CenterCrop(image_size),
                transforms.RandomCrop((image_size_width, image_size_height)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                    0.225])
            ])
        elif mode == "monochro":
            penalty_transform = transforms.Compose([
                transforms.Resize((2 * image_size_width, 2 * image_size_height)),
                transforms.RandomGrayscale(1.0),
                #transforms.CenterCrop(image_size),
                transforms.RandomCrop((image_size_width, image_size_height)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                    0.225])
            ])
        else:
            raise ValueError("Invalid mode. Choose 'color' or 'monochro'.")
        penalty_set = ImageFolderWithoutTarget(cfg.imagenet_train_path,
                                            transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=cfg.batch_size, shuffle=True,
                                    num_workers=4, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    teacher = get_pdn_small(out_channels)
    student = get_pdn_small(2 * out_channels)

    #weight_path = os.path.join("C:/Fastenerlnsp/retrain_app/pretraining", cfg.mode, "teacher_small_best_state.pth")
    if cfg.mode == "color":
        weight_path = os.path.join("D:/0032011/shisui_project/AI/EfficientAD/training/output/pretraining/teacher_small_color_final_state.pth")
    elif cfg.mode == "monochro":
        weight_path = os.path.join("D:/0032011/shisui_project/AI/EfficientAD/training/output/pretraining/teacher_small_monochro_final_state.pth")
    else:
        raise ValueError("Invalid mode. Choose 'color' or 'monochro'.")
    #weight_path = os.path.join("C:/Fastenerlnsp/retrain_app/pretraining/ImageNet/teacher_small_final_state.pth")
    state_dict = torch.load(weight_path, map_location='cpu')
    teacher.load_state_dict(state_dict) #TeacherモデルにPretrainingで作成した
    autoencoder = get_autoencoder_256_512(out_channels)

    # teacher frozen
    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.to(device)
        student.to(device)
        autoencoder.to(device)

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    #optimizer = torch.optim.Adam(itertools.chain(student.parameters(),
    #                                            autoencoder.parameters()),
    #                            lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(
    #    optimizer, step_size=int(0.95 * cfg.train_step), gamma=cfg.model.gamma)

    optimizer = torch.optim.AdamW(itertools.chain(student.parameters(),
                                                autoencoder.parameters()),
                                lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train_step)

    best_val_loss = float('inf')
    val_patience_counter = 0

    train_dataset_size = len(full_train_set)
    iterations_per_epoch = train_dataset_size // cfg.batch_size
    cfg.train_step = iterations_per_epoch * cfg.epochs

    val_interval = len(train_loader)

    scaler = GradScaler()

    tqdm_obj = tqdm(range(cfg.train_step))
    for iteration, (image_st, image_ae), image_penalty in zip(
            tqdm_obj, train_loader_infinite, penalty_loader_infinite):
        optimizer.zero_grad()
        with autocast("cuda"):
            if on_gpu:
                image_st = image_st.to(device)
                image_ae = image_ae.to(device)
                if image_penalty is not None:
                    image_penalty = image_penalty.to(device)

            with torch.no_grad():
                teacher_output_st = teacher(image_st)
                teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
            student_output_st = student(image_st)[:, :out_channels]

            distance_st = (teacher_output_st - student_output_st) ** 2
            #d_hard = torch.quantile(distance_st, q=0.999)

            flat = distance_st.flatten()
            max_elements = 100_000 # 10万要素までに制限

            if flat.numel() > max_elements:
                indices = torch.randperm(flat.numel(), device=flat.device)[:max_elements]
                flat = flat[indices]

            d_hard = torch.quantile(flat, q=0.999)

            loss_hard = torch.mean(distance_st[distance_st >= d_hard])

            # --- メモリ節約：不要なテンソルを削除 ---
            del teacher_output_st, student_output_st, distance_st
            torch.cuda.empty_cache()

            if image_penalty is not None:
                student_output_penalty = student(image_penalty)[:, :out_channels]
                loss_penalty = torch.mean(student_output_penalty**2)
                loss_st = loss_hard + loss_penalty
            else:
                loss_st = loss_hard

            with torch.no_grad():
                teacher_output_ae = teacher(image_ae)
                teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
            target_size = teacher_output_ae.shape[2:]  # (height, width)

            ae_output = autoencoder(image_ae)
            student_output_ae = student(image_ae)[:, out_channels:]

            distance_ae = (teacher_output_ae - ae_output)**2
            distance_stae = (ae_output - student_output_ae)**2
            loss_ae = torch.mean(distance_ae) # autoencoderの出力とteacherの出力の距離　autoencoderがteacherの出力を再構成するための損失
            loss_stae = torch.mean(distance_stae) # autoencoderの出力とstudentの出力の距離 studentがautoencoderの出力を再構成するための損失
            loss_total = cfg.loss_st * loss_st + cfg.loss_ae * loss_ae + cfg.loss_stae * loss_stae

            #loss_total.backward()
            #optimizer.step()
            #scheduler.step()

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if iteration % 10 == 0:
                tqdm_obj.set_description(
                    "Current loss: {:.4f}  ".format(loss_total.item()))

            if iteration % val_interval == 0:
                val_loss = evaluate_validation_loss(validation_loader, teacher, student, autoencoder, teacher_mean, teacher_std, device, cfg)
                print(f"Validation Loss: {val_loss:.4f}")

                torch.save(teacher.state_dict(), os.path.join(train_output_dir, 'teacher_state_temp.pth'))
                torch.save(student.state_dict(), os.path.join(train_output_dir, 'student_state_temp.pth'))
                torch.save(autoencoder.state_dict(), os.path.join(train_output_dir, 'autoencoder_state_temp.pth'))

                q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
                    validation_loader=validation_loader, teacher=teacher,
                    student=student, autoencoder=autoencoder,
                    teacher_mean=teacher_mean, teacher_std=teacher_std, st_para=cfg.map_st, ae_para=cfg.map_ae,out_channels=cfg.out_channels, gpu_id_no=cfg.gpu_id,
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

                # --- メモリ解放 ---
                del image_st, image_ae, image_penalty
                del teacher_output_ae, student_output_ae, ae_output
                del distance_ae, distance_stae, loss_ae, loss_stae, loss_total
                torch.cuda.empty_cache()

    teacher.eval()
    student.eval()
    autoencoder.eval()

    # bestモデルのパラメータ読み込み
    teacher.load_state_dict(torch.load(os.path.join(train_output_dir, 'teacher_state_best.pth')))
    student.load_state_dict(torch.load(os.path.join(train_output_dir, 'student_state_best.pth')))
    autoencoder.load_state_dict(torch.load(os.path.join(train_output_dir, 'autoencoder_state_best.pth')))

    para_file = os.path.join(train_output_dir, 'para.json')
    fix_json_file(para_file) #jsonファイルの自動修正
    para_json = open(para_file, 'r')
    para_dict = json.load(para_json)
    teacher_mean = torch.tensor(para_dict['teacher_mean'])[0].to(device)
    teacher_std = torch.tensor(para_dict['teacher_std'])[0].to(device)
    q_st_start = torch.tensor(para_dict['q_st_start']).to(device)
    q_st_end = torch.tensor(para_dict['q_st_end']).to(device)
    q_ae_start = torch.tensor(para_dict['q_ae_start']).to(device)
    q_ae_end = torch.tensor(para_dict['q_ae_end']).to(device)

    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean, st_para=cfg.map_st, ae_para=cfg.map_ae,
        teacher_std=teacher_std,  out_channels=cfg.out_channels, gpu_id_no=cfg.gpu_id, desc='Final map normalization')

    st_para = cfg.map_st
    ae_para = cfg.map_ae
    best_thresholds = computing_best_thresholds(validation_loader, teacher, student, autoencoder, teacher_mean, teacher_std, st_para, ae_para, cfg.out_channels,
        q_st_start, q_st_end, q_ae_start, q_ae_end, gpu_id)

    print(f"{cfg.mode} の閾値：{best_thresholds}" )

    if cfg.mode == "color":
        config_path = "./conf/threshold_color.yaml"

    elif cfg.mode == "monochro":
        config_path = "./conf/threshold_monochro.yaml"

    lock_path = config_path + ".lock"

    with FileLock(lock_path):
        cfg_th = OmegaConf.load(config_path)
        # 設定値を変更
        cfg_th.threshold = float(best_thresholds)
        # 保存
        OmegaConf.save(cfg_th, config_path)

@torch.no_grad()
def evaluate_validation_loss(validation_loader, teacher, student, autoencoder, teacher_mean, teacher_std, device, cfg):
    teacher.eval()
    student.eval()
    autoencoder.eval()

    total_loss = 0
    count = 0
    for image, _ in validation_loader:
        image = image.to(device)
        teacher_output = teacher(image)
        teacher_output = (teacher_output - teacher_mean) / teacher_std
        student_output = student(image)
        ae_output = autoencoder(image)

        loss_st = torch.mean((teacher_output - student_output[:, :384]) ** 2)
        loss_ae = torch.mean((teacher_output - ae_output) ** 2)
        loss_stae = torch.mean((ae_output - student_output[:, 384:]) ** 2)

        loss_total = cfg.loss_st * loss_st + cfg.loss_ae * loss_ae + cfg.loss_stae * loss_stae
        total_loss += loss_total.item()
        count += 1

    teacher.train()
    student.train()
    autoencoder.train()
    return total_loss / count

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std, st_para, ae_para, out_channels,
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
        if q_st_end - q_st_start == 0:
            map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start + 1e-6)
        else:
            map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)

    if q_ae_start is not None:
        if q_ae_end - q_ae_start==0:
            map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start + 1e-6)
        else:
            map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = st_para * map_st + ae_para * map_ae

    return map_combined, map_st, map_ae

def computing_best_thresholds(validation_loader, teacher, student, autoencoder, teacher_mean, teacher_std, st_para, ae_para, out_channels,
        q_st_start, q_st_end, q_ae_start, q_ae_end, gpu_id,
        desc='computing best thresholds', percentile_value=99):

    y_score = []

    # デバイスを作成
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        for image, _ in validation_loader:
            orig_width = 512
            orig_height = 256
            if on_gpu:
                image = image.to(device)

            map_combined, map_st, map_ae = predict(image, teacher, student, autoencoder, teacher_mean, teacher_std, st_para=st_para, ae_para=ae_para,out_channels=out_channels,
            q_st_start=q_st_start, q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end)

            map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
            map_combined = torch.nn.functional.interpolate(
                map_combined, (orig_height, orig_width), mode='bilinear')
            map_combined = map_combined[0, 0].cpu().numpy()
            y_score_image = np.max(map_combined)

            y_score.append(y_score_image)

    best_thresholds = np.percentile(y_score, percentile_value)

    return best_thresholds

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                    teacher_mean, teacher_std, st_para, ae_para, out_channels, gpu_id_no, desc='Map normalization'):
    on_gpu = torch.cuda.is_available()
    maps_st = []
    maps_ae = []
    # 使用したいGPUの番号を指定
    gpu_id = gpu_id_no
    # デバイスを作成
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.to(device)
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, st_para=st_para, ae_para=ae_para, out_channels=out_channels)
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

    # 使用したいGPUの番号を指定
    on_gpu = torch.cuda.is_available()
    gpu_id = 0
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

def output_score_feature(test_set, teacher, student, autoencoder, teacher_mean, teacher_std, st_para, ae_para, out_channels,
        q_st_start, q_st_end, q_ae_start, q_ae_end, gpu_id_no, test_output_dir=None, test_output_heatmap_dir=None,
        desc='Running inference'):
    features_list = []
    score_list = []
    path_list=[]

    val_default_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    on_gpu = torch.cuda.is_available()

    # 使用したいGPUの番号を指定
    gpu_id = gpu_id_no
    # デバイスを作成
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for image, target, path in tqdm(test_set, desc=desc):
            orig_width = image.width
            orig_height = image.height
            image = val_default_transform(image)
            image = image[None]
            if on_gpu:
                image = image.to(device)

            #処理速度計測に必要な関数
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            map_combined, map_st, map_ae = predict(
                image=image, teacher=teacher, student=student,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std, st_para=st_para, ae_para=ae_para, out_channels=out_channels, q_st_start=q_st_start, q_st_end=q_st_end,
                q_ae_start=q_ae_start, q_ae_end=q_ae_end)
            map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
            map_combined = torch.nn.functional.interpolate(
                map_combined, (orig_height, orig_width), mode='bilinear')
            map_feature = map_combined[0,0]
            map_feature = map_feature.view(map_feature.size(0), -1).cpu().numpy()
            map_combined = map_combined[0, 0].cpu().numpy()
            score = np.max(map_combined)

            features_list.append(map_feature)
            score_list.append(score)
            path_list.append(path)

    return features_list, score_list, path_list

def fix_json_file(path):
    with open(path, 'r') as f:
        content = f.read()
    # 末尾に } が多い場合、1個ずつ削る
    while True:
        try:
            json.loads(content)
            break
        except json.JSONDecodeError:
            content = content[:-1]
    with open(path, 'w') as f:
        f.write(content)

class Trainer:
    def __init__(self, cfg, mode: str, gpu_id: int):
        self.cfg = cfg
        self.mode = mode
        self.gpu_id = gpu_id

    def run(self):
        """
        学習の実行
        """
        self.cfg.mode = self.mode
        self.cfg.gpu_id = self.gpu_id
        device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
        train(self.cfg)  # train関数はcfgを使って学習を実行

class TrainingPipeline:
    def __init__(self, cfg):
        self.cfg = cfg

    def execute(self):

        p1 = multiprocessing.Process(target=partial(Trainer(self.cfg, "monochro", 0).run))
        p2 = multiprocessing.Process(target=partial(Trainer(self.cfg, "color", 0).run))
        p1.start()
        p2.start()
        p1.join()
        p2.join()

if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("./conf/config.yaml")
    pipeline = TrainingPipeline(cfg)
    pipeline.execute()
