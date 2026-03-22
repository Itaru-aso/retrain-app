import os
import torch
import pandas as pd
import numpy as np
import tifffile
import cv2
import matplotlib.pyplot as plt
import json
from omegaconf import DictConfig
from utils.common import get_autoencoder_256_512, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader, OpenCVResize
from torch.utils.data import DataLoader
from torchvision import transforms
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support, precision_recall_curve

# constants
seed = 42
random.seed(0)
on_gpu = torch.cuda.is_available()
out_channels = 384

default_transform = transforms.Compose([
    #OpenCVResize(width=512, height=384),#color
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main(cfg: DictConfig):

    dataset_path = os.path.join(cfg.dataset_path, cfg.target_color, cfg.mode)
    train_output_dir = os.path.join(cfg.model_dir, cfg.target_color, cfg.mode)

    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, 'test'))

    teacher = get_pdn_small(out_channels)
    student = get_pdn_small(2 * out_channels)
    autoencoder = get_autoencoder_256_512(out_channels)

    teacher.eval()
    student.eval()
    autoencoder.eval()

    gpu_id = 1
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    # create output dir
    test_output_dir = os.path.join(train_output_dir, 'anomaly_maps')
    test_output_heatmap_dir = os.path.join(train_output_dir, 'heat_maps')

    if os.path.isdir(test_output_dir):
        pass
    else:
        os.makedirs(test_output_dir)
    if os.path.isdir(test_output_heatmap_dir):
        pass
    else:
        os.makedirs(test_output_heatmap_dir)

    # bestモデルのパラメータ読み込み
    teacher.load_state_dict(torch.load(os.path.join(train_output_dir, 'teacher_state_best.pth')))
    student.load_state_dict(torch.load(os.path.join(train_output_dir, 'student_state_best.pth')))
    autoencoder.load_state_dict(torch.load(os.path.join(train_output_dir, 'autoencoder_state_best.pth')))

    para_file = os.path.join(train_output_dir, 'para.json')
    para_json = open(para_file, 'r')
    para_dict = json.load(para_json)
    teacher_mean = torch.tensor(para_dict['teacher_mean'])[0].cuda()
    teacher_std = torch.tensor(para_dict['teacher_std'])[0].cuda()
    q_st_start = torch.tensor(para_dict['q_st_start']).cuda()
    q_st_end = torch.tensor(para_dict['q_st_end']).cuda()
    q_ae_start = torch.tensor(para_dict['q_ae_start']).cuda()
    q_ae_end = torch.tensor(para_dict['q_ae_end']).cuda()

    y_true, y_score, file_list, defect_list, time_list = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, st_para=cfg.map_st, ae_para=cfg.map_ae, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end, test_output_dir=test_output_dir, test_output_heatmap_dir = test_output_heatmap_dir, desc='Final inference')
    #print('Final PR-AUC: {:.4f}'.format(pr_auc))

    output_df = pd.DataFrame({'path': file_list,
                            'defect': defect_list,
                            'y_true': y_true,
                            'score' : y_score,
                            'time' : time_list
                            })

    output_df.to_csv(os.path.join(train_output_dir, 'result_color_ex.csv'))

def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std, st_para, ae_para,
        q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None, test_output_heatmap_dir=None,
        desc='Running inference', use_percentile_threshold=False, percentile_value=99):
    y_true = []
    y_score = []
    y_predict = []
    file_list = []
    defect_list = []
    time_list = []

    # 使用したいGPUの番号を指定
    gpu_id = 1
    # デバイスを作成
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for image, target, path in tqdm(test_set, desc=desc):
            orig_width = image.width
            orig_height = image.height
            image = default_transform(image)
            image = image[None]
            if on_gpu:
                image = image.cuda()

            #処理速度計測に必要な関数
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            map_combined, map_st, map_ae = predict(image, teacher, student, autoencoder, teacher_mean, teacher_std, st_para=st_para, ae_para=ae_para,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None)
            map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
            map_combined = torch.nn.functional.interpolate(
                map_combined, (orig_height, orig_width), mode='bilinear')
            map_combined = map_combined[0, 0].cpu().numpy()
            y_score_image = np.max(map_combined)

            end.record()

            # GPU の処理が終わるのを待つ
            torch.cuda.synchronize()

            # 経過時間をミリ秒で取得
            elapsed_time_ms = start.elapsed_time(end)

            defect_class = os.path.basename(os.path.dirname(path))

            if test_output_dir is not None:
                img_nm = os.path.split(path)[1].split('.')[0]
                if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                    os.makedirs(os.path.join(test_output_dir, defect_class))
                file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
                tifffile.imwrite(file, map_combined)
                img_path = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
                image = cv2.imread(img_path, -1)
                colormap = plt.get_cmap('inferno')
                heatmap = (colormap(image) * 2**16).astype(np.uint16)[:,:,:3]
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                if not os.path.exists(os.path.join(test_output_heatmap_dir, defect_class)):
                    os.makedirs(os.path.join(test_output_heatmap_dir, defect_class))
                save_file = os.path.join(test_output_heatmap_dir, defect_class, img_nm + '.tiff')
                cv2.imwrite(save_file, heatmap)

            y_true_image = 0 if defect_class == 'good' else 1

            y_true.append(y_true_image)
            y_score.append(y_score_image)
            file_list.append(path)
            defect_list.append(defect_class)
            time_list.append(elapsed_time_ms)

    """
    # PR-AUC
    pr_auc = average_precision_score(y_true, y_score)

    # 閾値の決定方法
    if use_percentile_threshold:
        good_scores = [score for score, label in zip(y_score, y_true) if label == 0]
        best_thresholds = np.percentile(good_scores, percentile_value)
    else:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        best_thresholds = thresholds[np.argmax(f1_scores)]

    #最適閾値で異常検出の分類
    y_pred = [1 if score > best_thresholds else 0 for score in y_score]

    # Precision, Recall, F1スコアの計算
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    """

    return y_true, y_score, file_list, defect_list, time_list
    #pr_auc*100, f1, precision, recall, best_thresholds

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std, st_para=None, ae_para=None,
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
    map_combined = st_para * map_st + ae_para * map_ae
    return map_combined, map_st, map_ae

if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("./conf/config.yaml")
    main(cfg)
