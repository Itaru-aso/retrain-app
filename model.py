import torch
import json
import os
import sys
import numpy as np
import torch.nn as nn
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common import get_autoencoder, get_pdn_small

class EfficientADFullModel(torch.nn.Module):
    def __init__(self, mode, height, width, teacher, student, autoencoder,
                teacher_mean, teacher_std,
                st_para, ae_para,
                q_st_start=None, q_st_end=None,
                q_ae_start=None, q_ae_end=None):
        super().__init__()

        self.mode = mode
        
        self.height = height
        self.width = width

        self.teacher = teacher
        self.student = student
        self.autoencoder = autoencoder

        if self.mode == "monochro":
            #self.register_buffer("mean", torch.tensor([0.518, 0.518, 0.518]).view(1, 3, 1, 1))
            #self.register_buffer("std", torch.tensor([0.178, 0.178, 0.178]).view(1, 3, 1, 1))

            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        elif self.mode == "color":
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        else:
            raise ValueError("Invalid mode. Choose monochro or color")

        # パラメータを登録
        self.register_buffer("teacher_mean", teacher_mean.view(1, -1, 1, 1))
        self.register_buffer("teacher_std", teacher_std.view(1, -1, 1, 1))

        self.register_buffer("st_para", torch.tensor(st_para))
        self.register_buffer("ae_para", torch.tensor(ae_para))

        if q_st_start is not None:
            self.register_buffer("q_st_start", q_st_start)
            self.register_buffer("q_st_end", q_st_end)
        else:
            self.q_st_start = self.q_st_end = None

        if q_ae_start is not None:
            self.register_buffer("q_ae_start", q_ae_start)
            self.register_buffer("q_ae_end", q_ae_end)
        else:
            self.q_ae_start = self.q_ae_end = None

    def forward(self, x):
        out_channels = 384

        x = x / 255.0  # 0から1の範囲に正規化
        x = (x - self.mean) / self.std  # 標

        # モデル推論
        teacher_output = self.teacher(x)
        teacher_output = (teacher_output - self.teacher_mean) / self.teacher_std

        student_output = self.student(x)
        autoencoder_output = self.autoencoder(x)

        # map_stとmap_aeの計算
        map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2, dim=1, keepdim=True)
        map_ae = torch.mean((autoencoder_output - student_output[:, out_channels:])**2, dim=1, keepdim=True)

        # スケーリング
        if self.q_st_start is not None:
            map_st = 0.1 * (map_st - self.q_st_start) / (self.q_st_end - self.q_st_start)
        if self.q_ae_start is not None:
            map_ae = 0.1 * (map_ae - self.q_ae_start) / (self.q_ae_end - self.q_ae_start)

        # 合成
        map_combined = self.st_para * map_st + self.ae_para * map_ae

        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(map_combined, (self.height, self.width), mode='bilinear')
        #map_combined = map_combined[0, 0].cpu()

        #output = np.max(map_combined)
        #output = torch.amax(map_combined, dim=(2, 3))
        output = torch.max(torch.max(map_combined, dim=3)[0], dim=2)[0]

        #print(f"Anomaly score : {score:.4f}")

        return output

def test_efficientad_full_model(color_num, mode, image_path):
    out_channels = 384
    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    map_st = 0.9
    map_ae = 0.2

    # モデルの構築
    teacher = get_pdn_small(out_channels)
    student = get_pdn_small(2 * out_channels)
    autoencoder = get_autoencoder(out_channels)

    model_dir = r"C:\Fastenerlnsp\retrain_app\onnx_framework\model"

    input_dir = os.path.join(model_dir, "input", mode, color_num)
    output_dir = os.path.join(model_dir, "output", mode, color_num)

    # モデルの読み込み
    teacher.load_state_dict(torch.load(os.path.join(input_dir, 'teacher_state_best.pth'), map_location=device))
    student.load_state_dict(torch.load(os.path.join(input_dir, 'student_state_best.pth'), map_location=device))
    autoencoder.load_state_dict(torch.load(os.path.join(input_dir, 'autoencoder_state_best.pth'), map_location=device))

    para_dict = json.load(open(os.path.join(input_dir, 'para.json'), 'r'))
    teacher_mean = torch.tensor(para_dict['teacher_mean'])[0].to(device)
    teacher_std = torch.tensor(para_dict['teacher_std'])[0].to(device)
    #q_st_start = torch.tensor(para_dict['q_st_start'])
    #q_st_end = torch.tensor(para_dict['q_st_end'])
    #q_ae_start = torch.tensor(para_dict['q_ae_start'])
    #q_ae_end = torch.tensor(para_dict['q_ae_end'])

    # 画像読み込みと前処理（ToTensorのみ）
    from PIL import Image

    # 入力画像を [1, 3, H, W] の float32 (0〜255) に変換
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image, dtype=np.float32).transpose(2, 0, 1)[np.newaxis, ...]

    # NumPy配列 → PyTorch Tensor に変換
    image_tensor = torch.from_numpy(image_np).to(torch.float32)
    # GPU使用時は .to(device) を追加
    image_tensor = image_tensor.to(device)

    width, height = image.size
    # モデル統合
    model = EfficientADFullModel(
        height, width, teacher, student, autoencoder,
        teacher_mean, teacher_std,
        st_para=map_st, ae_para=map_ae,
        q_st_start=None, q_st_end=None,
        q_ae_start=None, q_ae_end=None
    ).to(device)
    model.eval()

    # 推論
    with torch.no_grad():
        score = model(image_tensor)

    print(f"Anomaly score: {score}")

def main():
    image_path = r"C:\Fastenerlnsp\retrain_app\onnx_framework\OK_image_111500_048_1.bmp"
    test_efficientad_full_model("316", "color", image_path)

if __name__ == "__main__":
    main()
