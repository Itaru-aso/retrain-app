import torch
import json
import os
import re
import torch.nn as nn
import torch.nn.functional as F
from utils.common import get_autoencoder_256_512, get_pdn_small
from omegaconf import OmegaConf
from model import EfficientADFullModel

class ModelExporter:
    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg.mode == 'monochro':
            self.input_dir = os.path.join(self.cfg.model_dir, self.cfg.target_color, cfg.mode)
        elif self.cfg.mode == 'color':
            self.input_dir = os.path.join(self.cfg.model_dir, self.cfg.target_color, cfg.mode)

        self.height = self.cfg.image_size.height
        self.width = self.cfg.image_size.width

        self.map_st = self.cfg.map_st
        self.map_ae = self.cfg.map_ae

        self.gpu_id = self.cfg.gpu_id
        self.device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')

        self.mode = self.cfg.mode

    def load_models(self):
        teacher_model = get_pdn_small(self.cfg.out_channels)
        student_model = get_pdn_small(2 * self.cfg.out_channels)
        autoencoder_model = get_autoencoder_256_512()

        teacher_model.load_state_dict(torch.load(os.path.join(self.input_dir, 'teacher_state_best.pth'), map_location=self.device))
        student_model.load_state_dict(torch.load(os.path.join(self.input_dir, 'student_state_best.pth'), map_location=self.device))
        autoencoder_model.load_state_dict(torch.load(os.path.join(self.input_dir, 'autoencoder_state_best.pth'), map_location=self.device))

        teacher_model.eval()
        student_model.eval()
        autoencoder_model.eval()

        return teacher_model, student_model, autoencoder_model

    def load_parameters(self):
        para_path = os.path.join(self.input_dir, 'para.json')

        # 1. ファイルを読み込み
        with open(para_path, 'r', encoding='utf-8') as f:
            raw = f.read()

        # 2. バランス確認して末尾の余分な } を削除
        opens = raw.count('{')
        closes = raw.count('}')
        to_trim = max(0, closes - opens)
        fixed = raw
        if to_trim:
            # 末尾の } を必要数だけ削除（空白含む）
            fixed = re.sub(r"\s*}\s*$", "", fixed, count=to_trim)

        opens = raw.count('[')
        closes = raw.count(']')
        to_trim = max(0, closes - opens)
        fixed = raw
        if to_trim:
            # 末尾の } を必要数だけ削除（空白含む）
            fixed = re.sub(r"\s*]\s*$", "", fixed, count=to_trim)

        # 3. バリデーション（JSONとして読み込めるか確認）
        try:
            para_dict = json.loads(fixed)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON修正後も読み込み失敗: {e}")

        # 4. 修正した場合は上書き保存
        if fixed != raw:
            with open(para_path, 'w', encoding='utf-8') as f:
                json.dump(para_dict, f, indent=4, ensure_ascii=False)

        # 5. Tensor化して返却
        return (
            torch.tensor(para_dict['teacher_mean'])[0].to(self.device),
            torch.tensor(para_dict['teacher_std'])[0].to(self.device),
            torch.tensor(para_dict['q_st_start']),
            torch.tensor(para_dict['q_st_end']),
            torch.tensor(para_dict['q_ae_start']),
            torch.tensor(para_dict['q_ae_end']),
        )

    def export_onnx(self):
        teacher_model, student_model, autoencoder_model = self.load_models()
        teacher_mean, teacher_std, q_st_start, q_st_end, q_ae_start, q_ae_end = self.load_parameters()

        
        if self.mode == "monochro" or self.mode == "color":
            # 統合モデル
            model = EfficientADFullModel(
                self.mode, self.height, self.width, teacher_model, student_model, autoencoder_model,
                teacher_mean, teacher_std,
                st_para=self.map_st, ae_para=self.map_ae,
                q_st_start=q_st_start, q_st_end=q_st_end,
                q_ae_start=q_ae_start, q_ae_end=q_ae_end
            ).to(self.device)
        """
        if self.mode == "monochro":
            # 統合モデル
            model = EfficientADFullModel(
                self.height, self.width, teacher_model, student_model, autoencoder_model,
                teacher_mean, teacher_std,
                st_para=self.map_st, ae_para=self.map_ae,
                q_st_start=None, q_st_end=None,
                q_ae_start=None, q_ae_end=None
            ).to(self.device)
        """

        model.eval()

        dummy_input = torch.randn(1, 3, self.height, self.width).to(self.device)*255
        #scripted_model = torch.jit.script(model)

        onnx_path = os.path.join(self.input_dir, f"{self.cfg.target_color}_{self.cfg.mode}_model.onnx")
        torch.onnx.export(
            model, dummy_input, onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        #print(f"Exported ONNX model to {onnx_path}")

if __name__ == '__main__':
    cfg = OmegaConf.load("./conf/config.yaml")
    exporter = ModelExporter(cfg)
    exporter.export_onnx()
