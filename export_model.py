import torch
import os
import json
from common import get_pdn_small, get_autoencoder_256_512
from model import EfficientADFullModel  # モデル定義を別ファイルにしている場合

def export_onnx_model(color_num, mode):
    out_channels = 384
    gpu_id = 1
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    map_st = 0.9
    map_ae = 0.2

    # モデルの構築
    teacher = get_pdn_small(out_channels).to(device)
    student = get_pdn_small(2 * out_channels).to(device)
    autoencoder = get_autoencoder_256_512(out_channels).to(device)

    model_dir = 'D:/0032011/shisui_project/AI/EfficientAD/output_onnx_model/model'

    input_dir = os.path.join(model_dir, "input", mode, color_num)
    output_dir = os.path.join(model_dir, "output", mode, color_num)

    # モデルの読み込み
    teacher.load_state_dict(torch.load(os.path.join(input_dir, 'teacher_state_best.pth'), map_location=device))
    student.load_state_dict(torch.load(os.path.join(input_dir, 'student_state_best.pth'), map_location=device))
    autoencoder.load_state_dict(torch.load(os.path.join(input_dir, 'autoencoder_state_best.pth'), map_location=device))

    para_dict = json.load(open(os.path.join(input_dir, 'para.json'), 'r'))
    teacher_mean = torch.tensor(para_dict['teacher_mean'])[0].to(device)
    teacher_std = torch.tensor(para_dict['teacher_std'])[0].to(device)
    q_st_start = torch.tensor(para_dict['q_st_start']).to(device)
    q_st_end = torch.tensor(para_dict['q_st_end']).to(device)
    q_ae_start = torch.tensor(para_dict['q_ae_start']).to(device)
    q_ae_end = torch.tensor(para_dict['q_ae_end']).to(device)

    # 入力サイズ（例: 256x512）
    height, width = 256, 512
    dummy_input = torch.randn(1, 3, height, width).to(device) * 255.0  # uint8画像を模擬

    # 統合モデル
    model = EfficientADFullModel(
        height, width, teacher, student, autoencoder,
        teacher_mean, teacher_std,
        st_para=map_st, ae_para=map_ae,
        q_st_start=None, q_st_end=None,
        q_ae_start=None, q_ae_end=None
    ).to(device)
    model.eval()

    export_path = os.path.join(output_dir, f"{color_num}_{mode}_model.onnx")

    # ONNXエクスポート
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={"input": {0: "batch_size"}, "map_combined": {0: "batch_size"}, "score": {0: "batch_size"}}
    )

    print(f"ONNX model exported to: {export_path}")

if __name__ == "__main__":
    export_onnx_model(color_num="316", mode="color")
