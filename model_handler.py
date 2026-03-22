import mlflow
import mlflow.onnx
import onnx
import os
import numpy as np
from omegaconf import OmegaConf
import onnxruntime as ort
from PIL import Image

from mlflow.tracking import MlflowClient


class ONNXModelHandler:
    def __init__(self, cfg):
        #self.model_path = model_path
        self.model = None
        self.cfg = cfg

    def load_model(self, model_path):
        self.model = onnx.load(model_path)
        onnx.checker.check_model(self.model)
        print(f"✅ ONNXモデル検証完了")

    def register_model(self, model_name):
        with mlflow.start_run():
            mlflow.set_tags({
                "project": "shisui",
                "architecture": "EfficientAD",
                "color": self.cfg.target_color,
                "mode": self.cfg.mode,
            })
            mlflow.onnx.log_model(onnx_model=self.model, artifact_path="onnx_model")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/onnx_model"
            mlflow.register_model(model_uri=model_uri, name=model_name)
            print(f"✅ モデル登録完了")

            # モデルにタグを追加
            client = MlflowClient()
            client.set_registered_model_tag(model_name, "project", "shisui")
            client.set_registered_model_tag(model_name, "architecture", "EfficientAD")
            client.set_registered_model_tag(model_name, "color", self.cfg.target_color)
            client.set_registered_model_tag(model_name, "mode", self.cfg.mode)


class ONNXModelValidator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        #image = image.resize((224, 224))  # モデルの入力サイズに合わせて調整
        image_array = np.array(image).astype(np.float32) # 画像をNumPy配列に変換
        image_array = np.transpose(image_array, (2, 0, 1))  # CHW形式に変換
        image_array = np.expand_dims(image_array, axis=0)  # バッチ次元追加
        return image_array
    def validate_directory(self, data_dir):
        with mlflow.start_run():
            score_list = []
            for filename in os.listdir(data_dir):
                if filename.lower().endswith(".bmp"):
                    file_path = os.path.join(data_dir, filename)
                    try:
                        input_data = self.preprocess_image(file_path)
                        result = self.session.run([self.output_name], {self.input_name: input_data})
                        anomaly_map = result[0].flatten()
                        anomaly_score = anomaly_map.max()
                        score_list.append(anomaly_score)
                        print(f"{file_path} の異常度: {anomaly_score}")

                    except Exception as e:
                        print(f"{filename} の検証中にエラーが発生しました: {e}")

            last_part = os.path.basename(data_directory)

            mlflow.log_metric(f"{last_part}_output_mean", np.mean(score_list))
            mlflow.log_metric(f"{last_part}_output_std", np.std(score_list))
            print(f"{filename} の検証結果: 平均={np.mean(score_list)}, 標準偏差={np.std(score_list)}")


if __name__ == '__main__':

    cfg = OmegaConf.load("./conf/config.yaml")
    model_file_name = f"{cfg.mode}_model.onnx"

    # 使用例
    model_path = os.path.join(cfg.model_dir, cfg.target_color, cfg.mode, model_file_name)  # 実際のモデルパスに置き換えてください
    model_name = f'EfficientAD_color_no_{cfg.target_color}_mode_{cfg.mode}'

    # モデル登録
    handler = ONNXModelHandler(cfg)
    handler.load_model(model_path)
    handler.register_model(model_name)

    data_directory = os.path.join(cfg.dataset_path, cfg.target_color, cfg.mode, "test", "ng")    # .bmp画像が格納された検証ディレクトリ

    validator = ONNXModelValidator(model_path)
    validator.validate_directory(data_directory)

