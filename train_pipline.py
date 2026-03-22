import multiprocessing
from omegaconf import OmegaConf
import torch
from train_func_monochro_old import train_monochro
from train_func_color import train_color
from model_exporter import ModelExporter
from model_handler import ONNXModelHandler


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
        if self.mode == "monochro":
            print(f"🟢 モノクロAIの学習を開始します... (GPU ID: {self.gpu_id})")
            train_monochro(self.cfg)
        elif self.mode == "color":
            print(f"🔵 カラーAIの学習を開始します... (GPU ID: {self.gpu_id})")
            train_color(self.cfg)
        else:
            print("⚠️ 不明なモードです。")

def run_trainer(cfg, mode, gpu_id):
    trainer = Trainer(cfg, mode, gpu_id)
    trainer.run()

class TrainingPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_handler = ONNXModelHandler(self.cfg)

    def execute(self):
        print("🚀 学習パイプラインを開始します...")

        from functools import partial

        p1 = multiprocessing.Process(target=partial(Trainer(self.cfg, "monochro", 0).run))
        p2 = multiprocessing.Process(target=partial(Trainer(self.cfg, "color", 0).run))
        p1.start()
        p2.start()
        p1.join()
        p2.join()

        # ONNXモデルのエクスポート
        for mode in ["monochro", "color"]:
            self.cfg.mode = mode
            exporter = ModelExporter(self.cfg)
            exporter.export_onnx()


if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("./conf/config.yaml")
    pipeline = TrainingPipeline(cfg)
    pipeline.execute()

