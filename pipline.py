import os
import shutil
import datetime
import multiprocessing
from omegaconf import OmegaConf
from ftplib import FTP
import torch
import cv2
from utils.common import check_json
from utils.ftp_common import upload_file_to_ftp, download_ftp_selected, is_directory
from utils.image_preprocessing import load_image_as_byte_array, process_image
from train_func_monochro import train_monochro
from train_func_color import train_color
from model_exporter import ModelExporter
from model_handler import ONNXModelHandler

class DatasetManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.target_color = cfg.target_color
        self.image_size_width = cfg.image_size.width
        self.image_size_height = cfg.image_size.height

        # モードごとのデータセットパス
        self.mode_paths = {
            "monochro": os.path.join(self.cfg.dataset_path, self.target_color, "monochro", "train"),
            "color": os.path.join(self.cfg.dataset_path, self.target_color, "color", "train")
        }

        # モードごとのモデル保存パス
        self.model_paths = {
            "monochro": os.path.join(self.cfg.model_dir, self.target_color, "monochro"),
            "color": os.path.join(self.cfg.model_dir, self.target_color, "color")
        }

        # モードごとのアノテーションデータ保存パス
        self.download_paths = {
            "monochro": os.path.join(self.cfg.download_dir, "monochro"),
            "color": os.path.join(self.cfg.download_dir, "color")
        }

    def _backup(self, source_paths, backup_root, subfolder, color_folder=True):
        """
        共通のバックアップ処理
        source_paths: モードごとのコピー元パス辞書
        backup_root: バックアップのルートディレクトリ
        subfolder: コピー先の末尾パス（例："train"）
        color_folder:色番のフォルダを作成するかどうか
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if color_folder:
            backup_base_dir = os.path.join(backup_root, self.target_color, timestamp)
        else:
            backup_base_dir = os.path.join(backup_root, timestamp)

        for mode, src_dir in source_paths.items():
            dst_dir = os.path.join(backup_base_dir, mode, subfolder)
            if os.path.exists(src_dir):
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                print(f"Copied from {src_dir} to {dst_dir}")
            else:
                print(f"Source directory {src_dir} does not exist.")

    def backup_dataset(self):
        """学習データのバックアップ作成"""
        self._backup(self.mode_paths, os.path.join(self.cfg.backup_dir,"dataset"),  "train")

    def backup_model(self):
        """モデルファイルのバックアップ作成"""
        self._backup(self.model_paths, os.path.join(self.cfg.backup_dir,"model"), "")

    def backup_annotated_data(self):
        """アノテーションデータのバックアップ作成"""
        self._backup(self.download_paths, os.path.join(self.cfg.backup_dir,"download", self.cfg.mode), "", color_folder=False)


    def process_annotated_images(self):
        """
        ダウンロードしたアノテーション画像セットを前処理して学習データセットに追加
        """

        for mode in ["monochro", "color"]:
            # 出力先ディレクトリの準備
            annotated_dataset_dir = os.path.join(self.mode_paths[mode], "annotated", "good")
            if os.path.exists(annotated_dataset_dir):
                shutil.rmtree(annotated_dataset_dir)
            os.makedirs(annotated_dataset_dir)

            # アノテーション画像のベースディレクトリパス
            #base_dir = os.path.join(self.cfg.download_dir, mode, "annotated_data", self.target_color)
            base_dir = os.path.join(self.cfg.download_dir, mode, self.target_color)

            # "good" を含むすべてのファイルパスを取得
            all_file_paths = []
            for root, dirs, files in os.walk(base_dir):
                if "good" in root.split(os.path.sep):
                    for file in files:
                        all_file_paths.append(os.path.join(root, file))
                elif "auto_good" in root.split(os.path.sep):
                    for file in files:
                        all_file_paths.append(os.path.join(root, file))

            # アノテーション画像の前処理と学習データセットへの保存
            for image_path in all_file_paths:
                image_data = load_image_as_byte_array(image_path)
                top, bottom, _ = process_image(image_data, self.image_size_width, self.image_size_height, mode)

                name, ext = os.path.splitext(os.path.basename(image_path))
                cv2.imwrite(os.path.join(annotated_dataset_dir, f"{name}_0{ext}"), top)
                cv2.imwrite(os.path.join(annotated_dataset_dir, f"{name}_1{ext}"), bottom)

                #file_path =os.path.join(annotated_dataset_dir, f"{name}_0{ext}")

            print(f"✅ {mode} モードのアノテーション画像を前処理して学習データに追加しました。")


class FTPManager:
    def __init__(self, cfg, host_config):
        self.cfg = cfg
        self.name = host_config.name
        self.host = host_config.host
        self.username = host_config.username
        self.password = host_config.password
        self.monochro_port = host_config.monochro_port
        self.color_port = host_config.color_port
        self.model_port = host_config.model_port
        self.start_path = cfg.ftp_common.start_path
        self.local_root = cfg.ftp_common.local_root

    def download_images(self):
        """
        検査PCにFTP接続してアノテーションされた画像セットをダウンロード
        """
        if self.cfg.mode == 'monochro':
            start_path = "/camera1_image/annotated_data"
            port = self.monochro_port
            local_root = os.path.join(self.cfg.download_dir, "monochro") # モノクロ画像の保存先ディレクトリパス
            local_root_num = os.path.join(self.cfg.download_dir, "monochro", self.cfg.target_color)

        elif self.cfg.mode == 'color':
            start_path = "/camera2_image/annotated_data"
            port = self.color_port
            local_root = os.path.join(self.cfg.download_dir, "color") # モノクロ画像の保存先ディレクトリパス
            local_root_num = os.path.join(self.cfg.download_dir, "color", self.cfg.target_color)

        ftp=FTP()
        ftp.connect(self.host, port, timeout=10)
        ftp.login(user=self.username, passwd=self.password)

        # 対象色だけダウンロードする、MDTM が取れないサーバ前提なら size_only=True にすると安全
        download_ftp_selected(
            ftp=ftp,
            remote_root=start_path,
            local_root=local_root,
            allowed_top_levels=[self.cfg.target_color],
            is_dir_func=is_directory,  # あなたの LIST ベース判定
            size_only=False            # MDTM を使って時刻比較したい場合は False（未対応なら自動フォールバック）
        )

    def upload_onnx_model(self):
        """
        再学習モデルのデプロイ
        """
        model_file_name = f"{self.cfg.target_color}_{self.cfg.mode}_model.onnx"
        upload_path = os.path.join("./")
        port = self.model_port
        if self.cfg.mode == 'monochro':
            onnx_file_path = os.path.join(self.cfg.model_dir, self.cfg.target_color, self.cfg.mode, model_file_name)  # ONNXモデルのパス

        elif self.cfg.mode == 'color':
            onnx_file_path = os.path.join(self.cfg.model_dir, self.cfg.target_color, self.cfg.mode, model_file_name)

        upload_file_to_ftp(
            host=self.host,
            port=port,
            username=self.username,
            password=self.password,
            local_file_path=onnx_file_path,  # アップロードしたいファイル
            remote_folder=upload_path  # アップロード先のリモートフォルダ
        )

class MultiFTPManager:
    """複数検査PCへの一括FTP操作"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.managers = [
            FTPManager(cfg, host_cfg)
            for host_cfg in cfg.ftp_hosts
        ]

    def download_images(self):
        # ループ前にダウンロード先を1回だけクリア（複数PCのデータをマージするため）
        for mode_name in ["monochro", "color"]:
            if self.cfg.mode == mode_name:
                local_root_num = os.path.join(self.cfg.download_dir, mode_name, self.cfg.target_color)
                shutil.rmtree(local_root_num, ignore_errors=True)

        for mgr in self.managers:
            try:
                print(f"📥 [{mgr.name}] からダウンロード中...")
                mgr.download_images()
                print(f"✅ [{mgr.name}] ダウンロード完了")
            except Exception as e:
                print(f"⚠ [{mgr.name}] からのダウンロード失敗（スキップ）: {e}")

    def upload_onnx_model(self):
        for mgr in self.managers:
            try:
                print(f"📤 [{mgr.name}] へアップロード中...")
                mgr.upload_onnx_model()
                print(f"✅ [{mgr.name}] アップロード完了")
            except Exception as e:
                print(f"⚠ [{mgr.name}] へのアップロード失敗（スキップ）: {e}")

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
            print(f"🟢 モノクロAIの学習を開始します... (GPU ID: {self.gpu_id}, color: {self.cfg.target_color})")
            train_monochro(self.cfg)
        elif self.mode == "color":
            print(f"🔵 カラーAIの学習を開始します... (GPU ID: {self.gpu_id}, color: {self.cfg.target_color})")
            train_color(self.cfg)
        else:
            print("⚠️ 不明なモードです。")

def run_trainer(cfg, mode, gpu_id):
    trainer = Trainer(cfg, mode, gpu_id)
    trainer.run()

class TrainingPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_manager = DatasetManager(self.cfg)
        self.ftp_manager = MultiFTPManager(self.cfg)
        self.model_handler = ONNXModelHandler(self.cfg)

    def execute(self):
        print("🚀 学習パイプラインを開始します...")
        print(f"📥 バックアップ作成中...")
        #self.dataset_manager.backup_dataset()
        self.dataset_manager.backup_model()
        #self.dataset_manager.backup_annotated_data()
        print(f"✅ バックアップ完了")

        # アノテーション画像のダウンロード
        for mode in ["monochro", "color"]:
            self.cfg.mode = mode
            self.ftp_manager.download_images()

        self.dataset_manager.process_annotated_images()
        from functools import partial

        p1 = multiprocessing.Process(target=partial(Trainer(self.cfg, "monochro", 0).run))
        p2 = multiprocessing.Process(target=partial(Trainer(self.cfg, "color", 0).run))
        p1.start()
        p2.start()
        p1.join()
        p2.join()

        # ONNXモデルのエクスポート
        for mode in ["monochro", "color"]:
            if mode == 'monochro':
                input_dir = os.path.join(self.cfg.model_dir, self.cfg.target_color, mode)
            elif mode == 'color':
                input_dir = os.path.join(self.cfg.model_dir, self.cfg.target_color, mode)
            check_json(input_dir)
            self.cfg.mode = mode
            exporter = ModelExporter(self.cfg)
            exporter.export_onnx()
            self.ftp_manager.upload_onnx_model()

            """Mlflowモデルの登録"""
            #model_path = os.path.join(self.cfg.model_dir, self.cfg.target_color, self.cfg.mode, f"{self.cfg.target_color}_{mode}_model.onnx")
            #model_name = f'EfficientAD_color_no_{self.cfg.target_color}_mode_{self.cfg.mode}'
            #self.model_handler.load_model(model_path)
            #self.model_handler.register_model(model_name)

            """ONNXモデルの検証, デプロイ判断
            validator = ONNXModelValidator(model_path)
            data_directory = os.path.join(cfg.dataset_path, cfg.target_color, cfg.mode, "test", "ng")    # .bmp画像が格納された検証ディレクトリ

            validator.validate_directory(data_directory)
            """

if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("./conf/config.yaml")
    pipeline = TrainingPipeline(cfg)
    pipeline.execute()

