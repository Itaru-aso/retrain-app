import os
import torch
import cv2
import shutil
import datetime
import multiprocessing
import json
from omegaconf import OmegaConf
from ftplib import FTP, error_perm

from train_model import train, output_score_feature
from utils.common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from utils.ftp_common import connect_and_download_tree, upload_file_to_ftp
from utils.image_preprocessing import load_image_as_byte_array, process_image

def run_train(mode:str, gpu_id:int) -> None:
    """訓練を実行する関数

    Args:
        mode (str): "monochro" または "color" のいずれかを指定
        gpu_id (int): 使用するGPUのID。0または1を指定することを想定しています。
    """
    cfg = OmegaConf.load("./conf/config.yaml")
    cfg.mode = mode
    cfg.gpu_id = gpu_id # GPU IDをcfgに追加しておくと便利


    # GPU設定
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # 学習関数を呼び出す
    train(cfg)

def process_and_save_images(file_paths:list[str], output_dir:str, image_size_width:int, image_size_height:int, mode:str) -> None:
    """画像を読み込み、前処理し、保存

    Args:
        file_paths (List): 検査PCからダウンロードした良品のアノテーション画像のファイルパスリスト
        output_dir (str): 前処理後の画像を保存する学習データセットのディレクトリパス
        image_size_width (int): リサイズ後の画像の幅
        image_size_height (int): リサイズ後の画像の高さ
        mode (str): "monochro" または "color" のいずれかを指定
    """
    for image_path in file_paths:
        image_data = load_image_as_byte_array(image_path)
        top, bottom, _ = process_image(image_data, image_size_width, image_size_height, mode)

        name, ext = os.path.splitext(os.path.basename(image_path))
        cv2.imwrite(os.path.join(output_dir, f"{name}_0{ext}"), top)
        cv2.imwrite(os.path.join(output_dir, f"{name}_1{ext}"), bottom)


if __name__ == '__main__':

    cfg = OmegaConf.load("./conf/config.yaml")

    target_color = cfg.target_color  # 実際の実行日を設定
    monochro_dataset_dir = os.path.join(cfg.dataset_path.monochro, "train", target_color, 'original', 'good') #モノクロ画像の学習データセットのパス
    color_dataset_dir = os.path.join(cfg.dataset_path.color, "train", target_color, 'original', 'good') #カラー画像の学習データセットのパス

    image_size_width = cfg.image_size.width  # リサイズ後の画像の幅
    image_size_height = cfg.image_size.height  # リサイズ後の画像の高さ

    """
    学習データのバックアップ作成
    """
    # 既存学習データフォルダをコピーしてバックアップ作成

    # バックアップ用のタイムスタンプ付きフォルダパスを定義
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_base_dir = os.path.join("./dataset/backup", target_color, timestamp)

    # コピー先ディレクトリの定義
    backup_dirs = {
        monochro_dataset_dir: os.path.join(backup_base_dir, "monochro", "train", "good"),
        color_dataset_dir: os.path.join(backup_base_dir, "color", "train", "good")
    }

    # コピー処理
    for src_dir, dst_dir in backup_dirs.items():
        os.makedirs(dst_dir, exist_ok=True)
        for filename in os.listdir(src_dir):
            src_file = os.path.join(src_dir, filename)
            dst_file = os.path.join(dst_dir, filename)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)


    """
    #検査PCにFTP接続してアノテーションされた画像セットをダウンロード
    """
    # 検査PCのFTPサーバーに接続してアノテーションされた画像セットをダウンロード
    connect_and_download_tree(
        host="172.28.204.23",
        username="ykk\\s001197",
        password="shisui@02",
        start_path="/camera1_image",
        local_root="./annotated_data"
    )

    """
    #ダウンロードしたアノテーション画像セットを前処理して学習データセットに追加
    """

    mode_paths = {
        "monochro": cfg.dataset_path.monochro,
        "color": cfg.dataset_path.color
    }

    for mode in ["monochro", "color"]:
        # 出力先ディレクトリの準備
        annotated_dataset_dir = os.path.join(mode_paths[mode], "train", target_color, "annotated", "good")
        if os.path.exists(annotated_dataset_dir):
            shutil.rmtree(annotated_dataset_dir)
        os.makedirs(annotated_dataset_dir)

        # アノテーション画像のベースディレクトリ
        base_dir = os.path.join("./annotated_data", mode, target_color)

        # "good" を含むすべてのファイルパスを取得
        all_file_paths = []
        for root, dirs, files in os.walk(base_dir):
            if "good" in root.split(os.path.sep):
                for file in files:
                    all_file_paths.append(os.path.join(root, file))

        # 前処理と保存
        process_and_save_images(all_file_paths, annotated_dataset_dir, image_size_width, image_size_height, mode)

    """
    #学習の実行
    """
    # 並列処理のためのmultiprocessingを使用
    # モノクロとカラーの学習を並列で実行
    p1 = multiprocessing.Process(target=run_train, args=("monochro", 0))
    p2 = multiprocessing.Process(target=run_train, args=("color", 1))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    """
    #再学習モデルの精度確認, デプロイ判断
    """
    """

    """
    #再学習モデルのデプロイ
    """
    # 再学習モデルのモデルファイルを検査PCにFTPでアップロード
    upload_file_to_ftp(
        host="172.28.204.23",
        username="ykk\\s001197",
        password="shisui@02",
        local_file_path="D:/0032011/shisui_project/AI/retrain_app/pretraining/teacher_medium_tmp.pth",  # アップロードしたいファイル
        remote_folder="/camera1_image/uploaded"  # アップロード先のリモートフォルダ
    )

    """

    """
    # bestモデルのパラメータ読み込み
    teacher.load_state_dict(torch.load(os.path.join(train_output_dir, 'teacher_state_best.pth')))
    student.load_state_dict(torch.load(os.path.join(train_output_dir, 'student_state_best.pth')))
    autoencoder.load_state_dict(torch.load(os.path.join(train_output_dir, 'autoencoder_state_best.pth')))

    para_file = os.path.join(train_output_dir, 'para.json')
    para_json = open(para_file, 'r')
    para_dict = json.load(para_json)
    teacher_mean = torch.tensor(para_dict['teacher_mean'])[0].to(device)
    teacher_std = torch.tensor(para_dict['teacher_std'])[0].to(device)
    q_st_start = torch.tensor(para_dict['q_st_start']).to(device)
    q_st_end = torch.tensor(para_dict['q_st_end']).to(device)
    q_ae_start = torch.tensor(para_dict['q_ae_start']).to(device)
    q_ae_end = torch.tensor(para_dict['q_ae_end']).to(device)

    teacher.to(device)
    student.to(device)
    autoencoder.to(device)

    teacher.eval()
    student.eval()
    autoencoder.eval()

    test_set = ImageFolderWithPath(
        os.path.join(cfg.monochro_dataset_path, 'test'))

    st_para = cfg.map_st
    ae_para = cfg.map_ae
    feature_list, score_list, path_list = output_score_feature(test_set, teacher, student, autoencoder, teacher_mean, teacher_std, st_para, ae_para, out_channels,
        q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None, test_output_heatmap_dir=None,
        desc='Running inference')

    # リストをタプルのリストに結合
    combined_list = list(zip(score_list, path_list))

    # スコアで降順に並び替え
    sorted_combined_list = sorted(combined_list, key=lambda x: x[0], reverse=True)

    # 並び替えたタプルのリストを元のリストに戻す
    sorted_score_list, sorted_path_list = zip(*sorted_combined_list)

    # タプルをリストに変換
    sorted_score_list = list(sorted_score_list)
    sorted_path_list = list(sorted_path_list)

    print(path_list)
    print(feature_list[0].shape)
    """
