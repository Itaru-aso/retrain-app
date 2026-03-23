# EfficientAD 再学習パイプライン

EfficientADベースの異常検知モデルを再学習するためのパイプラインアプリケーションです。
CustomTkinter GUIから色番を選択して学習を起動し、検査PCとのFTP通信による画像の取得からONNXモデルのデプロイまでを自動化します。

---

## 概要

```
[GUI で色番選択]
      |
      v
[FTP: 検査PCからアノテーション画像をダウンロード]（複数PC対応・失敗PC自動スキップ）
      |
      v
[前処理: クロップ・リサイズ・上下分割 → 学習データセットに追加]
      |
      v
[並列学習: モノクロモデル(GPU:0) / カラーモデル(GPU:0) を multiprocessing で同時実行]
      |
      v
[ONNXエクスポート: Teacher / Student / AutoEncoder を統合してONNXに変換]
      |
      v
[FTP: 検査PCへONNXモデルをアップロード]（複数PC対応・失敗PC自動スキップ）
```

---

## 主要コンポーネント

| ファイル | 役割 |
|---|---|
| `train_app.py` | CustomTkinter GUI。色番選択・実行ボタン・ログ表示 |
| `pipline.py` | メインパイプライン。各処理クラスの統合 |
| `train_func_color.py` | カラーモデルの学習関数 |
| `train_func_monochro.py` | モノクロモデルの学習関数 |
| `model.py` | EfficientADFullModel（推論時の統合モデル） |
| `model_exporter.py` | PyTorchモデルをONNX形式にエクスポート |
| `model_handler.py` | MLflow連携（モデル登録・検証） |
| `conf/config.yaml` | 全設定パラメータ |
| `utils/image_preprocessing.py` | 画像クロップ・リサイズ・上下分割 |
| `utils/ftp_common.py` | FTPダウンロード・アップロード・スキップ判定 |

---

## ディレクトリ構成

```
retrain_app_running/
├── train_app.py             # GUI起動エントリポイント
├── pipline.py               # パイプライン本体
├── train_func_color.py      # カラー学習関数
├── train_func_monochro.py   # モノクロ学習関数
├── model.py                 # EfficientADFullModel
├── model_exporter.py        # ONNXエクスポート
├── model_handler.py         # MLflow連携
├── conf/
│   ├── config.yaml          # メイン設定ファイル
│   ├── threshold_color.yaml # カラー閾値設定
│   └── threshold_monochro.yaml # モノクロ閾値設定
├── utils/
│   ├── common.py            # 共通ユーティリティ（PDN/Autoencoder構築等）
│   ├── ftp_common.py        # FTP操作ユーティリティ
│   ├── image_preprocessing.py # 画像前処理
│   └── image_dataset_resize.py
├── dataset/
│   └── {色番}/
│       ├── monochro/train/  # モノクロ学習データ
│       └── color/train/     # カラー学習データ
├── model/
│   ├── pretraining/         # 事前学習済みTeacherの重み
│   └── {色番}/
│       ├── monochro/        # モノクロモデルの重み・ONNX
│       └── color/           # カラーモデルの重み・ONNX
├── download/                # FTPでダウンロードしたアノテーション画像
└── backup/                  # 学習前バックアップ
```

---

## セットアップ

### 前提条件

- Python 3.9 以上
- CUDA対応GPU（CPU環境でも動作可、ただし学習時間が大幅に増加）

### 依存パッケージのインストール

```bash
pip install torch torchvision
pip install customtkinter
pip install omegaconf
pip install opencv-python
pip install onnx onnxruntime
pip install mlflow
pip install tqdm
```

### 事前学習済み重みの配置

カラー・モノクロ共通のTeacher事前学習済み重みを以下に配置してください。

```
model/pretraining/teacher_small_color_final_state.pth
```

### ImageNetデータセット（オプション）

学習ペナルティ項に使用します。不要な場合は `config.yaml` の `imagenet_train_path` を `none` に設定してください。

```
D:/shisui/ILSVRC/Data/CLS-LOC/train/   ← デフォルトパス（config.yamlで変更可）
```

---

## 設定ファイル（conf/config.yaml）

```yaml
ftp_hosts:
  - name: 検査PC_1
    host: 169.254.93.171
    username: ykk\shisui_PJ
    password: shisui@09
    monochro_port: 2121   # モノクロカメラ用FTPポート
    color_port: 2122      # カラーカメラ用FTPポート
    model_port: 2123      # モデルアップロード用FTPポート

target_color: '875'         # 対象色番（GUI起動時は上書きされる）
dataset_path: ./dataset     # 学習データセットのルート
model_dir: ./model          # モデル保存先
backup_dir: ./backup        # バックアップ先
download_dir: ./download    # FTPダウンロード先

train_step: 50000           # 最大学習ステップ数
batch_size: 4
gpu_id: 0
epochs: 80
out_channels: 384           # EfficientAD特徴チャンネル数
seed: 42
image_size:
  height: 256
  width: 512

imagenet_train_path: D:/shisui/ILSVRC/Data/CLS-LOC/train  # noneで無効化

model:
  lr: 0.0003
  weight_decay: 1.0e-05
  gamma: 0.1

map_st: 0.9                 # Student-Teacher異常マップの重み
map_ae: 0.2                 # Autoencoder異常マップの重み
early_stop_patience: 5      # 早期停止の待機エポック数
recall_threshold: 0.8
```

### 複数検査PCの追加

`ftp_hosts` リストにエントリを追加するだけで対応できます。

```yaml
ftp_hosts:
  - name: 検査PC_1
    host: 169.254.93.171
    # ...（省略）
  - name: 検査PC_2
    host: 169.254.93.172
    username: ykk\shisui_PJ
    password: shisui@09
    monochro_port: 2121
    color_port: 2122
    model_port: 2123
```

各PCへの接続が失敗した場合は警告メッセージを出力してスキップし、他のPCへの処理を継続します。

---

## 起動方法

### GUIで起動（通常運用）

```bash
python train_app.py
```

起動後の操作手順:

1. 左パネルの色番一覧からチェックボックスで対象色番を選択
   - 色番一覧は `./dataset/` 配下のフォルダ名から自動取得
   - 「すべて選択」ボタンで一括選択、「選択解除」で一括解除
2. 「選択した色番で実行」ボタンを押下
3. 右パネルのログエリアにパイプラインの進捗がリアルタイム表示される
4. 複数色番を選択した場合は順次実行（色番ごとに全パイプラインを完了してから次へ）

### コマンドラインで直接実行

`conf/config.yaml` の `target_color` を対象色番に変更してから実行します。

```bash
python pipline.py
```

---

## パイプライン詳細

### 1. バックアップ（DatasetManager.backup_model）

学習開始前に現在のモデルファイルをタイムスタンプ付きで `./backup/model/{色番}/{タイムスタンプ}/` にコピーします。

### 2. アノテーション画像のダウンロード（MultiFTPManager.download_images）

各検査PCのFTPサーバーから対象色番のアノテーション画像をダウンロードします。

- モノクロ: ポート `monochro_port` の `/camera1_image/annotated_data/{色番}/` から取得
- カラー: ポート `color_port` の `/camera2_image/annotated_data/{色番}/` から取得
- 既存ファイルはサイズとMDTMを比較してスキップ（差分ダウンロード）
- 接続失敗したPCは警告を出力してスキップし、他のPCに続行

### 3. 前処理（DatasetManager.process_annotated_images）

ダウンロードした画像をクロップ・リサイズして学習データセットに追加します。

| 処理 | モノクロ | カラー |
|---|---|---|
| クロップ範囲 | x:820〜1130 | 90度回転後 x:215〜1675 |
| 上下分割 | 上半分・下半分 | 上半分・下半分（下は垂直反転） |
| リサイズ後サイズ | 512x256 | 512x256 |

処理後の画像は `{dataset_path}/{色番}/{mode}/train/annotated/good/` に保存されます。

### 4. 並列学習（Trainer）

`multiprocessing.Process` を使い、モノクロとカラーを同時並列で学習します（どちらもGPU:0を使用）。

学習アルゴリズム（EfficientAD）:
- Teacher（PDN-Small, 384ch）: 事前学習済み重みを使用。学習中は固定（frozen）
- Student（PDN-Small, 768ch）: Teacher出力との差分を学習
- AutoEncoder（256x512対応カスタム構造）: 正常画像の再構成を学習
- ペナルティ項: ImageNetランダム画像に対しStudent出力が小さくなるよう正則化
- 2500ステップごとにバリデーション損失を計算し、最良モデルを `*_state_best.pth` として保存
- 早期停止: `early_stop_patience` 回連続でバリデーション損失が改善しない場合に停止

### 5. ONNXエクスポート（ModelExporter.export_onnx）

Teacher・Student・AutoEncoderの3モデルを `EfficientADFullModel` に統合し、ONNX形式でエクスポートします。

- 入力: `[batch, 3, height, width]`（0〜255のuint8相当のfloat32）
- 出力: `[batch, 1]`（異常スコア、最大ピクセル値）
- opset version: 11
- 出力ファイル: `model/{色番}/{mode}/{色番}_{mode}_model.onnx`

### 6. モデルのアップロード（MultiFTPManager.upload_onnx_model）

生成されたONNXモデルを各検査PCのFTPサーバーにアップロードします（ポート: `model_port`）。

---

## モデルアーキテクチャ（EfficientADFullModel）

```
入力画像 (0〜255)
    |
    v
正規化（ImageNet mean/std）
    |
    +---> Teacher (PDN-Small, 384ch) ─── frozen
    |         |
    +---> Student (PDN-Small, 768ch)
    |         |                    |
    |     前半384ch            後半384ch
    |         |                    |
    v         v                    v
AutoEncoder  map_st = mean((teacher - student[:384])^2)
             map_ae = mean((autoencoder - student[384:])^2)
                |
                v
         map_combined = 0.9 * map_st + 0.2 * map_ae
                |
                v
         bilinear upsample → max pooling → 異常スコア
```

---

## ファイル出力

学習完了後に生成されるファイル:

```
model/{色番}/{mode}/
├── teacher_state_best.pth      # 最良モデル（Teacher重み）
├── student_state_best.pth      # 最良モデル（Student重み）
├── autoencoder_state_best.pth  # 最良モデル（AutoEncoder重み）
├── teacher_state_temp.pth      # 最新チェックポイント（Teacher）
├── student_state_temp.pth      # 最新チェックポイント（Student）
├── autoencoder_state_temp.pth  # 最新チェックポイント（AutoEncoder）
├── para.json                   # 量子化パラメータ（teacher_mean, teacher_std等）
└── {色番}_{mode}_model.onnx    # ONNXエクスポート済みモデル
```

---

## トラブルシューティング

### `./dataset` フォルダが見つからない

GUIの色番一覧が空になります。`dataset/` ディレクトリを作成し、色番ごとのサブフォルダを用意してください。

```
dataset/
└── 875/
    ├── monochro/train/good/   ← 学習用正常画像
    └── color/train/good/      ← 学習用正常画像
```

### `./conf/config.yaml` が見つからない

実行ボタン押下時にエラーダイアログが表示されます。`conf/config.yaml` が存在するか確認してください。

### FTP接続エラー

該当PCをスキップして処理が続行されます。ログに `⚠ [{PC名}] からのダウンロード失敗（スキップ）: ...` と表示されます。IPアドレス・ポート番号・ネットワーク接続を確認してください。

### `teacher_small_color_final_state.pth` が見つからない

`model/pretraining/` ディレクトリに事前学習済みTeacherの重みファイルを配置してください。

### `para.json` の読み込みエラー

`ModelExporter` はJSON構文エラーを自動修正する機能を持っています。それでも失敗する場合は `para.json` を削除して再学習してください。

---

## 技術スタック

| 種別 | 技術 |
|---|---|
| 言語 | Python 3.9+ |
| 深層学習 | PyTorch, torchvision |
| GUI | CustomTkinter |
| 設定管理 | OmegaConf |
| 画像処理 | OpenCV |
| モデル変換 | ONNX (opset 11) |
| モデル管理 | MLflow（オプション） |
| FTP通信 | ftplib（Python標準ライブラリ） |
| 並列処理 | multiprocessing |
