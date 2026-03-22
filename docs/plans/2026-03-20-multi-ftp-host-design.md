# 設計: FTPManager 複数検査PC対応

## 概要

FTPManagerのダウンロード/アップロード先を複数の検査PCに対応させる。
各PCはIPアドレス・ポート・認証情報がそれぞれ異なる。

## 背景

現在の`FTPManager`は`config.yaml`の`ftp_info`から単一の検査PCの接続情報を読み取る設計。
検査PCが複数台ある環境に対応するため、複数宛先への一括操作を可能にする。

## 設計方針

- **アプローチ**: config.yamlの`ftp_info`をリスト形式(`ftp_hosts`)に拡張
- **エラー処理**: 失敗したPCはスキップして残りのPCの処理を続行
- **データ統合**: 全PCからのダウンロードデータは同一ディレクトリに統合

## 変更対象ファイル

1. `conf/config.yaml` - FTP設定構造の変更
2. `pipline.py` - FTPManager改修、MultiFTPManager新規追加、TrainingPipeline修正

## 詳細設計

### 1. config.yaml の構造変更

**変更前:**
```yaml
ftp_info:
  host: 169.254.93.171
  username: ykk\shisui_PJ
  password: shisui@09
  start_path: /camera1_image
  local_root: ./annotated_data
  monochro_port: 2121
  color_port: 2122
  model_port: 2123
```

**変更後:**
```yaml
ftp_common:
  start_path: /camera1_image
  local_root: ./annotated_data

ftp_hosts:
  - name: 検査PC_1
    host: 169.254.93.171
    username: ykk\shisui_PJ
    password: shisui@09
    monochro_port: 2121
    color_port: 2122
    model_port: 2123
  - name: 検査PC_2
    host: 192.168.250.201
    username: admin
    password: pass456
    monochro_port: 2121
    color_port: 2122
    model_port: 2123
```

- `start_path`と`local_root`は全PC共通のため`ftp_common`に分離
- PC固有の設定は`ftp_hosts`リストに格納
- `name`フィールドでログ出力時のPC識別に使用

### 2. FTPManager クラスの改修

`FTPManager.__init__`のシグネチャを変更し、1台分のホスト設定を受け取る。

```python
class FTPManager:
    """1台の検査PCに対するFTP操作"""
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
```

`download_images`と`upload_onnx_model`の内部ロジックは変更なし。

### 3. MultiFTPManager クラスの新規追加

複数PCへの一括操作を担当するファサードクラス。

```python
class MultiFTPManager:
    """複数検査PCへの一括FTP操作"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.managers = [
            FTPManager(cfg, host_cfg)
            for host_cfg in cfg.ftp_hosts
        ]

    def download_images(self):
        for mgr in self.managers:
            try:
                print(f"📥 {mgr.name} からダウンロード中...")
                mgr.download_images()
                print(f"✅ {mgr.name} ダウンロード完了")
            except Exception as e:
                print(f"⚠ {mgr.name} からのダウンロード失敗（スキップ）: {e}")

    def upload_onnx_model(self):
        for mgr in self.managers:
            try:
                print(f"📤 {mgr.name} へアップロード中...")
                mgr.upload_onnx_model()
                print(f"✅ {mgr.name} アップロード完了")
            except Exception as e:
                print(f"⚠ {mgr.name} へのアップロード失敗（スキップ）: {e}")
```

### 4. TrainingPipeline の変更

```python
# 変更前
self.ftp_manager = FTPManager(self.cfg)

# 変更後
self.ftp_manager = MultiFTPManager(self.cfg)
```

`download_images()`と`upload_onnx_model()`のインターフェースは同一のため、呼び出し側の変更は最小限。

## 影響範囲

- `train_pipline.py`: FTPを使用していないため変更不要
- `train_app.py`: `pipline.py`の`TrainingPipeline`を使用しているが、インターフェース変更なし
- 学習処理（`train_func_*.py`）: 変更なし
