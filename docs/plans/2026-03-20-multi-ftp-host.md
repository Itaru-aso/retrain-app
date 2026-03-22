# 複数検査PC FTP対応 実装計画

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** FTPManagerのダウンロード/アップロード先を複数の検査PCに対応させる

**Architecture:** config.yamlの`ftp_info`を`ftp_common`+`ftp_hosts`リストに分割。FTPManagerは1台のPC操作を担当し、新規MultiFTPManagerが複数PCへのループ+エラーハンドリングを担当する。TrainingPipelineはMultiFTPManagerを使用する。

**Tech Stack:** Python, OmegaConf, ftplib

**設計ドキュメント:** `docs/plans/2026-03-20-multi-ftp-host-design.md`

---

### Task 1: config.yaml のFTP設定構造を変更

**Files:**
- Modify: `conf/config.yaml:10-19`

**Step 1: config.yaml の `ftp_info` を `ftp_common` + `ftp_hosts` に書き換え**

`conf/config.yaml` の10-19行目を以下に置き換える:

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
```

注: 既存のPC情報をそのまま `ftp_hosts` の最初の要素として保持する。2台目以降は運用時に追加する。

**Step 2: config が正しく読み込めることを確認**

Run:
```bash
cd D:/0032011/GitLab/shisui/retrain_app_running && python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('./conf/config.yaml')
print('ftp_common.start_path:', cfg.ftp_common.start_path)
print('ftp_hosts count:', len(cfg.ftp_hosts))
print('ftp_hosts[0].name:', cfg.ftp_hosts[0].name)
print('ftp_hosts[0].host:', cfg.ftp_hosts[0].host)
print('OK')
"
```

Expected:
```
ftp_common.start_path: /camera1_image
ftp_hosts count: 1
ftp_hosts[0].name: 検査PC_1
ftp_hosts[0].host: 169.254.93.171
OK
```

---

### Task 2: FTPManager のコンストラクタを変更

**Files:**
- Modify: `pipline.py:118-128`

**Step 1: FTPManager.__init__ を書き換え**

`pipline.py` の118-128行目を以下に置き換える:

```python
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
```

注: `download_images`と`upload_onnx_model`の内部ロジックは変更不要。これらのメソッドは既に`self.host`, `self.username`等のインスタンス変数を参照しているため、コンストラクタの読み取り元が変わるだけで動作する。

**Step 2: インスタンス生成が正しく動作することを確認**

Run:
```bash
cd D:/0032011/GitLab/shisui/retrain_app_running && python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('./conf/config.yaml')
from pipline import FTPManager
mgr = FTPManager(cfg, cfg.ftp_hosts[0])
print('name:', mgr.name)
print('host:', mgr.host)
print('start_path:', mgr.start_path)
print('OK')
"
```

Expected:
```
name: 検査PC_1
host: 169.254.93.171
start_path: /camera1_image
OK
```

---

### Task 3: MultiFTPManager クラスを追加

**Files:**
- Modify: `pipline.py` (FTPManagerクラスの直後、Trainerクラスの直前に追加)

**Step 1: MultiFTPManager クラスを追加**

`pipline.py` のFTPManagerクラス定義の末尾（`upload_onnx_model`メソッドの後、`class Trainer`の前）に以下を挿入:

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

```

**Step 2: クラスが正しくインポート・生成できることを確認**

Run:
```bash
cd D:/0032011/GitLab/shisui/retrain_app_running && python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('./conf/config.yaml')
from pipline import MultiFTPManager
multi = MultiFTPManager(cfg)
print('managers count:', len(multi.managers))
print('first manager name:', multi.managers[0].name)
print('OK')
"
```

Expected:
```
managers count: 1
first manager name: 検査PC_1
OK
```

---

### Task 4: TrainingPipeline の FTPManager を MultiFTPManager に置き換え

**Files:**
- Modify: `pipline.py:215`

**Step 1: TrainingPipeline.__init__ を修正**

`pipline.py` の215行目を変更:

変更前:
```python
        self.ftp_manager = FTPManager(self.cfg)
```

変更後:
```python
        self.ftp_manager = MultiFTPManager(self.cfg)
```

注: `execute`メソッド内の `self.ftp_manager.download_images()` と `self.ftp_manager.upload_onnx_model()` の呼び出しは変更不要。MultiFTPManagerが同じメソッド名を提供しているため。

**Step 2: TrainingPipeline が正しく生成できることを確認**

Run:
```bash
cd D:/0032011/GitLab/shisui/retrain_app_running && python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('./conf/config.yaml')
cfg.target_color = '875'
from pipline import TrainingPipeline
pipeline = TrainingPipeline(cfg)
print('ftp_manager type:', type(pipeline.ftp_manager).__name__)
print('managers count:', len(pipeline.ftp_manager.managers))
print('OK')
"
```

Expected:
```
ftp_manager type: MultiFTPManager
managers count: 1
OK
```

---

### Task 5: 複数ホスト設定での動作確認

**Files:**
- Modify: `conf/config.yaml` (一時的に2台目を追加してテスト)

**Step 1: config.yaml に2台目のダミーホストを追加して構造テスト**

`conf/config.yaml` の `ftp_hosts` に2台目を追加:

```yaml
ftp_hosts:
  - name: 検査PC_1
    host: 169.254.93.171
    username: ykk\shisui_PJ
    password: shisui@09
    monochro_port: 2121
    color_port: 2122
    model_port: 2123
  - name: 検査PC_2_テスト
    host: 192.168.250.201
    username: test_user
    password: test_pass
    monochro_port: 2121
    color_port: 2122
    model_port: 2123
```

**Step 2: MultiFTPManager が2台分を認識することを確認**

Run:
```bash
cd D:/0032011/GitLab/shisui/retrain_app_running && python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('./conf/config.yaml')
from pipline import MultiFTPManager
multi = MultiFTPManager(cfg)
print('managers count:', len(multi.managers))
for mgr in multi.managers:
    print(f'  {mgr.name}: {mgr.host}')
print('OK')
"
```

Expected:
```
managers count: 2
  検査PC_1: 169.254.93.171
  検査PC_2_テスト: 192.168.250.201
OK
```

**Step 3: ダミーホストを削除して本番設定に戻す**

2台目の `検査PC_2_テスト` エントリを削除し、1台構成に戻す（実際の2台目は運用時に追加）。

---

### Task 6: 最終確認とコミット

**Step 1: 変更差分を確認**

Run: `git diff` で変更内容を確認

対象ファイル:
- `conf/config.yaml` — `ftp_info` → `ftp_common` + `ftp_hosts` 構造変更
- `pipline.py` — FTPManager改修、MultiFTPManager追加、TrainingPipeline修正

**Step 2: コミット**

```bash
git add conf/config.yaml pipline.py docs/plans/
git commit -m "機能追加: FTPManager を複数検査PC対応に拡張

- config.yaml の ftp_info を ftp_common + ftp_hosts リスト形式に変更
- FTPManager のコンストラクタを1台分のホスト設定を受け取る形式に改修
- MultiFTPManager クラスを新規追加（複数PCへの一括操作、失敗時スキップ）
- TrainingPipeline で MultiFTPManager を使用するよう変更"
```
