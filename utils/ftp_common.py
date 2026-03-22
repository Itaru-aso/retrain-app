from ftplib import FTP
import os

import os
from ftplib import FTP

import os
from ftplib import FTP

def is_directory(ftp, path):
    """
    Windows系FTPサーバーで、親ディレクトリのLIST出力から対象がディレクトリかを判定
    """
    parent_path, name = os.path.split(path.rstrip('/'))
    lines = []
    try:
        ftp.retrlines(f'LIST {parent_path}', lines.append)
    except Exception:
        return False

    for line in lines:
        if name in line and '<DIR>' in line:
            return True
    return False


import os
import datetime

def ftp_get_size(ftp, remote_path):
    """SIZE コマンドでリモートファイルサイズ（バイト）を取得。未対応時は None。"""
    try:
        size = ftp.size(remote_path)
        # ftplib.FTP.size は int を返す（未対応やディレクトリだとエラー）
        return int(size) if size is not None else None
    except Exception:
        return None

def ftp_get_mdtm_epoch(ftp, remote_path):
    """
    MDTM の結果を UNIX エポック秒（float）で返す。
    返却形式例: '213 20251223120130' や '213 YYYYMMDDhhmmss'
    未対応・失敗時は None。
    """
    try:
        resp = ftp.sendcmd(f"MDTM {remote_path}")
        # 応答から YYYYMMDDhhmmss を抽出
        # 典型応答: '213 20251223120130'
        ts = resp.split()[-1]
        dt = datetime.datetime.strptime(ts, "%Y%m%d%H%M%S")
        # サーバが UTC を返すことが一般的だが厳密にはサーバ依存。
        # ここでは UTC 想定で timestamp に変換。
        return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    except Exception:
        return None

def should_skip_download(local_path, remote_path, ftp, size_only=False):
    """
    ローカルに同名ファイルがある場合にスキップすべきかを判定。
    - size_only=True の場合、サイズ一致のみで判定。
    - MDTM が取得できる場合は時刻も考慮（ローカルの方が新しい/同等ならスキップ）。
    戻り値:
      ("skip", reason) / ("resume", offset) / ("full", reason)
    """
    if not os.path.exists(local_path):
        return ("full", "local_missing")

    local_size = os.path.getsize(local_path)
    remote_size = ftp_get_size(ftp, remote_path)

    if remote_size is None:
        # サイズが取れない場合は conservative にフルダウンロード
        return ("full", "remote_size_unavailable")

    if local_size == remote_size:
        if size_only:
            return ("skip", "size_equal")
        # MDTM 比較
        remote_mdtm = ftp_get_mdtm_epoch(ftp, remote_path)
        if remote_mdtm is None:
            # MDTM 取れないならサイズ一致でスキップ
            return ("skip", "size_equal_no_mdtm")
        local_mtime = os.path.getmtime(local_path)
        # ローカルが同時刻以上（新しい/同等）ならスキップ
        if local_mtime >= remote_mdtm:
            return ("skip", "size_equal_and_local_newer_or_equal")
        else:
            # リモートが新しい可能性 → フルダウンロードで置き換え
            return ("full", "size_equal_but_remote_newer")
    elif local_size < remote_size:
        # レジューム可能なら REST で追記
        return ("resume", local_size)
    else:
        # ローカルの方が大きい（壊れた/異なるファイル）→ フルダウンロードで上書き
        return ("full", "local_larger_than_remote")


def download_ftp_tree_with_skip(ftp, remote_path, local_path, is_dir_func, size_only=False):
    """
    再帰ダウンロード（フォルダ/ファイル）に、スキップ＆レジューム判定を組み込んだ版。
    - size_only=True: MDTM を使わず size 一致のみでスキップ。
    """
    os.makedirs(local_path, exist_ok=True)

    # ディレクトリ一覧
    try:
        ftp.cwd(remote_path)
        items = ftp.nlst()
    except Exception as e:
        print(f"⚠️ ディレクトリ取得失敗: {remote_path} - {e}")
        return

    for item in items:
        if item in ('.', '..'):
            continue

        remote_item_path = f"{remote_path.rstrip('/')}/{item}"
        local_item_path  = os.path.join(local_path, item)

        if is_dir_func(ftp, remote_item_path):
            # ディレクトリ：再帰
            download_ftp_tree_with_skip(ftp, remote_item_path, local_item_path, is_dir_func, size_only=size_only)
        else:
            # ファイル：スキップ/レジューム/フルの判定
            try:
                decision, info = should_skip_download(local_item_path, remote_item_path, ftp, size_only=size_only)

                if decision == "skip":
                    # print(f"⏭️ スキップ: {remote_item_path} ({info})")
                    continue
                elif decision == "resume":
                    offset = info  # 既存サイズ（バイト）
                    # retrbinary の rest 引数でレジューム
                    with open(local_item_path, 'ab') as f:  # 追記モード
                        ftp.retrbinary(f"RETR {remote_item_path}", f.write, rest=offset)
                    # print(f"↪️ レジューム完了: {remote_item_path} from {offset} bytes")
                else:  # "full"
                    with open(local_item_path, 'wb') as f:
                        ftp.retrbinary(f"RETR {remote_item_path}", f.write)
                    # print(f"⬇️ ダウンロード: {remote_item_path}")
            except Exception as e:
                print(f"❌ ファイルのダウンロードに失敗: {remote_item_path} - {e}")
                # 続行


def download_ftp_selected(ftp, remote_root, local_root, allowed_top_levels, is_dir_func, size_only=False):
    """
    remote_root 直下の allowed_top_levels に含まれるサブディレクトリのみ再帰ダウンロード。
    すでに存在するファイルはサイズ/MDTMでスキップ、足りない分はレジューム。

    remote_root（例: './annotated_data'）直下のサブディレクトリのうち、
    allowed_top_levels（例: ['010']）に含まれるものだけを再帰ダウンロードする。

    Parameters
    ----------
    ftp : ftplib.FTP インスタンス
    remote_root : str  例 './annotated_data' または '/annotated_data'
    local_root  : str  ローカルの保存先ルートディレクトリ
    allowed_top_levels : List[str]  ダウンロード対象（例 ['010'] or ['010','020']）
    is_dir_func : Callable(ftp, remote_path) -> bool  ディレクトリ判定関数
    """
    os.makedirs(local_root, exist_ok=True)

    remote_root_norm = remote_root.rstrip('/')
    if remote_root_norm.startswith('./'):
        remote_root_norm = '/' + remote_root_norm[2:]  # './x' -> '/x'

    allowed = set([name.strip() for name in allowed_top_levels if name and name.strip()])

    # トップ階層一覧
    try:
        items = ftp.nlst(remote_root_norm)
        top_names = []
        for it in items:
            nm = it.rsplit('/', 1)[-1]
            if nm in ('.', '..'):
                continue
            top_names.append(nm)
    except Exception as e:
        print(f"⚠️ トップ階層の一覧取得に失敗: {remote_root} - {e}")
        return

    for name in top_names:
        if name not in allowed:
            continue

        remote_path = f"{remote_root_norm}/{name}"
        local_path  = os.path.join(local_root, name)

        try:
            if is_dir_func(ftp, remote_path):
                download_ftp_tree_with_skip(ftp, remote_path, local_path, is_dir_func, size_only=size_only)
            else:
                # トップ直下がファイルの場合もスキップ判定
                decision, info = should_skip_download(local_path, remote_path, ftp, size_only=size_only)
                if decision == "skip":
                    continue
                elif decision == "resume":
                    with open(local_path, 'ab') as f:
                        ftp.retrbinary(f"RETR {remote_path}", f.write, rest=info)
                else:
                    with open(local_path, 'wb') as f:
                        ftp.retrbinary(f"RETR {remote_path}", f.write)
        except Exception as e:
            print(f"❌ ダウンロード失敗: {remote_path} - {e}")
            # 続行
            continue


def download_ftp_tree(ftp, remote_path, local_path):
    """
    FTP サーバーからファイルとディレクトリを再帰的にダウンロード
    """
    if os.path.isdir(local_path):
        pass
    else:
        os.makedirs(local_path)

    try:
        ftp.cwd(remote_path)
        items = ftp.nlst()
    except Exception as e:
        print(f"⚠️ ディレクトリ取得失敗: {remote_path} - {e}")
        return

    for item in items:
        if item in ('.', '..'):
            continue
        remote_item_path = f"{remote_path}/{item}"
        local_item_path = os.path.join(local_path, item)

        if is_directory(ftp, remote_item_path):
            download_ftp_tree(ftp, remote_item_path, local_item_path)
        else:
            try:
                with open(local_item_path, 'wb') as f:
                    ftp.retrbinary(f"RETR {remote_item_path}", f.write)
                #print(f"⬇️ ダウンロード成功: {remote_item_path}")
            except Exception as e:
                print(f"❌ ファイルのダウンロードに失敗: {remote_item_path} - {e}")



import os
"""
def download_ftp_selected(ftp, remote_root, local_root, allowed_top_levels, is_dir_func):


    # --- 前処理 ---
    # ローカルのルート作成
    os.makedirs(local_root, exist_ok=True)

    # remote_root の正規化（末尾のスラッシュ削除、先頭の './' は除去）
    remote_root_norm = remote_root.rstrip('/')
    if remote_root_norm.startswith('./'):
        remote_root_norm = remote_root_norm[2:]  # './annotated_data' -> 'annotated_data'
        remote_root_norm = '/' + remote_root_norm  # 絶対パス化: '/annotated_data'

    # 許可トップ階層の正規化（重複/空白除去）
    allowed = set([name.strip() for name in allowed_top_levels if name and name.strip()])

    # --- トップ階層の一覧取得 ---
    try:
        # nlst に絶対パスを渡す方が安全なサーバが多い
        items = ftp.nlst(remote_root_norm)
        # 例: '/annotated_data/010' と '010' 等が混在しうるため末尾名で正規化
        top_names = []
        for it in items:
            name = it.rsplit('/', 1)[-1]  # 末尾名抽出
            if name in ('.', '..'):
                continue
            top_names.append(name)
    except Exception as e:
        print(f"⚠️ トップ階層の一覧取得に失敗: {remote_root} - {e}")
        return  # 一覧が取れない場合はここで終了

    # --- フィルタしてダウンロード ---
    for name in top_names:
        if name not in allowed:
            # 指定外はスキップ
            # print(f"⏭️ スキップ: {name}")
            continue

        remote_path = f"{remote_root_norm}/{name}"
        local_path  = os.path.join(local_root, name)

        try:
            if is_dir_func(ftp, remote_path):
                # 指定トップ階層のみ再帰ダウンロード
                download_ftp_tree(ftp, remote_path, local_path)
            else:
                # トップ直下がファイルであるケース（念のため対応）
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, 'wb') as f:
                    ftp.retrbinary(f"RETR {remote_path}", f.write)
                # print(f"⬇️ ファイル保存: {remote_path}")
        except Exception as e:
            # ここで止まらず、他の項目に進みたいなら "continue" を使う
            print(f"❌ ダウンロード失敗: {remote_path} - {e}")
            # continue  # ← 続行したい場合はコメント解除
            # return    # ← 即座に関数を終了したい場合はこちらに切り替え
"""


def connect_and_download_tree(host, port, username, password, start_path, local_root):
    ftp=FTP()
    ftp.encoding = 'utf-8'
    ftp.connect(host, port, timeout=10)
    ftp.login(user=username, passwd=password)
    download_ftp_tree(ftp, start_path, local_root)

from ftplib import FTP
import os

def ensure_ftp_directory(ftp, remote_folder):
    """
    FTPサーバー上に指定されたフォルダが存在しない場合は、階層ごとに作成する。
    """
    original_dir = ftp.pwd()
    for part in remote_folder.strip("/").split("/"):
        if part == "":
            continue
        try:
            ftp.cwd(part)
        except Exception:
            try:
                ftp.mkd(part)
                ftp.cwd(part)
            except Exception as e:
                print(f"❌ ディレクトリ作成失敗: {part} - {e}")
                ftp.cwd(original_dir)
                raise
    ftp.cwd(original_dir)

def upload_file_to_ftp(host, port, username, password, local_file_path, remote_folder):
    print(f"📤 アップロード開始: {local_file_path} → {remote_folder}")
    try:
        ftp=FTP()
        ftp.encoding = 'utf-8'
        ftp.connect(host, port, timeout=10)
        ftp.login(user=username, passwd=password)

        # アップロード先フォルダが存在しない場合は作成
        ensure_ftp_directory(ftp, remote_folder)

        # フォルダに移動
        ftp.cwd(remote_folder)

        filename = os.path.basename(local_file_path)
        with open(local_file_path, 'rb') as f:
            ftp.storbinary(f'STOR ' + filename, f)

        print(f"✅ アップロード完了: {filename} → {remote_folder}")
    except Exception as e:
        print(f"❌ FTPアップロードエラー: {e}")


if __name__ == '__main__':
    #使用例

    host="192.168.250.201"
    username="ykk\\shisui_PJ"
    ftp_pass="shisui@03"

    try:
        ftp=FTP()
        ftp.connect(host, 2121, timeout=10)
        print("connect")
        ftp.login(user=username, passwd=ftp_pass)
        print("login")
        ftp.retrlines('LIST')

        ftp.quit()

    except Exception as e:
        import traceback
        traceback.print_exc()

    upload_file_to_ftp(
        host="192.168.250.201",
        port=2121,
        username="ykk\\shisui_PJ",
        password="shisui@03",
        local_file_path="C:/Fastenerlnsp/retrain_app/pretraining/teacher_medium_tmp_state.pth",  # アップロードしたいファイル
        remote_folder="/model/841/color/"  # アップロード先のリモートフォルダ
    )


    # 010 だけダウンロード（020 はスキップ）
    download_ftp_selected(
        ftp=ftp,
        remote_root='./annotated_data',           # or '/annotated_data'
        local_root='./downloaded_annotated_data',
        allowed_top_levels=['010'],
        is_dir_func=is_directory                  # あなたの LIST 判定関数
    )
    
    # 010 だけ対象、MDTM が取れないサーバ前提なら size_only=True にすると安全
    download_ftp_selected(
        ftp=ftp,
        remote_root='./annotated_data',
        local_root='./downloaded_annotated_data',
        allowed_top_levels=['010'],
        is_dir_func=is_directory,  # あなたの LIST ベース判定
        size_only=False            # MDTM を使って時刻比較したい場合は False（未対応なら自動フォールバック）
    )


    """
    connect_and_download_tree(
        host="192.168.250.201",
        username="ykk\\shisui_PJ",
        password="shisui@03",
        start_path="/camera1_image",
        local_root="./annotated_data"
    )
    """










