import os
from PIL import Image
import numpy as np
import cv2

class OpenCVResize:
    def __init__(self, width: int, height: int, interpolation=cv2.INTER_LINEAR):
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        resized_bgr = cv2.resize(img_bgr, (self.width, self.height), interpolation=self.interpolation)
        resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(resized_rgb)

# 入力・出力ディレクトリ
#input_dir = r"D:\0032011\shisui_project\AI\EfficientAD\dataset_raw\color\test\841\minogashi"
#output_dir = r"D:\0032011\shisui_project\AI\EfficientAD\dataset\384_512\841\color\test\minogashi"
input_dir = r"D:\0032011\shisui_project\Analyze\back_light\raw_data"
output_dir = r"D:\0032011\shisui_project\Analyze\back_light\resize_data"

# 出力ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# リサイズ処理の初期化（サイズ指定）
resize_transform = OpenCVResize(width=512, height=384)

# ファイル処理ループ
for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # 画像ファイルのみ処理
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        try:
            with Image.open(input_path) as img:
                resized_img = resize_transform(img)
                resized_img.save(output_path)
                print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

import cv2
import numpy as np

# ファイルパス
path1 = r"D:\0032011\shisui_project\AI\EfficientAD\dataset\841\monochro\train\good_\OK_image_093640_258_0.bmp"
path2 = r"D:\0032011\shisui_project\AI\EfficientAD\dataset\841\monochro\train\good\OK_image_093640_258_0.bmp"

# 画像読み込み（カラー）
img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
img2 = cv2.imread(path2, cv2.IMREAD_COLOR)

# 読み込み確認
if img1 is None or img2 is None:
    print("画像が読み込めません。")
else:
    # サイズ確認
    if img1.shape != img2.shape:
        print("画像サイズが異なります。")
    else:
        # 差分マスク作成
        diff = cv2.absdiff(img1, img2)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        num_diff_pixels = np.count_nonzero(diff_gray)
        total_pixels = diff_gray.size
        diff_ratio = num_diff_pixels / total_pixels * 100

        print(f"異なるピクセル数: {num_diff_pixels}")
        print(f"全体に対する差異の割合: {diff_ratio:.2f}%")

        # 差分画像を保存（オプション）
        cv2.imwrite("difference_mask.png", diff_gray)

