import cv2
import numpy as np
import time

def process_image(image_data, image_size_width, image_size_height, mode):
    """ 画像データを処理し、上半分と下半分をリサイズする関数

    Args:
        image_data (byte): 画像データのバイト配列
        mode (str): "monochro" または "color" のいずれかを指定

    Returns:
        resized_top_half (numpy.ndarray): 上半分のリサイズ画像
        resized_bottom_half (numpy.ndarray): 下半分のリサイズ画像
        elapsed_time (float): 処理にかかった時間
    """
    read_img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    if mode == "monochro":
        crop_rectangle = (820, 0, 1130, read_img.shape[0])
    elif mode == "color":
        read_img = cv2.rotate(read_img, cv2.ROTATE_90_CLOCKWISE)
        crop_rectangle = (215, 0, 1675, read_img.shape[0])
    else:
        print("color or monochroの指定がありません。")
        return None, None, 0

    x, y, w, h = crop_rectangle
    img = read_img[y:y+h, x:x+w]

    half_height = img.shape[0] // 2
    top_half = img[0:half_height, :]

    if mode == "color":
        bottom_half = cv2.flip(img[half_height:, :], 0)
    elif mode == "monochro":
        bottom_half = img[half_height:, :]

    new_size = (image_size_width, image_size_height)
    start_time = time.time()
    resized_top_half = cv2.resize(top_half, new_size, interpolation=cv2.INTER_NEAREST)
    elapsed_time = time.time() - start_time
    resized_bottom_half = cv2.resize(bottom_half, new_size, interpolation=cv2.INTER_NEAREST)

    return resized_top_half, resized_bottom_half, elapsed_time

def load_image_as_byte_array(file_path):
    """ 画像ファイルを読み込み、PNG形式でエンコードしてバイト配列に変換する関数

    Args:
        file_path (str): 画像ファイルのパス

    Returns:
        encoded_image.tobytes() (byte): 画像をPNG形式でエンコードしたバイト配列
    """

    # OpenCaaVで画像を読み込む
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"指定されたファイルが見つかりません: {file_path}")

    # PNG形式にエンコードしてバイト配列に変換
    success, encoded_image = cv2.imencode('.png', image)

    if not success:
        raise Exception("画像のエンコードに失敗しました。")

    return encoded_image.tobytes()

if __name__ == '__main__':
    image_data = load_image_as_byte_array("D:/0032011/shisui_project/AI/EfficientAD/ImageData/color/train/841/OK_image_093646_924.bmp")
    resized_top_half, resized_bottom_half, elapsed_time = process_image(image_data, "color")

    cv2.imwrite("resized_top_half.bmp", resized_top_half)
    cv2.imwrite("resized_bottom_half.bmp", resized_bottom_half)
    print(f"画像を .bmp 形式で保存しました。処理時間: {elapsed_time:.4f}秒")

    img_cs = cv2.imread("D:/0032011/shisui_project/AI/EfficientAD/dataset/841/color/train/good/OK_image_093646_924_0.bmp")
    img_py = cv2.imread("resized_top_half.bmp")

    print("完全一致:", np.array_equal(img_py, img_cs))
    print("近似一致:", np.allclose(img_py, img_cs))
