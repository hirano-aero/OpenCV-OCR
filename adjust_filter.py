# -*- coding: utf-8 -*-
import cv2
import sys
import pyocr
import pyocr.builders
from PIL import Image
import numpy as np

thresh = 0
dilate = 0
erode = 0
kernel = np.ones((3, 3), np.uint8)
median = 1
gaussian = 1
box = 1
max_val = 255


def main():
    imgfile = "clip_1.bmp"
    # 画像をグレースケールで読み込む
    img = cv2.imread(imgfile, 0)

    # ウィンドウの名前を設定
    cv2.namedWindow("img")
    cv2.namedWindow("retouch")

    # トラックバーのコールバック関数の設定
    cv2.createTrackbar("thresh", "img", 0, 50, changethresh)
    cv2.createTrackbar("dilate", "img", 0, 20, changedilate)
    cv2.createTrackbar("erode", "img", 0, 20, changeerode)
    cv2.createTrackbar("median", "img", 0, 11, changemedian)
    cv2.createTrackbar("gaussian", "img", 0, 11, changegaussian)
    cv2.createTrackbar("box", "img", 0, 11, changebox)

    # 色反転 白背景黒文字にする
    img = cv2.bitwise_not(img)
    cv2.imshow("img", img)

    # OCRモジュール読み込み
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    tool = tools[0]

    while(1):
        retouch = img
        if thresh > 0:
            _, retouch = cv2.threshold(retouch, thresh, max_val, cv2.THRESH_BINARY)
        if dilate > 0:
            retouch = cv2.dilate(retouch, kernel, iterations=300)
        if erode > 0:
            retouch = cv2.erode(retouch, kernel)
        if median > 0:
            retouch = cv2.medianBlur(retouch, median)
        if gaussian > 0:
            retouch = cv2.GaussianBlur(retouch, (gaussian, gaussian), 1, 1)
        if box > 0:
            retouch = cv2.boxFilter(retouch, -1, (box, box))

        # OCR表示
        text = tool.image_to_string(cv2pil(retouch),
                                    lang="eng",
                                    builder=pyocr.builders.WordBoxBuilder(tesseract_layout=6))
        # rectangleが赤色なのでカラーに変更
        right = cv2.cvtColor(retouch, cv2.COLOR_GRAY2BGR)
        left = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        print('-------------')
        i = 1
        for d in text:
            print(i, d.position, d.content)
            cv2.rectangle(right, d.position[0], d.position[1], (0, 0, 255), 2)
            cv2.putText(right, str(i), d.position[1], cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
            i += 1

        # 画像を結合する
        concat = concatImage(left, right)

        # 画像を表示する
        cv2.imshow("retouch", concat)
        k = cv2.waitKey(1500)
        if k == 'q':
            break


# 画像を結合する
def concatImage(left, right):
    dst = cv2.hconcat([left, right])
    return dst


# 2値化コールバック
def changethresh(pos):
    global thresh
    thresh = pos


# 膨張コールバック
def changedilate(pos):
    global dilate, kernel
    dilate = pos
    kernel = np.ones((pos, pos), np.uint8)


# 収縮コールバック
def changeerode(pos):
    global erode, kernel
    erode = pos
    kernel = np.ones((pos, pos), np.uint8)


# メディアンコールバック
def changemedian(pos):
    global median
    mod = pos % 2
    if mod > 0:
        median = pos
    else:
        median = pos + 1


# ガウシアンコールバック
def changegaussian(pos):
    global gaussian
    mod = pos % 2
    if mod > 0:
        gaussian = pos
    else:
        gaussian = pos + 1



# BOXコールバック
def changebox(pos):
    global box
    box = pos


def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
