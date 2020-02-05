#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyocr
import pyocr.builders
import cv2
from PIL import Image
import sys
import numpy as np

# https://qiita.com/derodero24/items/f22c22b22451609908ee


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


tools = pyocr.get_available_tools()

if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)

tool = tools[0]

langs = tool.get_available_languages()
print("Available languages: %s" % ", ".join(langs))

imagefile = "clip_00a.bmp"
cvimg = cv2.imread(imagefile, 0)
ret, dst = cv2.threshold(cvimg, 200, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("img",dst)

# 膨張
kernel = np.ones((3,3),np.uint8)
dst = cv2.dilate(dst, kernel, iterations=300)

# メディアン
# dst = cv2.medianBlur(dst,3)

# ガウシアン
# dst = cv2.GaussianBlur(dst, (3,3), 1, 1)

# BOX
dst = cv2.boxFilter(dst, -1, (1,1))

# 収縮
# kernel = np.ones((1,1),np.uint8)
# dst = cv2.erode(dst,kernel)

# 色反転 白背景黒文字にする
dst = cv2.bitwise_not(dst)

# cv2.imshow("Fil",dst)
# cv2.waitKey(100)

res = tool.image_to_string(cv2pil(dst),
                           config="nobatch digits")
#                           lang="eng",
#                           builder=pyocr.builders.WordBoxBuilder(tesseract_layout=6))
"""
out = cv2.imread(imagefile)
for d in res:
    print(d.content)
    print(d.position)
    cv2.rectangle(out, d.position[0], d.position[1], (0, 0, 255), 2)

cv2.imshow("img",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
cv2.destroyAllWindows()
