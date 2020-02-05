#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://www.tech-tech.xyz/opencv_threshold.html

import cv2

thresh = 0
max_val = 255
thresholdType = cv2.THRESH_BINARY_INV


# トラックバーで、しきい値を変更
def changethresh(pos):
    global thresh
    thresh = pos


# 画像をグレースケールで読み込む
img = cv2.imread("clip_00a.bmp", 0)
# ウィンドウの名前を設定
cv2.namedWindow("img")
cv2.namedWindow("thresh")
# トラックバーのコールバック関数の設定
cv2.createTrackbar("trackbar", "thresh", 0, 255, changethresh)
while(1):
    cv2.imshow("img", img)
    _, thresh_img = cv2.threshold(img, thresh, max_val, thresholdType)
    cv2.imshow("thresh", thresh_img)
    k = cv2.waitKey(1)
    if k == 'q':
        break

cv2.destroyAllWindows()
