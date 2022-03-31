import random
from openvino.inference_engine import IECore
import numpy as np
import time
import cv2.cv2 as cv
import pandas as pd
#
# cap1 = cv.VideoCapture("C:/image/video/COI-video/20210928/mov_area1_2021-09-10_14-00-01_600_2.mp4")
# cap2 = cv.VideoCapture("C:/image/video/COI-video/20210928/mov_area1_2021-09-10_14-00-01_600.mp4")
cap1 = cv.VideoCapture("C:/image/video/2.mp4")  # 视频文件读入
cap2 = cv.VideoCapture("C:/image/video/vibration.mp4")  # 视频文件读入
# 查看视频是否读取成功
print(cap1.isOpened())
print(cap2.isOpened())

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if (ret1 is not True) and (ret2 is not True):
        break
    width = int(cap1.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv.CAP_PROP_FRAME_HEIGHT))
    print("the size of frame width:{} heigt:{}".format(width,height))
    frame1 = cv.resize(frame1,(width//1,height//1))
    print("the size of frame1:{}".format(frame1.shape))
    frame2 = cv.resize(frame2, (width//1, height//1))
    print("the size of frame2:{}".format(frame2.shape))
    img = np.hstack((frame1, frame2))
    print("the size of img:{}".format(img.shape))
    img = cv.line(img, (1920,0), (3840,1088), (0, 255, 0), 10)
    img = cv.line(img, (0, 0), (1920, 544), (0, 0, 255), 10)
    img = cv.line(img, (0, 0), (1920, 1088), (0, 0, 255), 10)
    cv.namedWindow('cap1', cv.WINDOW_NORMAL)
    cv.imshow("cap1", img)
    # cv.namedWindow('cap2', cv.WINDOW_NORMAL)
    cv.imshow("cap2", frame2)
    c = cv.waitKey(50)
    if c == 27:
        break