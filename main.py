# 此文件系主文件，检测图片请运行此文件

# 组件库
from detector.preprocess import *
from detector.nms import *
# 第三方库
import cv2 as cv
import os
import configparser
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression
import numpy as np


# 设置文件目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cfg_path = os.path.join(BASE_DIR, 'detector', 'config.cfg')
pos_path = os.path.join(BASE_DIR, 'data', 'train', 'pos')
pred_path = os.path.join(BASE_DIR, 'data', 'predict')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 读取配置信息
    print('正在读取配置信息...')
    config = configparser.ConfigParser()
    config.read(cfg_path)
    detector_type = config.get('detect', 'detector_type')

    # ！！！注意，使用预训练好的模型时，窗口大小是固定的，按照下方的标准使用
    # ！！！之所以使用条件配置，而不是使用配置文件，是因为在配置文件中变量的格式不符合标准，而HOGDescriptor对参数有严格的标准
    # 设置特征提取器与行人检测器
    clf = cv.HOGDescriptor()
    if detector_type == 'default':
        winSize = (64, 128)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        print('正在设置特征提取器...')
        clf = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        print('正在设置行人检测器...')
        clf.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    elif detector_type == 'daimler':
        winSize = (48, 96)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        print('正在设置特征提取器...')
        clf = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        print('正在设置行人检测器...')
        clf.setSVMDetector(cv.HOGDescriptor_getDaimlerPeopleDetector())

    # 读取图片信息
    print('正在读取目标图片...')
    file_path = os.path.join(pred_path, '3_persons_b.jpg')
    img = read_file(file_path)
    detect_img = img.copy()
    detected_img = img.copy()
    detected_img_nms = img.copy()
    # 由于cv2的HOG只能提取灰度图的HOG，所以需要复制图片，并转化为灰度图
    detect_graph = cv.cvtColor(detect_img, cv.COLOR_RGB2GRAY)

    # 检测图片
    print('正在检测图片中的行人...')
    (windows, cls) = clf.detectMultiScale(detect_graph, winStride=(2, 2), padding=(8, 8), scale=1.25)
    # 绘制未进行非最大抑制的检测结果
    for (x, y, w, h) in windows:
        cv.rectangle(detected_img, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
    # 对检测结果进行费最大抑制处理
    # 此时的prob参数未知作用
    nms_windows = non_max_suppression(windows, probs=None, overlapThresh=0.3)

    # 绘制进行非最大抑制的检测结果
    for (x, y, w, h) in nms_windows:
        cv.rectangle(detected_img_nms, (x, y), (x + w, y + h),  color=(255, 0, 0), thickness=2)

    # 绘制检测结果
    print('检测结果:')
    plt.imshow(img)
    plt.show()
    plt.imshow(detected_img)
    plt.show()
    plt.imshow(detected_img_nms)
    plt.show()
