"""此文件包含用于预处理图片的组件"""

# 图像处理
from skimage.io import imread
# 通用库
import os


def read_file(file_path, as_gray=False):
    img = imread(file_path, as_gray=as_gray)
    return img


def read_dir(root_path, as_gray=False):
    imgs = []
    for root, dirs, files in os.walk(root_path, topdown=True):
        for file in files:
            _, ending = os.path.splitext(file)
            if ending == '.jpg' or ending == '.jpeg' or ending == '.png':
                imgs.append(imread(os.path.join(root_path, file), as_gray=as_gray))
    return imgs
