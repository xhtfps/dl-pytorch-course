"""
案例:
演示基础的图像操作.

图像分类:
    二值图:
        1通道,每个像素点由0,1组成
    灰度图:
        1通道,每个像素点的范围:[0,255]
    索引图:
        1通道,每个像素点的范围:[0,255],像素点表示颜色表的索引
    RGB真彩图:
        3通道,Red,Green,Blue,红绿蓝.

涉及到的API：
    imshow: 绘制图像
    imread: 加载图像
    imsave: 保存图像
"""

# 导包
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# 1. 绘制全黑全白图
def dm01():
    # 创建全黑图像
    # HWC: H: 高度 W: 宽度  C: 通道数
    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    # print(f'img1: {img1}')

    # 绘制图片
    plt.imshow(img1)
    plt.show()

    # 创建全白图像
    img2 = np.ones((200, 200, 3), dtype=np.uint8) * 255
    # 绘制图像
    plt.imshow(img2)
    plt.show()


# 2.加载图像
def dm02():
    # 加载图像 (路径基于脚本所在目录, 避免因运行目录不同而找不到文件)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, 'data', 'img.png')
    img = plt.imread(img_path)
    print(f'img: {img}')
    print(f'img.shape: {img.shape}')
    print(f'img.dtype: {img.dtype}')
    print(f'img.max(): {img.max()}')

    # 绘制图像
    plt.imshow(img)
    plt.show()

# 3. 测试
if __name__ == '__main__':
    # dm01()
    dm02()

