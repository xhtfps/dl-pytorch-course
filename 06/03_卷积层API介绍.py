"""
案例:
演示卷积层的API, 用于 提取图像的局部特征, 获取: 特征图(Feature Map)

卷积神经网络介绍:
概述:
    全称叫: Convolutional neural network, 即: 包含卷积层的神经网络。
组成:
    卷积层(Convolutional):
        用于提取图像的 局部特征, 结合 卷积核(每个卷积核 = 1个神经元) 实现, 处理后的结果叫: 特征图。
    池化层(Pooling):
        用于 降维, 降采样
    全连接层(Full Connected, fc, linear):
        用于 预测结果, 并输出结果的。
特征图计算方式:
    N = (W - F + 2*P) / S  + 1
    W: 输入图像的大小
    F: 卷积核的大小
    P: 填充的大小
    S: 步长
    N: 输出图像的大小(特征图大小)
"""
#导包
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 

# 1. 定义函数，用于图像的加载，卷积，特征图的可视化操作
def dm01():
    # 1.1 加载图像
    img = plt.imread("./06/data/img.png")#[..., :3]
    # 1.2 打印图像的信息
    # print(f'img: {img} img.shape: {img.shape}')
    # 1.3 将图像数据转换为张量
    img_tensor = torch.tensor(img, dtype = torch.float32)
    # 1.4 图像数据转换: HWC -> CHW
    img_tensor = img_tensor.permute(2, 0, 1)
    # print(f'img_tensor: {img_tensor} img_tensor.shape: {img_tensor.shape}')
    # 1.5 将图像数据转换为4D张量
    img_tensor = img_tensor.unsqueeze(0)
    # print(f'img_tensor: {img_tensor} img_tensor.shape: {img_tensor.shape}')
    # 1.6. 定义卷积层
    # 参1：输入通道数， 参2：输出通道数，参3：卷积核大小，参4：步长，参5：填充
    conv = nn.Conv2d(in_channels = 4, out_channels = 5, kernel_size = 3, stride = 1, padding = 1)
    # 1.7. 将图像数据输入卷积层
    feature_map = conv(img_tensor)
    # print(f'feature_map: {feature_map} feature_map.shape: {feature_map.shape}')
    # 1.8. 查看提取的4个特征图
    img1 = conv(img_tensor)[0]
    # print(f'img1: {img1} img1.shape: {img1.shape}')
    # 1.9. 将特征图数据转换为图像数据
    img2 = img1.permute(1, 2, 0)
    # print(f'img2: {img2} img2.shape: {img2.shape}')
    # 1.10. 可视化第一个特征图
    freature1 = img2[:, :, 4].detach().numpy()     # 第1通道的像素图
    plt.imshow(freature1)
    plt.show()


# 2. 测试
if __name__ == '__main__':
    dm01()
