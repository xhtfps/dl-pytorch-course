"""
案例:
    代码演示批量归一化，它(批量归一化)也属于正则化的一种，也是用于 缓解模型的 过拟合情况的。

批量归一化:
    思路:
        先对数据做标准化(会丢失一些信息)，然后再对数据做 缩放(λ，理解为: w权重) 和 平移(β，理解为: b偏置)，再找补回一些信息。
    应用场景:
        批量归一化在计算机视觉领域使用较多。

        BatchNorm1d：主要应用于全连接层或处理一维数据的网络，例如文本处理。它接收形状为 (N, num_features) 的张量作为输入。
        BatchNorm2d：主要应用于卷积神经网络，处理二维图像数据或特征图。它接收形状为 (N, C, H, W) 的张量作为输入。
        BatchNorm3d：主要用于三维卷积神经网络（3D CNN），处理三维数据，例如视频或医学图像。它接收形状为 (N, C, D, H, W) 的张量作为输入。

"""


# 导包
import torch
import torch.nn as nn

# 1.定义函数:处理二位数据的批量归一化
def batch_norm_2d():
    # 1.1 定义一个二维数据
    # 一张图片，3个通道，3*4的像素点
    data = torch.randn(1, 3, 3, 4)  # (N, C, H, W)
    print(f'原始数据: {data}')

    # 1.2 定义一个批量归一化层（BN层）
    # 参1：通道数，参2：一个小的常数，防止分母为0，参3：动量系数，参4：是否学习缩放和平移参数
    batch_norm = nn.BatchNorm2d(num_features=3, eps=1e-5, momentum=0.1, affine=True)  

    # 1.3 对数据进行批量归一化
    # 参1：输入数据，参2：是否使用训练模式
    data = batch_norm(data)
    print(f'批量归一化后: {data}')

# 2. 定义函数:处理一维数据的批量归一化
def batch_norm_1d():
    # 2.1 定义一个一维数据
    # 2个样本，每个样本有2个特征
    data = torch.randn(2, 2)
    print(f'原始数据: {data}')

    # 2.2 创建线性层
    linear = nn.Linear(2, 4)  # 输入特征数为2，输出特征数为4

    # 2.3 对数据进行线性变换
    data = linear(data)
    print(f'线性变换后: {data}')

    # 2.4 定义一个批量归一化层（BN层）
    # 参1：特征数，参2：一个小的常数，防止分母为0，参3：动量系数，参4：是否学习缩放和平移参数
    batch_norm = nn.BatchNorm1d(num_features=4, eps=1e-5, momentum=0.1, affine=True)

    # 2.5 对数据进行批量归一化
    # 参1：输入数据，参2：是否使用训练模式
    data = batch_norm(data)
    print(f'批量归一化后: {data}')




# 3. 测试
if __name__ == '__main__':
    # batch_norm_2d()
    batch_norm_1d()

