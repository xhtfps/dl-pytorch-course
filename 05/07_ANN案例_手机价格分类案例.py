"""
案例:
    ANN(人工神经网络)案例: 手机价格分类案例.

背景:
    基于手机的20列特征 → 预测手机的价格区间(4个区间), 可以用机器学习做, 也可以用深度学习做(推荐)

ANN案例的实现步骤：
    1. 准备数据集
    2. 搭建神经网络
    3. 训练神经网络
    4. 测试神经网络
"""

# 导包
import torch                                  # PyTorch框架，封装了张量的各种操作
from torch.utils.data import TensorDataset    # 数据集对象. 数据 → Tensor → 数据集 → 数据加载器
from torch.utils.data import DataLoader       # 数据加载器.
import torch.nn as nn                         # neural network, 封装了神经网络的各种操作
import torch.optim as optim                   # 优化器
from sklearn.model_selection import train_test_split  # 训练集和测试集的划分
import matplotlib.pyplot as plt               # 绘图
import numpy as np                            # 数组(矩阵)操作
import pandas as pd                           # 数据处理
import time                                    # 时间模块

# 1.定义函数，构建数据集
def create_dataset():
    # 1.1 准备数据集
    data = pd.read_csv('./data/手机价格预测.csv')
    # print(f'原始数据集: {data.head()}')
    # print(f'原始数据集大小: {data.shape}')

    # 1.2 获取x特征列和y标签列
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    # print(f'特征列: {x.head()}, {x.shape}')
    # print(f'标签列: {y.head()}, {y.shape}')

    # 1.3 数据预处理
    x = x.astype(np.float32)  # 转换为float32类型
    y = y.astype(np.int64)    # 转换为int64类型
    # print(f'特征列: {x.head()}, {x.shape}, 标签列: {y.head()}, {y.shape}')

    # 1.4 数据集的划分
    # 参1：特征列，参2：标签列，参3：测试集占比，参4：随机种子，参5：分层抽样
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)  # 分层抽样，保证训练集和测试集的类别分布一致

    # 5. 把数据转换成张量数据集：数据 → 张量 Tensor → 数据集 TensorDataset → 数据加载器 DataLoader
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))   # 训练集转换为张量数据集
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))      # 测试集转换为张量数据集
    # print(f'训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}')

    # 6. 把数据集封装成数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练集数据加载器
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 测试集数据加载器
    # print(f'训练集数据加载器大小: {len(train_loader)}, 测试集数据加载器大小: {len(test_loader)}')

    # 7. 返回结果
    return train_dataset, test_dataset, train_loader, test_loader, x_train.shape[1], len(np.unique(y))  # 返回训练集、测试集、训练集数据加载器、测试集数据加载器、输入特征数、输出标签数


# 5.测试
if __name__ == '__main__':
    # 5.1 构建数据集
    train_dataset, test_dataset, train_loader, test_loader, input_dim, output_dim = create_dataset()
    # 5.2 打印结果
    print(f'训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}, 输入特征数: {input_dim}, 输出标签数: {output_dim}')