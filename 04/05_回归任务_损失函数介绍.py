"""
案例:
    演示 回归任务的损失函数介绍.

回归任务常用损失函数如下:
MAE:  Mean Absolute Error, 平均绝对误差.
    公式:
        误差绝对值之和 / 样本总数
    类似于L1正则化, 权重可以降为0, 数据会变得稀疏.

    弊端:
        在0点不平滑, 可能错过最小值.

MSE:    Mean Squared Error, 均方误差.
    公式:
        误差平方之和 / 样本总数
    弊端:
        如果差值过大, 可能存在梯度爆炸的情况.

Smooth L1:
    就是基于MAE 和 MSE做的综合, 在 [-1, 1]是 L2(MSE), 其它段时L1.
    这样即解决了L1不平滑的问题(0点不可导, 可能错过最小值)
    又解决了L2(MSE)的 梯度爆炸的问题.

"""

# 导包
import torch
import torch.nn as nn

# 1. 定义函数，演示：MAE损失函数
def dm01():
    # 1.1 手动创建真实值 -> 上述的y_true
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)

    # 1.2 手动创建预测值 -> 上述的y_pred
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)

    # 1.3 创建MAE损失函数对象
    loss = nn.L1Loss()

    # 1.4 计算损失
    result = loss(y_pred ,y_true)

    # 1.5 打印结果
    print(f'MAE: {result}')

# 2. 定义函数，演示：MSE损失函数
def dm02():
    # 2.1 手动创建真实值 -> 上述的y_true
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)

    # 2.2 手动创建预测值 -> 上述的y_pred
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)

    # 2.3 创建MSE损失函数对象
    loss = nn.MSELoss()

    # 2.4 计算损失
    result = loss(y_pred ,y_true)

    # 2.5 打印结果
    print(f'MSE: {result}')

# 3. 定义函数，演示：Smooth L1损失函数
def dm03():
    # 3.1 手动创建真实值 -> 上述的y_true
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)

    # 3.2 手动创建预测值 -> 上述的y_pred
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)

    # 3.3 创建Smooth L1损失函数对象
    loss = nn.SmoothL1Loss()

    # 3.4 计算损失
    result = loss(y_pred ,y_true)

    # 3.5 打印结果
    print(f'Smooth L1: {result}')

# 4. 调用函数
if __name__ == '__main__':
    dm01()
    dm02()
    dm03()