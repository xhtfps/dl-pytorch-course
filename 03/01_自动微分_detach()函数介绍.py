"""
案例:
    演示 detach()函数的功能，解决 自动微分的弊端。

回顾:
自动微分 = 求导，即: 基于损失函数，计算梯度。
结合权重更新公式: W新 = W旧 - 学习率 * 梯度，来更新权重的。

问题:
一个张量一旦设置了 自动微分，这个张量就不能直接转成 numpy 的 ndarray对象了，需要通过 detach()函数解决
"""

# 导包
import torch
import numpy as np

def dm01():

    """
    演示PyTorch张量与NumPy数组之间的转换，以及自动微分的处理
    
    这个函数展示了如何处理带有requires_grad=True的张量转换为NumPy数组时的问题，
    并演示了使用detach()函数的正确方法。
    """
    #1. 定义一个张量，requires_grad=True表示需要计算梯度，dtype=torch.float指定浮点类型
    t1 = torch.tensor([1, 2, 3, 4, 5], requires_grad=True, dtype = torch.float)
    print(f't1 = {t1}, type = {type(t1)}')

    # #2. 尝试将 t1 转成 numpy 的 ndarray对象
    # n1 = t1.numpy()
    # print(f'n1 = {n1}, type = {type(n1)}')

    #3. 解决方法：使用 detach()函数，拷贝一份张量，然后转换
    t2 = t1.detach()
    print(f't2 = {t2}, type = {type(t2)}')

    #4. 测试t1,t2是否共享同一块内存 -> 共享
    t1.data[0] = 100
    print(f't1 = {t1}, type = {type(t1)}')
    print(f't2 = {t2}, type = {type(t2)}')
    print('-' * 30)

    #5. 查看t1和t2谁可以自动微分
    print(f't1.requires_grad = {t1.requires_grad}')
    print(f't2.requires_grad = {t2.requires_grad}')

    #6. 把t2转成 numpy 的 ndarray对象
    n2 = t2.numpy()
    print(f'n2 = {n2}, type = {type(n2)}')

    #7. 最终版
    n2 = t1.detach().numpy()
    print(f'n2 = {n2}, type = {type(n2)}')


if __name__ == '__main__':
    dm01()