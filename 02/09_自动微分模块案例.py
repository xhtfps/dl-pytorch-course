"""
案例:
    演示自动微分模块，循环实现 计算梯度，更新参数。

需求:
    求 loss = w**2 + 20 的极小值点 并打印y是最小值时 w的值(梯度)

解题步骤:
    1. 定义点 w=10 requires_grad=True dtype=torch.float32
    2. 定义函数 loss = w**2 + 20
    3. 利用梯度下降法 循环迭代1000 求最优解
    3.1 正向计算(前向传播)
    3.2 梯度清零 w.grad.zero_()
    3.3 反向传播
    3.4 梯度更新 w.data = w.data - 0.01 * w.grad

回顾：
    W新 = W旧 - 学习率 * 梯度
"""

# 导包
import torch

def dm01():
    """
    演示使用PyTorch进行梯度下降优化的基本过程
    通过迭代更新权重w来最小化损失函数loss = w^2 + 20
    """
    # 1. 定义点 w=10 requires_grad=True dtype=torch.float32
    #参1: 值，参2: 是否需要计算梯度，参3: 数据类型
    w = torch.tensor(10.0, requires_grad=True, dtype=torch.float32)

    # 2. 定义函数 loss = w**2 + 20
    loss = w**2 + 20    # 求导：loss' = 2w

    # 3. 利用梯度下降法 循环迭代1000 求最优解
    print(f'开始时权重初始值：{w.data},(0.01 * w.grad): 无, loss: {loss.item()}')   # 10, 无, 120

    #迭代100次
    for i in range(100):
        # 3.1 正向计算(前向传播)
        loss = w**2 + 20

        # 3.2 梯度清零 w.grad.zero_()   默认梯度会累加，所以需要清零
        #至此，（第一次时），还没有计算梯度，所以w.grad = None， 所以需要判断是否为None
        if w.grad is not None:
            w.grad.zero_()

        # 3.3 反向传播
        loss.sum().backward()

        # 3.4 梯度更新 w.data = w.data - 0.01 * w.grad
        if w.grad is not None:
            w.data = w.data - 0.01 * w.grad

        # 3.5 打印本次梯度更新后的权重值
        if w.grad is not None:
            print(f'第{i+1}次权重值：{w.data}, (0.01 * w.grad): {0.01 * w.grad:.5f}, loss: {loss.item():.5f}, 梯度值为：{w.grad}')

    # 4. 打印最终结果
    print(f'结束时权重值：{w.data}, 梯度: {w.grad}, loss: {loss.item()}')

if __name__ == '__main__':
    dm01()