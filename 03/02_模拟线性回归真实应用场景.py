"""
案例:
    演示自动微分的真实应用场景。

结论:
1. 先前向传播(正向传播) 计算出 预测值(z)
2. 基于损失函数，结合 预测值(z) 和 真实值(y)，来计算 梯度。
3. 结合权重更新公式 W新 = W旧 - 学习率 * 梯度，来更新 权重。
"""

# 导包
import torch

def dm01():
    """
    这是一个简单的神经网络训练示例，展示了前向传播、损失计算、反向传播和参数更新的完整过程。
    使用PyTorch框架实现了一个基本的线性模型，通过梯度下降法更新权重和偏置。
    """
    # 1. 定义x，表示：特征(输入数据) 假设：2行5列，全1张量
    x = torch.ones(2, 5)  # 创建一个2行5列的全1张量作为输入特征
    print(f'x : {x}')  # 打印输入特征x

    # 2. 定义y，表示：标签(输出数据) 假设：2行3列，全0张量
    y = torch.zeros(2, 3)  # 创建一个2行3列的全0张量作为目标输出
    print(f'y : {y}')  # 打印目标输出y

    # 3. 定义可自动微分的权重和偏置
    w = torch.randn(5, 3, requires_grad=True)   #x @ w + b = y  # 创建权重矩阵，形状为5x3，并启用梯度计算
    print(f'w : {w}')  # 打印权重w

    b = torch.randn(3, requires_grad=True)  # 创建偏置向量，形状为3，并启用梯度计算
    print(f'b : {b}')  # 打印偏置b

    criterion = torch.nn.MSELoss()      # neural network :神经网络  # 使用均方误差损失函数

    for i in range(100):  # 迭代100次
        # 4. 前向传播(正向传播) 计算出 预测值(z)
        z = x @ w + b  # 计算预测值z，使用矩阵乘法和偏置
        print(f'z : {z}')  # 打印预测值z

        # 5. 定义损失函数
        loss = criterion(z, y)              # loss = 损失  # 计算预测值z与目标值y之间的损失
        print(f'loss : {loss}')  # 打印损失值

        # 6. 基于损失函数，进行自动微分，结合反向传播，更新权重
        loss.sum().backward()  # 计算损失相对于所有可学习参数的梯度

        # 7. 打印w, b用来更新的梯度
        print(f'更新后的 w 的梯度 : {w.grad}')  # 打印权重w的梯度
        print(f'更新后的 b 的梯度 : {b.grad}')  # 打印偏置b的梯度

        # 8. 结合权重更新公式 W新 = W旧 - 学习率 * 梯度，来更新 权重
        if w.grad is not None:  # 检查w的梯度是否存在
            w.data = w.data - 0.01 * w.grad  # 使用学习率0.01更新权重w
        if b.grad is not None:  # 检查b的梯度是否存在
            b.data = b.data - 0.01 * b.grad  # 使用学习率0.01更新偏置b

        # 9. 清除梯度
        if w.grad is not None:  # 检查w的梯度是否存在
            w.grad.zero_()  # 清除权重w的梯度
        if b.grad is not None:  # 检查b的梯度是否存在
            b.grad.zero_()  # 清除偏置b的梯度

        print(f'更新后的 w : {w}')  # 打印更新后的权重w
        print(f'更新后的 b : {b}')  # 打印更新后的偏置b
        print(f'更新后的 loss : {loss}')  # 打印更新后的损失值

    print(f'最终 z : {z}')  # 打印最终预测值z
    print(f'最终 w : {w}')  # 打印最终权重w
    print(f'最终 b : {b}')  # 打印最终偏置b
    print(f'最终 loss : {loss}')  # 打印最终损失值

    

if __name__ == '__main__':
    dm01()