"""
案例:
    演示学习率衰减策略。

学习率衰减策略介绍:
    目的:
        较之于AdaGrad, RMSProp, Adam方式, 我们可以通过 等间隔, 指定间隔, 指数等方式, 来手动控制学习率的调整。

    分类:
        等间隔学习率衰减
        指定间隔学习率衰减
        指数学习率衰减

等间隔学习率衰减:
    step_size: 间隔的轮数, 即: 多少轮调整一次学习率。
    gamma:     学习率衰减系数, 即: lr新 = lr旧 * gamma

指定间隔学习率衰减:
    milestones = [50, 125, 160]    里边定义的是要调整学习率的 轮数.
    gamma:        学习率衰减系数, 即: lr新 = lr旧 * gamma

指数间隔学习率衰减:
    前期学习率衰减快, 中期慢, 后期更慢，更符合梯度下降规律。
    公式:
        lr新 = lr旧 * gamma ** epoch

总结:
    等间隔学习率衰减:
        优点:
            直观, 易于调试, 适用于 大批量数据.
        缺点:
            学习率变化较大, 可能跳过最优解.
        应用场景:
            大型数据集, 较为简单的任务.

    指定间隔学习率衰减:
        优点:
            易于调试, 稳定训练过程.
        缺点:
            在某些情况下可能衰减过快, 导致优化提前停滞.
        应用场景:
            对训练平稳性要求较高的任务.

    指数学习率衰减:
        优点:
            平滑, 且考虑历史更新, 收敛稳定性较强.
        缺点:
            超参调节较为复杂, 可能需要更多的资源.
        应用场景:
            高精度训练, 避免过快收敛.

"""

# 导包
import torch
from torch import optim
import matplotlib.pyplot as plt

# 1. 定义函数，演示：等间隔学习率衰减
def dm01():
    # 1.定义变量，记录初始值的学习率，训练的轮数，每轮训练的批次数
    lr, epochs, iterations = 0.1, 200, 10

    # 2.创建数据集    y_true, x, w
    # 真实值
    y_true = torch.tensor([0])
    # 输入特征
    x = torch.tensor([1.0], dtype=torch.float32)
    # 权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 3.定义优化器对象，动量法 -> 加速模型收敛，减少震荡
    # 参1：待优化的参数，参2：学习率，参3：动量系数
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)

    # 4.定义学习率衰减策略
    # 思路1：创建等间隔学习率衰减对象
    # 参1：优化器对象，参2：间隔轮数，参3：衰减系数
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # 5. 创建两个列表，分别表示：每次训练轮数，训练次数，每轮训练的学习率
    # epoch_list = [0, 1, 2, 3, ... 50, 51, 52, ... 100, 101, 101, ... 150, 151, ... 199]
    # lr_list = [0.1, 0.1, 0.1, ... 0.05,    ...    0.025,    ...      0.0125,  ...  ]
    lr_list, epoch_list = [], []

    # 6. 循环遍历训练轮数，进行具体的训练
    for epoch in range(epochs):         # epoch: 0 - 199

        # 将每轮训练的学习率记录到列表中
        lr_list.append(scheduler.get_last_lr())        #获取最后的lr（学习率） 
        epoch_list.append(epoch)

        # 循环遍历，每轮每批训练
        for iteration in range(iterations):
            # 6.1 前向传播
            y_pred = x * w
            # 6.2 计算损失值
            loss = (y_pred - y_true) ** 2
            # 6.3 梯度清零
            optimizer.zero_grad()
            # 6.4 反向传播
            loss.backward()
            # 6.5 更新参数
            optimizer.step()

        # 6.6 学习率衰减策略更新学习率
        scheduler.step()

    # 7. 打印结果：
    print(f'lr.list: {lr_list}')            # lr_list = [0.1, 0.1, 0.1, ... 0.05,    ...    0.025,    ...      0.0125,  ...  ]

    # 8. 绘制学习率衰减曲线
    # x轴：训练轮数， y轴：学习率
    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()

# 2. 定义函数，演示：指定间隔学习率衰减
def dm02():
    # 1.定义变量，记录初始值的学习率，训练的轮数，每轮训练的批次数
    lr, epochs, iterations = 0.1, 200, 10

    # 2.创建数据集    y_true, x, w
    # 真实值
    y_true = torch.tensor([0])
    # 输入特征
    x = torch.tensor([1.0], dtype=torch.float32)
    # 权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 3.定义优化器对象，动量法 -> 加速模型收敛，减少震荡
    # 参1：待优化的参数，参2：学习率，参3：动量系数
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)

    # 4.定义学习率衰减策略
    # 思路1：创建等间隔学习率衰减对象
    # 参1：优化器对象，参2：间隔轮数，参3：衰减系数
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # 思路2：创建指定间隔学习率衰减对象
    # 参1：优化器对象，参2：间隔列表，参3：衰减系数列表
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 125, 160], gamma=0.5)

    # 5. 创建两个列表，分别表示：每次训练轮数，训练次数，每轮训练的学习率
    # epoch_list = [0, 1, 2, 3, ... 50, 51, 52, ... 100, 101, 101, ... 150, 151, ... 199]
    # lr_list = [0.1, 0.1, 0.1, ... 0.05,...0.025,...0.0125,...]
    lr_list, epoch_list = [], []

    # 6. 循环遍历训练轮数，进行具体的训练
    for epoch in range(epochs):         # epoch: 0 - 199

        # 将每轮训练的学习率记录到列表中
        lr_list.append(scheduler.get_last_lr())        #获取最后的lr（学习率） 
        epoch_list.append(epoch)

        # 循环遍历，每轮每批训练
        for iteration in range(iterations):
            # 6.1 前向传播
            y_pred = x * w
            # 6.2 计算损失值
            loss = (y_pred - y_true) ** 2
            # 6.3 梯度清零
            optimizer.zero_grad()
            # 6.4 反向传播
            loss.backward()
            # 6.5 更新参数
            optimizer.step()

        # 6.6 学习率衰减策略更新学习率
        scheduler.step()

    # 7. 打印结果：
    print(f'lr.list: {lr_list}')            # lr_list = [0.1, 0.1, 0.1, ... 0.05,    ...    0.025,    ...      0.0125,  ...  ]

    # 8. 绘制学习率衰减曲线
    # x轴：训练轮数， y轴：学习率
    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()

# 3. 定义函数，演示：指数学习率衰减
def dm03():
    # 1.定义变量，记录初始值的学习率，训练的轮数，每轮训练的批次数
    lr, epochs, iterations = 0.1, 200, 10

    # 2.创建数据集    y_true, x, w
    # 真实值
    y_true = torch.tensor([0])
    # 输入特征
    x = torch.tensor([1.0], dtype=torch.float32)
    # 权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 3.定义优化器对象，动量法 -> 加速模型收敛，减少震荡
    # 参1：待优化的参数，参2：学习率，参3：动量系数
    optimizer = optim.SGD([w], lr=lr, momentum=0.95)

    # 4.定义学习率衰减策略
    # 思路1：创建等间隔学习率衰减对象
    # 参1：优化器对象，参2：间隔轮数，参3：衰减系数
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # 思路2：创建指定间隔学习率衰减对象
    # 参1：优化器对象，参2：间隔列表，参3：衰减系数列表
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 125, 160], gamma=0.5)

    # 思路3：创建指数学习率衰减对象
    # 参1：优化器对象，参2：衰减系数
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # 5. 创建两个列表，分别表示：每次训练轮数，训练次数，每轮训练的学习率
    # epoch_list = [0, 1, 2, 3, ... 50, 51, 52, ... 100, 101, 101, ... 150, 151, ... 199]
    # lr_list = [0.1, 0.1, 0.1, ... 0.05,...0.025,...0.0125,...]
    lr_list, epoch_list = [], []

    # 6. 循环遍历训练轮数，进行具体的训练
    for epoch in range(epochs):         # epoch: 0 - 199

        # 将每轮训练的学习率记录到列表中
        lr_list.append(scheduler.get_last_lr())        #获取最后的lr（学习率） 
        epoch_list.append(epoch)

        # 循环遍历，每轮每批训练
        for iteration in range(iterations):
            # 6.1 前向传播
            y_pred = x * w
            # 6.2 计算损失值
            loss = (y_pred - y_true) ** 2
            # 6.3 梯度清零
            optimizer.zero_grad()
            # 6.4 反向传播
            loss.backward()
            # 6.5 更新参数
            optimizer.step()

        # 6.6 学习率衰减策略更新学习率
        scheduler.step()

    # 7. 打印结果：
    print(f'lr.list: {lr_list}')            # lr_list = [0.1, 0.1, 0.1, ... 0.05,    ...    0.025,    ...      0.0125,  ...  ]

    # 8. 绘制学习率衰减曲线
    # x轴：训练轮数， y轴：学习率
    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()

# 4. 主函数
if __name__ == '__main__':
    # 调用函数
    # dm01()
    # dm02()
    dm03()
