"""
案例:
    演示近30天，天气分布情况。

结论:
    针对于 β(调节权重系数)来讲，其值越大说明: 越依赖指数加权平均，越不依赖本地的梯度值，数据就越: 平缓。
"""

import torch
import matplotlib.pyplot as plt

ELEMENT_MUMBER = 30

# 1. 实际平均温度
def dm01():
    # 固定随机种子
    torch.manual_seed(0)
    # 产生30天的随机温度
    temperature = torch.randn(size = [ELEMENT_MUMBER, ]) * 10
    print(temperature)
    # 绘制平均温度
    days = torch.arange(1, ELEMENT_MUMBER + 1, 1)
    plt.plot(days, temperature, color = 'r')
    plt.scatter(days, temperature)
    plt.show()

# 2. 指数加权平均温度
def dm02(beta):
    # 固定随机种子
    torch.manual_seed(0)
    # 产生30天的随机温度
    temperature = torch.randn(size = [ELEMENT_MUMBER, ]) * 10
    print(temperature)
    # 指数加权平均温度
    exp_weight_avg = []
    for idx, temp in enumerate(temperature, 1):
        # 第一个元素的EWA值等于自身
        if idx == 1:
            exp_weight_avg.append(temp)
            continue
        # 第二个元素的EWA值等于上一个EWA乘以β + 自身乘以(1-β)
        # idx-2 : 2-2 = 0, exp_weight_avg列表第一个值的下标
        new_temp = exp_weight_avg[idx - 2] * beta + temp * (1 - beta)
        exp_weight_avg.append(new_temp)

    # 绘制平均温度
    days = torch.arange(1, ELEMENT_MUMBER + 1, 1)
    plt.plot(days, exp_weight_avg, color = 'r')
    plt.scatter(days, temperature)
    plt.show()

# 3. 测试
if __name__ == '__main__':
    dm01()      # 不考虑权重系数，每个权重一致
    dm02(0.5)   # 考虑权重系数，越依赖指数加权平均，β越大，数据:就越平缓
    dm02(0.9)   