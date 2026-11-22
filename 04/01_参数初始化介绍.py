"""
案例:
  演示参数初始化的 7 种方式.

参数初始化的目的:
1. 防止梯度消失 或者 梯度爆炸.
2. 提高收敛速度.
3. 打破对称性.

参数初始化的方式:
无法打破对称性的:
  全0, 全1, 固定值
可以打破对称性的:
  随机初始化, 正态分布初始化, kaiming初始化, xavier初始化

总结:
1. 记忆 kaiming初始化, xavier初始化, 全0初始化.
2. 关于初始化的选择上:
   激活函数ReLU及其系列: 优先用 kaiming
   激活函数非ReLU: 优先用 xavier
   如果是浅层网络: 可以考虑使用 随机初始化
"""

# 导入模块
import torch.nn as nn       # 神经网络模块

# 1. 随机均匀分布初始化
def dm01():
    # 1. 创建线性层，输入维度为5，输出维度为3
    linear = nn.Linear(5, 3)
    # 2. 对权重进行随机初始化，使用均匀分布
    nn.init.uniform_(linear.weight)
    # 3. 对偏置进行随机初始化，使用均匀分布
    nn.init.uniform_(linear.bias)
    # 4. 打印权重和偏置
    print(linear.weight.data)
    print(linear.bias.data)

# 2. 固定初始化
def dm02():
    # 1. 创建线性层，输入维度为5，输出维度为3
    linear = nn.Linear(5, 3)
    # 2. 对权重进行固定初始化，使用固定值3
    nn.init.constant_(linear.weight, 3)
    # 3. 对偏置进行固定初始化，使用固定值3
    nn.init.constant_(linear.bias, 3)
    # 4. 打印权重和偏置
    print(linear.weight.data)
    print(linear.bias.data)

# 3. 全0初始化
def dm03():
    # 1. 创建线性层，输入维度为5，输出维度为3
    linear = nn.Linear(5, 3)
    # 2. 对权重进行全0初始化
    nn.init.zeros_(linear.weight)
    # 3. 对偏置进行全0初始化
    nn.init.zeros_(linear.bias)
    # 4. 打印权重和偏置
    print(linear.weight.data)
    print(linear.bias.data)

# 4. 全1初始化
def dm04():
    # 1. 创建线性层，输入维度为5，输出维度为3
    linear = nn.Linear(5, 3)
    # 2. 对权重进行全1初始化
    nn.init.ones_(linear.weight)
    # 3. 对偏置进行全1初始化
    nn.init.ones_(linear.bias)
    # 4. 打印权重和偏置
    print(linear.weight.data)
    print(linear.bias.data)

# 5. 正态分布初始化
def dm05():
    # 1. 创建线性层，输入维度为5，输出维度为3
    linear = nn.Linear(5, 3)
    # 2. 对权重进行正态分布初始化
    nn.init.normal_(linear.weight)
    # 3. 对偏置进行正态分布初始化
    nn.init.normal_(linear.bias)
    # 4. 打印权重和偏置
    print(linear.weight.data)
    print(linear.bias.data)

# 6. kaiming初始化
def dm06():
    # 1. 创建线性层，输入维度为5，输出维度为3
    linear = nn.Linear(5, 3)

    # kaiming正态分布初始化
    # 2. 对权重进行kaiming初始化
    nn.init.kaiming_normal_(linear.weight)
    # 3. 对偏置进行kaiming初始化
    nn.init.zeros_(linear.bias)
    # 4. 打印权重和偏置
    print(linear.weight.data)
    print(linear.bias.data)

    # kaiming均匀分布初始化
    # 2. 对权重进行kaiming初始化
    nn.init.kaiming_uniform_(linear.weight)
    # 3. 对偏置进行kaiming初始化
    nn.init.zeros_(linear.bias)
    # 4. 打印权重和偏置
    print(linear.weight.data)
    print(linear.bias.data)

# 7. xavier初始化
def dm07():
    # 1. 创建线性层，输入维度为5，输出维度为3
    linear = nn.Linear(5, 3)

    # xavier正态分布初始化
    # 2. 对权重进行xavier初始化
    nn.init.xavier_normal_(linear.weight)
    # 3. 对偏置进行xavier初始化
    nn.init.zeros_(linear.bias)
    # 4. 打印权重和偏置
    print(linear.weight.data)
    print(linear.bias.data)

    # xavier均匀分布初始化
    # 2. 对权重进行xavier初始化
    nn.init.xavier_uniform_(linear.weight)
    # 3. 对偏置进行xavier初始化
    nn.init.zeros_(linear.bias)
    # 4. 打印权重和偏置
    print(linear.weight.data)
    print(linear.bias.data)


if __name__ == '__main__':
    # dm01()    # 1. 均匀分布随机初始化
    # dm02()      # 2. 固定初始化
    # dm03()      # 3. 全0初始化
    # dm04()      # 4. 全1初始化
    # dm05()      # 5. 正态分布初始化
    # dm06()      # 6. kaiming初始化
    dm07()      # 7. xavier初始化