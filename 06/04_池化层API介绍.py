"""
案例:
    演示池化层相关操作。

池化层解释(Pooling):
    目的:
        降维。
    思路:
        最大池化。
        平均池化。
    特点:
        池化不会改变数据的 通道数。
"""

# 导包
import torch
import torch.nn as nn

# 1. 定义函数, 演示单通道池化
def dm01():
    # 1. 创建1个1通道3*3的输入数据
    input = torch.tensor([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    print("input:\n", input)

    # 2. 创建最大池化层对象
    # 参1：池化核大小， 参2：步长
    pool = nn.MaxPool2d(kernel_size=2, stride=1)
    print("pool:\n", pool)

    # 3. 调用池化层对象, 进行池化操作
    output = pool(input)
    print("output:\n", output)

    # 4. 创建平均池化层对象
    # 参1：池化核大小， 参2：步长
    pool = nn.AvgPool2d(kernel_size=2, stride=1)
    print("pool:\n", pool)

    # 5. 调用池化层对象, 进行池化操作
    output = pool(input)
    print("output:\n", output)

# 2. 定义函数, 演示多通道池化
def dm02():
    input = torch.tensor(
    [
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],

        [
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18]
        ],

        [
            [19, 20, 21],
            [22, 23, 24],
            [25, 26, 27]
        ]
    ], dtype=torch.float32)
    print("input:\n", input)

    # 2. 创建最大池化层对象
    # 参1：池化核大小， 参2：步长
    pool = nn.MaxPool2d(kernel_size=2, stride=1)
    print("pool:\n", pool)

    # 3. 调用池化层对象, 进行池化操作
    output = pool(input)
    print("output:\n", output)

    # 4. 创建平均池化层对象
    # 参1：池化核大小， 参2：步长
    pool = nn.AvgPool2d(kernel_size=2, stride=1)
    print("pool:\n", pool)

    # 5. 调用池化层对象, 进行池化操作
    output = pool(input)
    print("output:\n", output)

# 3. 测试
if __name__ == '__main__':
    # dm01()
    dm02()