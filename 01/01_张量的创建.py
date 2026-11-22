"""
案例：
    张量的创建

张量(Tensor)：
    pytorch框架属于最常用的深度学习框架，ANN（人工神经网络），CNN（卷积神经网络），RNN（循环神经网络）
    底层数据处理，都使用的是张量来处理的

    张量->存储同一类型元素的容器，且元素必须是数值

张量基本的创建方式：
    torch.tensor    根据指定数据创建张量
    torch.Tensor    根据形状创建张量，还可用来创建指定数据的张量
    torch.IntTensor, torch.FloatTensor, torch.DoubleTensor  创建指定类型的张量

"""

#导包
import torch
import numpy as np

#1.定义函数，演示： torch.tensor    根据指定数据创建张量

def dm01():
    #场景一：标量 -> 张量
    t1 = torch.tensor(10)
    print(f't1:{t1}, type:{type(t1)}')
    print('-'*30)

    #场景二：列表 -> 张量
    data = [[1, 2, 3],[4, 5, 6]]
    t2 = torch.tensor(data)
    print(f't2:{t2}, type:{type(t2)}')
    print('-'*30)

    #场景三：numpy nd数组 -> 张量
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.tensor(data)
    print(f't3:{t3}, type:{type(t3)}')
    print('-'*30)

    #场景四：尝试直接创建张量 指定维度
    t4 = torch.tensor([2, 3])
    print(f't4:{t4}, type:{type(t4)}')
    print('-'*30)

#2.定义函数，演示： torch.Tensor    根据形状创建张量，还可用来创建指定数据的张量

def dm02():
    #场景一：标量 -> 张量
    t1 = torch.Tensor(10)
    print(f't1:{t1}, type:{type(t1)}')
    print('-'*30)

    #场景二：列表 -> 张量
    data = [[1, 2, 3],[4, 5, 6]]
    t2 = torch.Tensor(data)
    print(f't2:{t2}, type:{type(t2)}')
    print('-'*30)

    #场景三：numpy nd数组 -> 张量
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.Tensor(data)
    print(f't3:{t3}, type:{type(t3)}')
    print('-'*30)

    #场景四：尝试直接创建张量 指定维度
    t4 = torch.Tensor(2, 3)
    print(f't4:{t4}, type:{type(t4)}')
    print('-'*30)

#3.定义函数，演示： torch.IntTensor, torch.FloatTensor, torch.DoubleTensor  创建指定类型的张量
def dm03():
    #场景一：标量 -> 张量
    t1 = torch.IntTensor(10)
    print(f't1:{t1}, type:{type(t1)}')
    print('-'*30)

    #场景二：列表 -> 张量
    data = [[1, 2, 3],[4, 5, 6]]
    t2 = torch.IntTensor(data)
    print(f't2:{t2}, type:{type(t2)}')
    print('-'*30)

    #场景三：numpy nd数组 -> 张量
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.IntTensor(data)
    print(f't3:{t3}, type:{type(t3)}')
    print('-'*30)

    #场景四：尝试直接创建张量 指定维度
    data = np.random.randint(0, 10, size=(2, 3))
    t4 = torch.FloatTensor(data)    #默认：float32
    print(f't4:{t4}, type:{type(t4)}')
    print('-'*30)

#4.定义测试函数
if __name__ == '__main__':
    dm01()
    #dm02()
    #dm03()