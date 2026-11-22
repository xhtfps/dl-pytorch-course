"""
案例：
    演示张量和numpy之间相互转换，以及如何从标量张量中提取器内容

涉及到的API:

    1.张量 -> numpy
        张量对象.numpy()            共享内存
        张量对象.numpy().copy()     不共享内存，链式编程写法

    2.numpy -> 张量
        torch.from_numpy()                共享内存
        torch.tensor(nd数组)              不共享内存
    3.标量张量 -> 标量
        张量对象.item()  将标量张量转换为标量

掌握：
    张量 -> numpy:      张量对象.detach().numpy()
    numpy -> 张量:      torch.tensor(nd数组)
    标量张量 -> 标量:    张量对象.item()
"""

#导包
import torch
import numpy as np

#1.定义函数，演示：张量->numpy
def dm01():
    #1.创建张量
    t1 = torch.tensor([1, 2, 3, 4, 5])
    print(f't1:{t1}, 张量类型:{type(t1)}')
    print('-'*30)

    #2.张量转numpy
    # n1 = t1.numpy()       #共享内存
    n1 = t1.numpy().copy()  #不共享内存
    print(f'n1:{n1}, 张量类型:{type(n1)}')
    print('-'*30)

    #3.演示上述方式是否共享内存
    n1[0] = 100
    print(f't1:{t1}')
    print(f'n1:{n1}')
    print('-'*30)


#2.定义函数，演示：numpy -> 张量
def dm02():
    #1.创建numpy数组
    n1 = np.array([11, 22, 33])
    print(f'n1:{n1}, 张量类型:{type(n1)}')

    #2.numpy转张量
    # t1 = torch.from_numpy(n1).type(torch.float32)   #转换加转类型
    t1 = torch.from_numpy(n1)       #共享内存
    print(f't1:{t1}, 张量类型:{type(t1)}')

    t2 = torch.tensor(n1)           #不共享内存
    print(f't2:{t2}, 张量类型:{type(t2)}')

    #3.演示上述方式是否共享内存
    n1[0] = 200
    print(f'n1:{n1}')
    print(f't1:{t1}')
    print(f't2:{t2}')

#3.定义函数，演示：标量张量 -> 标量
def dm03():
    #1.创建张量
    # t1 = torch.tensor(100.3, )
    t1 = torch.tensor([100.3, ]).type(torch.float32)
    print(f't1:{t1}, 张量类型:{type(t1)}')

    #2.从张量中提取内容
    a = t1.item()
    print(f'a:{a}, 张量类型:{type(a)}')

#4.测试
if __name__ == '__main__':
    # dm01()
    # dm02()
    dm03()
