"""
案例：
    张量的点乘和矩阵乘法操作

点乘:
    两个张量的维度一致, 对应元素做相应的操作
    API:
        t1 * t2
        t1.mul(t2)      #multiply: 乘法

矩阵乘法:
    要求: 第一个张量的列数等于第二个张量的行数(A列 = B行)
    结果: A行B列
    API:
        t1 @ t2
        t1.matmul(t2)
        t1.dot(t2)      扩展: 只针对一维张量有效

掌握：
    t1 * t2
    t1 @ t2
"""

#导包
import torch

#1.定义函数, 演: 张量点乘
def dm1():
    #1. 创建张量, 2行3列
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f't1:{t1}')

    #2. 创建张量, 2行3列
    t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f't2:{t2}')

    #3. 演示张量点乘操作
    t3 = t1 * t2
    # t3 = t1.multiply(t2)        #效果相同

    #4. 打印结果
    print(f't3:{t3}')

    print('-'*30)

#2.定义函数, 演示: 张量矩阵乘法
def dm2():
    #1. 创建张量, 2行3列
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f't1:{t1}')

    #2. 创建张量, 3行3列
    t2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f't2:{t2}')

    #3. 演示张量矩阵乘法操作
    t3 = t1 @ t2
    # t3 = t1.matmul(t2)        #效果相同

    #4. 打印结果
    print(f't3:{t3}')

    #5. 演示dot函数
    t4 = torch.tensor([1, 2, 3])
    t5 = torch.tensor([4, 5, 6])
    t6 = t4.dot(t5)
    print(f't6:{t6}')
    
    print('-'*30)

#3.演示
if __name__ == '__main__':
    # dm1()
    dm2()
    