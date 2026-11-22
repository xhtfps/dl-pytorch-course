"""
案例：
    张量的常用的运算函数

涉及到的API(函数)如下：
    sum(), max(), min(), mean()                     ->  都有dim参数, 0表示列, 1表示行
    pow(), sqrt(), exp(), log(), log2(), log10()    ->  无dim参数

掌握函数：
    sum(), max(), min(), mean(), pow()
"""

#导包
import torch


def dm1():
    #1.定义张量, 定义初值
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]]).type(torch.float)

    #2.演示有dim参数的函数
    #sum() 求和
    print(t1.sum(dim=0))    #按列求和
    print(t1.sum(dim=1))    #按行求和
    print(t1.sum())         #整体求和

    #max() 求最大值
    print(t1.max(dim=0))    #按列求最大值
    print(t1.max(dim=1))    #按行求最大值
    print(t1.max())         #整体求最大值

    #min() 求最小值
    print(t1.min(dim=0))    #按列求最小值
    print(t1.min(dim=1))    #按行求最小值
    print(t1.min())         #整体求最小值

    #mean() 求最小值
    print(t1.mean(dim=0))    #按列求平均值
    print(t1.mean(dim=1))    #按行求平均值
    print(t1.mean())         #整体求平均值
    print('-' * 30)

    #3.演示无dim参数的函数
    #pow() 求n次幂
    print(t1.pow(2))
    print(t1 ** 2)      #效果同上
    print('-' * 30)

    #sqrt() 求平方根
    print(t1.sqrt())
    print('-' * 30)

    #exp() e的n次幂, n是矩阵中的每个元素
    print(t1.exp())
    print('-' * 30)

    #log(), log2(), log10()
    print(t1.log())
    print('-' * 30)
    print(t1.log2())
    print('-' * 30)
    print(t1.log10())
    print('-' * 30)
#演示
if __name__ == '__main__':
    dm1()
    