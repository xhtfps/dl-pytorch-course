"""
案例:
    演示张量的拼接操作。

涉及到的API:
cat()     不改变维度数，拼接张量。除了拼接的那个维度外，其它维度数必须保持一致。
stack()   会改变维度数，拼接张量。所有的维度都必须保持一致。
"""

# 导包
import torch

#设置随机种子
torch.manual_seed(24)

def dm01():
    """
    张量拼接操作示例函数
    展示了PyTorch中两种张量拼接方法：cat和stack的区别和使用
    """
    # 1.创建张量
    # 创建一个2行3列的随机整数张量，数值范围在1到10之间
    t1 = torch.randint(1, 10,(2, 3))
    print(f't1:{t1}, shape:{t1.shape}')
    # 创建另一个2行3列的随机整数张量，数值范围在1到10之间
    t2 = torch.randint(1, 10,(2, 3))
    print(f't2:{t2}, shape:{t2.shape}')

    # 2.拼接张量
    # 思路1：cat拼接张量
    t3 = torch.cat((t1, t2), dim=0)     #（2，3）+（2，3）=（4，3）
    print(f't3:{t3}, shape:{t3.shape}')

    t4 = torch.cat((t1, t2), dim=1)     #（2，3）+（2，3）=（2，6）
    print(f't4:{t4}, shape:{t4.shape}')

    # t5 = torch.cat((t1, t2), dim=2)     #越界，报错
    # print(f't5:{t5}, shape:{t5.shape}')
    print('-' * 30)

    #思路2：stack拼接张量，可以是新维度，所有维度必须一致
    t6 = torch.stack((t1, t2), dim=0)     #（2，3）+（2，3）=（2，2，3）
    print(f't6:{t6}, shape:{t6.shape}')

    t7 = torch.stack((t1, t2), dim=1)     #（2，3）+（2，3）=（2，2，3）
    print(f't7:{t7}, shape:{t7.shape}')

    t8 = torch.stack((t1, t2), dim=2)     #（2，3）+（2，3）=（2，3，2）
    print(f't8:{t8}, shape:{t8.shape}')

    # t9 = torch.stack((t1, t2), dim=3)     #越界，报错
    # print(f't9:{t9}, shape:{t9.shape}')

if __name__ == '__main__':
    dm01()