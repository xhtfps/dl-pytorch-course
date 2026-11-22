"""
案例:
    演示张量的形状操作。

涉及到的API:
reshape():        不改变张量内容，改变张量的形状
unsqueeze():      在指定位置增加一个维度
squeeze():        在指定位置删除一个维度
transpose():      一次只能交换两个维度, 会改变张量的顺序
permute():        一次可以交换多个维度, 会改变张量的顺序
view():           只能修改连续张量的形状，连续张量 = 内存中的存储数据和张量中显示的顺序相同
contiguous():     把不连续的张量，变成连续的，即基于张量中显示的数据，修改内存中的存储顺序
is_contiguous():  判断张量是否连续

需要你掌握的函数:
    reshape(), unsqueeze(), permute(), view()
"""

# 导包
import torch

#指定随机种子
torch.manual_seed(24)

#1.定义函数，演示：reshape()函数
def dm01():
    """
    演示PyTorch中张量的reshape操作
    展示如何使用reshape()函数改变张量的形状
    """
    #1.定义2行3列的张量，演示：reshape()函数
    # 使用torch.randint创建一个2行3列的随机整数张量，数值范围在1到10之间
    t1 = torch.randint(1, 10, size = (2, 3))
    # 打印张量t1及其形状信息
    # 包括张量本身、形状(shape)、行数、列数和最后一个维度的大小
    print(f't1:{t1}, shanpe:{t1.shape}, row:{t1.shape[0]}, col:{t1.shape[1]}, {t1.shape[-1]}')

    #2.通过reshape()函数，将t1张量，变成3行，2列，1行6列，6行1列的
    t2 = t1.reshape(3, 2)
    print(f't2:{t2}, shanpe:{t2.shape}, row:{t2.shape[0]}, col:{t2.shape[1]}, {t1.shape[-1]}')

    #3.通过reshape()函数，将t1张量，变成2行5列
    # t3 = t1.reshape(2, 5)       #注意：张量元素个数不能变
    # print(f't3:{t3}')

    print('-' * 30)

#2.定义张量，演示：unsqueeze(), squeeze()函数
def dm02():
    """
    演示PyTorch中unsqueeze()和squeeze()函数的使用方法
    unsqueeze()用于在指定维度上增加一个大小为1的维度
    squeeze()用于删除所有大小为1的维度
    """
    #1.定义2行3列的张量，演示：unsqueeze()函数
    t1 = torch.randint(1, 10, size = (2, 3))  # 创建一个2行3列的随机整数张量，值范围在1到10之间
    print(f't1:{t1}, shape:{t1.shape}')     #(2, 3)

    #2.在0维增加一个维度
    t2 = t1.unsqueeze(0)
    print(f't2:{t2}, shape:{t2.shape}')     #(1, 2, 3)

    #3.在1维增加一个维度
    t3 = t1.unsqueeze(1)
    print(f't3:{t3}, shape:{t3.shape}')     #(2, 1, 3)

    #4.在2维增加一个维度
    t4 = t1.unsqueeze(2)
    print(f't4:{t4}, shape:{t4.shape}')     #(2, 3, 1)

    #5.在3维增加一个维度
    # t5 = t1.unsqueeze(3)       #注意：张量维度不能越界
    # print(f't5:{t5}, shape:{t5.shape}')       （2，3，？，1）
    print('-' * 30)

    #6.删除所有为1的维度
    t6 = torch.randint(1, 10, size = (2, 1, 3, 1, 1))
    print(f't6:{t6}, shape:{t6.shape}')     #(2, 1, 3, 1, 1)
    t7 = t6.squeeze()
    print(f't7:{t7}, shape:{t7.shape}')     #(2, 3)


#3.定义张量，演示：transpose(), permute()函数
def dm03():
    """
    演示PyTorch中张量的转置操作
    包括transpose()和permute()函数的使用方法
    """
    #1.定义2个3行4列的张量，演示：transpose()函数
    # 创建一个形状为(2, 3, 4)的随机整数张量，数值范围在1到10之间
    t1 = torch.randint(1, 10, size = (2, 3, 4))
    print(f't1:{t1}, shape:{t1.shape}')     #(2, 3, 4)
    print('-' * 30)

    #2.(2, 3, 4) -> (4, 3, 2)
    t2 = t1.transpose(0, 2)
    # t2 = t1.transpose(0, -1)        #注意：-1表示最后一个维度
    print(f't1:{t1}, shape:{t1.shape}')     #(2, 3, 4)
    print(f't2:{t2}, shape:{t2.shape}')     #(4, 3, 2)

    #3.(2, 3, 4) -> (4, 2, 3)
    t3 = t1.permute(2, 0, 1)
    print(f't3:{t3}, shape:{t3.shape}')     #(4, 2, 3)

#4.定义张量，演示：view(), contiguous(), is_contiguous()函数
def dm04():
    """
    演示PyTorch中张量的view()函数
    view()函数只能用于连续张量，即基于张量中显示的数据，修改内存中的存储顺序
    """
    #1.定义2个3行4列的张量，演示：view()函数
    # 创建一个形状为(2, 3, 4)的随机整数张量，数值范围在1到10之间
    # 注意：这里实际上创建的是(2, 3)形状的张量，注释可能有误
    t1 = torch.randint(1, 10, size = (2, 3))
    print(f't1:{t1}, shape:{t1.shape}')     #(2, 3, 4)

    #2.判断张量是否连续，即内存中的存储顺序和张量中显示的顺序是否相同
    # print(f't1 is contiguous:{t1.is_contiguous()}')     #True

    #3.通过view()函数，修改张量的形状，从2行3列变成3行2列
    t2 = t1.view(3, 2)
    print(f't2:{t2}, shape:{t2.shape}')     #(3, 2)
    print(f't2 is contiguous:{t2.is_contiguous()}')

    #4.transpose()函数交换维度，修改张量的形状，从2行3列变成3行2列
    t3 = t1.transpose(0, 1)
    print(f't3:{t3}, shape:{t3.shape}')     #(3, 2)
    print(f't3 is contiguous:{t3.is_contiguous()}')     #False

    #5.view()函数，修改张量的形状，从3行2列变成2行3列
    # t4 = t3.view(2, 3)       #注意：view()函数只能用于连续张量
    # print(f't4:{t4}, shape:{t4.shape}')     #(2, 3)

    #6.contiguous()函数，把不连续的张量，变成连续的，即基于张量中显示的数据，修改内存中的存储顺序
    t4 = t3.contiguous()
    print(f't4:{t4}, shape:{t4.shape}')     #(3, 2)

    #7.view()函数，修改张量的形状，从3行2列变成2行3列
    t5 = t4.view(2, 3)
    print(f't5:{t5}, shape:{t5.shape}')     #(2, 3)

#5.测试
if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    dm04()