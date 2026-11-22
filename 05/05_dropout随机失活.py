"""
案例:
    代码演示 随机失活.

正则化的作用:
    缓解模型的过拟合情况.

正则化的方式:
    L1正则化: 权重可以变为0, 相当于: 降维.
    L2正则化: 权重可以无限接近0
    DropOut: 随机失活, 每批次样本训练时, 随机让一部分神经元死亡, 防止一些特征对结果的影响较大(防止过拟合)
    BN(批量归一化): ...
"""

# 导包
import torch
import torch.nn as nn

# 1. 定义函数，演示：Dropout 随机失活
def dropout_demo():
    # 1. 创建隐藏层输入结果
    t1 = torch.randint(1, 10, (3, 4), dtype=torch.float32)
    print(f'隐藏层输入结果: {t1}')       

    # 2.进行下一层，加权求和和激活函数计算
    # 2.1 创建全连接层（充当线形成）
    # 参1：输入特征数, 参2：输出特征数
    linear = nn.Linear(4, 5)

    # 2.2 加权求和
    t2 = linear(t1)
    print(f'加权求和结果: {t2}')        

    # 2.3 激活函数计算
    t3 = torch.sigmoid(t2)
    print(f'激活函数计算结果: {t3}')   

    # 3. Dropout 随机失活，只有训练阶段有
    # 3.1 创建Dropout对象
    #参1：失活比例
    dropout = nn.Dropout(p=0.5)     #每个神经元有50%的概率被随机失活  

    # 3.2 Dropout随机失活
    t4 = dropout(t3)    
    print(f'Dropout随机失活结果: {t4}')

# 2. 测试
if __name__ == '__main__':
    dropout_demo()
