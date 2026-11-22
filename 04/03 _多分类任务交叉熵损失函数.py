"""
概述:
损失函数也叫成本函数，目标函数，代价函数，误差函数，就是用来衡量模型好坏(模型拟合情况)的。

分类:
分类问题:
	多分类交叉熵损失: CrossEntropyLoss
	二分类交叉熵损失: BCELoss

回归问题:
	MAE: Mean Absolute Error, 平均绝对误差。
	MSE: Mean Squared Error, 均方误差。
	Smooth L1: 结合上述两个的特点做的升级，优化。

多分类交叉熵损失: CrossEntropyLoss
设计思路：
Loss = - ∑y log(S(f(x)))

简单记忆：
x: 		样本
f(x): 	加权求和
S(f(x))：处理后的概率
y: 		样本x属于某一个类别的真实概率。

大白话解释：
损失函数结果 = 最小化正确类别概率的对数的最小化。

细节：
    CrossEntropyLoss = Softmax () + 损失计算，后续如果用这个损失函数，则：输出层不能加 softmax ()
"""

# 导入相关库
import torch
import torch.nn as nn


# 1. 多分类交叉熵损失函数
def dm01():
    """
    演示多分类交叉熵损失的计算过程
    该函数手动创建真实值和预测值，并计算它们之间的多分类交叉熵损失。
    这通常用于分类任务的模型训练中，用于衡量模型预测与真实标签之间的差异。
    """
    # 1.1 手动创建真实值 -> 上述的y
    y_true = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32)
    # y_true = torch.tensor([1, 2])

    # 1.2 手动创建预测值 -> 上述的S(f(x))
    y_pred = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]], dtype=torch.float32, requires_grad=True)

    # 1.3 创建多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()       # 平均损失，来源与参数：reduction: str = 'mean'

    # 1.4 计算损失
    loss = criterion(y_pred, y_true)
    
    # 1.5 打印损失
    print(f'损失值:{loss}' )


# 2.测试
if __name__ == '__main__':
    dm01()