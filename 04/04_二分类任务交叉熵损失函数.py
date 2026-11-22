"""
案例:
    演示二分类任务的损失函数.

二分类任务的损失函数(BCELoss):
    公式:
        Loss = -ylog(预测值) - (1 - y)log(1 - 预测值)
    细节:
        因为公式中没有包含Sigmoid激活函数, 所以使用BCELoss的时候, 还需要手动指定 Sigmoid.
"""

# 导包
import torch
import torch.nn as nn

# 1. 定义函数，演示：二分类损失函数
def dm01():
    # 1.1 手动创建真实值 -> 上述的y
        y_true = torch.tensor([0, 1, 0], dtype=torch.float)
        # y_true = torch.tensor([1, 2])

        # 1.2 手动创建预测值 -> 上述的预测值
        y_pred = torch.tensor([0.7152, 0.3874, 0.1162])

        # 1.3 创建多分类交叉熵损失函数
        criterion = nn.BCELoss()       # 平均损失，来源与参数：reduction: str = 'mean'

        # 1.4 计算损失
        loss = criterion(y_pred, y_true).detach().numpy()

        # 1.5 打印损失
        print(f'损失值:{loss}' )

# 2. 演示
if __name__ == '__main__':
    dm01()