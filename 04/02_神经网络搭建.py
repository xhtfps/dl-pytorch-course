"""
案例:
  演示神经网络搭建流程.

深度学习案例的4个步骤:
  1. 准备数据.
  2. 搭建神经网络
  3. 模型训练
  4. 模型测试

神经网络搭建流程:
  1. 定义一个类, 继承: nn.Module
  2. 在__init__()方法中, 搭建神经网络.
  3. 在 forward()方法中,完成: 前向传播.
"""

# 导包
import torch
import torch.nn as nn
from torchsummary import summary  # 计算模型参数,查看模型结构, pip install torchsummary -i https://mirrors.aliyun.com/pypi/simple/

# 1. 搭建神经网络，即定义一个类，继承nn.Module
class ModeDemo(nn.Module):
    """
    一个简单的神经网络模型演示类，继承自nn.Module。
    该模型包含两个隐藏层和一个输出层，使用了不同的激活函数。
    """
    # 1. 在__init__()方法中, 完成初始化：父类成员及神经网络搭建
    def __init__(self):
        # 1.1 调用父类成员
        super().__init__()

        # 1.2 定义神经网络 -> 隐藏层 + 输出层
        # 1.2.1 定义隐藏层1， 输入特征：3，输出特征：3
        self.linear1 = nn.Linear(3, 3)
        # 1.2.2 定义隐藏层2， 输入特征：3，输出特征：2
        self.linear2 = nn.Linear(3, 2)
        # 1.2.3 定义输出层， 输入特征：2，输出特征：2
        self.output = nn.Linear(2, 2)

        # 1.3 对隐藏层参数初始化
        # 1.3.1 初始化隐藏层1
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        # 1.3.2 初始化隐藏层2
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    # 2 前向传播：输入层->隐藏层->输出层
    def forward(self, x):
        # 2.1 第一层 隐藏层计算：加权求和 + 激活函数（sigmoid）
        #分解版
        # x = self.linear1(x)         # 加权求和
        # x = torch.sigmoid(x)        # 激活函数
        # 合并版
        x = torch.sigmoid(self.linear1(x))

        # 2.2 第二层 隐藏层计算：加权求和 + 激活函数（relu）
        x = torch.relu(self.linear2(x))
        
        # 2.3 第三层 输出层计算：加权求和 + 激活函数（Softmax）
        # dim = -1 表示最后一个维度, 按行计算
        x = torch.softmax(self.output(x), dim = -1)

        # 2.4 返回输出结果
        return x

# 2. 模型训练
def train():
    """
    训练函数，演示神经网络的基本工作流程
    包括模型创建、数据生成、前向传播、参数查看等步骤
    """
    # 1.创建模型对象
    my_model = ModeDemo()     # 底层自动调用了forward(), 即完成了前向传播
    # print(f'my_model:{my_model}')

    # 2. 创建数据集样本，随机生成
    data = torch.randn(5, 3)  # 5个样本，每个样本3个特征
    print(f'data:{data},\n data.shape:{data.shape},\n require_grad:{data.requires_grad}')

    # 3. 调用神经网络模型，进行模型训练
    output = my_model(data)
    print(f'output:{output},\n output.shape:{output.shape},\n require_grad:{output.requires_grad}')
    print('-' * 30)

    # 4. 计算和查看模型参数
    print('=========================计算模型参数=========================')
    # 参1: 模型对象，参2: 输入数据维度(样本数 ，特征数)
    summary(my_model, input_size=(5, 3), device='cpu')

    print('=========================计算模型参数=========================')
    # 遍历模型的所有参数，打印参数名称和对应的值
    for name, param in my_model.named_parameters():
        print(f'name:{name},\n param:{param}\n')


# 3. 模型测试
if __name__ == '__main__':
    train()
