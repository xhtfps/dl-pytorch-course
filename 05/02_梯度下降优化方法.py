"""
案例:
    演示 梯度下降优化方法.

    梯度下降相关介绍:
        概述:
            梯度下降是结合 本次损失函数的导数(作为梯度) 基于学习率 来更新权重的.
        公式:
            W新 = W旧 - 学习率 * 梯度
        存在的问题:
            1. 遇到平缓区域, 梯度下降(权重更新)可能会慢.
            2. 可能会遇到 鞍点(梯度为0)
            3. 可能会遇到 局部最小值.
        解决思路:
            从上述的 学习率 或者 梯度入手, 进行优化, 于是有了: 动量法Momentum, 自适应学习率AdaGrad, RMSProp, 综合衡量: Adam

    动量法Momentum:
        动量法公式：
            St = β * St-1 + (1 - β) * Gt

        解释：
            St:	    本次的指数移动加权平均结果。
            β：	    调节权重系数，越大，数据越平缓，历史指数移动加权平均 比重越大，本次梯度权重越小。
            St-1:	历史的指数移动加权平均结果。
            Gt:	    本次计算出的梯度(不考虑历史梯度)。

        加入动量法后的 梯度更新公式:
            W新 = W旧 - 学习率 * St

    自适应学习率：AdaGrad
        公式：
            累计平方梯度：
                St = St-1 + Gt * Gt

        解释：
            St：	累计平方梯度
            St-1：	历史累计平方梯度。
            Gt：	本次的梯度。

        学习率：
            学习率 = 学习率 / (sqrt(St) + 小常数)
            解释：
                小常数：1e-10，目的：防止分母变为0

        梯度下降公式：
            W新 = W旧 - 调整后的学习率 * Gt
        
        缺点：
            导致学习率过早过量降低，导致魔心后期学习率太小，难以找到最优解。

    自适应学习率：RMSProp -> AdaGrad的改进, 加入调和加权平均
        公式：
            指数加权平均，累计平方梯度：
                St = β * St-1 + (1 - β) * Gt * Gt

        解释：
            St：	累计平方梯度
            St-1：	历史累计平方梯度。
            Gt：	本次的梯度。
            β：	    调节权重系数，越大，数据越平缓，历史指数移动加权平均 比重越大，本次梯度权重越小。

        学习率：
            学习率 = 学习率 / (sqrt(St) + 小常数)
            解释：
                小常数：1e-10，目的：防止分母变为0

        梯度下降公式：
            W新 = W旧 - 调整后的学习率 * Gt
        
        优点：
            RMSProp通过引入衰减系数β，防止学习率过早过量降低，导致模型后期学习率太小，难以找到最优解。

    自适应矩估计：Adam(Adaptive Moment Estimation)
        思路：
            即优化学习率，又优化梯度。

        公式：
            一阶矩：算均值。
                Mt = β1 * Mt-1 + (1 - β1) * Gt        充当: 梯度
                St = β2 * St-1 + (1 - β2) * Gt * Gt  充当: 学习率

            二阶矩：梯度的方差。
                Mt^ = Mt / (1 - β1 ^ t)
                St^ = St / (1 - β2 ^ t)

        权重更新公式：
            W新 = W旧 - 学习率 / (sqrt(St^) + 小常数) * Mt^

    总结：如何选择梯度下降优化方法
        简单任务和较小的模型：
            SGD，动量法
        复杂任务或者有大量数据：
            Adam
        需要处理稀疏数据或者文本数据：
            AdaGrad，RMSProp
   
"""

# 导包
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义函数，演示：梯度下降将优化方法 -> 动量法（Momentum）
def dm01_monentum():
    # 1. 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 2. 定义损失函数
    criterion = ((w ** 2)/2.0).sum()

    # 3. 定义优化器 -> 基于SGD（随机梯度下降），加入参数momentum，即动量法
    # 优化器参数解释：参1：需要优化的参数，参2：学习率，参3：动量法参数
    optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9)     #momentum默认是0，只考虑本次梯度

    # 4. 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()        #梯度清零
    criterion.backward()         #反向传播
    optimizer.step()             #参数更新
    print(f'w: {w}, w.grad: {w.grad}')

    # 5. 重复上述步骤，第2次，更新权重参数
    # 5.1. 定义损失函数
    criterion = ((w ** 2)/2.0).sum()
    # 5.2. 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()        #梯度清零
    criterion.backward()         #反向传播
    optimizer.step()             #参数更新
    print(f'w: {w}, w.grad: {w.grad}')

# 2. 定义函数，演示：梯度下降将优化方法 -> 自适应学习率（AdaGrad）
def dm02_adagrad():
    # 1. 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 2. 定义损失函数
    criterion = ((w ** 2)/2.0).sum()

    # 3. 定义优化器（函数对象） 
    # 思路一 ：基于SGD（随机梯度下降），加入参数momentum，即动量法
    # 优化器参数解释：参1：需要优化的参数，参2：学习率，参3：动量法参数
    # optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9)     #momentum默认是0，只考虑本次梯度

    # 思路二 ：基于AdaGrad（自适应学习率）
    # 参1：需要优化的参数，参2：学习率
    optimizer = torch.optim.Adagrad([w], lr=0.01)     #momentum默认是0，只考虑本次梯度

    # 4. 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()        #梯度清零
    criterion.backward()         #反向传播
    optimizer.step()             #参数更新
    print(f'w: {w}, w.grad: {w.grad}')

    # 5. 重复上述步骤，第2次，更新权重参数
    # 5.1. 定义损失函数
    criterion = ((w ** 2)/2.0).sum()
    # 5.2. 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()        #梯度清零
    criterion.backward()         #反向传播
    optimizer.step()             #参数更新
    print(f'w: {w}, w.grad: {w.grad}')

# 3. 定义函数，演示：梯度下降将优化方法 -> RMSProp
def dm03_rmsprop():
    # 1. 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 2. 定义损失函数
    criterion = ((w ** 2)/2.0).sum()

    # 3. 定义优化器（函数对象） 
    # 思路一 ：基于SGD（随机梯度下降），加入参数momentum，即动量法
    # 优化器参数解释：参1：需要优化的参数，参2：学习率，参3：动量法参数
    # optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9)     #momentum默认是0，只考虑本次梯度

    # 思路二 ：基于AdaGrad（自适应学习率）
    # 参1：需要优化的参数，参2：学习率
    # optimizer = torch.optim.Adagrad([w], lr=0.01)     #momentum默认是0，只考虑本次梯度

    # 思路三 ：基于RMSProp（自适应学习率）
    # 参1：需要优化的参数，参2：学习率
    optimizer = torch.optim.RMSprop([w], lr=0.01, alpha=0.9)     #momentum默认是0，只考虑本次梯度

    # 4. 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()        #梯度清零
    criterion.backward()         #反向传播
    optimizer.step()             #参数更新
    print(f'w: {w}, w.grad: {w.grad}')

    # 5. 重复上述步骤，第2次，更新权重参数
    # 5.1. 定义损失函数
    criterion = ((w ** 2)/2.0).sum()
    # 5.2. 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()        #梯度清零
    criterion.backward()         #反向传播
    optimizer.step()             #参数更新
    print(f'w: {w}, w.grad: {w.grad}') 

# 4. 定义函数，演示：梯度下降将优化方法 -> Adam
def dm04_adam():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 2. 定义损失函数
    criterion = ((w ** 2)/2.0).sum()

    # 3. 定义优化器（函数对象） 
    # 思路一 ：基于SGD（随机梯度下降），加入参数momentum，即动量法
    # 优化器参数解释：参1：需要优化的参数，参2：学习率，参3：动量法参数
    # optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9)     #momentum默认是0，只考虑本次梯度

    # 思路二 ：基于AdaGrad（自适应学习率）
    # 参1：需要优化的参数，参2：学习率
    # optimizer = torch.optim.Adagrad([w], lr=0.01)     #momentum默认是0，只考虑本次梯度

    # 思路三 ：基于RMSProp（自适应学习率）
    # 参1：需要优化的参数，参2：学习率，参3：梯度衰减系数
    # optimizer = torch.optim.RMSprop([w], lr=0.01, alpha=0.9)     #momentum默认是0，只考虑本次梯度

    # 思路四 ：基于Adam（自适应矩估计）
    # 参1：需要优化的参数，参2：学习率，参3：一阶矩衰减系数，参4：二阶矩衰减系数
    optimizer = torch.optim.Adam([w], lr=0.01, betas=(0.9, 0.999))     #momentum默认是0，只考虑本次梯度

    # 4. 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()        #梯度清零
    criterion.backward()         #反向传播
    optimizer.step()             #参数更新
    print(f'w: {w}, w.grad: {w.grad}')

    # 5. 重复上述步骤，第2次，更新权重参数
    # 5.1. 定义损失函数
    criterion = ((w ** 2)/2.0).sum()
    # 5.2. 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()        #梯度清零
    criterion.backward()         #反向传播
    optimizer.step()             #参数更新
    print(f'w: {w}, w.grad: {w.grad}') 

# 5. 测试
if __name__ == '__main__':
    # dm01_monentum()
    # dm02_adagrad()
    # dm03_rmsprop()
    dm04_adam()