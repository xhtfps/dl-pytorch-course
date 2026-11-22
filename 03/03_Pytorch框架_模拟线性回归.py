# 导入相关模块
import torch
from torch.utils.data import TensorDataset      # 构造数据集对象
from torch.utils.data import DataLoader         # 数据加载器
from torch import nn                            # nn模块中有平方损失函数和假设函数
from torch import optim                         # optim模块中有优化器函数
from sklearn.datasets import make_regression    # 创建线性回归模型数据集
import matplotlib.pyplot as plt                 # 可视化

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 1. 定义函数，创建线性回归样本数据
def create_data():
    # 1 创建样本数据
    x, y, coef = make_regression(
        n_samples=100,      #100个样本点
        n_features=1,       #1个特征点
        noise=10,           #噪声：噪声越大，样本点越分散；噪声越小，样本点越集中
        coef=True,          #系数：True表示生成样本数据时，会使用指定的系数
        bias=14.5,          #偏置：14.5表示样本数据的截距
        random_state=3      #随机种子：设置随机种子，保证每次运行代码时，生成的样本数据相同
    )

    print(type(x))
    # 2 将样本数据转换为Tensor对象
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # 3 返回结果
    return x, y, coef

# 2. 定义函数，训练线性回归模型
def train(x, y, coef):
    #1 . 创建数据集对象，把tensor-> 数据集对象 -> 数据加载器
    dataset = TensorDataset(x, y)
    #2. 创建数据加载器对象
    # 参1：数据集对象 参2：每个批次样本个数 参3：是否打乱样本数据（训练集打乱，测试集不打乱）
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    #3. 创建线性回归模型
    #参1：输入特征维度 参2：输出特征维度
    model = nn.Linear(1, 1)
    #4. 定义损失函数
    criterion = nn.MSELoss()
    #5. 定义优化器
    #参1：模型参数 参2：学习率
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    #6. 具体的训练过程
    # 6.1. 定义变量，分别表示：训练轮数，训练损失，训练总损失值，训练总样本数
    epochs, loss_list, total_loss, total_sample = 100, [], 0.0, 0

    # 6.2. 开始训练，按轮训练
    for epoch in range(epochs):     #epoch的值：0-99
        # 6.3 每轮是分批次训练的，所以从数据加载器中获取批次数据
        for train_x, train_y in data_loader:        #7批（16，16，16，16，16，16，4）
            # 6.3.1 模型预测值
            y_pred = model(train_x)
            # 6.3.2 计算(每批的平均)损失
            loss = criterion(y_pred, train_y.reshape(-1, 1))        # -1 自动计算， 1 表示1列
            # 6.3.3 计算总损失和样本批次数
            total_loss += loss.item() * train_x.size(0)
            total_sample += train_x.size(0)
            # 6.3.4 梯度清零 + 反向传播 + 梯度更新
            optimizer.zero_grad()           #梯度清零
            loss.backward()           #反向传播，计算梯度
            optimizer.step()                #梯度更新

        # 6.4 把本轮的（平均）损失值，追加到列表中
        loss_list.append(total_loss / total_sample)
        print(f'第{epoch+1}轮，第{total_sample}批，平均损失值：{total_loss / total_sample}')

    # 7. 打印最终的训练结果
    print(f'{epochs}轮的平均损失值：{loss_list}')
    print(f'模型参数：权重{model.weight}, 偏置{model.bias}')

    # 8. 绘制损失曲线
    # x轴：训练轮数， y轴：损失值
    plt.plot(range(epochs), loss_list)
    plt.title('损失值曲线变换图')
    plt.show()

    # 9. 绘制预测值和真实值曲线
    # 9.1 绘制样点分布情况
    plt.scatter(x, y, coef)
    # 9.2 绘制预测曲线
    # x : 100个样本点
    y_pred = torch.tensor(data = [v * model.weight + model.bias for v in x]) 
    # 9.3 计算真实值
    y_true = torch.tensor(data = [v * coef + 14.5 for v in x])
    # 9.4 绘制预测值和真实值的折线图
    plt.plot(x, y_pred, color='red', label='预测值')
    plt.plot(x, y_true, color='green', label='真实值')
    # 9.5 设置图例
    plt.legend()
    plt.grid()
    # 9.6 显示图像
    plt.show()


# 3. 主程序入口
if __name__ == '__main__':
    # 3.1. 创建样本数据
    x, y, coef = create_data()
    print(f'x: {x}, y: {y}, coef: {coef}')

    # 3.2. 训练线性回归模型
    train(x, y, coef)
