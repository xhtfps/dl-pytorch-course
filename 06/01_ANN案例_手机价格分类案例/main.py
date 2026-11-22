"""
案例:
    ANN(人工神经网络)案例: 手机价格分类案例.
背景:
    基于手机的20列特征 → 预测手机的价格区间(4个区间), 可以用机器学习做, 也可以用深度学习做(推荐)
ANN案例的实现步骤：
    1. 准备数据集
    2. 搭建神经网络
    3. 训练神经网络
    4. 测试神经网络
"""
# 导包
import os
import random                                                # 随机数模块（配合种子复现训练结果）
import torch                                                # PyTorch框架，封装了张量的各种操作
from torch.utils.data import TensorDataset                  # 数据集对象. 数据 → Tensor → 数据集 → 数据加载器
from torch.utils.data import DataLoader                     # 数据加载器.
import torch.nn as nn                                       # neural network, 封装了神经网络的各种操作
import torch.optim as optim                                 # 优化器
from sklearn.model_selection import train_test_split        # 训练集和测试集的划分
import matplotlib.pyplot as plt                             # 绘图
import numpy as np                                          # 数组(矩阵)操作
import pandas as pd                                         # 数据处理
import time                                                 # 时间模块
from torchsummary import summary                            # 网络结构可视化

# 设备自动选择：有cuda就用GPU，没有就CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 随机种子：固定后训练可复现；想对比不同种子的准确率，改这里的数字即可
SEED = 42

def set_seed(seed):
    """固定所有随机源：random/numpy/torch/cuda，GPU上再用cuDNN确定模式保证完全可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # 确定性算法（略慢，但结果可复现）
    torch.backends.cudnn.benchmark = False      # 关闭自动调优（会引入随机性）

# 脚本所在目录 → 数据/模型绝对路径（避免相对路径依赖"当前工作目录"而报FileNotFoundError）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', '手机价格预测.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'phone_price.pth')

# 创建模型保存文件夹
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 1. 定义函数，构建数据集和数据加载器
def create_dataset():
    # 1.1 准备数据集
    data = pd.read_csv(DATA_PATH)
    # print(f'原始数据集: {data.head()}')
    # print(f'原始数据集大小: {data.shape}')
    # 1.2 获取x特征列和y标签列
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    # print(f'特征列: {x.head()}, {x.shape}')
    # print(f'标签列: {y.head()}, {y.shape}')
    # 1.3 数据预处理
    x = x.astype(np.float32)  # 转换为float32类型（归一化交给网络内部的BatchNorm1d）
    y = y.astype(np.int64)    # 转换为int64类型
    # print(f'特征列: {x.head()}, {x.shape}, 标签列: {y.head()}, {y.shape}')
    # 1.4 数据集的划分
    # 参1：特征列，参2：标签列，参3：测试集占比，参4：分层抽样
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)  # 分层抽样，保证训练集和测试集的类别分布一致
    # 再从训练集中划分出验证集（用于早停监控，不参与训练，避免测试集信息泄漏）
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)
    # 5. 把数据转换成张量数据集：数据 → 张量 Tensor → 数据集 TensorDataset → 数据加载器 DataLoader
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))   # 训练集转换为张量数据集
    val_dataset = TensorDataset(torch.tensor(x_val.values), torch.tensor(y_val.values))         # 验证集转换为张量数据集
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))      # 测试集转换为张量数据集
    # print(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}, 测试集大小: {len(test_dataset)}')
    # 6. 把数据集封装成数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 训练集数据加载器
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)      # 验证集数据加载器（早停监控用）
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)  # 测试集数据加载器
    # print(f'训练集数据加载器大小: {len(train_loader)}, 验证集数据加载器大小: {len(val_loader)}, 测试集数据加载器大小: {len(test_loader)}')
    # 7. 返回结果
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, x_train.shape[1], len(np.unique(y))  # 返回训练集、验证集、测试集、对应数据加载器、输入特征数、输出标签数

# 2. 搭建神经网络
class PhonePrice(nn.Module):
    # 1. 在init魔法方法中，初始化父类成员，即搭建神经网络
    def __init__(self, input_dim, output_dim):  #输入：20，输出：4
        # 1.1. 初始化父类成员
        super(PhonePrice, self).__init__()
        # 1.2. 搭建神经网络
        # 隐藏层1
        self.linear1 = nn.Linear(input_dim, 256)  # 输入层20个神经元, 输出层256个神经元
        self.bn1 = nn.BatchNorm1d(256)            # 批归一化1：num_features = linear1的输出维度256
        # 隐藏层2
        self.linear2 = nn.Linear(256, 128)  # 输入层256个神经元, 输出层128个神经元
        self.bn2 = nn.BatchNorm1d(128)            # 批归一化2：num_features = linear2的输出维度128
        # 隐藏层3
        self.linear3 = nn.Linear(128, 64)  # 输入层128个神经元, 输出层64个神经元
        self.bn3 = nn.BatchNorm1d(64)             # 批归一化3：num_features = linear3的输出维度64
        # 隐藏层4
        self.linear4 = nn.Linear(64, 32)  # 输入层64个神经元, 输出层32个神经元
        self.bn4 = nn.BatchNorm1d(32)             # 批归一化4：num_features = linear4的输出维度32
        # Dropout随机失活：训练时以30%概率随机丢弃神经元，防止过拟合；eval()测试时自动关闭
        self.dropout = nn.Dropout(p=0.3)
        # 输出层
        self.output = nn.Linear(32, output_dim)  # 输入层32个神经元, 输出层4个神经元
    # 2. 前向传播方法forward（）
    def forward(self, x):
        # 2.1 隐藏层1：加权求和 + 批归一化 + 激活函数ReLU（标准顺序：Linear → BN → ReLU）
        x = torch.relu(self.bn1(self.linear1(x)))
        x = self.dropout(x)                       # 随机失活30%
        # 2.2 隐藏层2：加权求和 + 批归一化 + 激活函数ReLU
        x = torch.relu(self.bn2(self.linear2(x)))
        x = self.dropout(x)                       # 随机失活30%
        # 2.3 隐藏层3：加权求和 + 批归一化 + 激活函数ReLU
        x = torch.relu(self.bn3(self.linear3(x)))
        x = self.dropout(x)                       # 随机失活30%
        # 2.4 隐藏层4：加权求和 + 批归一化 + 激活函数ReLU
        x = torch.relu(self.bn4(self.linear4(x)))
        x = self.dropout(x)                       # 随机失活30%
        # 2.5 输出层：加权求和 + 激活函数Softmax
        # 正常写法，但是不需要，后续用多分类交叉熵损失函数 CrossEntropyLoss ()
        # x = torch.softmax(self.output(x), dim=1)  # 输出层使用Softmax激活函数，dim=1表示按行计算
        x = self.output(x)
        # 2.4 返回结果
        return x

# 3. 训练神经网络
def train(train_loader, val_loader, input_dim, output_dim):
    # 3.1 搭建神经网络
    model = PhonePrice(input_dim, output_dim).to(device)

    # 3.1.1 权重/偏置初始化（He适配ReLU，Xavier给无BN兜底的输出层，偏置全0）
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if m.out_features == output_dim:          # 输出层：32→4，无BN/激活兜底
                nn.init.xavier_uniform_(m.weight)
            else:                                     # 隐藏层：有BN+ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)                # 偏置全0

    model.apply(init_weights)  # 递归应用到所有子模块（Linear）

    # 3.2 定义损失函数, 使用多分类交叉熵损失函数 CrossEntropyLoss ()
    criterion = nn.CrossEntropyLoss()
    # 3.3 定义优化器, 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)         # 学习率0.001，权重衰减1e-4
    # 3.4 模型训练
    # 3.5 记录训练轮数
    num_epochs = 100
    # 3.5.1 学习率余弦退火：lr从0.001平滑降至~0，让训练收敛更彻底
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # 3.5.2 早停：监控验证集准确率，连续patience轮无提升则提前停止，并保存最优权重
    patience, no_improve, best_val_acc, best_state = 40, 0, 0.0, None
    # 3.5.3 记录每轮的训练损失和验证准确率（用于绘制训练曲线）
    train_losses, val_accs = [], []
    # 3.6 开始每轮训练
    for epoch in range(num_epochs):
        # 4.2.1 定义变量：记录每次的损失值， 训练批次数
        total_loss, batch_num = 0.0, 0
        # 4.2.2 定义变量，表示训练开始的时间
        start_time = time.time()
        # 4.2.3 开始训练
        for x, y in train_loader:
            # 1 切换模式
            model.train()  # 切换到训练模式
            # 2 把数据放到GPU上
            x, y = x.to(device), y.to(device)
            # 3 模型预测
            y_pred = model(x)
            # 4 计算损失值
            loss = criterion(y_pred, y)
            # 5 梯度清零
            optimizer.zero_grad()
            # 6 反向传播
            loss.backward()
            # 7 更新参数
            optimizer.step()
            # 8 累加损失值
            total_loss += loss.item()      #本轮的每批次的平均损失值累加
            batch_num += 1                 #训练批次数累加
        # 4.2.4 每轮结束：更新学习率 + 验证集评估 + 早停判断
        scheduler.step()  # 更新学习率（余弦退火）
        model.eval()      # 切换到评估模式（关闭Dropout/BN用running统计）
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_correct += (torch.argmax(model(x), dim=1) == y).sum().item()
        val_acc = val_correct / len(val_loader.dataset)
        train_losses.append(total_loss / batch_num)  # 记录本轮平均损失
        val_accs.append(val_acc)                     # 记录本轮验证准确率
        print(f'第{epoch+1}轮训练结束，耗时: {time.time() - start_time:.2f}秒，平均损失值: {total_loss / batch_num:.4f}，验证集准确率: {val_acc:.4f}')
        # 早停判断：验证集提升则记录最优权重，否则累计无提升次数
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}  # 深拷贝最优权重
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'早停：连续{patience}轮验证集无提升，停在第{epoch+1}轮，最优验证准确率: {best_val_acc:.4f}')
                break
    # 3.5 多轮训练结束，恢复最优权重并保存模型（参数）
    # 参1：模型对象的参数（权重矩阵，偏置矩阵），参2：模型保存的名字
    # print(f'\n\n模型的参数信息: {model.state_dict()}\n\n')
    if best_state is not None:
        model.load_state_dict(best_state)  # 用验证集上最优的权重，而非最后一轮
    torch.save(model.state_dict(), MODEL_PATH)

    # 3.6 绘制训练曲线：左侧训练损失，右侧验证准确率
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 解决中文乱码
    plt.rcParams['axes.unicode_minus'] = False                      # 正常显示负号
    plt.figure(figsize=(10, 4))
    # 左图：训练损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失', color="#1b69d0", linewidth=1.5)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('训练损失曲线')
    plt.grid(alpha=0.3); plt.legend()
    # 右图：验证准确率
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='验证准确率', color='#0f9d58', linewidth=1.5)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('验证准确率曲线')
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    curve_path = os.path.join(BASE_DIR, 'training_curves.png')
    plt.savefig(curve_path, dpi=150)   # 保存到脚本目录
    plt.show()                          # 弹窗显示
    print(f'训练曲线已保存: {curve_path}')

# 4. 测试神经网络
def evaluate(test_loader, test_dataset, input_dim, output_dim):
    # 1. 创建神经网络
    model = PhonePrice(input_dim, output_dim).to(device)
    # 2. 加载模型参数
    model.load_state_dict(torch.load(MODEL_PATH))
    # 3. 定义变量，记录预测正确的样本数
    correct_num = 0
    # 4. 从数据加载器中获取数据
    with torch.no_grad():
        for x, y in test_loader:
            # 4.1 切换到评估模式
            model.eval()
            # 4.2 把数据放到GPU上
            x, y = x.to(device), y.to(device)
            # 4.3 模型预测
            y_pred = model(x)
            # 4.4 根据加权求和，得到类别，用 argmax () 获取最大值对应的下标，就是类别
            y_pred = torch.argmax(y_pred, dim=1)    # dim = 1表示按行计算，即每个样本的预测类别
            # 4.5 计算预测正确的样本数
            correct_num += (y_pred == y).sum().item()
    # 5. 计算准确率
    accuracy = correct_num / len(test_dataset)
    print(f"\n测试集准确率: {accuracy:.4f}")
    return accuracy 

# 5. 测试
if __name__ == '__main__':
    # 5.0 固定随机种子（修改顶部SEED可复现/对比不同种子的准确率）
    set_seed(SEED)
    print(f"本次训练随机种子: {SEED}")
    # 5.1 构建数据集
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, input_dim, output_dim = create_dataset()
    # print(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}, 测试集大小: {len(test_dataset)}, 输入特征数: {input_dim}, 输出标签数: {output_dim}')
    # 5.2 搭建神经网络
    # model = PhonePrice(input_dim, output_dim).to(device)  # 模型对象，指定设备
    # 计算模型参数
    # 参1：模型对象，参2：输入数据的形状(不包含batch_size)，例如：(16, 20)表示16个样本，每个样本有20个特征
    # summary(model, input_size=(16, input_dim), device=str(device))  # device='cpu' or 'cuda'
    # 5.3 训练神经网络（验证集用于早停监控）
    train(train_loader, val_loader, input_dim, output_dim)
    # 5.4 测试神经网络
    evaluate(test_loader, test_dataset, input_dim, output_dim)
