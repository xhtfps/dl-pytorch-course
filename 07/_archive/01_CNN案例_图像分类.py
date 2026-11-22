"""
案例:
    演示CNN的综合案例，图像分类。

回顾: 深度学习项目的步骤
    1. 准备数据集.
        这里我们用的时候 计算机视觉模块 torchvision自带的 CIFAR10数据集，包含6W张（32,32,3）的图片，5W张训练集，1W张测试集，10个分类，每个分类6K张图片。
        你需要单独安装一下 torchvision包，即: pip install torchvision
    2. 搭建(卷积)神经网络
    3. 模型训练.
    4. 模型测试.

卷积层:
    提取图像的局部特征 → 特征图(Feature Map), 计算方式: N = math.floor((W - F + 2P) / S + 1
池化层：
    降维，有最大池化和平均池化
    池化只在HW上操作，不涉及通道数
"""

# 导包
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip       # pip install torchvision -i https://mirrors.aliyun.com/pypi/simple/
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import matplotlib.pyplot as plt
from torchsummary import summary

# 每批次样本数
BATCH_SIZE = 256

# 1. 准备数据集
def creat_dataset():
    # 1.1 数据预处理：
    # 训练集：随机裁剪(四周补4像素) + 随机水平翻转 + 转张量 + 标准化（数据增强，防过拟合、提准确率）
    transform_train = Compose([
        RandomCrop(32, padding=4),                    # 随机裁剪：先补4像素再随机取32x32
        RandomHorizontalFlip(),                       # 随机水平翻转（50%概率）
        ToTensor(),                                   # 转张量并归一化到[0,1]
        Normalize((0.4914, 0.4822, 0.4465),           # 标准化：减均值
                  (0.2470, 0.2435, 0.2616)),          # 除标准差（CIFAR-10标准值）
    ])
    # 测试集：只转张量+标准化，不做增强（保证评价一致性）
    transform_test = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    # 1.2 准备训练集
    #参1：数据集存放路径，参2：是否为训练集，参3：数据预处理，参4：是否下载数据集
    train_dataset = CIFAR10(root='./07/data', train=True, transform=transform_train, download=False)
    # 1.3 准备测试集
    #参1：数据集存放路径，参2：是否为训练集，参3：数据预处理，参4：是否下载数据集
    test_dataset = CIFAR10(root='./07/data', train=False, transform=transform_test, download=False)
    # 1.3 创建数据集加载器
    # {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    # 1.4 返回数据集
    return train_loader, test_loader

# 2. 搭建（卷积）神经网络
class CNN(nn.Module):
    # 1. 定义网络结构
    def __init__(self):
        super(CNN, self).__init__()
        # 2.1 定义卷积层1
        # 参1：输入通道数，参2：输出通道数，参3：卷积核大小，参4：步长，参5：填充
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        # 2.2 定义批量归一化层1（对卷积输出的6个通道做归一化，加速收敛）
        self.bn1 = nn.BatchNorm2d(num_features=6)
        # 2.3 定义池化层1
        # 参1：池化核大小，参2：步长
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2.4 定义卷积层2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
        # 2.5 定义批量归一化层2
        self.bn2 = nn.BatchNorm2d(num_features=16)
        # 2.6 定义池化层2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2.5 定义全连接层
        # 第1个全连接层
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        # 第2个全连接层
        self.fc2 = nn.Linear(120, 84)
        # Dropout层：训练时随机丢弃50%的神经元，防止过拟合（测试时自动关闭）
        self.dropout = nn.Dropout(0.5)
        # 输出层
        self.out = nn.Linear(84, 10)

    # 2. 定义前向传播
    def forward(self, x):
        # 2.1 卷积层1
        x = self.conv1(x)
        # 2.2 批量归一化1
        x = self.bn1(x)
        # 2.3 激活函数
        x = torch.relu(x)
        # 2.4 池化层1
        x = self.pool(x)
        # 2.5 卷积层2
        x = self.conv2(x)
        # 2.6 批量归一化2
        x = self.bn2(x)
        # 2.7 激活函数
        x = torch.relu(x)
        # 2.8 池化层2
        x = self.pool2(x)
        # 2.7 展平
        x = x.view(-1, 16 * 6 * 6)
        # 2.8 全连接层1
        x = self.fc1(x)
        # 2.9 激活函数
        x = torch.relu(x)
        # 2.10 Dropout（随机丢弃50%，防过拟合）
        x = self.dropout(x)
        # 2.11 全连接层2
        x = self.fc2(x)
        # 2.11 激活函数
        x = torch.relu(x)
        # 2.12 输出层
        x = self.out(x)

        return x

# 3. 模型训练
def train(model, train_loader, device):
    # 3.1 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 3.2 训练轮数
    epochs = 100    # 训练轮数
    # 3.3 定义优化器（weight_decay=L2正则，防过拟合）
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    # 3.4 定义余弦退火学习率调度器：每个epoch后lr沿余弦曲线下降，20轮从0.001降到接近0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # 3.5 训练开始前，打印关键配置信息
    print('=' * 60)
    print('开始训练 CNN 图像分类模型')
    print(f'  运行设备      : {device}')
    print(f'  训练集样本数  : {len(train_loader.dataset)} 张')
    print(f'  每批样本数    : {BATCH_SIZE}')
    print(f'  每轮训练步数  : {len(train_loader)} 步')
    print(f'  训练轮数      : {epochs} 轮')
    print(f'  损失函数      : CrossEntropyLoss（交叉熵）')
    print(f'  优化器        : Adam，初始学习率 {optimizer.param_groups[0]["lr"]}，weight_decay=5e-4')
    print(f'  学习率策略    : 余弦退火 CosineAnnealingLR（{epochs}轮从0.01逐渐降到接近0）')
    print('=' * 60)
    for epoch in range(epochs):
        # 3.3.1 记录开始时间
        start = time.time()
        # 3.3.2 训练模式
        model.train()
        # 3.3.3 遍历训练集
        epoch_loss = 0.0                    # 本轮累计损失，用于最后算平均
        epoch_correct = 0                   # 本轮预测正确的样本数（用于算训练准确率）
        epoch_total = 0                     # 本轮已处理的样本总数
        for i, (images, labels) in enumerate(train_loader):
            # 3.3.4 将数据放到GPU上
            images = images.to(device)
            labels = labels.to(device)
            # 3.3.5 前向传播
            outputs = model(images)
            # 3.3.6 计算损失
            loss = loss_fn(outputs, labels)
            epoch_loss += loss.item()       # 累计本轮损失
            # 统计训练准确率：比较预测类别与真实标签
            # 注意：训练中开启Dropout和数据增强，数值会比测试时略低，属正常现象
            _, predicted = torch.max(outputs.data, 1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += labels.size(0)
            # 3.3.7 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 3.3.9 记录结束时间
        end = time.time()
        # 3.3.10 每轮结束打印平均损失、训练准确率、学习率和耗时
        avg_loss = epoch_loss / len(train_loader)
        epoch_acc = 100 * epoch_correct / epoch_total   # 训练集准确率 = 正确数 / 总数
        cur_lr = optimizer.param_groups[0]['lr']
        print(f'  第{epoch + 1}/{epochs}轮完成 | 平均损失 {avg_loss:.4f} | 训练准确率 {epoch_acc:.2f}% | 学习率 {cur_lr:.6f} | 耗时 {end - start:.1f} 秒')
        # 3.3.11 余弦退火：更新下一轮的学习率（lr会沿余弦曲线下降）
        scheduler.step()
    # 3.4 保存模型（先确保目录存在）
    os.makedirs('./07/model', exist_ok=True)
    torch.save(model.state_dict(), './07/model/cnn_model.pth')
    print('=' * 60)
    print('训练完成，模型权重已保存到 ./07/model/cnn_model.pth')
    print('=' * 60)
    # 3.5 返回模型
    return model

# 4. 模型测试
def evaluate(model, test_loader, device):
    # 4.1 检查模型文件是否存在
    if not os.path.exists('./07/model/cnn_model.pth'):
        raise FileNotFoundError('模型文件 cnn_model.pth 不存在，请先运行训练!')
    # 4.2 加载模型（map_location 确保权重加载到当前设备，避免 cuda/cpu 不匹配）
    model.load_state_dict(torch.load('./07/model/cnn_model.pth', map_location=device))
    # 4.2 测试模式
    model.eval()
    # 4.3 遍历测试集
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            # 4.4 将数据放到GPU上
            images = images.to(device)
            labels = labels.to(device)
            # 4.5 前向传播
            outputs = model(images)
            # 4.6 预测结果
            _, predicted = torch.max(outputs.data, 1)
            # 4.7 统计
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # 4.8 打印测试结果
        accuracy = 100 * correct / total
        print('=' * 60)
        print('模型测试结果：')
        print(f'  测试样本数  : {total} 张')
        print(f'  正确预测数  : {correct} 张')
        print(f'  测试准确率  : {accuracy:.2f}%')
        print('=' * 60)
    # 4.9 返回模型
    return model

# 5. 测试
if __name__ == '__main__':
    # 1. 准备数据集
    train_loader, test_loader = creat_dataset()
    # print(f'训练集大小:{train_loader.dataset}, 测试集大小:{test_loader.dataset}') 

    # 图像展示
    # plt.figure(figsize=(2, 2))
    # plt.imshow(train_loader.dataset[0][0].permute(1, 2, 0))  # (3,32,32) -> (32,32,3)
    # plt.title(train_loader.dataset[0][1])
    # plt.show()

    # 2. 搭建（卷积）神经网络
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 有GPU用GPU，否则CPU
    model = CNN().to(device)
    print(f'使用设备: {device}')
    if device.type == 'cuda':
        print(f'GPU型号: {torch.cuda.get_device_name(0)}  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
    # 参1：模型，参2：输入形状
    # summary(model, (3, 32, 32), batch_size=BATCH_SIZE)

    # 3. 模型训练
    train(model, train_loader, device)

    # 4. 模型测试
    evaluate(model, test_loader, device)