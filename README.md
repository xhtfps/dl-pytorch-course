# PyTorch 深度学习课程代码

从张量、自动微分到 ANN/CNN/LSTM 的 PyTorch 课程代码与课堂笔记，按课次目录组织，可直接运行。

## 1. 目录一览

| 目录 | 内容 |
| --- | --- |
| `01/` | 张量基础：创建、形状变换、广播、索引 |
| `02/` | 自动微分与计算图（`backward` / `grad`） |
| `03/` | 线性回归、梯度下降与优化器 |
| `04/` | ANN 全连接网络、损失函数与训练循环 |
| `05/` | CNN：卷积、池化、批归一化 |
| `06/` | RNN / LSTM 序列建模 |
| `07/` | 综合实战（数据加载、训练、评估、保存） |
| `课堂笔记/` | 课程笔记截图与推导图 |

## 2. 环境准备

推荐独立虚拟环境，避免污染系统 Python：

```powershell
conda create -n robot-pose python=3.11 -y
conda activate robot-pose
pip install torch torchvision numpy matplotlib pandas jupyter
```

无 NVIDIA GPU 时默认 CPU 版即可，所有脚本均可在 CPU 跑通（小 epoch / 小 batch）。

## 3. 运行方式

```powershell
# 单脚本
python .\01\<脚本名>.py

# 笔记本
jupyter lab
```

每个目录内的脚本相互独立，不存在跨目录依赖。数据集与模型权重不入库，首次运行前按 [`07/_archive/README.md`](07/_archive/README.md) 下载 CIFAR-10。

## 4. 常见问题

- **`torch.cuda.is_available()` 为 False**：装的是 CPU 版，装 GPU 版需 `pip install torch --index-url https://download.pytorch.org/whl/cu121`。
- **中文路径报错**：Windows 下 Dataset 的 `num_workers>0` 与中文路径有兼容问题，先把 `num_workers` 设为 0。
- **显存不足**：先把 batch size 减半，再考虑混合精度。
- **matplotlib 中文变方块**：`plt.rcParams['font.sans-serif']=['SimHei']`。

## 5. 维护说明

- 数据集与权重不入库：CIFAR-10 原始批次（约 186 MB）已移除，重新获取方式见 [`07/_archive/README.md`](07/_archive/README.md)。
- 本仓库课程内容偏向“过一遍原理”，不包含正式论文复现代码（复现代码在对应论文仓库/本地工程中）。

