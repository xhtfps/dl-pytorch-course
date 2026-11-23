<div align="center">

# PyTorch 深度学习课程代码

**从张量、自动微分到 CNN/LSTM：按课次拆开的可运行脚本**

![language](https://img.shields.io/badge/language-Python-3776ab?style=flat-square)
![stack](https://img.shields.io/badge/stack-PyTorch%202.x-ee4c2c?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![last commit](https://img.shields.io/github/last-commit/xhtfps/dl-pytorch-course?style=flat-square)

[环境准备](#2-环境准备) · [常见问题](#4-常见问题) · [全部仓库索引](https://github.com/xhtfps/xhtfps)

</div>

## 为什么按课次拆目录

学 PyTorch 最容易卡在「环境跑不起来」和「一个脚本什么都干」。这里每节课一个目录、脚本彼此独立、不跨目录依赖——**跑不通就只可能是这一节的问题**，CPU 也能全部跑通。

## 亮点

- **七节课递进**：张量 → 自动微分 → 线性回归 → ANN → CNN → RNN/LSTM → 综合实战。
- **无需 GPU 也能跑**：小 epoch、小 batch 的配置留好了，先跑通再谈加速。
- **数据与权重不入库**：CIFAR-10 批次已移出，下载方式写在 `07/_archive/README.md`。
- **环境可复现**：conda 建环境的命令与依赖清单在正文，不靠记忆。

## 结构

```mermaid
flowchart LR
  T["01 张量基础"] --> G["02 自动微分"]
  G --> L["03 线性回归"]
  L --> A["04 ANN"]
  A --> C["05 CNN"]
  C --> R["06 RNN / LSTM"]
  R --> P["07 综合实战<br/>数据加载·训练·评估·保存"]
```

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
## 路线图

- [x] 七节课脚本与课堂笔记整理入库，CPU 可跑通
- [x] CIFAR-10 原始批次移出，改下载说明
- [ ] 补一份「常见报错 → 一行修复」速查
- [ ] 加一个统一的数据加载工具，减少各脚本重复代码

## 参与与反馈

- 遇到问题请走 [Issue 模板](.github/ISSUE_TEMPLATE)，注明课次、Python 与 torch 版本。
- 提 PR 前先读 [CONTRIBUTING.md](CONTRIBUTING.md)；安全问题按 [SECURITY.md](SECURITY.md) 私下反馈。
- 如果这个仓库帮你省过时间，点个 ⭐ 是最省事的支持。

## 许可证

本仓自写代码以 [MIT](LICENSE) 分发；PyTorch 与数据集按各自许可证使用。
