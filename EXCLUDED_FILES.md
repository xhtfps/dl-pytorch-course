# 排除文件摘要（EXCLUDED_FILES）

本仓库保存课程代码与笔记，训练权重、数据集与 IDE 缓存不入库。逐文件明细原为 CSV，已删除，仅留本摘要。

| 项 | 数值 |
| --- | --- |
| 排除条目 | 6 条 |
| 排除体积 | 约 **0.07 GB** |

| 条目数 | 原因 |
| --- | --- |
| 5 | 类型 `.pth`（训练权重） |
| 1 | 目录 `.vscode` |

另：`07/_archive/data/cifar-10-batches-py/` 下的 CIFAR-10 原始批次（约 186 MB）已从仓库移除，重新获取方式见 [`07/_archive/README.md`](07/_archive/README.md)。
