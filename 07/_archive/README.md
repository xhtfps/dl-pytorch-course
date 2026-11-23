# `07/_archive` 说明

本目录存放第 7 课的归档文件。原 `data/cifar-10-batches-py/` 下的 CIFAR-10 原始批次（`data_batch_1~5`、`test_batch`，约 186 MB）已从仓库移除——它们是可以随时重新生成的公开数据集，不适合放进 Git。

## 重新获取数据

```python
import torchvision
ds = torchvision.datasets.CIFAR10(root="07/_archive/data", train=True, download=True)
```

或者在 Linux/macOS 用官方脚本：

```bash
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz -C 07/_archive/data
```

下载完成后目录结构与 `torchvision` 默认一致（`cifar-10-batches-py/`），脚本内的相对路径无需改动。

## 为什么不入库

- 数据集每次都能从官方源确定地重新下载，属于"派生物"，入库只会让仓库变大、且与 TorchVision 版本脱钩。
- 训练权重（`*.pth` / `*.pt` / `*.ckpt`）同理，已在 `.gitignore` 中忽略；需要长期保存的模型走 Release 附件。
