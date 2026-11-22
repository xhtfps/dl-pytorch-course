"""
案例:
    演示词嵌入层的API应用。

RNN介绍:
    全称叫Recurrent neural network, 循环神经网络, 主要处理序列数据的。
    序列数据: 后边数据对前边的数据有依赖, 例如: 天气预测, 股市分析, 文本生成...

组成:
    词嵌入层
    循环网络层
    输出层

词嵌入层介绍(作用):
    把词(或者 词对应的索引)转成 词向量.
"""

# 导包
import torch
import jieba          # jieba分词器
import torch.nn as nn

# 1. 定义函数，用于演示词嵌入层的API应用，如何把词转成词向量
def word2vec():
    # 1.1 定义词嵌入层
    text = '虽然你是对手，但你很不错哦。'

    # 1.2 分词
    words = jieba.lcut(text)
    # print(words)

    # 1.3 定义词嵌入层
    # 参1: 词汇表大小，参2: 词向量维度
    embedding = nn.Embedding(len(words), 128)
    # print(embedding)

    # 1.4 获取词索引
    for i, word in enumerate(words):
        # print(f'{i} : {word}')

        # 1.5 获取词向量
        word_vector = embedding(torch.tensor(i))
        # print(word_vector)


# 2. 调用函数
if __name__ == '__main__':
    word2vec()  
