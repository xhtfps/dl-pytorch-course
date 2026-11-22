"""
案例:
    RNN案例，基于杰伦歌词来训练模型，用给定的起始词，结合长度，来进行 AI歌词生成。

    步骤:
        1. 获取数据，进行分词，构建词典，构建数据集
        2. 数据预处理，准备数据集
        3. 搭建RNN神经网络
        4. 训练模型
        5. 测试预测

    运行环境: conda 环境 robot-pose (torch 2.13.0+cu130, jieba 0.42.1)
    运行命令: D:/conda/envs/robot-pose/python.exe 04_RNN案例_AI歌词生成器.py
"""

# 导包
import os
import torch
import jieba
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import time
from pypinyin import lazy_pinyin, Style   # 拼音库: 提取行末词的韵母, 实现押韵

# ==================== 1. 获取数据，进行分词，构建词典，构建数据集 ====================
def build_vocab():
    # 1. 定义变量，记录：去重后的所有词，每行文本结果
    unique_words, all_words = [], []

    # 2. 读取文件，进行分词，构建词典
    with open('./07/data/jaychou_lyrics.txt', 'r', encoding='utf-8') as f:
        for line in f:
            # 2.1 分词（含行尾 \n，会被 jieba 切为独立 token）
            words = jieba.lcut(line)
            # 2.2 记录所有词
            all_words.extend(words)
            # 2.3 遍历分词结果，去重后记录到 unique_words
            for word in words:
                if word not in unique_words:
                    unique_words.append(word)

    # 3. 获取语料库词的总数
    word_count = len(unique_words)
    print(f'语料库词的总数: {word_count}')

    # 4. 构建词表：key:词，value:索引
    word_dict = {word: index for index, word in enumerate(unique_words)}

    # 5. 构建数据集：整个语料库转成索引序列（一维）
    corpus_idx = [word_dict[word] for word in all_words]

    # 6. 返回结果
    return word_dict, corpus_idx, word_count, all_words

# ==================== 2. 数据预处理，准备数据集 ====================
class MyDataset(Dataset):
    """把语料索引序列切成 (输入, 标签) 样本对，标签是输入右移一位"""

    def __init__(self, corpus_idx, word_dict, num_chars):
        # 1. 获取数据集（一维索引序列）
        self.corpus_idx = corpus_idx
        # 2. 获取词表
        self.word_dict = word_dict
        # 3. 获取语料库词的总数
        self.word_count = len(self.corpus_idx)
        # 4. 保存每个样本的字符数（关键修复：存成实例属性）
        self.num_chars = num_chars
        # 5. 计算句子数量：减1保证每个样本都能取到完整标签、不越界
        self.num_sentences = (self.word_count - 1) // self.num_chars

    def __len__(self):
        # 返回句子数量
        return self.num_sentences

    def __getitem__(self, index):
        # 1. 获取起始索引
        start = index * self.num_chars
        # 2. 获取结束索引：多取 1 个，用于构造右移一位的标签
        end = start + self.num_chars + 1
        # 3. 获取输入序列：前 num_chars 个字符
        sentence = self.corpus_idx[start: end - 1]
        # 4. 获取标签序列：输入序列右移一位，与输入等长
        label = self.corpus_idx[start + 1: end]
        # 5. 转成张量返回（长整型，词索引）
        return torch.tensor(sentence, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# ==================== 3. 搭建RNN神经网络 ====================
class MyModel(nn.Module):
    """
    组成:
        词嵌入层 nn.Embedding : 词索引 -> 词向量 (word_count -> embedding_dim)
        循环网络层 nn.RNN/LSTM: 逐步处理词向量, 生成每个时间步的隐藏状态
        全连接层 nn.Linear   : 隐藏状态 -> 词表概率分布 (hidden_size -> word_count)

    性能增强:
        cell_type='lstm' 时用 LSTM 替代 vanilla RNN, 缓解梯度消失, 长依赖记忆更强
    """

    def __init__(self, word_count, embedding_dim=128, hidden_size=256, num_layers=2, cell_type='lstm'):
        super(MyModel, self).__init__()
        # 1. 词嵌入层: 词表大小 -> 词向量维度
        self.embedding = nn.Embedding(num_embeddings=word_count, embedding_dim=embedding_dim)
        # 2. 循环网络层: 可选 RNN 或 LSTM, batch_first=True 让输入形状为 [batch, seq_len, feature]
        rnn_cls = nn.LSTM if cell_type == 'lstm' else nn.RNN
        self.rnn = rnn_cls(input_size=embedding_dim, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        # 3. 全连接层: 隐藏状态 -> 词表大小的概率分布
        self.fc = nn.Linear(in_features=hidden_size, out_features=word_count)
        # 记录层数/隐藏维度/单元类型, forward 里初始化隐藏状态要用
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cell_type = cell_type

    def forward(self, x, h=None):
        """
        :param x: 词索引序列 [batch, seq_len]
        :param h: 初始隐藏状态, None 则全零初始化
                 RNN : [num_layers, batch, hidden_size]
                 LSTM: (h_0, c_0) 两个同形状张量的元组
        :return: out [batch, seq_len, word_count] 每个时间步的词表概率(logits)
                 h   RNN: [num_layers, batch, hidden_size] / LSTM: (h_n, c_n) 最后一步状态
        """
        # 1. 词索引 -> 词向量 [batch, seq_len, embedding_dim]
        emb = self.embedding(x)
        # 2. 未传隐藏状态时, 全零初始化 (LSTM 需要 (h, c) 两个状态)
        if h is None:
            z = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            h = (z, z.clone()) if self.cell_type == 'lstm' else z
        # 3. 循环网络计算: [batch, seq_len, hidden_size] + 最后一步隐藏状态
        out, h = self.rnn(emb, h)
        # 4. 全连接映射到词表: [batch, seq_len, word_count]
        out = self.fc(out)
        return out, h

# ==================== 4. 训练模型 ====================
def train(model, dataloader, epochs=20, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 交叉熵损失: 自动对 word_count 维做 softmax, 计算每个位置预测词的损失
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss, start_time = 0, time.time()
        for x, y in dataloader:
            # x, y: [batch, num_chars]
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            # out: [batch, num_chars, word_count]
            out, _ = model(x)
            # 展平: [batch*num_chars, word_count] 与 [batch*num_chars] 算损失
            loss = criterion(out.reshape(-1, out.size(-1)), y.reshape(-1))
            loss.backward()
            # 梯度裁剪: 把梯度范数限制在5以内, 防止RNN梯度爆炸导致loss震荡/不收敛
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}  Loss: {total_loss / len(dataloader):.4f}  耗时: {time.time() - start_time:.1f}s')

# ==================== 5. 测试预测 ====================
def predict(model, word_dict, start_words, max_len=50, temperature=0.6, top_k=5):
    """
    给定起始词, 逐词预测, 生成 max_len 个新词
    性能增强: 温度采样 + top-k 采样替代贪心(argmax), 避免生成重复/死循环

    :param model: 训练好的模型
    :param word_dict: 词->索引 词表
    :param start_words: 起始词, 如 "我想"
    :param max_len: 预测的新词个数
    :param temperature: 温度系数, <1 更确定/保守, >1 更随机/发散, 0.8 常用
    :param top_k: 只从前 k 个概率最高的词里采样, 过滤掉离谱的低概率词, 10 常用
    :return: 拼接后的完整歌词字符串
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    # 索引 -> 词 反查表
    index2word = {v: k for k, v in word_dict.items()}

    # 1. 起始词分词 -> 索引序列
    idxs = [word_dict[w] for w in jieba.lcut(start_words) if w in word_dict]
    if not idxs:
        raise ValueError(f'起始词"{start_words}"分词后都不在词表中')

    # 2. 逐词预测
    with torch.no_grad():
        h = None
        input_seq = torch.tensor([idxs], device=device).long()  # [1, len]
        for _ in range(max_len):
            # 第一次喂完整起始词; 之后每次只喂最后1个词, 并带上一步隐藏状态 h
            out, h = model(input_seq if h is None else input_seq[:, -1:], h)
            # 取最后一个时间步的 logits, 温度缩放后 top-k 采样
            logits = out[0, -1] / temperature                # 温度缩放: 压低/抬高分布尖锐度
            topk_vals, topk_idx = torch.topk(logits, top_k)  # 只保留概率最高的 k 个词
            probs = torch.softmax(topk_vals, dim=-1)         # 转概率
            next_idx = topk_idx[torch.multinomial(probs, 1)].item()  # 按概率随机采样
            idxs.append(next_idx)
            input_seq = torch.tensor([[next_idx]], device=device).long()

    # 3. 索引 -> 词 -> 拼接成歌词
    return ''.join(index2word[i] for i in idxs)

# ==================== 5.1 韵律增强: 韵脚词典 + 完整歌词生成 ====================
def _rhyme(word):
    """提取词尾字的韵母, 作为押韵依据 (如 '家' -> 'a', '天' -> 'ian')"""
    if not word:
        return ''
    try:
        py = lazy_pinyin(word[-1], style=Style.FINALS)   # FINALS=只取韵母, 不带声调
        return py[0] if py else ''
    except Exception:
        return ''


def build_rhyme_dict(word_dict):
    """
    从歌词语料构建 韵母 -> [词索引] 词典
    方法: 每行歌词 jieba 分词, 取行末词, 按尾字韵母归类
    过滤词数 < 5 的韵组, 保证押韵时有足够候选词可选
    """
    rhyme_dict = {}
    with open('./07/data/jaychou_lyrics.txt', 'r', encoding='utf-8') as f:
        for line in f:
            words = [w for w in jieba.lcut(line) if w.strip()]   # 去掉空白/换行
            if not words:
                continue
            last = words[-1]                        # 行末词 = 韵脚词
            r = _rhyme(last)
            if r and last in word_dict:
                rhyme_dict.setdefault(r, set()).add(word_dict[last])
    # 只保留候选词 >= 5 个的韵组
    return {r: list(v) for r, v in rhyme_dict.items() if len(v) >= 5}


def _sample(logits, temperature, top_k, device):
    """温度 + top-k 采样"""
    logits = logits / temperature
    topk_vals, topk_idx = torch.topk(logits, top_k)
    probs = torch.softmax(topk_vals, dim=-1)
    return topk_idx[torch.multinomial(probs, 1)].item()


def _sample_rhyme(logits, candidates, temperature, device):
    """
    押韵采样: 只允许从指定韵脚组的候选词里选
    用模型概率加权 → 既押韵, 又尽量通顺 (概率高的词优先)
    """
    mask = torch.full_like(logits, float('-inf'))
    mask[candidates] = logits[candidates]           # 非候选词概率置 -inf
    logits = mask / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def generate_song(model, word_dict, start_words, num_lines=6, max_line_len=14,
                  temperature=0.6, top_k=5, rhyme_dict=None):
    """
    生成一段完整的、押韵的歌词 (ABAB式: 偶数行押同一韵)

    断句自然: 按歌词语料学到的行结构, 模型输出换行符时结束本行, 行宽有上限
    有韵律  : 第2行行尾确定"主韵", 第4/6行行尾强制从主韵候选词中采样

    :param model: 训练好的模型
    :param word_dict: 词->索引 词表
    :param start_words: 起始歌词
    :param num_lines: 生成行数 (默认6行 = 一段)
    :param max_line_len: 每行最多词数 (行宽上限, 防超长句)
    :param temperature: 温度, 0.6 保守
    :param top_k: 采样候选数
    :param rhyme_dict: 韵脚词典, 不传则自动构建
    :return: 多行歌词字符串
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    index2word = {v: k for k, v in word_dict.items()}
    if rhyme_dict is None:
        rhyme_dict = build_rhyme_dict(word_dict)

    # 1. 起始词分词 -> 索引序列
    idxs = [word_dict[w] for w in jieba.lcut(start_words) if w in word_dict]
    if not idxs:
        raise ValueError(f'起始词"{start_words}"分词后都不在词表中')

    main_rhyme = None        # 主韵脚 (第2行确定)
    lines = []               # 每行一个词索引列表
    h = None
    input_seq = torch.tensor([idxs], device=device).long()

    with torch.no_grad():
        for li in range(num_lines):
            line = []
            # 偶数行(行号1,3,5)行尾押主韵; 第2行(li==1)自由定韵
            need_rhyme = (li >= 1 and li % 2 == 1)
            attempts = 0                    # 本行总步数保护, 防止连续采到换行符时死循环

            while len(line) < max_line_len:
                attempts += 1
                if attempts > max_line_len * 30:   # 异常保护: 强制收尾
                    break

                out, h = model(input_seq if h is None else input_seq[:, -1:], h)
                logits = out[0, -1]

                # ---- 行尾处理: 到达行宽上限, 收尾 ----
                if len(line) == max_line_len - 1:
                    while True:
                        if need_rhyme and main_rhyme and main_rhyme in rhyme_dict:
                            nxt = _sample_rhyme(logits, rhyme_dict[main_rhyme], temperature, device)
                        else:
                            nxt = _sample(logits, temperature, top_k, device)
                        word = index2word[nxt]
                        if word in ('\n', ' '):
                            continue            # 行尾不能是空白, 重采
                        break
                    # 第2行: 用行尾词确定主韵
                    if li == 1 and main_rhyme is None:
                        main_rhyme = _rhyme(word)
                    line.append(nxt)
                    break

                # ---- 行中采样: 采到换行符则本行结束(断句交给模型) ----
                nxt = _sample(logits, temperature, top_k, device)
                word = index2word[nxt]
                if word == '\n':
                    if not line:
                        continue            # 本行还没词, 不能空行, 重采
                    break
                line.append(nxt)
                input_seq = torch.tensor([[nxt]], device=device).long()

            # 行尾词续接下一行 (防御: 行不可能为空, 因为空行会强制重采)
            if not line:
                line.append(idxs[0])        # 兜底: 用起始词第一个词
            lines.append(line)
            input_seq = torch.tensor([[line[-1]]], device=device).long()

    # 拼接: 每行内部无空格, 行间换行
    return '\n'.join(''.join(index2word[i] for i in line) for line in lines)


# ==================== 6. 主流程 ====================
if __name__ == '__main__':
    # 0. 设备: 优先 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if device.type == 'cuda':
        print(f'GPU型号: {torch.cuda.get_device_name(0)}  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')

    # 1. 获取数据，进行分词，构建词典，构建数据集
    word_dict, corpus_idx, word_count, all_words = build_vocab()

    # 2. 数据预处理，准备数据集
    num_chars = 32          # 每个样本的字符数(时间步长度)
    dataset = MyDataset(corpus_idx, word_dict, num_chars)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'样本总数: {len(dataset)}')

    # 3. 搭建循环神经网络 (默认LSTM, 想用vanilla RNN把 cell_type 改成 'rnn')
    cell_type = 'lstm'
    model = MyModel(word_count, embedding_dim=128, hidden_size=256, num_layers=2, cell_type=cell_type).to(device)
    print(f'模型: {cell_type.upper()} | 参数量: {sum(p.numel() for p in model.parameters()):,}')

    # 4. 检测已保存模型: 可加载直接预测, 或重新训练
    # 模型文件名带 cell_type, 避免 LSTM/RNN 权重混用导致加载失败
    model_path = f'./07/model/{cell_type}_model.pth'
    if os.path.exists(model_path):
        choice = input('检测到已保存模型，输入 1 加载后直接生成歌词，直接回车重新训练: ').strip()
        if choice == '1':
            model.load_state_dict(torch.load(model_path, map_location=device))
            print('模型加载完成')
        else:
            train(model, dataloader, epochs=100)
            torch.save(model.state_dict(), model_path)
            print(f'模型已保存到 {model_path}')
    else:
        train(model, dataloader, epochs=100)
        torch.save(model.state_dict(), model_path)
        print(f'模型已保存到 {model_path}')

    # 5. 交互式歌词生成: 输入任意起始歌词, 生成一段完整押韵歌词
    print('\n========== 歌词生成 ==========')
    print('输入起始歌词后回车，生成一段押韵歌词（6行）；输入 quit 退出')
    rhyme_dict = build_rhyme_dict(word_dict)   # 构建韵脚词典(一次即可)
    print(f'韵脚词典: {len(rhyme_dict)} 个韵组')
    while True:
        start_words = input('请输入起始歌词: ').strip()
        if start_words.lower() == 'quit':
            print('已退出')
            break
        if not start_words:
            continue
        try:
            print('\n' + generate_song(model, word_dict, start_words, rhyme_dict=rhyme_dict) + '\n')
        except ValueError as e:
            print(f'错误: {e}')
