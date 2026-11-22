"""
案例:
    LSTM案例，字符级 AI歌词生成器（带韵律/断句优化）。

    为什么用字符级(而非词级)?
        词级: 5703个词 vs 4.3万词小语料 = 每个词平均只出现7次, 模型学不熟, 生成容易复读/生造
        字符级: 词表只有 ~2000 个字符, 每个字符出现几百次, 数据密度高10倍, 生成更流畅

    与词级(04)的区别:
        1. 不再用 jieba 分词, 直接按"字符"建词表
        2. 标点符号也是词表成员 → 模型自己学会在哪里断句(逗号/句号/换行)
        3. 韵律增强: 提取行末字符的韵母, 偶数行行尾强制押同一韵

    运行: D:\conda\envs\robot-pose\python.exe 05_LSTM案例_AI歌词生成器_字符级.py
"""

# 导包
import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import time
from pypinyin import lazy_pinyin, Style   # 拼音库: 提取行末字的韵母, 实现押韵

DATA_PATH = './07/data/jaychou_lyrics_clean.txt'   # 优化版: 繁转简+去英文/数字/符号
MODEL_PATH = './07/model/char_lstm_model.pth'


# ==================== 1. 获取数据, 构建字符级词表与索引序列 ====================
def build_vocab():
    # 1. 读取全文
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    # 2. 所有出现的字符去重, 排序(保证词表顺序稳定, 可复现)
    unique_chars = sorted(set(text))
    # 3. 词表: 字符 -> 索引
    word_dict = {c: i for i, c in enumerate(unique_chars)}
    # 4. 全文转索引序列
    corpus_idx = [word_dict[c] for c in text]
    print(f'字符表大小: {len(unique_chars)} | 语料总字符数: {len(text)}')
    return word_dict, corpus_idx, len(unique_chars)


# ==================== 2. 数据预处理, 准备数据集 ====================
class MyDataset(Dataset):
    """把字符索引序列切成 (输入, 标签) 样本对, 标签是输入右移一位"""

    def __init__(self, corpus_idx, word_dict, num_chars):
        self.corpus_idx = corpus_idx
        self.word_dict = word_dict
        self.word_count = len(self.corpus_idx)
        self.num_chars = num_chars
        self.num_sentences = (self.word_count - 1) // self.num_chars

    def __len__(self):
        return self.num_sentences

    def __getitem__(self, index):
        start = index * self.num_chars
        end = start + self.num_chars + 1
        sentence = self.corpus_idx[start: end - 1]
        label = self.corpus_idx[start + 1: end]
        return torch.tensor(sentence, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ==================== 3. 搭建LSTM神经网络 ====================
class MyModel(nn.Module):
    """字符嵌入 -> LSTM -> 全连接(字符表概率分布)"""

    def __init__(self, char_count, embedding_dim=128, hidden_size=256, num_layers=2):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=char_count, embedding_dim=embedding_dim)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=char_count)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x, h=None):
        emb = self.embedding(x)
        if h is None:
            z = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            h = (z, z.clone())   # LSTM 需要 (h, c) 两个状态
        out, h = self.rnn(emb, h)
        out = self.fc(out)
        return out, h


# ==================== 4. 训练模型 ====================
def train(model, dataloader, epochs=100, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss, start_time = 0, time.time()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out.reshape(-1, out.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # 梯度裁剪
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}  Loss: {total_loss / len(dataloader):.4f}  耗时: {time.time() - start_time:.1f}s')


# ==================== 5. 韵律增强: 韵脚词典 + 完整歌词生成 ====================
def _rhyme(ch):
    """提取字符的韵母, 作为押韵依据 (如 '家' -> 'a', '天' -> 'ian')"""
    try:
        py = lazy_pinyin(ch, style=Style.FINALS)
        return py[0] if py else ''
    except Exception:
        return ''


def build_rhyme_dict(word_dict):
    """
    从歌词语料构建 韵母 -> [字符索引] 词典
    方法: 每行歌词取行末字符(去掉空白/标点), 按韵母归类
    """
    rhyme_dict = {}
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            # 取行内最后一个非空白、非标点的字符作为韵脚
            chars = [c for c in line.strip() if not c.isspace() and c not in '，。！？、；：""''（）《》【】…—']
            if not chars:
                continue
            last = chars[-1]
            r = _rhyme(last)
            if r and last in word_dict:
                rhyme_dict.setdefault(r, set()).add(word_dict[last])
    # 只保留候选字符 >= 3 个的韵组
    return {r: list(v) for r, v in rhyme_dict.items() if len(v) >= 3}


def _sample(logits, temperature, top_k, penalized=None, penalty=1.3):
    """
    温度 + top-k 采样 (带重复惩罚)
    :param penalized: 已出现过的字符索引列表, 对这些字符的概率打折扣, 防止自我强化循环
    :param penalty: 惩罚系数, >1 降低已出现字符被再采样的概率 (1.3 常用)
    """
    if penalized is not None and len(penalized) > 0:
        logits = logits.clone()
        p = torch.tensor(penalized, device=logits.device, dtype=torch.long)
        pos = logits[p] > 0
        logits[p[pos]] /= penalty      # 正logits除以惩罚
        logits[p[~pos]] *= penalty     # 负logits乘以惩罚(抬高, 效果同降低)
    logits = logits / temperature
    topk_vals, topk_idx = torch.topk(logits, top_k)
    probs = torch.softmax(topk_vals, dim=-1)
    return topk_idx[torch.multinomial(probs, 1)].item()


def _sample_rhyme(logits, candidates, temperature):
    """押韵采样: 只允许从指定韵脚组的候选字符里选, 模型概率加权保证通顺"""
    mask = torch.full_like(logits, float('-inf'))
    mask[candidates] = logits[candidates]
    logits = mask / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def _sample_valid(logits, temperature, top_k, penalized, penalty, valid_mask):
    """
    采样一个"合法字符"(汉字/常见标点): 最多重试50次, 仍失败则取合法字符中logits最高的
    :param valid_mask: 布尔张量, 标记词表每个索引是否为合法字符
    """
    for _ in range(50):
        nxt = _sample(logits, temperature, top_k, penalized, penalty)
        if valid_mask[nxt]:
            return nxt
    logits2 = logits.clone()
    logits2[~valid_mask] = float('-inf')
    return int(torch.argmax(logits2).item())


def generate_song(model, word_dict, start_words, num_lines=8, max_line_len=12, min_line_len=5,
                  temperature=0.7, top_k=10, rhyme_dict=None, rep_penalty=1.5):
    """
    生成一段完整的、押韵的歌词

    断句自然: 每行至少 min_line_len 个字符, 之后模型采到空白(行尾信号)才断句
    有韵律  : 第2行行尾确定"主韵", 第4/6/8行行尾强制押主韵 (ABAB式)
    不重复  : 重复惩罚 rep_penalty, 防止"我我我我"式自我强化循环
    纯净    : 只允许汉字/常见标点, 过滤英文、特殊符号

    :param start_words: 起始歌词(字符串, 字符级直接逐字生成)
    :return: 多行歌词字符串
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    index2word = {v: k for k, v in word_dict.items()}
    if rhyme_dict is None:
        rhyme_dict = build_rhyme_dict(word_dict)

    # 起始词逐字符取词表内字符
    idxs = [word_dict[c] for c in start_words if c in word_dict]
    if not idxs:
        raise ValueError(f'起始歌词"{start_words}"的字符都不在词表中')

    # 合法字符掩码: 汉字(Unicode CJK) + 常见中文标点, 过滤英文/数字/特殊符号
    PUNCT = '，。！？、；：·'
    valid_mask = torch.tensor([
        ('\u4e00' <= index2word[i] <= '\u9fff') or (index2word[i] in PUNCT)
        for i in range(len(word_dict))
    ], device=device)
    # 空白字符标记表
    isspace = torch.tensor([index2word[i].isspace() for i in range(len(word_dict))], device=device)

    main_rhyme = None
    lines = []
    h = None
    used = list(idxs)                 # 已出现字符索引, 用于重复惩罚
    input_seq = torch.tensor([idxs], device=device).long()

    with torch.no_grad():
        for li in range(num_lines):
            line = []
            need_rhyme = (li >= 1 and li % 2 == 1)   # 偶数行押主韵
            attempts = 0

            while len(line) < max_line_len:
                attempts += 1
                if attempts > max_line_len * 30:
                    break

                out, h = model(input_seq if h is None else input_seq[:, -1:], h)
                logits = out[0, -1]

                # ---- 行尾处理: 到达行宽上限, 收尾 ----
                if len(line) == max_line_len - 1:
                    while True:
                        if need_rhyme and main_rhyme and main_rhyme in rhyme_dict:
                            nxt = _sample_rhyme(logits, rhyme_dict[main_rhyme], temperature)
                        else:
                            nxt = _sample_valid(logits, temperature, top_k, used, rep_penalty, valid_mask)
                        ch = index2word[nxt]
                        if ch.isspace():
                            continue          # 行尾不能是空白
                        break
                    if li == 1 and main_rhyme is None:
                        main_rhyme = _rhyme(ch)
                    line.append(nxt)
                    used.append(nxt)
                    break

                # ---- 行中采样 ----
                if len(line) < min_line_len:
                    # 行开头/前段: 强制合法非空白字符, 保证每行够长、断句自然
                    nxt = _sample_valid(logits, temperature, top_k, used, rep_penalty, valid_mask)
                else:
                    # 行中后段: 正常采样, 采到空白 = 模型认为句子结束
                    nxt = _sample(logits, temperature, top_k, penalized=used, penalty=rep_penalty)
                    if index2word[nxt].isspace():
                        # 断句前, 押韵行补一个押韵字作为行尾 (ABAB韵脚约束)
                        if need_rhyme and main_rhyme and main_rhyme in rhyme_dict:
                            rhyme_nxt = _sample_rhyme(logits, rhyme_dict[main_rhyme], temperature)
                            line.append(rhyme_nxt)
                            used.append(rhyme_nxt)
                        break
                    if not valid_mask[nxt]:
                        nxt = _sample_valid(logits, temperature, top_k, used, rep_penalty, valid_mask)
                line.append(nxt)
                used.append(nxt)
                input_seq = torch.tensor([[nxt]], device=device).long()

            if not line:
                line.append(idxs[0])
            # 第2行结束: 用行尾字确定主韵 (若行尾处理未设置)
            if li == 1 and main_rhyme is None and line:
                main_rhyme = _rhyme(index2word[line[-1]])
            lines.append(line)
            input_seq = torch.tensor([[line[-1]]], device=device).long()

    return '\n'.join(''.join(index2word[i] for i in line) for line in lines)


# ==================== 6. 主流程 ====================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if device.type == 'cuda':
        print(f'GPU型号: {torch.cuda.get_device_name(0)}')

    # 1. 字符级词表
    word_dict, corpus_idx, char_count = build_vocab()

    # 2. 数据集
    num_chars = 64
    dataset = MyDataset(corpus_idx, word_dict, num_chars)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    print(f'样本总数: {len(dataset)}')

    # 3. 模型
    model = MyModel(char_count).to(device)
    print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')

    # 4. 检测已保存模型
    if os.path.exists(MODEL_PATH):
        choice = input('检测到已保存模型，输入 1 加载后直接生成歌词，直接回车重新训练: ').strip()
        if choice == '1':
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print('模型加载完成')
        else:
            train(model, dataloader, epochs=100)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f'模型已保存到 {MODEL_PATH}')
    else:
        train(model, dataloader, epochs=100)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f'模型已保存到 {MODEL_PATH}')

    # 5. 交互式生成一段完整押韵歌词
    print('\n========== 歌词生成 ==========')
    print('输入起始歌词后回车，生成一段押韵歌词（8行）；输入 quit 退出')
    rhyme_dict = build_rhyme_dict(word_dict)
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
