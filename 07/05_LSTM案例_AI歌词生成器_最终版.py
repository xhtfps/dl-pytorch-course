# -*- coding: utf-8 -*-
"""
案例:
    LSTM 歌词生成器 · 最终版 (字符级)
    整合 05~08 全部有效优化, 目标: 生成"更像杰伦"的歌词

    为什么是字符级(而非词级)?
        词表 2354 字符 × 5.9万字符 → 每个字出现25次, 模型学得熟 (val loss 1.36)
        词级 5520 词 × 3.4万词     → 每词仅6次, 语序学不会 (val loss 7.02, 已实测否决)

    四项优化:
        1. 风格引导 : 采样时给杰伦风格字表加低偏置(+0.3), 用词偏向"周氏意象"
                      (注意: 偏置不能高, 实测 1.0 会自我强化成"默默默默默")
        2. 结构化   : 数据含 <副歌> 标记(内部用私有字符 \ue000 表示),
                      生成时主歌4行 + 副歌4行, 有"歌"的骨架
        3. 模型升级 : 字符级 LSTM hidden 512×2 + dropout 0.3
        4. 训练优化 : 验证集早停 + ReduceLROnPlateau 学习率衰减

    数据: data/jaychou_lyrics_structured.txt (清洗版 + 副歌标记)
    运行: D:\conda\envs\robot-pose\python.exe 09_LSTM案例_AI歌词生成器_最终版.py
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pypinyin import lazy_pinyin, Style

DATA_PATH = './07/data/jaychou_lyrics_structured.txt'   # 结构化: 清洗 + 副歌标记
MODEL_PATH = './07/model/lyrics_final_model.pth'
TAG = '\ue000'                  # 副歌标记 (私有区字符, 1个token, 不会与歌词冲突)
TAG_TEXT = '<副歌>'             # 数据文件里的标记原文, 读入时替换

NUM_CHARS = 64                  # 字符级窗口
BATCH_SIZE = 64
EPOCHS = 300                    # 上限, 早停决定
LR = 1e-3
PATIENCE = 15                   # 验证 loss 连续15轮不降 → 早停
CLIP = 5.0
STYLE_BIAS = 0.3                # 风格字低偏置 (实测 >1.0 会自我强化重复)

# 杰伦风格字: 由风格词拆字 (口语周氏 + 文山意象)
STYLE_CHARS = set(''.join([
    '安静', '沉默', '回忆', '微笑', '眼泪', '晴天', '雨天', '你说', '我想', '记得',
    '从前', '故事', '背影', '温柔', '假装', '离开', '等候', '想念', '拥抱', '结局',
    '最美', '烟雨', '江南', '墨', '月', '风', '稻香', '青花', '菊花', '古道', '燕',
    '檀香', '骤雨', '朱砂', '马蹄', '长亭', '夜色', '蝉鸣', '萤火虫', '篱笆', '炊烟',
    '落花', '涟漪', '天涯', '琴声', '月色', '枫叶', '搁浅', '放晴', '小时候',
]))


# ==================== 1. 字符级词表 (含副歌标记) ====================
def build_vocab():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    # 数据里的 <副歌> 标记 → 单个私有字符 (1 token)
    text = text.replace(TAG_TEXT, TAG)
    unique_chars = sorted(set(text))
    word_dict = {c: i for i, c in enumerate(unique_chars)}
    corpus_idx = [word_dict[c] for c in text]
    print(f'字符表大小: {len(unique_chars)} | 语料总字符数: {len(text)} | 副歌标记在词表: {TAG in word_dict}')
    return word_dict, corpus_idx, len(unique_chars)


# ==================== 2. 数据集 ====================
class MyDataset(Dataset):
    def __init__(self, corpus_idx, num_chars):
        self.corpus_idx = corpus_idx
        self.num_chars = num_chars
        self.num_sentences = (len(corpus_idx) - 1) // num_chars

    def __len__(self):
        return self.num_sentences

    def __getitem__(self, index):
        start = index * self.num_chars
        end = start + self.num_chars + 1
        sentence = self.corpus_idx[start: end - 1]
        label = self.corpus_idx[start + 1: end]
        return torch.tensor(sentence, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ==================== 3. 模型: 字符级 LSTM 升级版 ====================
class MyModel(nn.Module):
    def __init__(self, char_count, embedding_dim=128, hidden_size=512, num_layers=2):
        super().__init__()
        self.char_count = char_count
        self.embedding = nn.Embedding(char_count, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, char_count)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x, h=None):
        emb = self.embedding(x)
        if h is None:
            z = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            h = (z, z.clone())
        out, h = self.lstm(emb, h)
        return self.fc(out), h


# ==================== 4. 训练: 验证集早停 + 学习率衰减 ====================
def train(model, dataloader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    best_val, best_epoch, bad_epochs = float('inf'), 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out.reshape(-1, model.char_count), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            total += loss.item()
        tr = total / len(dataloader)

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out, _ = model(x)
                vtotal += criterion(out.reshape(-1, model.char_count), y.reshape(-1)).item()
        va = vtotal / len(val_loader)
        scheduler.step(va)

        if va < best_val:
            best_val, best_epoch, bad_epochs = va, epoch, 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            bad_epochs += 1
        if epoch % 10 == 0 or bad_epochs == 0:
            print(f'Epoch {epoch:3d} | train {tr:.4f} | val {va:.4f} | best {best_val:.4f}@{best_epoch} | lr {optimizer.param_groups[0]["lr"]:.1e}')
        if bad_epochs >= PATIENCE:
            print(f'早停: 最优 Epoch {best_epoch}, val loss {best_val:.4f}')
            break
    print(f'训练完成, 模型已保存 {MODEL_PATH}')


# ==================== 5. 韵律 ====================
def _rhyme(ch):
    try:
        py = lazy_pinyin(ch, style=Style.FINALS)
        return py[0] if py else ''
    except Exception:
        return ''


def build_rhyme_dict(word_dict):
    rhyme_dict = {}
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            chars = [c for c in line.strip() if not c.isspace() and c not in '，。！？、；：""''（）《》【】…—']
            if not chars:
                continue
            last = chars[-1]
            r = _rhyme(last)
            if r and last in word_dict:
                rhyme_dict.setdefault(r, set()).add(word_dict[last])
    return {r: list(v) for r, v in rhyme_dict.items() if len(v) >= 3}


# ==================== 6. 采样 ====================
def _sample(logits, temperature, top_k, penalized=None, penalty=1.3,
            style_idx=None, style_bias=STYLE_BIAS, bigram_pairs=None, bigram_penalty=2.5):
    """
    温度 + top-k 采样 (带重复惩罚 + 风格字低偏置 + 双字序列惩罚)
    :param style_idx: 风格字索引列表, 采样时加低偏置 (抬高周氏用词概率)
    :param bigram_pairs: 已出现过的 (前字, 后字) 索引对, 对后字打折 → 治"儿啦呜"式交替循环
    """
    if penalized is not None and len(penalized) > 0:
        logits = logits.clone()
        p = torch.tensor(penalized, device=logits.device, dtype=torch.long)
        pos = logits[p] > 0
        logits[p[pos]] /= penalty
        logits[p[~pos]] *= penalty
    if bigram_pairs:
        logits = logits.clone()
        for _, nxt in bigram_pairs:      # 只惩罚后字
            logits[nxt] /= bigram_penalty
    if style_idx and style_bias:
        logits = logits.clone()
        logits[style_idx] += style_bias
    logits = logits / temperature
    topk_vals, topk_idx = torch.topk(logits, top_k)
    probs = torch.softmax(topk_vals, dim=-1)
    return topk_idx[torch.multinomial(probs, 1)].item()


def _sample_rhyme(logits, candidates, temperature):
    mask = torch.full_like(logits, float('-inf'))
    mask[candidates] = logits[candidates]
    logits = mask / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def _sample_valid(logits, temperature, top_k, penalized, penalty, valid_mask, style_idx,
                  bigram_pairs=None):
    for _ in range(50):
        nxt = _sample(logits, temperature, top_k, penalized, penalty, style_idx, STYLE_BIAS,
                      bigram_pairs)
        if valid_mask[nxt]:
            return nxt
    logits2 = logits.clone()
    logits2[~valid_mask] = float('-inf')
    return int(torch.argmax(logits2).item())


# ==================== 7. 生成: 主歌4行 + 副歌4行 ====================
def generate_song(model, word_dict, start_words, num_lines=8, max_line_len=12,
                  min_line_len=5, temperature=0.7, top_k=10,
                  rhyme_dict=None, rep_penalty=1.5):
    """
    最终版生成:
        结构: 主歌4行(偶数行押韵) + <副歌>标记 + 副歌4行(偶数行押韵)
        风格: 全局低偏置风格字 (+0.3), 不破坏语序
        纯净: 只生成汉字/常见标点, 标记字符 \ue000 只由代码插入, 采样禁止
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    index2word = {v: k for k, v in word_dict.items()}
    if rhyme_dict is None:
        rhyme_dict = build_rhyme_dict(word_dict)

    idxs = [word_dict[c] for c in start_words if c in word_dict]
    if not idxs:
        raise ValueError(f'起始歌词"{start_words}"的字符都不在词表中')

    PUNCT = '，。！？、；：·'
    valid_mask = torch.tensor([
        ('\u4e00' <= index2word[i] <= '\u9fff') or (index2word[i] in PUNCT)
        for i in range(len(word_dict))
    ], device=device)
    isspace = torch.tensor([index2word[i].isspace() for i in range(len(word_dict))], device=device)
    # 风格字索引 (只保留在词表里的)
    style_idx = [word_dict[c] for c in STYLE_CHARS if c in word_dict]
    tag_idx = word_dict[TAG]

    def gen_block(seed_idxs, h, style_bias):
        """生成一个4行块: 偶数行押主韵; 返回 (lines, h)"""
        main_rhyme = None
        lines = []
        used = list(seed_idxs)
        used_pairs = set()              # 已出现的 (前字,后字) 对 → 双字重复惩罚
        prev_idx = seed_idxs[-1]
        input_seq = torch.tensor([seed_idxs], device=device).long()
        with torch.no_grad():
            for li in range(4):
                line = []
                need_rhyme = (li >= 1 and li % 2 == 1)
                attempts = 0
                while len(line) < max_line_len:
                    attempts += 1
                    if attempts > max_line_len * 30:
                        break
                    # 主歌首次: 喂全部种子建立状态; 之后每次只喂最近1个字符
                    out, h = model(input_seq if h is None else input_seq[:, -1:], h)
                    logits = out[0, -1]
                    if len(line) == max_line_len - 1:
                        while True:
                            if need_rhyme and main_rhyme and main_rhyme in rhyme_dict:
                                nxt = _sample_rhyme(logits, rhyme_dict[main_rhyme], temperature)
                            else:
                                nxt = _sample_valid(logits, temperature, top_k, used, rep_penalty, valid_mask, style_idx, used_pairs)
                            ch = index2word[nxt]
                            if ch.isspace():
                                continue
                            break
                        if li == 1 and main_rhyme is None:
                            main_rhyme = _rhyme(ch)
                        line.append(nxt)
                        used.append(nxt)
                        used_pairs.add((prev_idx, nxt))
                        prev_idx = nxt
                        break
                    if len(line) < min_line_len:
                        nxt = _sample_valid(logits, temperature, top_k, used, rep_penalty, valid_mask, style_idx, used_pairs)
                    else:
                        nxt = _sample(logits, temperature, top_k, penalized=used, penalty=rep_penalty,
                                      style_idx=style_idx, style_bias=style_bias, bigram_pairs=used_pairs)
                        if index2word[nxt].isspace():
                            if need_rhyme and main_rhyme and main_rhyme in rhyme_dict:
                                rn = _sample_rhyme(logits, rhyme_dict[main_rhyme], temperature)
                                line.append(rn)
                                used.append(rn)
                                used_pairs.add((prev_idx, rn))
                                prev_idx = rn
                            break
                        if not valid_mask[nxt]:
                            nxt = _sample_valid(logits, temperature, top_k, used, rep_penalty, valid_mask, style_idx, used_pairs)
                    line.append(nxt)
                    used.append(nxt)
                    used_pairs.add((prev_idx, nxt))
                    prev_idx = nxt
                    input_seq = torch.tensor([[nxt]], device=device).long()
                if not line:
                    line.append(seed_idxs[-1])
                if li == 1 and main_rhyme is None and line:
                    main_rhyme = _rhyme(index2word[line[-1]])
                lines.append(line)
                input_seq = torch.tensor([[line[-1]]], device=device).long()
        return lines, h

    # 主歌 (低偏置风格)
    verse, h = gen_block(idxs, None, style_bias=STYLE_BIAS)
    # 插入副歌标记, 让模型"看见"标记再续写
    with torch.no_grad():
        _, h = model(torch.tensor([[tag_idx]], device=device).long(), h)
    # 副歌种子: 标记 + 主歌末句末字
    chorus_seed = [tag_idx, verse[-1][-1]]
    chorus, _ = gen_block(chorus_seed, h, style_bias=STYLE_BIAS)

    return '\n'.join(''.join(index2word[i] for i in line) for line in verse + [[]] + chorus)


# ==================== 8. 主流程 ====================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device} | GPU: {torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"}')

    word_dict, corpus_idx, char_count = build_vocab()
    dataset = MyDataset(corpus_idx, NUM_CHARS)
    n = len(dataset)
    n_train = int(n * 0.9)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n - n_train])
    dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    vdl = DataLoader(val_ds, batch_size=BATCH_SIZE)
    print(f'样本: {n} (训练 {n_train} / 验证 {n - n_train})')

    model = MyModel(char_count).to(device)
    print(f'模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    if os.path.exists(MODEL_PATH):
        choice = input('检测到已保存模型，输入 1 加载后直接生成，直接回车重新训练: ').strip()
        if choice == '1':
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print('模型加载完成')
        else:
            train(model, dl, vdl, device)
    else:
        train(model, dl, vdl, device)

    print('\n========== 最终版歌词生成 ==========')
    print('结构: 主歌4行 + 副歌4行 | 偶数行押韵 | 周氏风格字引导')
    print('输入起始歌词后回车生成；输入 quit 退出')
    rhyme_dict = build_rhyme_dict(word_dict)
    print(f'韵脚词典: {len(rhyme_dict)} 个韵组')
    while True:
        start_words = input('\n请输入起始歌词: ').strip()
        if start_words.lower() == 'quit':
            break
        if not start_words:
            continue
        try:
            print('\n' + generate_song(model, word_dict, start_words, rhyme_dict=rhyme_dict) + '\n')
        except ValueError as e:
            print(f'错误: {e}')
