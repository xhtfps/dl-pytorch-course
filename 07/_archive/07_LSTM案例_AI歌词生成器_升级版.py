# -*- coding: utf-8 -*-
"""
案例:
    LSTM 歌词生成器 · 升级版 (词级)
    目标: 生成"更像杰伦"的歌词

    相比 04/05 的四项升级:
        1. 风格引导   : 采样时给杰伦风格词表加 logits 偏置 (用词立刻"周氏")
        2. 结构化标记 : 数据含 <副歌> 标记, 生成时主歌4行 + 副歌4行 (有"歌"的骨架)
        3. 模型升级   : 词级回归 + Embedding256 + LSTM hidden512×3层 + dropout0.3
        4. 训练优化   : 验证集早停 + ReduceLROnPlateau 学习率衰减

    数据: data/jaychou_lyrics_structured.txt (清洗版 + 副歌标记, 6236行)
    运行: D:\conda\envs\robot-pose\python.exe 07_LSTM案例_AI歌词生成器_升级版.py
"""
import torch
import torch.nn as nn
import jieba

DATA_PATH = './07/data/jaychou_lyrics_structured.txt'
MODEL_PATH = './07/model/lyrics_v2_model.pth'
TAG = '<副歌>'          # 结构化标记 token

NUM_CHARS = 32          # 词级窗口: 每样本 32 个词
BATCH_SIZE = 64
EPOCHS = 300            # 上限, 实际由早停决定
LR = 3e-4               # 小数据+大模型 → 用小学习率防过拟合
PATIENCE = 25           # 验证 loss 连续 25 轮不降 → 早停 (小数据 val 波动大)
CLIP = 5.0              # 梯度裁剪
WEIGHT_DECAY = 1e-4     # L2 正则
LABEL_SMOOTH = 0.1      # 标签平滑, 防过拟合

# ============ 1. 杰伦风格词表 (生成时 logits 偏置) ============
STYLE_WORDS = [
    # 口语周氏 (高频生活化表达)
    '安静', '沉默', '回忆', '微笑', '眼泪', '晴天', '雨天', '你说', '我想', '记得',
    '从前', '故事', '背影', '温柔', '假装', '离开', '等候', '想念', '拥抱', '结局',
    '最美', '放晴', '搁浅', '简单', '发如雪', '安静', '慢动作', '小时候',
    # 文山意象 (古典文艺词)
    '烟雨', '江南', '墨', '月', '风', '稻香', '青花', '菊花', '古道', '燕',
    '檀香', '骤雨', '朱砂', '马蹄', '长亭', '夜色', '蝉鸣', '萤火虫', '篱笆', '炊烟',
    '落花', '涟漪', '天涯', '琴声', '月色', '枫叶',
]
STYLE_BIAS = 0.5        # 风格词 logits 加成强度
REP_PENALTY = 1.3       # 重复惩罚
TEMPERATURE = 0.7
TOP_K = 8


# ============ 2. 词表构建 (含 <副歌> 整词) ============
def build_vocab():
    unique_words, all_words = [], []
    for line in open(DATA_PATH, 'r', encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        # 标记行整体作为一个词, 不经过 jieba
        words = [TAG] if line == TAG else jieba.lcut(line)
        all_words.extend(words)
        for w in words:
            if w not in unique_words:
                unique_words.append(w)
    word_dict = {w: i for i, w in enumerate(unique_words)}
    corpus_idx = [word_dict[w] for w in all_words]
    return word_dict, corpus_idx, len(unique_words)


# ============ 3. 数据集 (滑动窗口, 标签右移) ============
class MyDataset(torch.utils.data.Dataset):
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
        return torch.tensor(sentence), torch.tensor(label)


# ============ 4. 模型: 词级 LSTM 升级版 ============
class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=512, num_layers=3):
        super().__init__()
        self.vocab_size = vocab_size
        # 词嵌入: 词ID -> 256维词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 3层LSTM, 层间 dropout=0.5 防过拟合 (只有多层时生效)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            batch_first=True, dropout=0.5)
        # 输出映射: 512维记忆 -> 全词表分数
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)                 # [B, N, 256]
        out, _ = self.lstm(emb)                 # [B, N, 512]
        return self.fc(out)                     # [B, N, vocab]


# ============ 5. 训练: 验证集早停 + 学习率衰减 ============
def train(model, dataloader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # 标签平滑: 目标分布不再是 one-hot, 而是 0.9/0.1 摊开 → 防过拟合
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    # 学习率衰减: 验证 loss 5 轮不降 → 减半
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    best_val, best_epoch, bad_epochs = float('inf'), 0, 0
    for epoch in range(1, EPOCHS + 1):
        # --- 训练 ---
        model.train()
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                   # [B, N, vocab]
            loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(dataloader)

        # --- 验证 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits.reshape(-1, model.vocab_size),
                                      y.reshape(-1)).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)                # 按验证 loss 衰减 lr

        # --- 早停: 保存最优验证模型 ---
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            bad_epochs = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            bad_epochs += 1
        if epoch % 10 == 0 or bad_epochs == 0:
            print(f'Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | best {best_val:.4f}@{best_epoch} | lr {optimizer.param_groups[0]["lr"]:.1e}')
        if bad_epochs >= PATIENCE:
            print(f'早停: {PATIENCE} 轮未提升, 停止。最优模型在 Epoch {best_epoch}, val loss {best_val:.4f}')
            break
    print(f'训练完成, 最优 Epoch {best_epoch}, val loss {best_val:.4f}')


# ============ 6. 韵律: 行尾词韵母分组 ============
def _rhyme(word):
    """取词尾字的韵母 (简单韵母表, 够教学用)"""
    if not word:
        return ''
    try:
        from pypinyin import pinyin, Style
        last_char = word[-1]
        if not ('\u4e00' <= last_char <= '\u9fff'):
            return ''
        s = pinyin(last_char, style=Style.FINALS)[0][0]
        return s
    except Exception:
        return ''


def build_rhyme_dict(word_dict):
    """统计语料中每行的行尾词, 按韵母分组 → {韵母: [行尾词...]}"""
    rhyme = {}
    for line in open(DATA_PATH, encoding='utf-8'):
        line = line.strip()
        if not line or line == TAG:
            continue
        words = jieba.lcut(line)
        last_word = words[-1]
        r = _rhyme(last_word)
        if r:
            rhyme.setdefault(r, []).append(last_word)
    return rhyme


# ============ 7. 采样 ============
def _logits_with_style(logits, word_dict, style_words, style_bias):
    """对风格词表加 logits 偏置"""
    for w in style_words:
        idx = word_dict.get(w)
        if idx is not None:
            logits[idx] += style_bias
    return logits


def _sample(model, last_idx, h, c, word_dict, style_bias=STYLE_BIAS,
            temperature=TEMPERATURE, top_k=TOP_K, rep_penalty=REP_PENALTY,
            used_words=None):
    """采样下一个词: 温度 + top-k + 重复惩罚 + 风格偏置"""
    model.eval()
    with torch.no_grad():
        emb = model.embedding(last_idx)         # [1,1,256]
        out, (h, c) = model.lstm(emb, (h, c))   # [1,1,512]
        logits = model.fc(out)[0, -1].clone()   # [vocab]

    # 重复惩罚: 已出现的词 logits 打折
    if used_words:
        for w in used_words:
            idx = word_dict.get(w)
            if idx is not None:
                logits[idx] /= rep_penalty

    # 歌词行内禁止采样结构标记 <副歌> (防止标记混入正文)
    tag_idx = word_dict.get(TAG)
    if tag_idx is not None:
        logits[tag_idx] = float('-inf')

    # 风格偏置
    logits = _logits_with_style(logits, word_dict, STYLE_WORDS, style_bias)

    # 温度 + top-k
    logits = logits / temperature
    topk = torch.topk(logits, min(top_k, logits.size(0)))
    mask = torch.full_like(logits, float('-inf'))
    mask[topk.indices] = logits[topk.indices]
    probs = torch.softmax(mask, dim=-1)
    next_idx = torch.multinomial(probs, 1).item()
    return next_idx, h, c


def _sample_rhyme_word(rhyme_dict, target_rhyme, word_dict, used_words=None):
    """从目标韵组中采样一个行尾词 (韵律约束)"""
    candidates = rhyme_dict.get(target_rhyme, [])
    if not candidates:
        return None
    # 简单按出现频次加权采样, 避开已用词
    from collections import Counter
    freq = Counter(c for c in candidates if used_words is None or c not in used_words)
    if not freq:
        return None
    words, counts = zip(*freq.most_common(30))
    total = sum(counts)
    import random
    r = random.random() * total
    acc = 0
    for w, c in zip(words, counts):
        acc += c
        if r <= acc:
            return w
    return words[0]


# ============ 8. 生成: 主歌4行 + 副歌4行 ============
def generate_song(model, word_dict, start_text, rhyme_dict, device,
                  num_lines=8, max_words=9, min_words=3):
    """生成结构: 主歌4行(偶数行押韵) + <副歌>标记 + 副歌4行(偶数行押韵+强风格)"""
    id2w = {i: w for w, i in word_dict.items()}   # 索引 -> 词 (查词 O(1))
    # 起始词分词 → 索引 (过滤词表外的)
    seed_words = [w for w in jieba.lcut(start_text) if w in word_dict]
    if not seed_words:
        seed_words = [start_text[0]] if start_text[0] in word_dict else [id2w[0]]

    def gen_block(seed_words, style_bias):
        """生成一个 4 行块 (奇数行自由, 偶数行强制押韵)"""
        model.eval()
        h = torch.zeros(model.lstm.num_layers, 1, model.lstm.hidden_size).to(device)
        c = torch.zeros(model.lstm.num_layers, 1, model.lstm.hidden_size).to(device)
        lines = []
        used_words = []
        # 种子喂入: 逐个词过一遍 LSTM, 建立状态
        prev_word = seed_words[-1]
        for w in seed_words:
            idx = torch.tensor([[word_dict[w]]], device=device)
            _, h, c = _sample(model, idx, h, c, word_dict, style_bias=style_bias,
                              used_words=used_words)
            used_words.append(w)
        # 主韵: 由第2行行尾词决定, 后续偶数行都押它
        main_rhyme = None
        for line_no in range(1, 5):
            is_even = (line_no % 2 == 0)
            words = []
            # 偶数行预留 1 个词位给押韵词
            max_now = max_words - 1 if is_even else max_words
            for _ in range(max_now):
                # 输入 = 上一个词 (行首用上一行末词), 采样下一个词
                last_idx = torch.tensor([[word_dict[prev_word]]], device=device)
                next_idx, h, c = _sample(model, last_idx, h, c, word_dict,
                                         style_bias=style_bias, used_words=used_words)
                w = id2w[next_idx]
                words.append(w)
                used_words.append(w)
                prev_word = w
                # 到句号/叹号且长度达标 → 断行
                if len(words) >= min_words and w and w[-1] in '。！？':
                    break
            # 偶数行: 末尾词换成押韵词 (韵律约束)
            if is_even:
                target = main_rhyme if main_rhyme else _rhyme(words[-1])
                if main_rhyme is None:
                    main_rhyme = target
                rw = _sample_rhyme_word(rhyme_dict, target, word_dict, used_words)
                if rw:
                    words = words[:-1] + [rw]
                    used_words.append(rw)
                    prev_word = rw
                elif words:
                    words = words[:-1]
                    prev_word = words[-1]
            # 保证行非空
            if not words:
                words = [id2w[0]]
                prev_word = words[0]
            lines.append(''.join(words))
        return lines

    # 主歌 (基础风格偏置)
    verse = gen_block(seed_words, style_bias=STYLE_BIAS)
    # 副歌: 以 <副歌> 标记开头 + 主歌最后一句 (先清掉行内可能残留的标记), 风格偏置更强
    last_line = verse[-1].replace(TAG, '') if verse[-1] else ''
    chorus_seed = [TAG] + list(jieba.lcut(last_line))[:6] if last_line else [TAG]
    chorus = gen_block(chorus_seed, style_bias=STYLE_BIAS + 1.0)

    return '\n'.join(verse + [''] + chorus)


# ============ 9. 主流程 ============
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    # 加载/构建
    word_dict, corpus_idx, vocab_size = build_vocab()
    print(f'词表大小: {vocab_size} | 语料词数: {len(corpus_idx)} | 副歌标记: {TAG}')

    # 数据集: 90/10 训练/验证
    dataset = MyDataset(corpus_idx, NUM_CHARS)
    n = len(dataset)
    n_train = int(n * 0.9)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n - n_train])
    dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    vdl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE)
    print(f'样本: {n} (训练 {n_train} / 验证 {n-n_train})')

    model = MyModel(vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {n_params/1e6:.2f}M')

    # 训练 (已有模型可跳过)
    import os
    if os.path.exists(MODEL_PATH):
        print('检测到已训练模型, 加载权重')
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        train(model, dl, vdl, device)

    # 交互生成
    print('\n' + '=' * 50)
    print('杰伦风歌词生成器 (主歌4行 + 副歌4行, 偶数行押韵)')
    print('输入起始歌词后回车生成; 输入 quit 退出')
    print('=' * 50)
    rhyme_dict = build_rhyme_dict(word_dict)
    print(f'韵脚词典: {len(rhyme_dict)} 个韵组')
    while True:
        start = input('\n请输入起始歌词: ').strip()
        if start.lower() in ('quit', 'exit', 'q'):
            break
        if not start:
            continue
        song = generate_song(model, word_dict, start, rhyme_dict, device)
        print('生成歌词:')
        print(song)

