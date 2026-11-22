# -*- coding: utf-8 -*-
"""结构化数据: 检测副歌重复块, 首次出现处插入 <副歌> 标记 → jaychou_lyrics_structured.txt"""
from collections import Counter

SRC = r'D:\ML_project\07\data\jaychou_lyrics_clean.txt'
DST = r'D:\ML_project\07\data\jaychou_lyrics_structured.txt'
TAG = '<副歌>'

lines = [l for l in open(SRC, encoding='utf-8').read().split('\n') if l.strip()]
print(f'clean 数据: {len(lines)} 行')

# 1. 统计连续 L 行块 (L=2..6) 的出现次数
blocks = Counter()
for L in range(2, 7):
    for i in range(len(lines) - L + 1):
        blocks[(tuple(lines[i:i + L]), L)] += 1

# 2. 贪心标记: 优先长块; 同一内容块只标记第一次出现
out = []
seen_blocks = set()
i = 0
tagged_lines = 0
while i < len(lines):
    best_L = 0
    best_blk = None
    for L in range(6, 1, -1):
        if i + L <= len(lines):
            blk = tuple(lines[i:i + L])
            if blocks[(blk, L)] >= 2:
                best_L, best_blk = L, blk
                break
    if best_L >= 2 and best_blk not in seen_blocks:
        out.append(TAG)          # 首次出现 → 插标记
        tagged_lines += 1
        seen_blocks.add(best_blk)
    out.extend(lines[i:i + best_L] if best_L >= 2 else [lines[i]])
    i += best_L if best_L >= 2 else 1

open(DST, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print(f'结构化数据: {len(out)} 行 | <副歌> 标记数: {tagged_lines}')
print(f'样例前 40 行:')
for l in out[:40]:
    print(' ', l)
