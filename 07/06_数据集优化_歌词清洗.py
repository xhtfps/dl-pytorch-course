# -*- coding: utf-8 -*-
"""
案例:
    数据集优化: 歌词清洗
    把原始 jaychou_lyrics.txt 清洗为 jaychou_lyrics_clean.txt

    优化内容:
        1. 繁转简   : opencc t2s, 统一简体 (原数据有 249 处繁体)
        2. 去噪音   : 删除英文/日文/数字/特殊符号 (原数据有 481 个英文字符)
        3. 统一格式 : 删除空格/全角空格, 去掉行首尾标点 (让行尾字=汉字, 押韵分析更准)
        4. 行过滤   : 非中文为主的行(<50%中文)直接丢弃 (共丢54行)

    效果: 词表 2583 -> 2354, 生成不再出现繁体/英文乱入

    运行: D:\conda\envs\robot-pose\python.exe 06_数据集优化_歌词清洗.py
"""
import opencc
import re

SRC = './07/data/jaychou_lyrics.txt'
DST = './07/data/jaychou_lyrics_clean.txt'

cc = opencc.OpenCC('t2s')   # 繁转简
text = open(SRC, encoding='utf-8').read()
print(f'原数据: {len(text)} 字符')

# 1. 繁转简
text = cc.convert(text)
print(f'繁转简后: {len(text)} 字符')

# 2. 逐行清洗
lines = text.split('\n')
clean_lines = []
dropped = 0
for line in lines:
    line = line.strip()
    if not line:
        continue
    # 2.1 跳过非中文为主的行 (英文/日文/纯符号行)
    zh_count = len(re.findall(r'[\u4e00-\u9fff]', line))
    if zh_count / max(len(line), 1) < 0.5:
        dropped += 1
        continue
    # 2.2 删除英文、数字
    line = re.sub(r'[a-zA-Z0-9]+', '', line)
    # 2.3 只保留: 汉字 + 常见中文标点 (去掉特殊符号/全角空格)
    line = re.sub(r'[^\u4e00-\u9fff，。！？、；：·…—～「」『』（）《》]', '', line)
    # 2.4 去掉行首尾标点 (行尾字应为汉字, 便于押韵分析)
    line = line.strip('，。！？、；：·…—～「」『』（）《》')
    # 2.5 连续重复标点压缩
    line = re.sub(r'([，。！？、；：])\1+', r'\1', line)
    if line:
        clean_lines.append(line)

clean = '\n'.join(clean_lines) + '\n'
open(DST, 'w', encoding='utf-8').write(clean)

print(f'清洗后: {len(clean)} 字符 | 有效行: {len(clean_lines)} | 丢弃行: {dropped}')

# 3. 词表对比与残留检查
chars = sorted(set(clean))
print(f'原词表 2583 -> 清洗后 {len(chars)}')
bad = [c for c in set(clean)
       if not ('\u4e00' <= c <= '\u9fff')
       and not c.isspace()
       and c not in '，。！？、；：·…—～「」『』（）《》']
print(f'残留异常字符: {bad if bad else "无"}')
