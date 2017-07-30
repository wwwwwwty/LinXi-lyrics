# coding: utf-8

import jieba
import thulac
import re
from config import *

delete_words = ['林夕', '歌曲', '曲名', '歌手', '专辑', '作词', '作曲', '编曲', '编制', '监制',
                '曲:', '编:', '监:', '词:', 'rap:', 'by:', '黄贯中', '陈奕迅', '雷颂德', '阿牛',
                '1', '2', '3', '4', '5', '6', '7', '8', '9', '0','Composer', 'Lyricist',
                'Arranger', 'Producer', '主唱', '演唱', '编辑', '制作', '专辑', '合：', '陈：',
                '汪：', '唱：', '曲：', '词：', '编：', '监：', '歌名：', '罗大佑', '/']

# delete_chars = ['《', '》', ':', '：', '\n', '*', '＊', '#', '.', '', '@', '~', '!', '$', '[', ']', '△', '+', '-', '…']

def generate_pure_lyric(file):

    """
    文本预处理：非歌词替换，异常符号替换，非中文替换
    """
    new = []
    with open(file, 'r') as f:
        text = f.readlines()
        # text = text[:3]
        for line in text:
            add_flag = 1
            line = line.strip('\n')

            # 去掉单行人名
            if (len(line) < 4) & (len(line) > 0):
                add_flag = 0

            # 去掉非歌词行
            for words in delete_words:
                pattern = re.compile(words)
                if pattern.search(line):
                    add_flag = 0
                    break

            if add_flag == 1:
                line = line.replace(' ', ';')
                line = line.replace('\n', '.')
                # 去掉括号中的内容
                pattern = re.compile(r'\(.*\)')
                if pattern.search(line):
                    line = pattern.sub('', line)

                # 去掉:前的内容
                pattern = re.compile(r'.*\:')
                if pattern.search(line):
                    line = pattern.sub('', line)

                # 只保留中文
                pattern = re.compile(r"[^\u4e00-\u9fa5\s]")
                line = pattern.sub('', line)
                line = line.replace('  ', ' ')
                new.append(line)
    # print(new)
    return new

def write_lyric(text, file):

    """
    写入纯歌词文件
    """
    with open(file, 'w') as f:
        for x in text:
            if x.strip() != '':
                f.write(x + ';' + '\n')
            else:
                f.write('\n')

def split_lyric_jieba(text, file):
    """
    jieba分词
    """
    with open(file, 'w') as f:
        for lines in text:
            res = jieba.lcut(lines, cut_all=False)
            if len(res) != 0:
                for str in res:
                    f.write(str + ' ')
                f.write(';' + '\n')
            else:
                f.write('\n')

def split_lyric_thulac(input_file, output_file):
    """
    thulac分词
    """
    thu = thulac.thulac(seg_only = True)
    thu.cut_f(input_file, output_file)

if __name__ == "__main__":

    pure = generate_pure_lyric(ORIGINAL_DATA_PATH)

    # write_lyric(pure, PURE_DATA_PATH)

    # split_lyric_jieba(pure, SPLIT_DATA_PATH)
    split_lyric_thulac(PURE_DATA_PATH, 'data/split_thulac.txt')
