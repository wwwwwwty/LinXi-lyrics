import os
import sys
import numpy as np
from time import time
from gensim.models import Word2Vec
from config import *

class LoadCorpora(object):
    def __init__(self, s):
        self.path = s

    def __iter__(self):
        f = open(self.path, 'r')
        for line in f:
            yield line.split(' ')


class Word2Vector():
    '''
    利用分词结果构造Word2Vector
    '''
    def __init__(self, model_path = WORD2VEC_MODEL_PATH, model_size = WORD2VEC_SIZE, model_min = WORD2VEC_MIN_COUNT):

        self.model_path = model_path
        self.model_size = model_size
        self.model_min = model_min

        self.generate_model()

    def generate_model(self):

        if not os.path.exists(self.model_path):
            sentences = LoadCorpora(SPLIT_DATA_PATH)
            t_start = time()
            model = Word2Vec(sentences, size = self.model_size, min_count = self.model_min, workers=8)
            model.save(self.model_path)
            print('Word2vec finised, cost {:.3f} second:'.format(time() - t_start))

    def display_model(self):

        if os.path.exists(self.model_path):
            model = Word2Vec.load(self.model_path)
            print('词典中词的个数：', len(model.wv.vocab))

        intrested_words = ('爱情', '值得', '幸福', '眼泪', '拥抱')
        for word in intrested_words:
            result = model.most_similar(word)
            print('与', word, '最相近的词：')
            for w, s in result:
                print('\t', w, s)


class Normal_words():
    '''
    根据词典将单词映射为整数值，并输出batch
    '''
    def __init__(self, batch_size, seq_length):
        self.text = self.load_data(SPLIT_DATA_PATH)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.generate_model()
        self.create_batch()

    def load_data(self, file):
        with open(file, 'r') as f:
            text = f.read()

        data = text.split()
        return data

    def generate_model(self):
        # 构造词典及映射
        self.vocab = set(self.text)
        # map[word:num]
        self.vocab_to_int = {w: idx for idx, w in enumerate(self.vocab)}
        # map[num:word]
        self.int_to_vocab = {idx: w for idx, w in enumerate(self.vocab)}
        # 转换文本为整数 数字表示的文本集合
        self.int_text = [self.vocab_to_int[w] for w in self.text]


    def display_model(self):
        print('Total words: {}'.format(len(self.text)))
        print('Vocab size: {}'.format(len(self.vocab)))
        print('Text size: {}'.format(len(self.int_text)))
        print('Num_batch: {}'.format(self.num_batch))
        print('Batch_size: {}'.format(self.batch_size))
        print('Sequence_length: {}'.format(self.seq_length))
        print('X_batch: {}'.format(np.array(self.x_batch).shape))
        print('Y_batch: {}'.format(np.array(self.y_batch).shape))
        print('Batch: {}'.format(np.array(self.batch).shape))

    def create_batch(self):
        self.num_batch = int(len(self.int_text) / (self.batch_size * self.seq_length))
        # 按batch数取整
        self.int_text = np.array(self.int_text[:self.num_batch * self.batch_size * self.seq_length])
        # y是x后移一位的结果
        data_x = self.int_text
        data_y = np.zeros_like(data_x)
        data_y[:-1], data_y[-1] = data_x[1:], data_x[0]
        # reshape成(num_batch, batch_size, seq_length)尺寸
        self.x_batch = np.split(data_x.reshape(self.batch_size, -1), self.num_batch, 1)
        self.y_batch = np.split(data_y.reshape(self.batch_size, -1), self.num_batch, 1)
        self.batch = np.stack((self.x_batch, self.y_batch), axis=1)



if __name__ == '__main__':

    # w2v = Word2Vector()
    # w2v.display_model()

    normal = Normal_words(BATCH_SIZE, SEQ_LENGTH)
    normal.display_model()

