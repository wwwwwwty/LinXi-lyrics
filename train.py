import tensorflow as tf
import numpy as np
import collections
from time import time
from config import *
from get_batch import *
from rnn import *

def train():
    '''
    模型训练
    '''
    train_graph = tf.Graph()
    with train_graph.as_default():
        data = Normal_words(BATCH_SIZE, SEQ_LENGTH)
        vocab_size = len(data.vocab)
        model = My_Rnn(RNN_SIZE,
                       RNN_LAYER,
                       KEEP_PROB,
                       BATCH_SIZE,
                       SEQ_LENGTH,
                       EMBED_DIMENSION,
                       vocab_size)
        batches = data.batch

    with tf.Session(graph=train_graph) as sess:
        start_time = time()
        sess.run(tf.global_variables_initializer())
        for epoch in range(NUM_EPOCH):
            state = sess.run(model.initial_state, {model.inputs: batches[0][0]})
            # state = sess.run(model.cells.zero_state(1, tf.float32))
            # state = model.initial_state.eval()
            for batch_i, (x, y) in enumerate(batches):
                feed = {model.inputs: x,
                        model.targets: y,
                        model.initial_state: state,
                        model.learning_rate: LEARNING_RATE}
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)

                # 每训练一定阶段对结果进行打印
                if (epoch * len(batches) + batch_i) % SHOW_BATCHES == 0:
                    print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                            epoch,
                            batch_i,
                            len(batches),
                            train_loss))

        # 保存模型
        saver = tf.train.Saver()
        saver.save(sess, MODEL_PATH)
        print('Model Trained and Saved, cost {:.3f} second'.format(time()- start_time))


def get_tensors(loaded_graph):
    '''
    获取模型训练结果参数

    参数
    ---
    loaded_graph: 从文件加载的tensroflow graph
    '''
    inputs = loaded_graph.get_tensor_by_name('inputs:0')
    initial_state = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state = loaded_graph.get_tensor_by_name('final_state:0')
    probs = loaded_graph.get_tensor_by_name('probs:0')
    return inputs, initial_state, final_state, probs


def sample(probabilities, current, int_to_vocab):
    '''
    选择单词进行文本生成，用来以一定的概率生成下一个词

    参数
    ---
    probabilities: 下一个词的生成概率
    int_to_vocab: 映射表
    '''
    # 获取词频
    words_freq = get_words_frequency()
    # 对概率排序选出10个
    # sort = np.argsort(probabilities,kind='quicksourt')
    # sort = sort[:10]
    # print('当前词:', current, ',可能选择的10个词:')
    # for n in range(10):
    #     print(n, ':', int_to_vocab[sort[n]])
    # print('\n')
    # word = int_to_vocab[sort[0]]

    # 按照出现概率随机选出50个词
    print('current word:', current)
    result = np.random.choice(len(probabilities), 10, p=probabilities)
    word = int_to_vocab[result[0]]
    # 词频过小不考虑
    while words_freq[word] < 2:
        print('ignore word:', word)
        result = np.random.choice(len(probabilities), 10, p=probabilities)
        word = int_to_vocab[result[0]]
    print('\n')

    # choice_set = set(result)
    # for _, index in enumerate(choice_set):
    #     print('next word maybe:', int_to_vocab[index])
    # print('\n')
    return word

def sample2(probabilities,word,int_to_vocab):
    '''
    对于前一个词是否为空格，采取不同采样方式
    '''
    if word == ' ':
        section_prob = np.cumsum(probabilities)
        sum_prob = np.sum(probabilities)
        index = (int(np.searchsorted(section_prob, np.random.rand(1) * sum_prob)))
    else:
        index = np.argmax(p)

    word = int_to_vocab[index]
    return word

def predict(start_word, gen_length):
    '''
    根据模型生成歌词
    '''
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # 加载模型
        loader = tf.train.import_meta_graph(MODEL_PATH + '.meta')
        loader.restore(sess, MODEL_PATH)

        # 获取训练的结果参数
        input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

        gen_sentences = [start_word]
        prev_state = sess.run(initial_state, {input_text: np.array([[1]])})
        # prev_state = initial_state.eval()
        origin_data = Normal_words(BATCH_SIZE, SEQ_LENGTH)

        # 生成句子
        for n in range(gen_length):
            # 拿出最后seq_length个词作为输入
            dyn_input = [[origin_data.vocab_to_int[word] for word in gen_sentences[-SEQ_LENGTH:]]]
            # 开始阶段长度不足一个seq_length
            dyn_seq_length = len(dyn_input[0])

            # 预测
            feed = {input_text: dyn_input,
                    initial_state: prev_state}
            probabilities, prev_state = sess.run([probs, final_state], feed)

            # 每次从最后一个字符的概率中选择
            pred_word = sample(probabilities[dyn_seq_length - 1], gen_sentences[n], origin_data.int_to_vocab)

            gen_sentences.append(pred_word)

        lyrics = '/'.join(gen_sentences)
        # lyrics = lyrics.replace(';', '\n')
        # lyrics = lyrics.replace('.', ' ')
        # lyrics = lyrics.replace(' ', '')

        print(lyrics)

def get_words_frequency():

    with open(SPLIT_DATA_PATH) as f:
        words_box = []
        for line in f:
            words_box.extend(line.strip().split())

    return collections.Counter(words_box)


if __name__ == '__main__':

    # train()

    predict('心情', 20)

