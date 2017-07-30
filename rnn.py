import tensorflow as tf
from tensorflow.contrib import seq2seq
import numpy as np
import datetime
from config import *


class My_Rnn():
    def __init__(self, rnn_size, rnn_layers, keep_prob, batch_size, seq_length, embed_dim, vocab_size):
        '''
        获取RNN初始化参数

        参数
        ---
        rnn_size: 每层网络的节点数
        rnn_layers: RNN网络层数
        keep_prob: 节点保留率
        batch_size: batch大小
        seq_length: 序列长度
        embed_dim: 输入词向量维度
        vocab_size: 输入词典大小
        '''
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.init_inputs()
        self.build_lstm()
        self.build_network()
        self.optimize()

    def init_inputs(self):
        '''
        设置输入输出tensor
        '''
        self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    def build_lstm(self):
        '''
        建立lstm单元并堆叠
        '''
        def single_lstm():
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
            # lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = self.keep_prob)

        self.cells = tf.contrib.rnn.MultiRNNCell([single_lstm() for _ in range(self.rnn_layers)])
        input_data_shape = tf.shape(self.inputs)
        self.initial_state = self.cells.zero_state(input_data_shape[0], tf.float32)
        self.initial_state = tf.identity(self.initial_state, 'initial_state')
        print('Initial_state:', self.initial_state.shape)

    def build_network(self):
        '''
        建立网络
        '''
        embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_dim], -1, 1))
        # 返回embedding的input_data行（多行）
        self.embed = tf.nn.embedding_lookup(embedding, self.inputs)

        self.outputs, self.final_state = tf.nn.dynamic_rnn(
            self.cells,
            self.embed,
            dtype=tf.float32)
        # self.outputs, self.last_state = tf.contrib.legacy_seq2seq.rnn_decoder(
        #     self.inputs,
        #     self.initial_state,
        #     self.cells,
        #     scope='rnnlm')
        self.final_state = tf.identity(self.final_state, 'final_state')
        self.logits = tf.contrib.layers.fully_connected(self.outputs, self.vocab_size, activation_fn=None)
        self.probs = tf.nn.softmax(self.logits, name='probs')

    def optimize(self):
        '''
        优化配置
        '''
        input_shape = tf.shape(self.inputs)
        # 损失函数
        loss = seq2seq.sequence_loss(
            self.logits,
            self.targets,
            tf.ones([input_shape[0], input_shape[1]]))
        # self.cost = tf.reduce_sum(loss) / (self.batch_size * self.seq_length)
        self.cost = loss
        # 优化函数
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # Gradient Clipping
        gradients = optimizer.compute_gradients(self.cost)
        capped_gradients = [(tf.clip_by_value(grad, GRID_CAP*(-1), GRID_CAP), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)

