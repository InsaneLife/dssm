#!/usr/bin/env python
# encoding=utf-8
'''
@Time    :   2020/10/17 11:38:00
@Author  :   zhiyang.zzy
@Contact :   zhiyangchou@gmail.com
@Desc    :   使用bert做分类。
1. 对于sentence pair，直接将两个句子输入，然后用sep分割输入，然后使用cls的输出作为类别预测的输入。
'''

# here put the import lib
import time
import numpy as np
import tensorflow as tf
import random
import paddlehub as hub
from sklearn.metrics import accuracy_score
import math
from keras.layers import Dense, Subtract, Lambda
import keras.backend as K
from keras.regularizers import l2
import nni

import data_input
from config import Config
from .base_model import BaseModel

random.seed(9102)


def cosine_similarity(a, b):
    c = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), axis=1))
    d = tf.sqrt(tf.reduce_sum(tf.multiply(b, b), axis=1))
    e = tf.reduce_sum(tf.multiply(a, b), axis=1)
    f = tf.multiply(c, d)
    r = tf.divide(e, f)
    return r


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)

class BertClassifier(BaseModel):
    def __init__(self, cfg, is_training=1):
        super(BertClassifier, self).__init__(cfg, is_training)
    pass

    def add_placeholder(self):
        # 预测时只用输入query即可，将其embedding为向量。
        self.q_ids = tf.placeholder(
            tf.int32, shape=[None, None], name='query_batch')
        self.q_mask_ids = tf.placeholder(
            tf.int32, shape=[None, None], name='q_mask_ids')
        self.q_seg_ids = tf.placeholder(
            tf.int32, shape=[None, None], name='q_seg_ids')
        self.q_seq_length = tf.placeholder(
            tf.int32, shape=[None], name='query_sequence_length')
        self.is_train_place = tf.placeholder(
            dtype=tf.bool, name='is_train_place')
        # label
        self.sim_labels = tf.placeholder(
            tf.float32, shape=[None], name="sim_labels")

    def forward(self):
        # 获取cls的输出
        q_emb, _, self.q_e = self.share_bert_layer(
            self.is_train_place, self.q_ids, self.q_mask_ids, self.q_seg_ids, use_bert_pre=1)
        predict_prob = Dense(units=1, activation='sigmoid')(q_emb)
        self.predict_prob = tf.reshape(predict_prob, [-1])
        self.predict_idx = tf.cast(tf.greater_equal(predict_prob, 0.5), tf.int32)
        with tf.name_scope('Loss'):
            # Train Loss
            loss = tf.losses.log_loss(self.sim_labels, self.predict_prob)
            self.loss = tf.reduce_mean(loss)
            tf.summary.scalar('loss', self.loss)

    def build(self):
        self.add_placeholder()
        self.forward()
        self.add_train_op(self.cfg['optimizer'],
                          self.cfg['learning_rate'], self.loss)
        self._init_session()
        self._add_summary()
        pass

    def feed_batch(self, out_ids1, m_ids1, seg_ids1, seq_len1, label=None, is_test=0):
        is_train = 0 if is_test else 1
        fd = {
            self.q_ids: out_ids1, self.q_mask_ids: m_ids1,
            self.q_seg_ids: seg_ids1,
            self.q_seq_length: seq_len1,
            self.is_train_place: is_train}
        if label:
            fd[self.sim_labels] = label
        return fd

    def run_epoch(self, epoch, d_train, d_val):
        steps = int(math.ceil(float(len(d_train)) / self.cfg['batch_size']))
        progbar = tf.keras.utils.Progbar(steps)
        # 每个 epoch 分batch训练
        batch_iter = data_input.get_batch(
            d_train, batch_size=self.cfg['batch_size'])
        for i, (out_ids1, m_ids1, seg_ids1, seq_len1, label) in enumerate(batch_iter):
            fd = self.feed_batch(out_ids1, m_ids1, seg_ids1, seq_len1, label)
            # a = self.sess.run([self.is_train_place, self.q_e], feed_dict=fd)
            _, cur_loss = self.sess.run(
                [self.train_op, self.loss], feed_dict=fd)
            progbar.update(i + 1, [("loss", cur_loss)])
        # 训练完一个epoch之后，使用验证集评估，然后预测， 然后评估准确率
        dev_acc = self.eval(d_val)
        nni.report_intermediate_result(dev_acc)
        print("dev set acc:", dev_acc)
        return dev_acc

    def eval(self, test_data):
        pbar = data_input.get_batch(
            test_data, batch_size=self.cfg['batch_size'], is_test=1)
        val_label, val_pred = [], []
        for (out_ids1, m_ids1, seg_ids1, seq_len1, label) in pbar:
            val_label.extend(label)
            fd = self.feed_batch(out_ids1, m_ids1, seg_ids1, seq_len1, is_test=1)
            pred_labels, pred_prob = self.sess.run(
                [self.predict_idx, self.predict_prob], feed_dict=fd)
            val_pred.extend(pred_labels)
        test_acc = accuracy_score(val_label, val_pred)
        return test_acc

    def predict(self, test_data):
        pbar = data_input.get_batch(
            test_data, batch_size=self.cfg['batch_size'], is_test=1)
        val_pred, val_prob = [], []
        for (t1_ids, t1_len, t2_ids, t2_len) in pbar:
            fd = self.feed_batch(t1_ids, t1_len, t2_ids, t2_len, is_test=1)
            pred_labels, pred_prob = self.sess.run(
                [self.predict_idx, self.predict_prob], feed_dict=fd)
            val_pred.extend(pred_labels)
            val_prob.extend(pred_prob)
        return val_pred, val_prob


if __name__ == "__main__":
    start = time.time()
    # 读取配置
    conf = Config()
    # 读取数据
    dataset = hub.dataset.LCQMC()
    data_train, data_val, data_test = data_input.get_lcqmc()
    # data_train = data_train[:10000]
    print("train size:{},val size:{}, test size:{}".format(
        len(data_train), len(data_val), len(data_test)))
    model = SiamenseRNN(conf)
    model.fit(data_train, data_val, data_test)
    pass
