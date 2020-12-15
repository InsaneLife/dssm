#!/usr/bin/env python
# encoding=utf-8
'''
@Time    :   2020/10/17 11:38:00
@Author  :   zhiyang.zzy
@Contact :   zhiyangchou@gmail.com
@Desc    :   siamense network, 使用曼哈顿距离、cos相似度进行实验。
1. 使用预训练词向量。2. 使用lcqmc数据集实验。3. 添加预测。
todo: add triplet loss
'''

# here put the import lib
from os import name
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

def siamese_loss(out1,out2,y,Q=5):
    # 使用欧式距离，概率使用e^{-x}
    Q = tf.constant(Q, name="Q",dtype=tf.float32)
    E_w = tf.sqrt(tf.reduce_sum(tf.square(out1-out2),1))   
    pos = tf.multiply(tf.multiply(y,2/Q),tf.square(E_w))
    neg = tf.multiply(tf.multiply(1-y,2*Q),tf.exp(-2.77/Q*E_w))                
    loss = pos + neg
    loss = tf.reduce_mean(loss)
    prob = tf.exp(-E_w)
    return loss, prob

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


class SiamenseRNN(BaseModel):
    def __init__(self, cfg, is_training=1):
        # config来自于yml, 或者config.py 文件。
        self.cfg = cfg
        # if not is_training: dropout=0
        self.is_training = is_training
        if not is_training:
            self.cfg['dropout'] = 0
        self.build()
        pass
    pass

    def share_encoder(self, query_batch, query_seq_length, keep_prob_place):
        with tf.variable_scope('word_embeddings_layer', reuse=tf.AUTO_REUSE):
            # 这里可以加载预训练词向量
            _word_embedding = tf.get_variable(name="word_embedding_arr", dtype=tf.float32,
                                              shape=[self.cfg['nwords'], self.cfg['word_dim']])
            query_embed = tf.nn.embedding_lookup(
                _word_embedding, query_batch, name='query_batch_embed')
        with tf.variable_scope('RNN', reuse=tf.AUTO_REUSE):
            # Abandon bag of words, use GRU, you can use stacked gru
            cell_fw = tf.contrib.rnn.GRUCell(
                self.cfg['hidden_size_rnn'], reuse=tf.AUTO_REUSE)   # , reuse=tf.AUTO_REUSE
            cell_bw = tf.contrib.rnn.GRUCell(
                self.cfg['hidden_size_rnn'], reuse=tf.AUTO_REUSE)
            # query
            (_, _), (query_output_fw, query_output_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, query_embed,
                                                                                         sequence_length=query_seq_length,
                                                                                         dtype=tf.float32)
            query_rnn_output = tf.concat(
                [query_output_fw, query_output_bw], axis=-1)
            query_rnn_output = tf.nn.dropout(query_rnn_output, keep_prob_place)
            # TODO： 使用mean pooling， 或者self attention 来代替最后一个states
        return query_rnn_output

    def cos_sim(self, query_rnn_output, doc_rnn_output):
        with tf.name_scope('Cosine_Similarity'):
            # Cosine similarity
            # query_norm = sqrt(sum(each x^2))
            query_norm = tf.sqrt(tf.reduce_sum(tf.square(query_rnn_output), 1))
            # doc_norm = sqrt(sum(each x^2))
            doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_rnn_output), 1))

            # 内积
            prod = tf.reduce_sum(tf.multiply(
                query_rnn_output, doc_rnn_output), axis=1)
            # 模相乘
            mul = tf.multiply(query_norm, doc_norm)
            # cos_sim_raw = query * doc / (||query|| * ||doc||)
            # cos_sim_raw = tf.truediv(prod, tf.multiply(query_norm, doc_norm))
            cos_sim_raw = tf.divide(prod, mul)
            predict_prob = tf.sigmoid(cos_sim_raw)
            predict_idx = tf.cast(tf.greater_equal(
                predict_prob, 0.5), tf.int32)
        return predict_prob, predict_idx

    def l1_distance(self, query_rnn_output, doc_rnn_output):
        l1_distance_layer = Lambda(
            lambda tensors: K.abs(tensors[0] - tensors[1]))
        l1_distance = l1_distance_layer([query_rnn_output, doc_rnn_output])
        l1_distance = tf.concat([l1_distance, query_rnn_output, doc_rnn_output], axis=-1)
        predict_prob = Dense(units=1, activation='sigmoid')(l1_distance)
        # bs * 1
        predict_prob = tf.reshape(predict_prob, [-1])
        predict_idx = tf.cast(tf.greater_equal(predict_prob, 0.5), tf.int32)
        return predict_prob, predict_idx

    def forward(self):
        # 共享的encode来编码query
        query_rnn_output = self.share_encoder(
            self.query_batch, self.query_seq_length, self.keep_prob_place)
        self.query_rnn_output = query_rnn_output
        self.q_emb = query_rnn_output
        doc_rnn_output = self.share_encoder(
            self.doc_batch, self.doc_seq_length, self.keep_prob_place)
        # 计算cos相似度：
        # self.predict_prob, self.predict_idx = self.cos_sim(query_rnn_output, doc_rnn_output)
        # 使用原文曼哈顿距离
        self.predict_prob, self.predict_idx = self.l1_distance(
            query_rnn_output, doc_rnn_output)

        with tf.name_scope('Loss'):
            # Train Loss
            # cross_entropy = -tf.reduce_mean(self.sim_labels * tf.log(tf.clip_by_value(self.predict_prob,1e-10,1.0))+(1-self.sim_labels) * tf.log(tf.clip_by_value(1-self.predict_prob,1e-10,1.0)))
            loss = tf.losses.log_loss(self.sim_labels, self.predict_prob)
            self.loss = tf.reduce_mean(loss)
            tf.summary.scalar('loss', self.loss)
        # with tf.name_scope('Accuracy'):
        #     correct_prediction = tf.equal(tf.argmax(prob, 1), 0)
        #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #     tf.summary.scalar('accuracy', accuracy)

    def add_placeholder(self):
        with tf.name_scope('input'):
            # 预测时只用输入query即可，将其embedding为向量。
            self.query_batch = tf.placeholder(
                tf.int32, shape=[None, None], name='query_batch')
            self.doc_batch = tf.placeholder(
                tf.int32, shape=[None, None], name='doc_batch')
            self.query_seq_length = tf.placeholder(
                tf.int32, shape=[None], name='query_sequence_length')
            self.doc_seq_length = tf.placeholder(
                tf.int32, shape=[None], name='doc_seq_length')
            # label
            self.sim_labels = tf.placeholder(
                tf.float32, shape=[None], name="sim_labels")
            self.keep_prob_place = tf.placeholder(tf.float32, name='keep_prob')

    def build(self):
        self.add_placeholder()
        self.forward()
        self.add_train_op(self.cfg['optimizer'],
                          self.cfg['learning_rate'], self.loss)
        self._init_session()
        self._add_summary()
        pass

    def feed_batch(self, t1_ids, t1_len, t2_ids, t2_len, label=None, is_test=0):
        keep_porb = 1 if is_test else self.cfg['keep_porb']
        fd = {
            self.query_batch: t1_ids, self.doc_batch: t2_ids, self.query_seq_length: t1_len,
            self.doc_seq_length: t2_len, self.keep_prob_place: keep_porb}
        if label:
            fd[self.sim_labels] = label
        return fd

    def eval(self, test_data):
        pbar = data_input.get_batch(
            test_data, batch_size=self.cfg['batch_size'], is_test=1)
        val_label, val_pred = [], []
        for (t1_ids, t1_len, t2_ids, t2_len, label) in pbar:
            val_label.extend(label)
            fd = self.feed_batch(t1_ids, t1_len, t2_ids, t2_len, is_test=1)
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

    def run_epoch(self, epoch, data_train, data_val):
        steps = int(math.ceil(float(len(data_train)) / self.cfg['batch_size']))
        progbar = tf.keras.utils.Progbar(steps)
        # 每个 epoch 分batch训练
        batch_iter = data_input.get_batch(
            data_train, batch_size=self.cfg['batch_size'])
        for i, (t1_ids, t1_len, t2_ids, t2_len, label) in enumerate(batch_iter):
            fd = self.feed_batch(t1_ids, t1_len, t2_ids, t2_len, label)
            # a = sess.run([query_norm, doc_norm, prod, cos_sim_raw], feed_dict=fd)
            _, cur_loss = self.sess.run(
                [self.train_op, self.loss], feed_dict=fd)
            progbar.update(i + 1, [("loss", cur_loss)])
        # 训练完一个epoch之后，使用验证集评估，然后预测， 然后评估准确率
        dev_acc = self.eval(data_val)
        print("dev set acc:", dev_acc)
        return dev_acc


class SiamenseBert(SiamenseRNN):
    def __init__(self, cfg, is_training=1):
        super(SiamenseBert, self).__init__(cfg, is_training)
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

        self.d_ids = tf.placeholder(
            tf.int32, shape=[None, None], name='doc_batch')
        self.d_mask_ids = tf.placeholder(
            tf.int32, shape=[None, None], name='d_mask_ids')
        self.d_seg_ids = tf.placeholder(
            tf.int32, shape=[None, None], name='d_seg_ids')
        self.d_seq_length = tf.placeholder(
            tf.int32, shape=[None], name='doc_seq_length')
        self.is_train_place = tf.placeholder(
            dtype=tf.bool, name='is_train_place')
        # label
        self.sim_labels = tf.placeholder(
            tf.float32, shape=[None], name="sim_labels")
        self.keep_prob_place = tf.placeholder(tf.float32, name='keep_prob')
    def siamese_loss(self, out1, out2, y, Q=5.0):
        Q = tf.constant(Q, dtype=tf.float32)
        E_w = tf.sqrt(tf.reduce_sum(tf.square(out1-out2),1))   
        pos = tf.multiply(tf.multiply(y,2/Q),tf.square(E_w))
        neg = tf.multiply(tf.multiply(1-y,2*Q),tf.exp(-2.77/Q*E_w))                
        loss = pos + neg                 
        loss = tf.reduce_mean(loss)              
        return loss
    def contrastive_loss(self, model1, model2, y, margin=0.5):
        with tf.name_scope("contrastive-loss"):
            distance = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keepdims=True))
            similarity = y * tf.square(distance)                                           # keep the similar label (1) close to each other
            dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance), 0))        # give penalty to dissimilar label if the distance is bigger than margin
            return tf.reduce_mean(dissimilarity + similarity) / 2
    def forward(self):
        # 获取cls的输出
        q_emb, _, self.q_e = self.share_bert_layer(
            self.is_train_place, self.q_ids, self.q_mask_ids, self.q_seg_ids, use_bert_pre=1)
        d_emb, _, self.d_e = self.share_bert_layer(
            self.is_train_place, self.d_ids, self.d_mask_ids, self.d_seg_ids, use_bert_pre=1)
        self.q_emb = q_emb
        # 计算cos相似度：
        # self.predict_prob, self.predict_idx = self.cos_sim(q_emb, d_emb)
        # 使用原文曼哈顿距离
        self.predict_prob, self.predict_idx = self.l1_distance(q_emb, d_emb)
        with tf.name_scope('Loss'):
            # Train Loss
            # cross_entropy = -tf.reduce_mean(self.sim_labels * tf.log(tf.clip_by_value(self.predict_prob,1e-10,1.0))+(1-self.sim_labels) * tf.log(tf.clip_by_value(1-self.predict_prob,1e-10,1.0)))
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

    def feed_batch(self, out_ids1, m_ids1, seg_ids1, seq_len1, out_ids2, m_ids2, seg_ids2, seq_len2, label=None, is_test=0):
        keep_porb = 1 if is_test else self.cfg['keep_porb']
        is_train = 0 if is_test else 1
        fd = {
            self.q_ids: out_ids1, self.q_mask_ids: m_ids1,
            self.q_seg_ids: seg_ids1,
            self.q_seq_length: seq_len1,
            self.d_ids: out_ids2,
            self.d_mask_ids: m_ids2,
            self.d_seg_ids: seg_ids2,
            self.d_seq_length: seq_len2,
            self.keep_prob_place: keep_porb,
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
        for i, (out_ids1, m_ids1, seg_ids1, seq_len1, out_ids2, m_ids2, seg_ids2, seq_len2, label) in enumerate(batch_iter):
            fd = self.feed_batch(out_ids1, m_ids1, seg_ids1, seq_len1,
                                 out_ids2, m_ids2, seg_ids2, seq_len2, label)
            # a = self.sess.run([self.q_emb1, self.q_e, self.d_e], feed_dict=fd)
            _, cur_loss = self.sess.run(
                [self.train_op, self.loss], feed_dict=fd)
            progbar.update(i + 1, [("loss", cur_loss)])
        # 训练完一个epoch之后，使用验证集评估，然后预测， 然后评估准确率
        dev_acc = self.eval(d_val)
        print("dev set acc:", dev_acc)
        return dev_acc

    def eval(self, test_data):
        pbar = data_input.get_batch(
            test_data, batch_size=self.cfg['batch_size'], is_test=1)
        val_label, val_pred = [], []
        for (out_ids1, m_ids1, seg_ids1, seq_len1, out_ids2, m_ids2, seg_ids2, seq_len2, label) in pbar:
            val_label.extend(label)
            fd = self.feed_batch(out_ids1, m_ids1, seg_ids1, seq_len1, out_ids2, m_ids2, seg_ids2, seq_len2, is_test=1)
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
        
    def predict_embedding(self, test_data):
        pbar = data_input.get_batch(
            test_data, batch_size=self.cfg['batch_size'], is_test=1)
        val_embed = []
        for (out_ids1, m_ids1, seg_ids1, seq_len1) in pbar:
            fd = {
                self.q_ids: out_ids1, self.q_mask_ids: m_ids1,
                self.q_seg_ids: seg_ids1,
                self.q_seq_length: seq_len1,
                self.keep_prob_place: 1,
                self.is_train_place: 0
            }
            pred_embedding = self.sess.run(self.q_emb, feed_dict=fd)
            val_embed.extend(pred_embedding)
        return val_embed


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
