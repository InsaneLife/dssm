# coding=utf8
"""
python=3.5
TensorFlow=1.2.1
"""

import pandas as pd
from scipy import sparse
import collections
import random
import time
import numpy as np
import tensorflow as tf
import data_input

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir', 'Summaries', 'Summaries directory')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 80000, 'Number of steps to run trainer.')
flags.DEFINE_integer('epoch_steps', 2000, "Number of steps in one epoch.")
flags.DEFINE_integer('pack_size', 2000, "Number of batches in one pickle pack.")
flags.DEFINE_integer('test_pack_size', 200, "Number of batches in one pickle pack.")
flags.DEFINE_bool('gpu', 0, "Enable GPU or not")

start = time.time()
# 是否加BN层
norm, epsilon = False, 0.001

TRIGRAM_D = 6231
# negative sample
NEG = 4
# query batch size
query_BS = 100
# batch size
BS = query_BS * NEG
L1_N = 400
L2_N = 120

# 读取数据
train_size, test_size = 1000000, 100000
data_path = 'D:\data\dssm/hy_test.csv'
data_sets = data_input.get_search_data(data_path, train_size, test_size)


def add_layer(inputs, in_size, out_size, activation_function=None):
    wlimit = np.sqrt(6.0 / (in_size + out_size))
    Weights = tf.Variable(tf.random_uniform([in_size, out_size], -wlimit, wlimit))
    biases = tf.Variable(tf.random_uniform([out_size], -wlimit, wlimit))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def mean_var_with_update(ema, fc_mean, fc_var):
    ema_apply_op = ema.apply([fc_mean, fc_var])
    with tf.control_dependencies([ema_apply_op]):
        return tf.identity(fc_mean), tf.identity(fc_var)


def batch_normalization(x, phase_train, out_size):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        out_size:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[out_size]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[out_size]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


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


with tf.name_scope('input'):
    query_batch = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='QueryBatch')
    doc_positive_batch = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='DocBatch')
    doc_negative_batch = tf.sparse_placeholder(tf.float32, shape=[None, TRIGRAM_D], name='DocBatch')
    on_train = tf.placeholder(tf.bool)

with tf.name_scope('FC1'):
    # 激活函数在BN之后，所以此处为None
    query_l1 = add_layer(query_batch, TRIGRAM_D, L1_N, activation_function=None)
    doc_positive_l1 = add_layer(doc_positive_batch, TRIGRAM_D, L1_N, activation_function=None)
    doc_negative_l1 = add_layer(doc_negative_batch, TRIGRAM_D, L1_N, activation_function=None)

with tf.name_scope('BN1'):
    query_l1 = batch_normalization(query_l1, on_train, L1_N)
    doc_l1 = batch_normalization(tf.concat([doc_positive_l1, doc_negative_l1], axis=0), on_train, L1_N)
    doc_positive_l1 = tf.slice(doc_l1, [0, 0], [query_BS, -1])
    doc_negative_l1 = tf.slice(doc_l1, [query_BS, 0], [-1, -1])
    query_l1_out = tf.nn.relu(query_l1)
    doc_positive_l1_out = tf.nn.relu(doc_positive_l1)
    doc_negative_l1_out = tf.nn.relu(doc_negative_l1)

# with tf.name_scope('Drop_out'):
#     keep_prob = tf.placeholder("float")
#     query_l1_out = tf.nn.dropout(query_l1_out, keep_prob)
#     doc_positive_l1_out = tf.nn.dropout(doc_positive_l1_out, keep_prob)
#     doc_negative_l1_out = tf.nn.dropout(doc_positive_l1_out, keep_prob)


with tf.name_scope('FC2'):
    query_l2 = add_layer(query_batch, L1_N, L2_N, activation_function=None)
    doc_positive_l2 = add_layer(doc_positive_batch, L1_N, L2_N, activation_function=None)
    doc_negative_l2 = add_layer(doc_negative_batch, L1_N, L2_N, activation_function=None)


with tf.name_scope('BN2'):
    query_l2 = batch_normalization(query_l2, on_train, L2_N)
    doc_l2 = batch_normalization(tf.concat([doc_positive_l2, doc_negative_l2], axis=0), on_train, L2_N)
    doc_positive_l2 = tf.slice(doc_l2, [0, 0], [query_BS, -1])
    doc_negative_l2 = tf.slice(doc_l2, [query_BS, 0], [-1, -1])

    query_y = tf.nn.relu(query_l2)
    doc_positive_y = tf.nn.relu(doc_positive_l2)
    doc_negative_y = tf.nn.relu(doc_negative_l2)
    # query_y = tf.contrib.slim.batch_norm(query_l2, activation_fn=tf.nn.relu)

with tf.name_scope('Merge_Negative_Doc'):
    # 合并负样本，tile可选择是否扩展负样本。
    doc_y = tf.tile(doc_positive_y, [1, 1])
    for i in range(NEG):
        for j in range(query_BS):
            # slice(input_, begin, size)切片API
            doc_y = tf.concat([doc_y, tf.slice(doc_negative_y, [j * NEG + i, 0], [1, -1])], 0)

with tf.name_scope('Cosine_Similarity'):
    # Cosine similarity
    # query_norm = sqrt(sum(each x^2))
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [NEG + 1, 1])
    # doc_norm = sqrt(sum(each x^2))
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))

    prod = tf.reduce_sum(tf.multiply(tf.tile(query_y, [NEG + 1, 1]), doc_y), 1, True)
    norm_prod = tf.multiply(query_norm, doc_norm)

    # cos_sim_raw = query * doc / (||query|| * ||doc||)
    cos_sim_raw = tf.truediv(prod, norm_prod)
    # gamma = 20
    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, query_BS])) * 20

with tf.name_scope('Loss'):
    # Train Loss
    # 转化为softmax概率矩阵。
    prob = tf.nn.softmax(cos_sim)
    # 只取第一列，即正样本列概率。
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])
    loss = -tf.reduce_sum(tf.log(hit_prob))
    tf.summary.scalar('loss', loss)

with tf.name_scope('Training'):
    # Optimizer
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

# with tf.name_scope('Accuracy'):
#     correct_prediction = tf.equal(tf.argmax(prob, 1), 0)
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

with tf.name_scope('Test'):
    average_loss = tf.placeholder(tf.float32)
    loss_summary = tf.summary.scalar('average_loss', average_loss)

with tf.name_scope('Train'):
    train_average_loss = tf.placeholder(tf.float32)
    train_loss_summary = tf.summary.scalar('train_average_loss', train_average_loss)


def pull_all(query_in, doc_positive_in, doc_negative_in):
    query_in = query_in.tocoo()
    doc_positive_in = doc_positive_in.tocoo()
    doc_negative_in = doc_negative_in.tocoo()
    query_in = tf.SparseTensorValue(
        np.transpose([np.array(query_in.row, dtype=np.int64), np.array(query_in.col, dtype=np.int64)]),
        np.array(query_in.data, dtype=np.float),
        np.array(query_in.shape, dtype=np.int64))
    doc_positive_in = tf.SparseTensorValue(
        np.transpose([np.array(doc_positive_in.row, dtype=np.int64), np.array(doc_positive_in.col, dtype=np.int64)]),
        np.array(doc_positive_in.data, dtype=np.float),
        np.array(doc_positive_in.shape, dtype=np.int64))
    doc_negative_in = tf.SparseTensorValue(
        np.transpose([np.array(doc_negative_in.row, dtype=np.int64), np.array(doc_negative_in.col, dtype=np.int64)]),
        np.array(doc_negative_in.data, dtype=np.float),
        np.array(doc_negative_in.shape, dtype=np.int64))

    return query_in, doc_positive_in, doc_negative_in


def pull_batch(query_data, doc_positive, doc_negative, batch_id):
    query_in = query_data[batch_id * query_BS:(batch_id + 1) * query_BS, :]
    doc_positive_in = doc_positive[batch_id * query_BS:(batch_id + 1) * query_BS, :]
    doc_negative_in = doc_negative[batch_id * query_BS * NEG:(batch_id + 1) * query_BS * NEG, :]

    query_in, doc_positive_in, doc_negative_in = pull_all(query_in, doc_positive_in, doc_negative_in)
    return query_in, doc_positive_in, doc_negative_in


def feed_dict(on_training, Train, batch_id, drop_out_prob):
    if Train:
        batch_id = int(random.random() * (FLAGS.epoch_steps - 1))
        query_in, doc_positive_in, doc_negative_in = pull_batch(data_sets.query_train_data,
                                                                data_sets.doc_train_positive,
                                                                data_sets.doc_train_negative, batch_id)
    else:
        drop_out_prob = 1.0
        query_in, doc_positive_in, doc_negative_in = pull_batch(data_sets.query_test_data, data_sets.doc_test_positive,
                                                                data_sets.doc_test_negative, batch_id)
    return {query_batch: query_in, doc_positive_batch: doc_positive_in, doc_negative_batch: doc_negative_in,
            on_train: on_training}


# config = tf.ConfigProto()  # log_device_placement=True)
# config.gpu_options.allow_growth = True
# if not FLAGS.gpu:
# config = tf.ConfigProto(device_count= {'GPU' : 0})

# 创建一个Saver对象，选择性保存变量或者模型。
saver = tf.train.Saver()
# with tf.Session(config=config) as sess:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    # test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test', sess.graph)

    start = time.time()
    for step in range(FLAGS.max_steps):
        batch_id = step % FLAGS.epoch_steps

        sess.run(train_step, feed_dict=feed_dict(True, True, batch_id % FLAGS.pack_size, 0.5))

        if batch_id == 1:
            end = time.time()
            # print(sess.run(doc_l1, feed_dict=feed_dict(True, batch_id % FLAGS.pack_size, 0.5)).shape)
            # train loss
            epoch_loss = 0
            for i in range(FLAGS.pack_size):
                loss_v = sess.run(loss, feed_dict=feed_dict(False, True, i, 1))
                epoch_loss += loss_v

            epoch_loss /= (FLAGS.pack_size * query_BS)
            train_loss = sess.run(train_loss_summary, feed_dict={train_average_loss: epoch_loss})
            train_writer.add_summary(train_loss, step + 1)

            print("\nEpoch #%-5d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" %
                  (step / FLAGS.epoch_steps, epoch_loss, end - start))

            # test loss
            start = time.time()
            epoch_loss = 0
            for i in range(FLAGS.test_pack_size):
                loss_v = sess.run(loss, feed_dict=feed_dict(False, False, i, 1))
                epoch_loss += loss_v
            epoch_loss /= (FLAGS.test_pack_size * query_BS)
            test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
            train_writer.add_summary(test_loss, step + 1)
            # test_writer.add_summary(test_loss, step + 1)
            print("Epoch #%-5d | Test  Loss: %-4.3f | Calc_LossTime: %-3.3fs" %
                  (step / FLAGS.epoch_steps, epoch_loss, start - end))

    # 保存模型
    save_path = saver.save(sess, "model/model_1.ckpt")
    print("Model saved in file: ", save_path)
