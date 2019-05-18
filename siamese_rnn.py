# coding=utf8
"""
python=3.5
TensorFlow=1.2.1
"""

import time
import numpy as np
import tensorflow as tf
import data_input
from config import Config
import random

random.seed(9102)

start = time.time()
# 是否加BN层
norm, epsilon = False, 0.001

# TRIGRAM_D = 21128
TRIGRAM_D = 100
# query batch size
query_BS = 100

# 读取数据
conf = Config()
data_train = data_input.get_data_siamese_rnn(conf.file_train)
data_vali = data_input.get_data_siamese_rnn(conf.file_vali)
# print(len(data_train['query']), query_BS, len(data_train['query']) / query_BS)
train_epoch_steps = int(len(data_train) / query_BS) - 1
vali_epoch_steps = int(len(data_vali) / query_BS) - 1


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


def get_cosine_score(query_arr, doc_arr):
    # query_norm = sqrt(sum(each x^2))
    pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(query_arr), 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(doc_arr), 1))
    pooled_mul_12 = tf.reduce_sum(tf.multiply(query_arr, doc_arr), 1)
    cos_scores = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="cos_scores")
    return cos_scores


with tf.name_scope('input'):
    # 预测时只用输入query即可，将其embedding为向量。
    query_batch = tf.placeholder(tf.int32, shape=[None, None], name='query_batch')
    doc_batch = tf.placeholder(tf.int32, shape=[None, None], name='doc_batch')
    doc_label_batch = tf.placeholder(tf.float32, shape=[None], name='doc_label_batch')
    query_seq_length = tf.placeholder(tf.int32, shape=[None], name='query_sequence_length')
    doc_seq_length = tf.placeholder(tf.int32, shape=[None], name='pos_seq_length')
    on_train = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32, name='drop_out_prob')

with tf.name_scope('word_embeddings_layer'):
    _word_embedding = tf.get_variable(name="word_embedding_arr", dtype=tf.float32,
                                      shape=[conf.nwords, TRIGRAM_D])
    query_embed = tf.nn.embedding_lookup(_word_embedding, query_batch, name='query_batch_embed')
    doc_embed = tf.nn.embedding_lookup(_word_embedding, doc_batch, name='doc_positive_embed')

with tf.name_scope('RNN'):
    # Abandon bag of words, use GRU, you can use stacked gru
    # query_l1 = add_layer(query_batch, TRIGRAM_D, L1_N, activation_function=None)  # tf.nn.relu()
    # doc_positive_l1 = add_layer(doc_positive_batch, TRIGRAM_D, L1_N, activation_function=None)
    # doc_negative_l1 = add_layer(doc_negative_batch, TRIGRAM_D, L1_N, activation_function=None)
    if conf.use_stack_rnn:
        cell_fw = tf.contrib.rnn.GRUCell(conf.hidden_size_rnn, reuse=tf.AUTO_REUSE)
        stacked_gru_fw = tf.contrib.rnn.MultiRNNCell([cell_fw], state_is_tuple=True)
        cell_bw = tf.contrib.rnn.GRUCell(conf.hidden_size_rnn, reuse=tf.AUTO_REUSE)
        stacked_gru_bw = tf.contrib.rnn.MultiRNNCell([cell_fw], state_is_tuple=True)
        (output_fw, output_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(stacked_gru_fw, stacked_gru_bw)
        # not ready, to be continue ...
    else:
        cell_fw = tf.contrib.rnn.GRUCell(conf.hidden_size_rnn, reuse=tf.AUTO_REUSE)
        cell_bw = tf.contrib.rnn.GRUCell(conf.hidden_size_rnn, reuse=tf.AUTO_REUSE)
        # query
        (_, _), (query_output_fw, query_output_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, query_embed,
                                                                                     sequence_length=query_seq_length,
                                                                                     dtype=tf.float32)
        query_rnn_output = tf.concat([query_output_fw, query_output_bw], axis=-1)
        query_rnn_output = tf.nn.dropout(query_rnn_output, keep_prob)
        # doc_pos
        (_, _), (doc_pos_output_fw, doc_pos_output_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                         doc_embed,
                                                                                         sequence_length=doc_seq_length,
                                                                                         dtype=tf.float32)
        doc_rnn_output = tf.concat([doc_pos_output_fw, doc_pos_output_bw], axis=-1)
        doc_rnn_output = tf.nn.dropout(doc_rnn_output, keep_prob)

with tf.name_scope('Cosine_Similarity'):
    # Cosine similarity
    cos_sim = get_cosine_score(query_rnn_output, doc_rnn_output)
    cos_sim_prob = tf.clip_by_value(cos_sim, 1e-8, 1.0)

with tf.name_scope('Loss'):
    # Train Loss
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=doc_label_batch, logits=cos_sim)
    losses = tf.reduce_sum(cross_entropy)
    tf.summary.scalar('loss', losses)
    pass

with tf.name_scope('Training'):
    # Optimizer
    train_step = tf.train.AdamOptimizer(conf.learning_rate).minimize(losses)
    pass
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


def pull_batch(data_map, batch_id):
    query_index, title_index, label_index, dsize = range(4)
    cur_data = data_map[batch_id * query_BS:(batch_id + 1) * query_BS]
    query_in = [x[query_index] for x in cur_data]
    doc_in = [x[title_index] for x in cur_data]
    label = [x[label_index] for x in cur_data]

    return query_in, doc_in, label


def feed_dict(on_training, data_set, batch_id, drop_prob):
    query_in, doc_in, label = pull_batch(data_set, batch_id)
    # query_in, doc_in, label = np.array(query_in), np.array(doc_in), np.array(label)
    query_len = len(query_in)
    query_seq_len = [conf.max_seq_len] * query_len
    pos_seq_len = [conf.max_seq_len] * query_len
    return {query_batch: query_in, doc_batch: doc_in, doc_label_batch: label, on_train: on_training,
            keep_prob: drop_prob, query_seq_length: query_seq_len, doc_seq_length: pos_seq_len}


# config = tf.ConfigProto()  # log_device_placement=True)
# config.gpu_options.allow_growth = True
# if not config.gpu:
# config = tf.ConfigProto(device_count= {'GPU' : 0})

# 创建一个Saver对象，选择性保存变量或者模型。
saver = tf.train.Saver()
# with tf.Session(config=config) as sess:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(conf.summaries_dir + '/train', sess.graph)

    start = time.time()
    for epoch in range(conf.num_epoch):
        random.shuffle(data_train)
        for batch_id in range(train_epoch_steps):
            # print(batch_id)
            sess.run(train_step, feed_dict=feed_dict(True, data_train, batch_id, 0.5))
        end = time.time()
        # train loss
        epoch_loss = 0
        for i in range(train_epoch_steps):
            loss_v = sess.run(losses, feed_dict=feed_dict(False, data_train, i, 1))
            epoch_loss += loss_v

        epoch_loss /= (train_epoch_steps)
        train_loss = sess.run(train_loss_summary, feed_dict={train_average_loss: epoch_loss})
        train_writer.add_summary(train_loss, epoch + 1)
        print("\nEpoch #%d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" %
              (epoch, epoch_loss, end - start))

        # test loss
        start = time.time()
        epoch_loss = 0
        for i in range(vali_epoch_steps):
            loss_v = sess.run(losses, feed_dict=feed_dict(False, data_vali, i, 1))
            epoch_loss += loss_v
        epoch_loss /= (vali_epoch_steps)
        test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
        train_writer.add_summary(test_loss, epoch + 1)
        # test_writer.add_summary(test_loss, step + 1)
        print("Epoch #%d | Test  Loss: %-4.3f | Calc_LossTime: %-3.3fs" %
              (epoch, epoch_loss, start - end))

    # 保存模型
    save_path = saver.save(sess, "model/model_1.ckpt")
    print("Model saved in file: ", save_path)
