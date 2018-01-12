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
import multi_view_data_input

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir', 'Summaries', 'Summaries directory')
flags.DEFINE_float('learning_rate', 0.05, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 800000, 'Number of steps to run trainer.')
flags.DEFINE_integer('epoch_steps', 200, "Number of steps in one epoch.")
flags.DEFINE_integer('test_pack_size', 3185, "Number of steps in one epoch.")
flags.DEFINE_bool('gpu', 0, "Enable GPU or not")

start = time.time()
# user feature维度
user_dimension = 17309
# 负样本个数
NEG = 4
# positive batch size
user_BS = 100
# batch size
# BS = user_BS * (NEG + 1)
# 第1层网络的单元数目
L1_N = 400
# 第1层网络的单元数目
L2_N = 120

# 读取数据
# train_size, test_size = 1000000, 100000
# data_sets = multi_view_data_input.load_data()
data_sets = multi_view_data_input.get_data()
user_dimension = data_sets.TRIGRAM_D
# view1维度
view1_dimension = data_sets.app_number
view2_dimension = data_sets.music_number
view3_dimension = data_sets.novel_number
# view1 训练集大小
view1_size = data_sets.app_his.shape[0]
view2_size = data_sets.music_his.shape[0]
view3_size = data_sets.novel_his.shape[0]
total_size = view1_size + view2_size + view3_size
# view1 测试集大小
view1_size_test = data_sets.app_his_test.shape[0]
view2_size_test = data_sets.music_his_test.shape[0]
view3_size_test = data_sets.novel_his_test.shape[0]
# 测试集package size
flags.test_pack_size = int((view1_size_test + view2_size_test + view3_size_test) / user_BS)


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
    user_batch = tf.sparse_placeholder(tf.float32, shape=[None, user_dimension], name='user_batch')
    view1_batch = tf.sparse_placeholder(tf.float32, shape=[None, view1_dimension], name='view1_batch')
    view2_batch = tf.sparse_placeholder(tf.float32, shape=[None, view2_dimension], name='view2_batch')
    view3_batch = tf.sparse_placeholder(tf.float32, shape=[None, view3_dimension], name='view3_batch')
    active_view = tf.placeholder(tf.int32, name='active_view_number')
    on_train = tf.placeholder(tf.bool)

with tf.name_scope('User_View'):
    with tf.name_scope('User_FC1'):
        user_fc1_par_range = np.sqrt(6.0 / (user_dimension + L1_N))
        user_weight1 = tf.Variable(tf.random_uniform([user_dimension, L1_N], -user_fc1_par_range, user_fc1_par_range))
        user_bias1 = tf.Variable(tf.random_uniform([L1_N], -user_fc1_par_range, user_fc1_par_range))
        # variable_summaries(user_weight1, 'L1_weights')
        # variable_summaries(user_bias1, 'L1_biases')

        user_l1 = tf.sparse_tensor_dense_matmul(user_batch, user_weight1) + user_bias1
        user_l1_out = tf.nn.relu(user_l1)

    with tf.name_scope('User_FC2'):
        user_fc2_par_range = np.sqrt(6.0 / (L1_N + L2_N))
        user_weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -user_fc2_par_range, user_fc2_par_range))
        user_bias2 = tf.Variable(tf.random_uniform([L2_N], -user_fc2_par_range, user_fc2_par_range))
        # variable_summaries(user_weight2, 'L2_weights')
        # variable_summaries(user_bias2, 'L2_biases')

        user_l2 = tf.matmul(user_l1_out, user_weight2) + user_bias2
        user_y = tf.nn.relu(user_l2)

with tf.name_scope('Item_view1'):
    with tf.name_scope('Item_FC1'):
        view1_fc1_par_range = np.sqrt(6.0 / (view1_dimension + L1_N))
        view1_weight1 = tf.Variable(tf.random_uniform([view1_dimension, L1_N], -view1_fc1_par_range, view1_fc1_par_range))
        view1_bias1 = tf.Variable(tf.random_uniform([L1_N], -view1_fc1_par_range, view1_fc1_par_range))
        # variable_summaries(item_weight1, 'L1_weights')
        # variable_summaries(item_bias1, 'L1_biases')
        view1_positive_l1 = tf.sparse_tensor_dense_matmul(view1_batch, view1_weight1) + view1_bias1
        view1_positive_l1_out = tf.nn.relu(view1_positive_l1)

    with tf.name_scope('Item_FC2'):
        view1_fc2_par_range = np.sqrt(6.0 / (L1_N + L2_N))
        view1_weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -view1_fc2_par_range, view1_fc2_par_range))
        view1_bias2 = tf.Variable(tf.random_uniform([L2_N], -view1_fc2_par_range, view1_fc2_par_range))
        # variable_summaries(item_weight2, 'L2_weights')
        # variable_summaries(item_bias2, 'L2_biases')

        view1_positive_l2 = tf.matmul(view1_positive_l1_out, view1_weight2) + view1_bias2
        view1_positive_y = tf.nn.relu(view1_positive_l2)

with tf.name_scope('Item_view2'):
    with tf.name_scope('Item_FC1'):
        view2_fc1_par_range = np.sqrt(6.0 / (view2_dimension + L1_N))
        view2_weight1 = tf.Variable(tf.random_uniform([view2_dimension, L1_N], -view2_fc1_par_range, view2_fc1_par_range))
        view2_bias1 = tf.Variable(tf.random_uniform([L1_N], -view2_fc1_par_range, view2_fc1_par_range))
        # variable_summaries(item_weight1, 'L1_weights')
        # variable_summaries(item_bias1, 'L1_biases')
        view2_positive_l1 = tf.sparse_tensor_dense_matmul(view2_batch, view2_weight1) + view2_bias1
        view2_positive_l1_out = tf.nn.relu(view2_positive_l1)

    with tf.name_scope('Item_FC2'):
        view2_fc2_par_range = np.sqrt(6.0 / (L1_N + L2_N))
        view2_weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -view2_fc2_par_range, view2_fc2_par_range))
        view2_bias2 = tf.Variable(tf.random_uniform([L2_N], -view2_fc2_par_range, view2_fc2_par_range))
        # variable_summaries(item_weight2, 'L2_weights')
        # variable_summaries(item_bias2, 'L2_biases')

        view2_positive_l2 = tf.matmul(view2_positive_l1_out, view2_weight2) + view2_bias2
        view2_positive_y = tf.nn.relu(view2_positive_l2)

with tf.name_scope('Item_view3'):
    with tf.name_scope('Item_FC1'):
        view3_fc1_par_range = np.sqrt(6.0 / (view3_dimension + L1_N))
        view3_weight1 = tf.Variable(tf.random_uniform([view3_dimension, L1_N], -view3_fc1_par_range, view3_fc1_par_range))
        view3_bias1 = tf.Variable(tf.random_uniform([L1_N], -view3_fc1_par_range, view3_fc1_par_range))
        # variable_summaries(item_weight1, 'L1_weights')
        # variable_summaries(item_bias1, 'L1_biases')
        view3_positive_l1 = tf.sparse_tensor_dense_matmul(view3_batch, view3_weight1) + view3_bias1
        view3_positive_l1_out = tf.nn.relu(view3_positive_l1)

    with tf.name_scope('Item_FC2'):
        view3_fc2_par_range = np.sqrt(6.0 / (L1_N + L2_N))
        view3_weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -view3_fc2_par_range, view3_fc2_par_range))
        view3_bias2 = tf.Variable(tf.random_uniform([L2_N], -view3_fc2_par_range, view3_fc2_par_range))
        # variable_summaries(item_weight2, 'L2_weights')
        # variable_summaries(item_bias2, 'L2_biases')

        view3_positive_l2 = tf.matmul(view3_positive_l1_out, view3_weight2) + view3_bias2
        view3_positive_y = tf.nn.relu(view3_positive_l2)

with tf.name_scope('Make_Negative_Item'):
    # 合并负样本，tile可选择是否扩展负样本。
    # 判断激活哪一个view。
    if active_view == 1:
        item_y = tf.tile(view1_positive_y, [1, 1])
    elif active_view == 2:
        item_y = tf.tile(view2_positive_y, [1, 1])
    else:
        item_y = tf.tile(view3_positive_y, [1, 1])

    item_y_temp = tf.tile(item_y, [1, 1])
    # batch内随机负采样。
    for i in range(NEG):
        rand = int((random.random() + i) * user_BS / NEG)
        item_y = tf.concat([item_y,
                            tf.slice(item_y_temp, [rand, 0], [user_BS - rand, -1]),
                            tf.slice(item_y_temp, [0, 0], [rand, -1])], 0)

with tf.name_scope('Cosine_Similarity'):
    # Cosine similarity
    # query_norm = sqrt(sum(each x^2))
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(user_y), 1, True)), [NEG + 1, 1])
    # doc_norm = sqrt(sum(each x^2))
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(item_y), 1, True))
    # query * doc
    prod = tf.reduce_sum(tf.multiply(tf.tile(user_y, [NEG + 1, 1]), item_y), 1, True)
    # ||query|| * ||doc||
    norm_prod = tf.multiply(query_norm, doc_norm)
    # cos_sim_raw = query * doc / (||query|| * ||doc||)
    cos_sim_raw = tf.truediv(prod, norm_prod)
    # gamma = 20
    # shape = [user_BS, NEG + 1]，第一列是正样本cos相似度。
    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, user_BS])) * 20

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


def convert_to_sparse_tensor(data_in):
    data_in = data_in.tocoo()
    data_in = tf.SparseTensorValue(
        np.transpose([np.array(data_in.row, dtype=np.int64), np.array(data_in.col, dtype=np.int64)]),
        np.array(data_in.data, dtype=np.float),
        np.array(data_in.shape, dtype=np.int64))
    return data_in


def pull_batch(user_data, item_positive, batch_id):
    batch_id = int(batch_id)
    user_in = user_data[batch_id * user_BS:(batch_id + 1) * user_BS, :]
    item_positive_in = item_positive[batch_id * user_BS:(batch_id + 1) * user_BS, :]
    user_in, item_positive_in = convert_to_sparse_tensor(user_in), convert_to_sparse_tensor(item_positive_in)
    return user_in, item_positive_in


def feed_dict(on_verify, Train, batch_id):
    view1_batch_in = convert_to_sparse_tensor(sparse.csr_matrix(([], ([], [])), shape=(user_BS, view1_dimension)))
    view2_batch_in = convert_to_sparse_tensor(sparse.csr_matrix(([], ([], [])), shape=(user_BS, view2_dimension)))
    view3_batch_in = convert_to_sparse_tensor(sparse.csr_matrix(([], ([], [])), shape=(user_BS, view3_dimension)))
    active_view_in = 1
    if Train:
        if batch_id <= view1_size / user_BS:
            batch_id = batch_id if batch_id < view1_size / user_BS - 1 else batch_id - 1
            active_view_in = 1
            user_batch_in, view1_batch_in = pull_batch(data_sets.app_search, data_sets.app_his, batch_id)
        elif view1_size / user_BS < batch_id <= (view1_size + view2_size) / user_BS:
            batch_id -= view1_size / user_BS
            batch_id = batch_id if batch_id < view2_size / user_BS - 1 else batch_id - 1
            active_view_in = 2
            user_batch_in, view2_batch_in = pull_batch(data_sets.music_search, data_sets.music_his, batch_id)
        else:
            batch_id -= view1_size / user_BS + view2_size / user_BS
            batch_id = batch_id if batch_id < view3_size / user_BS - 1 else batch_id - 1
            active_view_in = 3
            user_batch_in, view3_batch_in = pull_batch(data_sets.novel_search, data_sets.novel_his, batch_id)


    else:
        if batch_id <= view1_size_test / user_BS:
            batch_id = batch_id if batch_id < view1_size_test / user_BS - 1 else batch_id - 1
            active_view_in = 1
            user_batch_in, view1_batch_in = pull_batch(data_sets.app_search_test, data_sets.app_his_test, batch_id)
        elif view1_size_test / user_BS < batch_id <= (view1_size_test + view2_size_test) / user_BS:
            batch_id -= view1_size_test / user_BS
            batch_id = batch_id if batch_id < view2_size_test / user_BS - 1 else batch_id - 1
            active_view_in = 2
            user_batch_in, view2_batch_in = pull_batch(data_sets.music_search_test, data_sets.music_his_test, batch_id)
        else:
            batch_id -= view1_size_test / user_BS + view2_size_test / user_BS
            batch_id = batch_id if batch_id < view3_size_test / user_BS - 1 else batch_id - 1
            active_view_in = 3
            user_batch_in, view3_batch_in = pull_batch(data_sets.novel_search_test, data_sets.novel_his_test, batch_id)

    return {user_batch: user_batch_in,
            view1_batch: view1_batch_in,
            view2_batch: view2_batch_in,
            view3_batch: view3_batch_in,
            active_view: active_view_in,
            on_train: on_verify}


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
        batch_id = int(random.random() * (total_size / user_BS - 1))
        # print(batch_id)
        sess.run(train_step, feed_dict=feed_dict(True, True, batch_id))

        if step % FLAGS.epoch_steps == 0:
            # train loss
            loss_v = sess.run(loss, feed_dict=feed_dict(False, True, batch_id))

            loss_v /= user_BS
            train_loss = sess.run(train_loss_summary, feed_dict={train_average_loss: loss_v})
            train_writer.add_summary(train_loss, step + 1)
            end = time.time()
            print("\nEpoch #%-5d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" %
                  (step / FLAGS.epoch_steps, loss_v, end - start))

            # test loss
            epoch_loss = 0
            for i in range(FLAGS.test_pack_size):
                loss_v = sess.run(loss, feed_dict=feed_dict(False, False, i))
                epoch_loss += loss_v
            epoch_loss /= (FLAGS.test_pack_size * user_BS)
            test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
            train_writer.add_summary(test_loss, step + 1)
            start = time.time()
            print("Epoch #%-5d | Test  Loss: %-4.3f | Calc_LossTime: %-3.3fs" %
                  (step / FLAGS.epoch_steps, epoch_loss, start - end))

    # 保存模型
    save_path = saver.save(sess, "model/model_1.ckpt")
    print("Model saved in file: ", save_path)
