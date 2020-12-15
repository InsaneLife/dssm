#!/usr/bin/env python
# encoding=utf-8
'''
Author: 	zhiyang.zzy 
Date: 		2020-10-25 11:07:55
Contact: 	zhiyangchou@gmail.com
FilePath: /dssm/base_model.py
Desc: 		基础模型，包含基本功能
'''
# here put the import lib


# here put the import lib
import numpy as np
import os
import tensorflow as tf
import nni
# from tensorflow.python.ops import rnn_cell_impl as core_rnn_cell
import logging
from collections import defaultdict
from .bert import modeling_v1 as modeling, tokenization, optimization
# logging.basicConfig(level=logging.DEBUG)


class TriplteLoss(object):
    # https://blog.csdn.net/u013082989/article/details/83537370
    @staticmethod
    def _pairwise_distance(embeddings, squared=False):
        '''
        计算两两embedding的距离
        ------------------------------------------
        Args：
            embedding: 特征向量， 大小（batch_size, vector_size）
            squared:   是否距离的平方，即欧式距离

        Returns：
            distances: 两两embeddings的距离矩阵，大小 （batch_size, batch_size）
        '''
        # 矩阵相乘,得到（batch_size, batch_size），因为计算欧式距离|a-b|^2 = a^2 -2ab + b^2,
        # 其中 ab 可以用矩阵乘表示
        dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
        # dot_product对角线部分就是 每个embedding的平方
        square_norm = tf.diag_part(dot_product)
        # |a-b|^2 = a^2 - 2ab + b^2
        # tf.expand_dims(square_norm, axis=1)是（batch_size, 1）大小的矩阵，减去 （batch_size, batch_size）大小的矩阵，相当于每一列操作
        distances = tf.expand_dims(
            square_norm, axis=1) - 2.0 * dot_product + tf.expand_dims(square_norm, axis=0)
        distances = tf.maximum(distances, 0.0)   # 小于0的距离置为0
        if not squared:          # 如果不平方，就开根号，但是注意有0元素，所以0的位置加上 1e*-16
            distances = distances + mask * 1e-16
            distances = tf.sqrt(distances)
            distances = distances * (1.0 - mask)    # 0的部分仍然置为0
        return distances
    @staticmethod
    def _get_triplet_mask(labels):
        '''
        得到一个3D的mask [a, p, n], 对应triplet（a, p, n）是valid的位置是True
        ----------------------------------
        Args:
            labels: 对应训练数据的labels, shape = (batch_size,)
        
        Returns:
            mask: 3D,shape = (batch_size, batch_size, batch_size)
        
        '''

        # 初始化一个二维矩阵，坐标(i, j)不相等置为1，得到indices_not_equal
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        # 因为最后得到一个3D的mask矩阵(i, j, k)，增加一个维度，则 i_not_equal_j 在第三个维度增加一个即，(batch_size, batch_size, 1), 其他同理
        i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
        i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
        j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
        # 想得到i!=j!=k, 三个不等取and即可, 最后可以得到当下标（i, j, k）不相等时才取True
        distinct_indices = tf.logical_and(tf.logical_and(
            i_not_equal_j, i_not_equal_k), j_not_equal_k)

        # 同样根据labels得到对应i=j, i!=k
        label_equal = tf.equal(tf.expand_dims(labels, 0),
                            tf.expand_dims(labels, 1))
        i_equal_j = tf.expand_dims(label_equal, 2)
        i_equal_k = tf.expand_dims(label_equal, 1)
        valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))
        # mask即为满足上面两个约束，所以两个3D取and
        mask = tf.logical_and(distinct_indices, valid_labels)
        return mask
    @staticmethod
    def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
        '''
        triplet loss of a batch
        -------------------------------
        Args:
            labels:     标签数据，shape = （batch_size,）
            embeddings: 提取的特征向量， shape = (batch_size, vector_size)
            margin:     margin大小， scalar
            
        Returns:
            triplet_loss: scalar, 一个batch的损失值
            fraction_postive_triplets : valid的triplets占的比例
        '''
        # 得到每两两embeddings的距离，然后增加一个维度，一维需要得到（batch_size, batch_size, batch_size）大小的3D矩阵
        # 然后再点乘上valid 的 mask即可
        pairwise_dis = _pairwise_distance(embeddings, squared=squared)
        anchor_positive_dist = tf.expand_dims(pairwise_dis, 2)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(
            anchor_positive_dist.shape)
        anchor_negative_dist = tf.expand_dims(pairwise_dis, 1)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(
            anchor_negative_dist.shape)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        mask = _get_triplet_mask(labels)
        mask = tf.to_float(mask)
        triplet_loss = tf.multiply(mask, triplet_loss)
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        # 计算valid的triplet的个数，然后对所有的triplet loss求平均
        valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        fraction_postive_triplets = num_positive_triplets / \
            (num_valid_triplets + 1e-16)

        triplet_loss = tf.reduce_sum(triplet_loss) / \
            (num_positive_triplets + 1e-16)
        return triplet_loss, fraction_postive_triplets


class BaseModel(object):
    def __init__(self, cfg, is_training=1):
        # config来自于yml文件。
        self.cfg = cfg
        # 通过cfg 解析出多少个 word, intent, action, 等
        # if not is_training: dropout=0
        self.is_training = is_training
        if not is_training:
            self.cfg['dropout'] = 0
        self.build()

    def __del__(self):
        # self.sess.close()
        pass

    def _init_session(self):
        # https://zhuanlan.zhihu.com/p/78998468
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())
        # saver = tf.train.Saver(max_to_keep=None)
        self.saver = tf.train.Saver()

    def restore_session(self, dir_model):
        print("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)

    def _add_summary(self):
        self.merged = tf.summary.merge_all()
        if not os.path.exists(self.cfg['summaries_dir']):
            os.makedirs(self.cfg['summaries_dir'])
        self.file_writer = tf.summary.FileWriter(
            self.cfg['summaries_dir'], self.sess.graph)

    def save_session(self):
        if not os.path.exists(self.cfg['checkpoint_dir']):
            os.makedirs(self.cfg['checkpoint_dir'])
        self.saver.save(self.sess, self.cfg['checkpoint_dir'])

    def init_from_pre_dir(self, pre_dir):
        tvars = tf.trainable_variables()
        (assignment, init_variable_names) = modeling.get_assignment_map_from_checkpoint(
            tvars, pre_dir)
        tf.train.init_from_checkpoint(pre_dir, assignment)

    @staticmethod
    def get_params_count():
        params_count = np.sum([np.prod(v.get_shape().as_list())
                               for v in tf.trainable_variables()])
        print("params_count", params_count)
        return params_count

    ####################   基本功能： fit, evaluate, predict  #####################
    def fit(self, train, dev, test=None):
        '''
        @description: 模型训练
        @param {type} 
        @return: 
        '''
        best_score, nepoch_no_imprv = -1, 0
        for epoch in range(self.cfg["num_epoch"]):
            print("Epoch {:} out of {:}".format(
                epoch + 1, self.cfg["num_epoch"]))
            score = self.run_epoch(epoch, train, dev)
            if score > best_score:
                nepoch_no_imprv = 0
                self.save_session()
                best_score = score
                print("- new best score!")
                if test:
                    test_acc = self.eval(test)
                    # self.print_eval_result(test_result)
                    print("test sf acc:{}".format(test_acc))
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.cfg["epoch_no_imprv"]:
                    print(
                        "- early stopping {} epoches without improvement".format(nepoch_no_imprv))
                    nni.report_final_result(best_score)
                    break
            pass
        pass

    def eval(self, test):
        '''
        @description: 测试集评测
        @param {type} 
        @return: 
        '''
        pass

    def predict(self):
        '''
        @description: 无标注数据评测
        @param {type} 
        @return: 
        '''
        pass

    ####################   模型模块  #####################
    def _state_lstm(self, input_emb, input_length, initial_state, hidden_size, variable_scope="StateLSTM"):
        with tf.variable_scope(variable_scope):
            cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
            initial_state = tf.nn.rnn_cell.LSTMStateTuple(
                initial_state, initial_state)
            _output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_emb,
                                                      sequence_length=input_length,
                                                      dtype=tf.float32,
                                                      initial_state_fw=initial_state,
                                                      initial_state_bw=initial_state)
            (output_fw, output_bw), ((_, state_fw), (_, state_bw)) = _output
            output = tf.concat([output_fw, output_bw], axis=-1)
            state = tf.concat([state_fw, state_bw], axis=-1)

        return output, state

    def _concat_lstm(self, input_emb, input_length, extra_emb, hidden_size, variable_scope="ConcatLSTM"):
        """
        input_emb: [batch_size, nstep, hidden_size]
        extra_emb: [batch_size, hidden_size]
        """
        with tf.variable_scope(variable_scope):
            nstep = input_emb.shape[1].value
            # [batch_size, nstep, hidden_size]
            expand_extra_emb = tf.tile(tf.expand_dims(
                extra_emb, axis=1), multiples=[1, nstep, 1])
            input_emb = tf.concat([input_emb, expand_extra_emb], axis=-1)

            cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
            _output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_emb,
                                                      sequence_length=input_length,
                                                      dtype=tf.float32)
            (output_fw, output_bw), ((_, state_fw), (_, state_bw)) = _output
            output = tf.concat([output_fw, output_bw], axis=-1)
            state = tf.concat([state_fw, state_bw], axis=-1)

        return output, state

    def _train_op(self):
        lr_m = self.cfg['optimizer'].lower()
        with tf.variable_scope("train_op"):
            optimizer = self._get_optimizer(lr_m)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            for grad, var in grads_and_vars:
                # grad = tf.Print(grad, [grad], "{} grad: ".format(var.name))
                if grad is not None:
                    tf.summary.histogram(var.op.name + "/gradients", grad)
            if self.cfg['clip'] > 0:
                grads, variables = zip(*grads_and_vars)
                grads, gnorm = tf.clip_by_global_norm(grads, self.cfg['clip'])
                self.train_op = optimizer.apply_gradients(zip(grads, variables),
                                                          global_step=tf.train.get_global_step())
            else:
                self.train_op = optimizer.minimize(
                    self.loss, global_step=tf.train.get_global_step())

    ####################   基础模块  #####################
    def _add_word_embedding_matrix(self,):
        # 如果有预训练矩阵，从其中导入
        self.embedding_file = self.cfg['meta_dir'] + \
            self.cfg.get('embedding_trimmed', None)
        if self.embedding_file and self.cfg["use_pretrained"]:
            embedding_matrix = np.load(self.embedding_file)["embeddings"]
            self.embedding_matrix = tf.Variable(
                embedding_matrix, name='embedding_matrix', dtype=tf.float32)
            pass
        else:
            self.embedding_matrix = tf.get_variable(name="embedding_matrix",
                                                    dtype=tf.float32,
                                                    shape=[self.cfg["word_num"], self.cfg["embedding_dim"]])

    def add_bert_layer(self, use_bert_pre=1):
        self.bert_config = modeling.BertConfig.from_json_file(
            self.cfg["bert_dir"] + self.cfg["bert_config"])
        bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_train_place,
            input_ids=self.query_ids,
            input_mask=self.mask_ids,
            token_type_ids=self.seg_ids,
            use_one_hot_embeddings=False)

        if use_bert_pre:
            tvars = tf.trainable_variables()
            bert_init_dir = self.cfg["bert_dir"] + \
                self.cfg["bert_init_checkpoint"]
            (assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   bert_init_dir)
            tf.train.init_from_checkpoint(bert_init_dir, assignment)

        bert_output_seq_ori = bert_model.get_sequence_output()
        bert_output_shape = tf.shape(bert_output_seq_ori)
        self.bert_output_seq_ori = bert_output_seq_ori
        # bs, seq, 768
        bert_output_seq = tf.strided_slice(
            bert_output_seq_ori, [0, 1, 0], bert_output_shape, [1, 1, 1])
        nsteps = tf.shape(bert_output_seq)[1]
        self.bert_output_seq = tf.reshape(
            bert_output_seq, [-1, nsteps, self.bert_config.hidden_size])
        self.cls_output = bert_model.get_pooled_output()
        self.embedding_table = bert_model.embedding_table
        # mask onehot
        bert_mask_shape = tf.shape(self.mask_ids)
        self.seq_mask_ids = tf.strided_slice(
            self.mask_ids, [0, 1], bert_mask_shape, [1, 1])
        self.word_mask_ids = tf.expand_dims(
            tf.cast(self.seq_mask_ids, tf.float32), -1)

    def share_bert_layer(self, is_train_place, query_ids, mask_ids, seg_ids, use_bert_pre=1):
        self.bert_config = modeling.BertConfig.from_json_file(
            self.cfg["bert_dir"] + self.cfg["bert_config"])
        bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_train_place,
            input_ids=query_ids,
            input_mask=mask_ids,
            token_type_ids=seg_ids,
            use_one_hot_embeddings=False,
            scope="bert")
        if use_bert_pre:
            tvars = tf.trainable_variables()
            bert_init_dir = self.cfg["bert_dir"] + \
                self.cfg["bert_init_checkpoint"]
            (assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   bert_init_dir)
            tf.train.init_from_checkpoint(bert_init_dir, assignment)
        bert_output_seq = bert_model.get_sequence_output()

        # 默认使用cls输出
        pooled = bert_model.get_pooled_output()
        embedding_table = bert_model.embedding_table
        input_mask_ = tf.cast(tf.expand_dims(mask_ids, axis=-1), dtype=tf.float32)
        if self.cfg['sentence_embedding_type'] == "avg":
            # 最后一层avg pooling
            pooled = tf.reduce_sum(bert_output_seq * input_mask_, axis=1) / tf.reduce_sum(input_mask_, axis=1)
        elif self.cfg['sentence_embedding_type'].startswith("avg-last-last-"):
            # 使用最后的第n层 avg pooling
            n_last = int(self.cfg['sentence_embedding_type'][-1])
            sequence = bert_model.all_encoder_layers[-n_last] # [batch_size, seq_length, hidden_size]
            pooled = tf.reduce_sum(sequence * input_mask_, axis=1) / tf.reduce_sum(input_mask_, axis=1)
        elif self.cfg['sentence_embedding_type'].startswith("avg-last-"):
            # 使用最后的n层 avg pooling
            pooled = 0
            n_last = int(self.cfg['sentence_embedding_type'][-1])
            for i in range(n_last):
                sequence = bert_model.all_encoder_layers[-i]
                pooled += tf.reduce_sum(sequence * input_mask_, axis=1) / tf.reduce_sum(input_mask_, axis=1)
            pooled /= float(n_last)
        elif self.cfg['sentence_embedding_type'].startswith("avg-last-concat-"):
            pooled = []
            n_last = int(self.cfg['sentence_embedding_type'][-1])
            for i in range(n_last):
                sequence = bert_model.all_encoder_layers[-i]
                pooled += [tf.reduce_sum(sequence * input_mask_, axis=1) / tf.reduce_sum(input_mask_, axis=1)]
            pooled = tf.concat(pooled, axis=-1)
        return pooled, bert_output_seq, embedding_table

    def _dropout(self, input_emb, ratio=None):
        if not self.is_training:
            return input_emb
        if ratio:
            return tf.layers.dropout(input_emb, ratio)
        else:
            return tf.layers.dropout(input_emb, self.cfg['dropout'])

    def _bigru(self, input_emb, input_length, hidden_size, variable_scope="BiGRU"):
        with tf.variable_scope(variable_scope):
            cell_fw = tf.nn.rnn_cell.GRUCell(hidden_size)
            cell_bw = tf.nn.rnn_cell.GRUCell(hidden_size)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_emb,
                                                              input_length, dtype=tf.float32)
        return tf.concat(outputs, axis=-1), tf.concat(states, axis=-1)

    def _bilstm(self, input_emb, input_length, hidden_size, variable_scope="BilSTM"):
        with tf.variable_scope(variable_scope):
            cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
            _output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_emb,
                                                      input_length, dtype=tf.float32)
            (output_fw, output_bw), ((_, state_fw), (_, state_bw)) = _output

        return tf.concat([output_fw, output_bw], axis=-1), tf.concat([state_fw, state_bw], axis=-1)

    def _iterable_dilated_cnn(self, embeddings):
        """
        :param embeddings: [batch_size, steps, embedding_dim]
        :return:
        """
        embedding_dim = embeddings.get_shape()[-1]
        with tf.variable_scope("id_cnn"):
            cnn_input = tf.expand_dims(embeddings, 1)
            initial_layer_filter_shape = [
                1, self.cfg.filter_width, embedding_dim, self.cfg.filter_num]
            initial_layer_w = tf.get_variable("initial_layer_w", shape=initial_layer_filter_shape,
                                              initializer=tf.contrib.layers.xavier_initializer())
            initial_layer_b = tf.get_variable("initial_layer_b",
                                              initializer=tf.constant(0.01, shape=[self.cfg.filter_num]))
            initial_layer_output = tf.nn.conv2d(cnn_input, initial_layer_w, strides=[1, 1, 1, 1],
                                                padding="SAME", name="initial_layer")
            initial_layer_output = tf.nn.relu(tf.nn.bias_add(
                initial_layer_output, initial_layer_b), name="relu")

            atrous_input = initial_layer_output
            atrous_layers_output = []
            atrous_layers_output_dim = 0
            for block in range(self.cfg.repeat_times):
                for i in range(len(self.cfg.idcnn_layers)):
                    layer_name = "conv_{}".format(i)
                    dilation = self.cfg.idcnn_layers[i]
                    with tf.variable_scope("atrous_conv_{}".format(i), reuse=tf.AUTO_REUSE):
                        filter_shape = [1, self.cfg.filter_width,
                                        self.cfg.filter_num, self.cfg.filter_num]
                        conv_w = tf.get_variable("{}_w".format(layer_name), shape=filter_shape,
                                                 initializer=tf.contrib.layers.xavier_initializer())
                        conv_b = tf.get_variable("{}_b".format(
                            layer_name), shape=[self.cfg.filter_num])
                        conv_output = tf.nn.convolution(atrous_input, conv_w, dilation_rate=[1, dilation],
                                                        padding="SAME", name=layer_name)
                        conv_output = tf.nn.relu(
                            tf.nn.bias_add(conv_output, conv_b))
                        if i == len(self.cfg.idcnn_layers) - 1:
                            atrous_layers_output.append(conv_output)
                            atrous_layers_output_dim += self.cfg.filter_num
                        atrous_input = conv_output
            output = tf.concat(axis=3, values=atrous_layers_output)
            return tf.squeeze(output, [1])

    def add_train_op(self, learning_method, learning_rate, loss, clip=-1):
        learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                   global_step=tf.train.get_or_create_global_step(),
                                                   decay_steps=self.cfg['decay_step'],
                                                   decay_rate=self.cfg['lr_decay'])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        _lr_m = learning_method.lower()
        with tf.variable_scope("train_step"):
            if _lr_m == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif _lr_m == 'lazyadam':
                optimizer = tf.contrib.opt.LazyAdamOptimizer(
                    learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
            elif _lr_m == "adagrad":
                optimizer = tf.train.AdagradOptimizer(learning_rate)
            elif _lr_m == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            elif _lr_m == "rmsprop":
                optimizer = tf.train.RMSPropOptimizer(learning_rate)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))
            with tf.control_dependencies(update_ops):
                if clip > 0:
                    grads, variables = zip(*optimizer.compute_gradients(loss))
                    grads, gnorm = tf.clip_by_global_norm(grads, clip)
                    self.train_op = optimizer.apply_gradients(zip(grads, variables),
                                                              global_step=tf.train.get_global_step())
                else:
                    # 梯度截断
                    # params = tf.trainable_variables()
                    # all_gradients = tf.gradients(loss, all_variables, stop_gradients=stop_tensors)
                    self.train_op = optimizer.minimize(
                        loss, global_step=tf.train.get_global_step())

            return self.train_op

    @staticmethod
    def label_smoothing(inp, ls_epsilon):
        """
        From the paper: "... employed label smoothing of epsilon = 0.1. This hurts perplexity,
        as the model learns to be more unsure, but improves accuracy and BLEU score."
        Args:
            inp (tf.tensor): one-hot encoding vectors, [batch, seq_len, vocab_size]
        """
        vocab_size = inp.shape.as_list()[-1]
        smoothed = (1.0 - ls_epsilon) * inp + (ls_epsilon / vocab_size)
        return smoothed


if __name__ == "__main__":
    model = BaseModel("s")
    pass
