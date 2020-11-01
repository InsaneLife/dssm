#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2020/11/02 00:06:44
@Author  :   Zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   
'''

# here put the import lib
from model.bert_classifier import BertClassifier
import os
import time
from numpy.lib.arraypad import pad
from tensorflow.python.ops.gen_io_ops import write_file
import yaml
import logging
import argparse
logging.basicConfig(level=logging.INFO)
import data_input
from config import Config
from model.siamese_network import SiamenseRNN, SiamenseBert
from data_input import Vocabulary, get_test
from util import write_file
from flask import Flask
app = Flask(__name__)

@app.route('/hello/<q1>/<q2>')
def hello_world(q1, q2):
    # print('Hello World! %s, %s' % (q1, q2))
    test_arr, query_arr  = data_input.get_test_bert_by_arr([[q1, q2]], vocab, is_merge=1)
    # print("test_arr:", test_arr)
    test_label, test_prob = model.predict(test_arr)
    # print("test label", test_label)
    return 'Hello World! {}:{}'.format(q1 + "-" + q2, test_prob[0])

if __name__ == '__main__':
    # 读取配置
    # conf = Config()
    cfg_path = "./configs/bert_classify.yml"
    cfg = yaml.load(open(cfg_path, encoding='utf-8'), Loader=yaml.FullLoader)
    # vocab: 将 seq转为id，
    vocab = Vocabulary(meta_file='./data/vocab.txt', max_len=cfg['max_seq_len'], allow_unk=1, unk='[UNK]', pad='[PAD]')
    # 读取数据
    # test_arr, query_arr  = data_input.get_test_bert(file_, vocab, is_merge=1)
    # print("test size:{}".format(len(test_arr)))
    model = BertClassifier(cfg)
    model.restore_session(cfg["checkpoint_dir"])
    app.run()
    # 输入url测试，例如：http://127.0.0.1:5000/hello/今天天气/明天天气