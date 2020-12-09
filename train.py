#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2020/10/25 22:28:30
@Author  :   zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   训练相似度模型
1. siamese network，分别使用 cosine、曼哈顿距离
2. triplet loss
'''

# here put the import lib
from model.bert_classifier import BertClassifier
import os
import time
from numpy.lib.arraypad import pad
import nni
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

def train_siamese():
    # 读取配置
    # conf = Config()
    cfg_path = "./configs/config.yml"
    cfg = yaml.load(open(cfg_path, encoding='utf-8'), Loader=yaml.FullLoader)
    # 读取数据
    data_train, data_val, data_test = data_input.get_lcqmc()
    # data_train = data_train[:100]
    print("train size:{},val size:{}, test size:{}".format(
        len(data_train), len(data_val), len(data_test)))
    model = SiamenseRNN(cfg)
    model.fit(data_train, data_val, data_test)
    pass

def predict_siamese(file_='./results/'):
    # 加载配置
    cfg_path = "./configs/config.yml"
    cfg = yaml.load(open(cfg_path, encoding='utf-8'), Loader=yaml.FullLoader)
    # 将 seq转为id，
    vocab = Vocabulary(meta_file='./data/vocab.txt', max_len=cfg['max_seq_len'], allow_unk=1, unk='[UNK]', pad='[PAD]')
    test_arr, query_arr = get_test(file_, vocab)
    # 加载模型
    model = SiamenseRNN(cfg)
    model.restore_session(cfg["checkpoint_dir"])
    test_label, test_prob = model.predict(test_arr)
    out_arr = [x + [test_label[i]] + [test_prob[i]] for i, x in enumerate(query_arr)]
    write_file(out_arr, file_ + '.siamese.predict', )
    pass

def train_siamese_bert():
    # 读取配置
    # conf = Config()
    cfg_path = "./configs/config_bert.yml"
    cfg = yaml.load(open(cfg_path, encoding='utf-8'), Loader=yaml.FullLoader)
    # 自动调参的参数，每次会更新一组搜索空间中的参数
    tuner_params= nni.get_next_parameter()
    cfg.update(tuner_params)
    # vocab: 将 seq转为id，
    vocab = Vocabulary(meta_file='./data/vocab.txt', max_len=cfg['max_seq_len'], allow_unk=1, unk='[UNK]', pad='[PAD]')
    # 读取数据
    data_train, data_val, data_test = data_input.get_lcqmc_bert(vocab)
    # data_train = data_train[:100]
    print("train size:{},val size:{}, test size:{}".format(
        len(data_train), len(data_val), len(data_test)))
    model = SiamenseBert(cfg)
    model.fit(data_train, data_val, data_test)
    pass

def predict_siamese_bert(file_="./results/input/test"):
    # 读取配置
    # conf = Config()
    cfg_path = "./configs/config_bert.yml"
    cfg = yaml.load(open(cfg_path, encoding='utf-8'), Loader=yaml.FullLoader)
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # vocab: 将 seq转为id，
    vocab = Vocabulary(meta_file='./data/vocab.txt', max_len=cfg['max_seq_len'], allow_unk=1, unk='[UNK]', pad='[PAD]')
    # 读取数据
    test_arr, query_arr  = data_input.get_test_bert(file_, vocab)
    print("test size:{}".format(len(test_arr)))
    model = SiamenseBert(cfg)
    model.restore_session(cfg["checkpoint_dir"])
    test_label, test_prob = model.predict(test_arr)
    out_arr = [x + [test_label[i]] + [test_prob[i]] for i, x in enumerate(query_arr)]
    write_file(out_arr, file_ + '.siamese.bert.predict', )
    pass

def train_bert():
    # 读取配置
    # conf = Config()
    cfg_path = "./configs/bert_classify.yml"
    cfg = yaml.load(open(cfg_path, encoding='utf-8'), Loader=yaml.FullLoader)
    # 自动调参的参数，每次会更新一组搜索空间中的参数
    tuner_params= nni.get_next_parameter()
    cfg.update(tuner_params)
    # vocab: 将 seq转为id，
    vocab = Vocabulary(meta_file='./data/vocab.txt', max_len=cfg['max_seq_len'], allow_unk=1, unk='[UNK]', pad='[PAD]')
    # 读取数据
    data_train, data_val, data_test = data_input.get_lcqmc_bert(vocab, is_merge=1)
    # data_train = data_train[:100]
    print("train size:{},val size:{}, test size:{}".format(
        len(data_train), len(data_val), len(data_test)))
    model = BertClassifier(cfg)
    model.fit(data_train, data_val, data_test)
    pass

def predict_bert(file_="./results/input/test"):
    # 读取配置
    # conf = Config()
    cfg_path = "./configs/bert_classify.yml"
    cfg = yaml.load(open(cfg_path, encoding='utf-8'), Loader=yaml.FullLoader)
    # vocab: 将 seq转为id，
    vocab = Vocabulary(meta_file='./data/vocab.txt', max_len=cfg['max_seq_len'], allow_unk=1, unk='[UNK]', pad='[PAD]')
    # 读取数据
    test_arr, query_arr  = data_input.get_test_bert(file_, vocab, is_merge=1)
    print("test size:{}".format(len(test_arr)))
    model = BertClassifier(cfg)
    model.restore_session(cfg["checkpoint_dir"])
    test_label, test_prob = model.predict(test_arr)
    out_arr = [x + [test_label[i]] + [test_prob[i]] for i, x in enumerate(query_arr)]
    write_file(out_arr, file_ + '.bert.predict', )
    pass

def siamese_bert_sentence_embedding(file_="./results/input/test.single"):
    # 输入一行是一个query，输出是此query对应的向量
    # 读取配置
    cfg_path = "./configs/config_bert.yml"
    cfg = yaml.load(open(cfg_path, encoding='utf-8'), Loader=yaml.FullLoader)
    cfg['batch_size'] = 64
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # vocab: 将 seq转为id，
    vocab = Vocabulary(meta_file='./data/vocab.txt', max_len=cfg['max_seq_len'], allow_unk=1, unk='[UNK]', pad='[PAD]')
    # 读取数据
    test_arr, query_arr = data_input.get_test_bert_single(file_, vocab)
    print("test size:{}".format(len(test_arr)))
    model = SiamenseBert(cfg)
    model.restore_session(cfg["checkpoint_dir"])
    test_label = model.predict_embedding(test_arr)
    test_label = [",".join([str(y) for y in x]) for x in test_label]
    out_arr = [[x, test_label[i]] for i, x in enumerate(query_arr)]
    print("write to file...")
    write_file(out_arr, file_ + '.siamese.bert.embedding', )
    pass

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="bert", type=str, help="train/predict")
    ap.add_argument("--mode", default="train", type=str, help="train/predict")
    ap.add_argument("--file", default="./results/input/test", type=str, help="train/predict")
    args = ap.parse_args()
    if args.mode == 'train' and args.method == 'rnn':
        train_siamese()
    elif args.mode == 'predict' and args.method == 'rnn':
        predict_siamese(args.file)
    elif args.mode == 'train' and args.method == 'bert_siamese':
        train_siamese_bert()
    elif args.mode == 'predict' and args.method == 'bert_siamese':
        predict_siamese_bert(args.file)
    elif args.mode == 'train' and args.method == 'bert':
        train_bert()
    elif args.mode == 'predict' and args.method == 'bert':
        predict_bert(args.file)
    elif args.mode == 'predict' and args.method == 'bert_siamese_embedding':
        # 此处输出句子的 embedding，如果想要使用向量召回
        # 建议训练模型的时候，损失函数使用功能和faiss一致的距离度量，例如faiss中使用是l2，那么损失函数用l2
        # faiss距离用cos，损失函数用cosin，或者损失中有一项是cosin相似度损失
        siamese_bert_sentence_embedding(args.file)
