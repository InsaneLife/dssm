#!/usr/bin/env python
# encoding=utf-8
'''
Author: 	zhiyang.zzy 
Date: 		2019-09-25 21:59:54
Contact: 	zhiyangchou@gmail.com
FilePath: /dssm/config.py
Desc: 		
'''


def load_vocab(file_path):
    word_dict = {}
    with open(file_path, encoding='utf8') as f:
        for idx, word in enumerate(f.readlines()):
            word = word.strip()
            word_dict[word] = idx
    return word_dict


class Config(object):
    def __init__(self):
        self.vocab_map = load_vocab(self.vocab_path)
        self.nwords = len(self.vocab_map)

    unk = '[UNK]'
    pad = '[PAD]'
    vocab_path = './data/vocab.txt'
    # file_train = './data/oppo_round1_train_20180929.mini'
    # file_train = './data/oppo_round1_train_20180929.txt'
    # file_vali = './data/oppo_round1_vali_20180929.mini'
    file_vali = './data/oppo_round1_vali_20180929.txt'
    file_train = file_vali
    max_seq_len = 40
    hidden_size_rnn = 100
    use_stack_rnn = False
    learning_rate = 0.001
    # max_steps = 8000
    num_epoch = 300
    summaries_dir = './Summaries/'
    gpu = 0
    word_dim = 100
    batch_size = 64
    keep_porb = 0.5


if __name__ == '__main__':
    conf = Config()
    print(len(conf.vocab_map))
    pass
