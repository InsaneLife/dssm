#!/usr/bin/env python
# encoding=utf-8
import json
from config import Config

# 配置文件
conf = Config()


def gen_word_set(file_path, out_path='./data/words.txt'):
    word_set = set()
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 4:
                continue
            prefix, query_pred, title, tag, label = spline
            if label == '0':
                continue
            cur_arr = [prefix, title]
            query_pred = json.loads(query_pred)
            for w in prefix:
                word_set.add(w)
            for each in query_pred:
                for w in each:
                    word_set.add(w)
    with open(word_set, 'w', encoding='utf8') as o:
        for w in word_set:
            o.write(w + '\n')
    pass


def convert_word2id(query, vocab_map):
    ids = []
    for w in query:
        if w in vocab_map:
            ids.append(vocab_map[w])
        else:
            ids.append(vocab_map[conf.unk])
    while len(ids) < conf.max_seq_len:
        ids.append(vocab_map[conf.pad])
    return ids[:conf.max_seq_len]


def get_data(file_path):
    """
    gen datasets, convert word into word ids.
    :param file_path:
    :return: [[query, pos sample, 4 neg sample]], shape = [n, 6]
    """
    data_arr = {'query': [], 'doc_pos': [], 'doc_neg': []}
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 4:
                continue
            prefix, query_pred, title, tag, label = spline
            if label == '0':
                continue
            cur_arr = []
            query_pred = json.loads(query_pred)
            # only 4 negative sample
            for each in query_pred:
                if each == title:
                    continue
                cur_arr.append(convert_word2id(each, conf.vocab_map))
            if len(cur_arr) >= 4:
                data_arr['query'].append(convert_word2id(prefix, conf.vocab_map))
                data_arr['doc_pos'].append(convert_word2id(title, conf.vocab_map))
                data_arr['doc_neg'].extend(cur_arr[:4])
            pass
    return data_arr


if __name__ == '__main__':
    # prefix, query_prediction, title, tag, label
    # query_prediction 为json格式。
    file_train = './data/oppo_round1_train_20180929.txt'
    file_vali = './data/oppo_round1_vali_20180929.txt'
    # data_train = get_data(file_train)
    data_train = get_data(file_vali)
    print(len(data_train['query']), len(data_train['doc_pos']), len(data_train['doc_neg']))
    pass
