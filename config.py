#!/usr/bin/env python
# encoding=utf-8


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
    file_train = './data/oppo_round1_train_20180929_mini.txt'
    # file_train = './data/oppo_round1_train_20180929.txt'
    file_vali = './data/oppo_round1_vali_20180929_mini.txt'
    # file_vali = './data/oppo_round1_vali_20180929.txt'
    max_seq_len = 10
    hidden_size_rnn = 100
    use_stack_rnn = False
    learning_rate = 0.01
    # max_steps = 8000
    num_epoch = 100
    summaries_dir = './Summaries/'
    gpu = 0


if __name__ == '__main__':
    conf = Config()
    print(len(conf.vocab_map))
    pass
