#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2020/10/25 22:28:30
@Author  :   zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   
'''

# here put the import lib
import os
import time
import yaml

import data_input
from config import Config
from model.siamese_network import SiamenseRNN

def train_siamese():
    start = time.time()
    # 读取配置
    # conf = Config()
    cfg_path = "./config.yml"
    cfg = yaml.load(open(cfg_path, encoding='utf-8'), Loader=yaml.FullLoader)
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # 读取数据
    data_train, data_val, data_test = data_input.get_lcqmc()
    # data_train = data_train[:10000]
    print("train size:{},val size:{}, test size:{}".format(
        len(data_train), len(data_val), len(data_test)))
    model = SiamenseRNN(cfg)
    model.fit(data_val, data_val, data_test)
    pass

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    train_siamese()