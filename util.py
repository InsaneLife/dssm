#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2020/10/13 20:33:50
@Author  :   zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   
'''

# here put the import lib
import os 
import six
import time

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def read_file(file_:str, splitter:str=None):
    out_arr = []
    with open(file_, encoding="utf-8") as f: 
        out_arr = [x.strip("\n") for x in f.readlines()]
        if splitter:
            out_arr = [x.split(splitter) for x in out_arr]
    return out_arr

def write_file(out_arr:list, file_:str, splitter='\t'):
    with open(file_, 'w', encoding='utf-8') as out:
        for line in out_arr:
            out.write(splitter.join([str(x) for x in line]) + '\n')

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
if __name__ == "__main__":
    pass