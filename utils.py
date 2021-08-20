# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 19:57:44 2020

@author: cm
"""

import time
import pandas as pd
import numpy as np


def time_now_string():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def cut_list(data, size):
    """
    data: a list
    size: the size of cut
    """
    return [data[i * size:min((i + 1) * size, len(data))] for i in range(int(len(data) - 1) // size + 1)]


def select(data, ids):
    return [data[i] for i in ids]


def shuffle_one(a1):
    ran = np.arange(len(a1))
    np.random.shuffle(ran)
    return a1[ran]


def load_txt(file):
    with  open(file, encoding='utf-8', errors='ignore') as fp:
        lines = fp.readlines()
        lines = [l.strip() for l in lines]
        print("Load data from file (%s) finished !" % file)
    return lines


def save_txt(file, lines):
    lines = [l + '\n' for l in lines]
    with  open(file, 'w+', encoding='utf-8') as fp:
        fp.writelines(lines)
    return "Write data to txt finished !"


def load_csv(file, header=0, encoding="utf-8"):
    return pd.read_csv(file,
                       encoding=encoding,
                       header=header,
                       error_bad_lines=False)


def save_csv(dataframe, file, header=True, index=None, encoding="utf-8"):
    return dataframe.to_csv(file,
                            mode='w+',
                            header=header,
                            index=index,
                            encoding=encoding)


def save_excel(dataframe, file, header=True, sheetname='Sheet1'):
    return dataframe.to_excel(file,
                              header=header,
                              sheet_name=sheetname)


def load_excel(file, header=0):
    return pd.read_excel(file,
                         header=header,
                         )


def load_vocabulary(file_vocabulary_label):
    """
    Load vocabulary to dict
    """
    vocabulary = load_txt(file_vocabulary_label)
    dict_id2label, dict_label2id = {}, {}
    for i, l in enumerate(vocabulary):
        dict_id2label[i] = str(l)
        dict_label2id[str(l)] = i
    return dict_id2label, dict_label2id


def shuffle_two(a1, a2):
    ran = np.arange(len(a1))
    np.random.shuffle(ran)
    a1_ = [a1[l] for l in ran]
    a2_ = [a2[l] for l in ran]
    return a1_, a2_


if __name__ == '__main__':
    print('')
