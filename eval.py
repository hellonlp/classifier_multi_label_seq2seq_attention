# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:56:43 2020

@author: cm
"""

import pandas as pd
from classifier_multi_label_seq2seq_attention.utils import load_csv, load_excel, save_csv, shuffle_two, save_txt
from classifier_multi_label_seq2seq_attention.predict import get_labels, get_label
from classifier_multi_label_seq2seq_attention.utils import cut_list

if __name__ == '__main__':
    ## 加载数据
    f = 'F:/github/classifier_multi_label_seq2seq_attention/data/test.csv'
    df = load_csv(f, encoding='utf-8').fillna('')
    contents = df['content'].tolist()
    labels = df['label'].tolist()
    labels_ = ['|' if str(l) in ['0', ''] else l for l in labels]

    ## 预测
    contents_bloc = cut_list(contents, 100)
    predicts = []
    for i, l in enumerate(contents_bloc):
        predict = get_labels(l)
        predict_ = []
        for p in predict:
            predict_.append('/'.join(p))
        predicts.extend(predict_)
        if i % 10 == 0:
            print(i)
    print(len(contents), len(predicts))

    ## 计算准召率 
    TP, FN, FP, TN = 0, 0, 0, 0
    k = 0
    for i, l in enumerate(predicts):
        k = k + 1
        label = labels_[i]
        if label != '|':
            if set(label.split('/')) & set(l.split('/')) != set():
                if set(label.split('/')) == set(l.split('/')):
                    TP = TP + 1
                else:
                    TP = TP + 0.5
                    FN = FN + 0.5
            else:
                FN = FN + 1
        elif label == '|':
            if label == l:
                TN = TN + 1
            else:
                FP = FP + 1
        else:
            print('error', i, l)
    print('TP:', TP)
    print('FN:', FN)
    print('FP:', FP)
    print('TN:', TN)
    print('召回率：', TP / (TP + FN))
    print('精确率：', TP / (TP + FP))
    print('F1:', 2 * TP / (2 * TP + FN + FP))
    print('总体准确率：', (TP + TN) / (TP + TN + FN + FP))
    print(TP + TN + FN + FP)
