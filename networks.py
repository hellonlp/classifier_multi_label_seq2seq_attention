# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:03:12 2020

@author: cm
"""

import os
import tensorflow as tf
from classifier_multi_label_seq2seq_attention import modeling
from classifier_multi_label_seq2seq_attention import optimization
from classifier_multi_label_seq2seq_attention.modules import encoder, decoder
from classifier_multi_label_seq2seq_attention.hyperparameters import Hyperparamters as hp
from classifier_multi_label_seq2seq_attention.utils import time_now_string
from classifier_multi_label_seq2seq_attention.classifier_utils import ClassifyProcessor

num_labels = hp.num_labels
processor = ClassifyProcessor()
bert_config_file = os.path.join(hp.bert_path, 'albert_config.json')
bert_config = modeling.AlbertConfig.from_json_file(bert_config_file)


class NetworkAlbertSeq2Seq(object):
   ...


if __name__ == '__main__':
    #  Load model
    albert = NetworkAlbertSeq2Seq(is_training=True)
