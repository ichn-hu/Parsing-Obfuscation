"""
This file is the pipe for passing data objects
Notice that there is a potential bug if we are going to use the model in a distributed way
A radical change is needed when we are facing that situation
Best practice is write-once strategy, once a data object is mounted, it will never be changed again
"""
from types import *
# from data.input import Inputs
hloss = None  # for entropy loss for feed forward obfuscator
parser_input = None  # for parser input, once initialized will never change again, safely make it available anywhere

train_data_a = None
num_train_data_a = None
train_data_b = None
num_train_data_b = None

target_word_ids = []  # the original ids of word in emb_weight in ascent order, of size cfg_g.num_out_words
target_emb_weight = None  # the embedding of target words w.r.t the order in target_word_ids

NNP_vocab = None

tbx_writer = None

pretrained_parser = None
