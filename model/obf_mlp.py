# TODO: rename this file to obfuscator

import torch
from torch import nn as nn
from net.generator import FeedForwardGenerator, Seq2SeqGenerator
from model.parser.model.max_original import BiRecurrentConvBiAffine
from data import pipe
import config
cfg = config.cfg


"""
1. build parser
2. hijacking
"""

class Obfuscator(nn.Module):
    def __init__(self, parser):
        super(Obfuscator, self).__init__()
        self.parser = parser

    def forward(self, *args, **kwargs):
         return self.parser.forward(*args, **kwargs)

    def loss(self, *args, **kwargs):
        return self.parser.loss(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.parser.decode(*args, **kwargs)

    def decode_mst(self, *args, **kwargs):
        return self.parser.decode_mst(*args, **kwargs)


class FeedForwardObfuscator(Obfuscator):
    # TODO: remove the pass of word_alphabet and char_alpha_bet, use pipe
    def __init__(self, word_ids, parser: BiRecurrentConvBiAffine, word_alphabet, char_alphabet):
        super(FeedForwardObfuscator, self).__init__(parser)
        # import ipdb
        # ipdb.set_trace()
        new_emb = FeedForwardGenerator(word_ids, parser.word_embedd, word_alphabet, char_alphabet)
        new_emb.to(cfg.device)
        self.parser.word_embedd = new_emb
        self.to(cfg.device)


class Seq2SeqObfuscator(Obfuscator):
    def __init__(self, parser):
        super(Seq2SeqObfuscator, self).__init__(parser)
        generator = Seq2SeqGenerator(parser.word_embedd)
        generator.to(cfg.device)
        self.parser.word_embedd = generator
        self.to(cfg.device)


