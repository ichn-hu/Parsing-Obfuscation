import torch
from torch import nn
from net.generator import Seq2SeqPGGenerator
from net.parser import BiRecurrentConvBiAffine
from data.fileio import NUM_SYMBOLIC_TAGS

from data import pipe
import config
cfg = config.cfg


class Seq2SeqPGObfuscator(nn.Module):
    """
    We implement a decoupled obfuscator
    """

    def __init__(self):
        super(Seq2SeqPGObfuscator, self).__init__()
        self.generator = Seq2SeqPGGenerator()
        self.parser = BiRecurrentConvBiAffine()

    def forward(self, word, char, pos, mask, lens, arcs=None, rels=None):
        (word_g, char_g, pos_g), (word_c, char_c, pos_c), p_policy = self.generator(word, char, pos, mask)

        if arcs is not None:
            arc_reward, rel_reward = self.parser.reward(word_g, char_g, pos_g, arcs, rels, mask, lens)
            arc_critic, rel_critic = self.parser.reward(word_c, char_c, pos_c, arcs, rels, mask, lens)
            # reward is the negative of the loss

            g = (arc_reward - arc_critic) + (rel_reward - rel_critic)
            g.detach_()

            return (g * p_policy).sum()
        else:
            arcs, rels = self.parser.decode_mst(word_g, char_g, pos_g, mask, lens, None, NUM_SYMBOLIC_TAGS)
            ori_arcs, ori_rels = self.parser.decode_mst(word, char, pos, mask, lens, None, NUM_SYMBOLIC_TAGS)
            return (arcs, rels, word_g), (ori_arcs, ori_rels, word)
