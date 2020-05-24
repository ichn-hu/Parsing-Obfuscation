# pylint: disable=invalid-name,missing-docstring
import torch
from torch import nn as nn
from net.functional.gumbel import gumbel_softmax
from utils import tensor_map, get_char_tensor, get_word_by_ids, dynamic_char_scatter, is_privacy_term
from net.functional.loss import hloss

import infra.config
from model.parser.model.nn.variational_rnn import VarMaskedFastLSTM

import config
from config.utils import MrDict, AttrDict
from data.fileio import PAD_ID_WORD, ROOT
from data import pipe
cfg = config.cfg


class ObfEncoder(nn.Module):
    def __init__(self, n_word, m_emb, word_ids, emb_weight):
        """
        emb_weight of (n_word, m_emb) is the tensor of the embedding of proper noun or noun of number n_word,
        m_emb is the embedded dimension, in this most simplistic setting, we use a MLP to select the word to
        be changed
        :param n_word:
        :param m_emb:
        :param emb_weight:
        """
        super(ObfEncoder, self).__init__()
        self.n = n_word
        self.m = m_emb
        self.word_ids = word_ids
        self.new_id = {self.word_ids[reid]: reid for reid in range(n_word)}
        self.embedder = nn.Embedding(n_word, m_emb, padding_idx=-1, _weight=emb_weight)
        self.embedder.weight.requires_grad = False  # freeze the embedding, don't have to update the embedding anymore

        self.MLP = nn.Sequential(
            nn.Linear(m_emb, 128),
            nn.ReLU(),
            nn.Linear(128, n_word)
        )
        self.to(cfg.device)

    def forward(self, inp, t):
        """
        inp of (n)
        t is the temperature for gumbel softmax
        :param inp:
        :return:
        """
        # import ipdb
        # ipdb.set_trace()
        inp = tensor_map(inp, lambda x: self.new_id[x])
        x = self.embedder(inp)
        # x of (*, n, m_emb)
        x = self.MLP(x)
        # x of (*, n, n_word)
        pipe.hloss = hloss(x)

        x = gumbel_softmax(x, t)
        # x of (*, n, n_word) is a one-hot vector
        # TODO: remove the assert
        assert round(x.sum().item()) == len(inp)
        y = x.argmax(dim=-1)
        x = torch.matmul(x, self.embedder.weight)
        # x of (*, n, m_emb)
        return x, y


class InpEncoder(nn.Module):
    def __init__(self, cfg=None):
        # @InpEncoderCfg
        super().__init__()
        if cfg is None:
            cfg_p = config.cfg.BiaffineAttnParser
            cfg = AttrDict({
                "use_word": True,
                "use_char": True,
                "use_pos": cfg_p.use_pos,
                "use_dist": False, #cfg_p.use_dist,
                "word_dim": cfg_p.word_dim,
                "char_dim": cfg_p.char_dim,
                "pos_dim": cfg_p.pos_dim,
                "num_filters": cfg_p.num_filters
                }, fixed=True, blob=True)
        inputs = pipe.parser_input

        input_dim = 0
        kernel_size = 3

        if cfg.use_word:
            input_dim += cfg.word_dim
            self.word_embedd = nn.Embedding(inputs.num_words, cfg.word_dim, _weight=inputs.embedd_word)
        else:
            self.word_embedd = None

        if cfg.use_pos:
            input_dim += cfg.pos_dim
            self.pos_embedd = nn.Embedding(inputs.num_pos, cfg.pos_dim, _weight=inputs.embedd_pos)
        else:
            self.pos_embedd = None

        if cfg.use_char:
            input_dim += cfg.num_filters
            self.char_embedd = nn.Embedding(inputs.num_chars, cfg.char_dim, _weight=inputs.embedd_char)
            self.conv1d = nn.Conv1d(cfg.char_dim, cfg.num_filters, kernel_size, padding=kernel_size - 1)
        else:
            self.char_embedd = None
        
    def forward(self, input_word, input_char, input_pos):
        inputs = []

        if self.word_embedd:
            if input_word.dim() == 2:
                word = self.word_embedd(input_word)
            else:
                word = input_word  # relaxation
            # apply dropout on input
            # word = self.drop(word)
            inputs.append(word)

        if self.char_embedd:
            # [batch, length, char_length, char_dim]
            char = self.char_embedd(input_char)
            char_size = char.size()
            # first transform to [batch *length, char_length, char_dim]
            # then transpose to [batch * length, char_dim, char_length]
            char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
            # put into cnn [batch*length, char_filters, char_length]
            # then put into maxpooling [batch * length, char_filters]
            char, _ = self.conv1d(char).max(dim=2)
            # reshape to [batch, length, char_filters]
            char = torch.tanh(char).view(char_size[0], char_size[1], -1)
            # apply dropout on input
            # char = self.drop(char)
            # concatenate word and char [batch, length, word_dim+char_filter]
            inputs.append(char)

        if self.pos_embedd:
            # [batch, length, pos_dim]
            pos = self.pos_embedd(input_pos)
            # apply dropout on input
            # pos = self.drop(pos)
            inputs.append(pos)

        inp = torch.cat(inputs, dim=2)
        return inp  # dict not needed for simple behavior module, don't perplex yourself


class InpWithDistEncoder(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg_p = config.cfg.BiaffineAttnParser
            cfg = AttrDict({
                "use_word": True,
                "use_char": True,
                "use_pos": cfg_p.use_pos,
                "use_dist": cfg_p.use_dist,
                "word_dim": cfg_p.word_dim,
                "char_dim": cfg_p.char_dim,
                "pos_dim": cfg_p.pos_dim,
                "num_filters": cfg_p.num_filters
                }, fixed=True, blob=True)
        inputs = pipe.parser_input

        pos_alphabet = inputs.pos_alphabet
        self.privacy_terms = []

        if cfg.use_dist:
            for (pos, pid) in pos_alphabet.items():
                if is_privacy_term(pos):
                    self.privacy_terms.append(pid)

        input_dim = 0
        kernel_size = 3

        if cfg.use_word:
            input_dim += cfg.word_dim
            self.word_embedd = nn.Embedding(inputs.num_words, cfg.word_dim, _weight=inputs.embedd_word)
        else:
            self.word_embedd = None

        if cfg.use_pos:
            input_dim += cfg.pos_dim
            self.pos_embedd = nn.Embedding(inputs.num_pos, cfg.pos_dim, _weight=inputs.embedd_pos)
        else:
            self.pos_embedd = None

        if cfg.use_char:
            input_dim += cfg.num_filters
            self.char_embedd = nn.Embedding(inputs.num_chars, cfg.char_dim, _weight=inputs.embedd_char)
            self.conv1d = nn.Conv1d(cfg.char_dim, cfg.num_filters, kernel_size, padding=kernel_size - 1)
        else:
            self.char_embedd = None
        
    def forward(self, input_word, input_char, input_pos):
        inputs = []

        if len(self.privacy_terms) != 0:
            pri_mask = torch.zeros_like(input_word).byte()
            for pid in self.privacy_terms:
                pri_mask |= input_pos == pid

            bs, ls = input_word.shape
            dis = torch.zeros_like(input_word).float()
            for i in range(bs):
                last = -1000
                for j in range(ls):
                    if pri_mask[i][j] > 0:
                        last = j
                    dis[i][j] = j - last

                last = 1000
                for j in range(ls):
                    k = ls - j - 1
                    if pri_mask[i][k] > 0:
                        last = k
                    dis[i][k] = min(dis[i][k], last - k)
            inputs.append(dis.reshape(bs, ls, 1))

        if self.word_embedd:
            if input_word.dim() == 2:
                word = self.word_embedd(input_word)
            else:
                word = input_word  # relaxation
            # apply dropout on input
            # word = self.drop(word)
            inputs.append(word)

        if self.char_embedd:
            # [batch, length, char_length, char_dim]
            char = self.char_embedd(input_char)
            char_size = char.size()
            # first transform to [batch *length, char_length, char_dim]
            # then transpose to [batch * length, char_dim, char_length]
            char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
            # put into cnn [batch*length, char_filters, char_length]
            # then put into maxpooling [batch * length, char_filters]
            char, _ = self.conv1d(char).max(dim=2)
            # reshape to [batch, length, char_filters]
            char = torch.tanh(char).view(char_size[0], char_size[1], -1)
            # apply dropout on input
            # char = self.drop(char)
            # concatenate word and char [batch, length, word_dim+char_filter]
            inputs.append(char)

        if self.pos_embedd:
            # [batch, length, pos_dim]
            pos = self.pos_embedd(input_pos)
            # apply dropout on input
            # pos = self.drop(pos)
            inputs.append(pos)

        inp = torch.cat(inputs, dim=2)
        return inp  # dict not needed for simple behavior module, don't perplex yourself

