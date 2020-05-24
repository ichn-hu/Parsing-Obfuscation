# pylint: disable=invalid-name
import torch
from torch import nn
from data import pipe
from data.fileio import PAD_ID_WORD
from utils import tensor_map, get_word_by_ids, get_char_tensor, dynamic_char_scatter, get_lut, word_to_char

from net.encoder import ObfEncoder
from model.parser.model.nn.variational_rnn import VarMaskedFastLSTM
from net.functional.gumbel import gumbel_sample, gumbel_softmax, gumbel_sample2
from net.functional.loss import hloss
from net.encoder import InpEncoder

from utils import get_temperature

import config
from config.utils import MrDict, AttrDict
cfg = config.cfg


class FeedForwardGenerator(torch.nn.Module):
    # TODO: remove word_ids as parameter, use pipe.target_word_ids
    def __init__(self, word_ids, word_embedd, word_alphabet, char_alphabet):
        """
        in the generator, we try to hijack the biaffine parser model provided by Max
        thanks for the dynamic feature of python, we want to substitute the embedd_word
        all the ablation study is done here, notice that once done, we need the result
        from the final decoder. It seems impossible to this end. But let's see.
        :param n_word: number of words that need to be substituted
        :param word_ids: a list of size n_word, is the id in word_embedd that should be considered
        to substitute, it seems what we are doing is to find an optimal choice of substitution? we
        can't give more data?
        :param word_embedd: the word_embedd from the biaffine parser to be hijacked
        """
        super(FeedForwardGenerator, self).__init__()
        word_ids = sorted(word_ids)

        self.word_embedd = word_embedd
        self.word_ids = word_ids
        self.word_ids_set = set(word_ids)
        m_emb = word_embedd.weight.size(-1)
        weight = torch.index_select(word_embedd.weight, 0, torch.tensor(word_ids, device=cfg.device))
        # only want weights that we care, by index_select
        self.obfenc = ObfEncoder(len(word_ids), m_emb, word_ids, weight)
        self.word_alphabet = word_alphabet
        self.char_alphabet = char_alphabet
        self.step = 0
        self.last_t = 0

    def forward(self, inp_word, inp_char, activate_obf=False):
        """
        :param inp_word: the input for word_embedd, typically (bs, len)
        :param inp_char: (bs, len, mc) TODO: check the max_char (mc)
        :return: embedded inp_word, and obfuscated inp_char
        """
        x = self.word_embedd(inp_word)  # (bs, len, m_emb)
        if not activate_obf:
            return x, inp_word, inp_char

        idx = tensor_map(inp_word, lambda wid: wid in self.word_ids_set)
        spots = inp_word.reshape(-1)[idx.reshape(-1) > 0]  # where words should be substituted
        if spots.numel() > 0:
            # import ipdb
            # ipdb.set_trace()
            if self.training:
                self.step += 1
                t = cfg_t.get_temperature(self.step, True)
                if self.last_t != t:
                    print("\n--- Temperature {:.4f} -> {:.4f} ---\n".format(self.last_t, t))
                    self.last_t = t
            y, obf_idx = self.obfenc(spots, self.last_t)  # TODO: add temperature annealing
            # y is the substituted word for spots, in its embedded form (len(spots), emb_word)
            # obf_idx is what thoses spots changed to in word_ids, (len(spots))

            obf_idx = tensor_map(obf_idx, lambda oid: self.word_ids[oid])
            # changed to id in word_alphabet
            ori_words = get_word_by_ids(spots, self.word_alphabet)
            obf_words = get_word_by_ids(obf_idx, self.word_alphabet)
            # print(' '.join(["{:>10}->{:10}".format(*t) for t in zip(ori_words, obf_words)]))

            idx = idx.reshape(*idx.shape, 1).type(dtype=torch.uint8)
            x.masked_scatter_(idx, y)
            # because obfuscated word may have more than mc characters
            obf_char, tmc = get_char_tensor(obf_words, self.char_alphabet)
            inp_char = dynamic_char_scatter(inp_char, idx, obf_char, tmc)
            inp_word = inp_word.clone()
            inp_word.masked_scatter_(idx.squeeze(), obf_idx)

            # print("ok")
        else:
            print("spots.numel() == 0")
        return x, inp_word, inp_char


class KeywordsPreservingSTGeneratorObsolete(torch.nn.Module):
    # TODO: remove word_ids as parameter, use pipe.target_word_ids
    def __init__(self, keywords, tgtwords):
        super(KeywordsPreservingSTGenerator, self).__init__()
        cfg_g = AttrDict({
            "word_dim": 100,
            "hidden_size": 128
        })
        self.cfg = cfg_g
        inputs = pipe.parser_input

        self.keywords = sorted(keywords)
        self.tgtwords = sorted(tgtwords)
        self.keywords_set = set(keywords)
        self.nkey = len(keywords)
        self.ntgt = len(tgtwords)
        self.nwrd = inputs.num_words

        # TODO: the word_emb is not fixed!
        self.word_emb = nn.Embedding(self.nwrd, cfg_g.word_dim, _weight=inputs.word_table)
        self.mlp = nn.Sequential(
                nn.Linear(cfg_g.word_dim, cfg_g.hidden_size),
                nn.ReLU(),
                nn.Linear(cfg_g.hidden_size, self.ntgt)
        )
        self.step = 0
        self.last_t = 0
        self.lut = get_lut(inputs.word_alphabet, inputs.char_alphabet).long().to(cfg.device)
        self.word_emb_weight = None
        self.word_emb_tgt = None
        # word table for the parser, will be reloaded when the parser is resumed

    def forward(self, inp_word, inp_char, inp_pos):
        """
        :param inp_word: the input for word_embedd, typically (bs, len)
        :param inp_char: (bs, len, mc) TODO: check the max_char (mc)
        :return: embedded inp_word, and obfuscated inp_char
        """

        bs, ls = inp_word.shape
        # TODO: __ROOT shouldn't be in keywords
        idx = tensor_map(inp_word, lambda wid: wid in self.keywords_set)
        spots = inp_word.reshape(-1)[idx.reshape(-1) > 0]  # where words should be substituted
        h = 0
        if spots.numel() > 0:
            if self.training:
                self.step += 1
                t = cfg_t.get_temperature(self.step, True)
                if self.last_t != t:
                    print("\n--- Temperature {:.4f} -> {:.4f} ---\n".format(self.last_t, t))
                    self.last_t = t
            x = self.word_emb(spots)
            # x of (bs, ls, es)
            x = self.mlp(x)
            # x of (bs, ls, self.ntgt)
            if self.cfg.use_hloss: h = hloss(x).mean() * 0.03
            x = gumbel_softmax(x, self.last_t)
            # x of (bs, ls, self.ntgt), one-hot
            y = x.argmax(dim=-1)
            y = tensor_map(y, lambda wid: self.tgtwords[wid])
            x_emb = torch.matmul(x, self.word_emb_tgt)  # pylint: disable=no-member
            # y of (bs, ls)
            word = inp_word.clone()
            word.masked_scatter_(idx.byte(), y)
            char = word_to_char(word, self.lut)
            word_emb = self.word_emb_weight[word]
            word_emb.masked_scatter_(idx.unsqueeze(-1).byte(), x_emb)
        else:
            word = inp_word
            char = inp_char
            word_emb = self.word_emb_weight[word]
        return word, word_emb, inp_char, inp_pos, h


class KeywordsPreservingGenerator(torch.nn.Module):
    # TODO: remove word_ids as parameter, use pipe.target_word_ids
    # Generator should be agnostic to how it is used, no matter straghter through or REINFORCE
    def __init__(self, keywords, tgtwords):
        super().__init__()
        inputs = pipe.parser_input

        self.keywords = sorted(keywords)
        self.tgtwords = sorted(tgtwords)
        self.keywords_set = set(keywords)
        self.nkey = len(keywords)
        self.ntgt = len(tgtwords)
        self.nwrd = inputs.num_words

        # TODO: the word_emb is not fixed!
        self.word_emb = nn.Embedding(self.nwrd, cfg_g.word_dim, _weight=inputs.word_table)
        self.mlp = nn.Sequential(
                nn.Linear(cfg_g.word_dim, cfg_g.hidden_size),
                nn.ReLU(),
                nn.Linear(cfg_g.hidden_size, self.ntgt)
        )
        self.step = 0
        self.last_t = 1
        self.lut = get_lut(inputs.word_alphabet, inputs.char_alphabet).long().to(cfg.device)
        self.word_emb_weight = None
        self.word_emb_tgt = None
        # word table for the parser, will be reloaded when the parser is resumed

    def forward(self, inp_word, inp_char, inp_pos):
        """
        :param inp_word: the input for word_embedd, typically (bs, len)
        :param inp_char: (bs, len, mc) TODO: check the max_char (mc)
        :return: embedded inp_word, and obfuscated inp_char
        """

        bs, ls = inp_word.shape
        # TODO: __ROOT shouldn't be in keywords
        idx = tensor_map(inp_word, lambda wid: wid in self.keywords_set)
        spots = inp_word.reshape(-1)[idx.reshape(-1) > 0]  # where words should be substituted
        log_p = torch.zeros([bs, ls]).float().to(cfg.device)
        h = torch.zeros([1]).float().to(cfg.device).sum()
        ret = {"obf_mask": idx.float()}

        if spots.numel() > 0:
            if self.training:
                self.step += 1
                t = cfg_t.get_temperature(self.step, True)
                if self.last_t != t:
                    print("\n--- Temperature {:.4f} -> {:.4f} ---\n".format(self.last_t, t))
                    self.last_t = t
            x = self.word_emb(spots)
            # x of (bs, ls, es)
            x = self.mlp(x)
            log_p.masked_scatter_(idx.byte(), x.max(dim=-1)[0])
            # x of (bs, ls, self.ntgt)
            if cfg_g.use_hloss: h = hloss(x).mean() * 0.03
            sft_x = gumbel_softmax(x, self.last_t)
            # x of (bs, ls, self.ntgt), one-hot
            y = sft_x.argmax(dim=-1)
            y = tensor_map(y, lambda wid: self.tgtwords[wid])
            x_emb = torch.matmul(sft_x, self.word_emb_tgt)  # pylint: disable=no-member
            # y of (bs, ls)
            word = inp_word.clone()
            word.masked_scatter_(idx.byte(), y)
            char = word_to_char(word, self.lut)
            word_emb = self.word_emb_weight[word]
            word_emb.masked_scatter_(idx.unsqueeze(-1).byte(), x_emb)
            ret["obf_word"] = word
            ret["obf_word_emb"] = word_emb  # for straight-through gradient estimator
            ret["obf_char"] = char
            ret["obf_pos"] = inp_pos
        else:
            ret["obf_word"] = inp_word
            ret["obf_word_emb"] = self.word_emb_weight[word]
            ret["obf_char"] = inp_char
            ret["obf_pos"] = inp_pos

        ret["log_p"] = log_p  # for REINFORCE, hard to say how on earth should I use this...
        ret["loss_h"] = h

        return ret



