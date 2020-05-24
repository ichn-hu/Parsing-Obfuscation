import torch
from torch import nn
from data import pipe
from data.fileio import PAD_ID_WORD, UNK_ID
from data.input import get_word_alltag

from utils import tensor_map, get_word_by_ids, get_char_tensor, dynamic_char_scatter, get_lut, word_to_char

from net.encoder import ObfEncoder
from model.parser.model.nn.variational_rnn import VarMaskedFastLSTM
from net.functional.gumbel import gumbel_sample, gumbel_softmax, gumbel_sample2
from net.functional.loss import hloss
from net.encoder import InpEncoder, InpWithDistEncoder

from utils import get_temperature

import config
from config.utils import MrDict, AttrDict
cfg = config.cfg

def is_privacy_term(pos_tag):
    pri_terms = config.cfg.Generator.pri_term # pylint: disable=no-member
    return pos_tag in pri_terms


class TagSpecCtxGenerator(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config.cfg.AlltagCopyCtxGenerator

        self.cfg = cfg

        inputs = pipe.parser_input
        tgtwords = get_word_alltag(inputs.word_alphabet)
        # the tgt words will be determined by obf_term, not pri_term
        self.tgtwords = tgtwords
        self.lut = get_lut(inputs.word_alphabet, inputs.char_alphabet).long().to(config.cfg.device)

        # self.inp_enc = InpEncoder()
        self.inp_enc = InpWithDistEncoder()

        self.ctx_rnn = VarMaskedFastLSTM(cfg.inp_dim, cfg.ctx_fs, cfg.num_layers,
                batch_first=True, bidirectional=cfg.bidirectional, dropout=cfg.p_rnn_drop)

        self.inp_drop = nn.Dropout(cfg.p_inp_drop)
        self.ctx_drop = nn.Dropout(cfg.p_ctx_drop)

        hs = cfg.ctx_fs * 2 if cfg.bidirectional else cfg.ctx_fs

        pos_lut = { pos: "pos_" + str(i) for i, pos in enumerate(tgtwords.keys()) }
        self.pos_lut = pos_lut
        self.dec = nn.ModuleDict({
            pos_lut[pos]: nn.Sequential(
                nn.Linear(hs, len(words)),
                nn.LogSoftmax(dim=-1)
            ) for pos, words in self.tgtwords.items()
        })
        self.aneal = MrDict({"step": 0, "t": 1, "nstep": 500, "r": -1e-5}, fixed=True)
        self.copy_classifier = nn.Sequential(
                nn.Linear(hs, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.LogSoftmax(dim=-1)
            )
        self.psr_emb_lut = {}
        self.psr_weight = None

        # self.atk_emb_lut = {}
        # self.atk_weight = None

    # def update_emb_weight(self, psr_weight, atk_weight):
    def update_emb_weight(self, psr_weight):
        self.psr_emb_lut = {}
        # self.atk_emb_lut = {}
        self.psr_weight = psr_weight.clone()
        # self.atk_weight = atk_weight.clone()
        for pos, words in self.tgtwords.items():
            self.psr_emb_lut[pos] = psr_weight[words]
            # self.atk_emb_lut[pos] = atk_weight[words]

    def forward(self, inp_word, inp_char, inp_pos, inp_mask, **kwargs):
        bs, ls = inp_word.shape

        zeros_like = torch.zeros_like  # pylint: disable=no-member
        cat = torch.cat  # pylint: disable=no-member
        matmul = torch.matmul  # pylint: disable=no-member

        inp = self.inp_enc(inp_word, inp_char, inp_pos)
        inp = self.inp_drop(inp)
        ctx, _ = self.ctx_rnn(inp, mask=inp_mask)
        ctx = self.ctx_drop(ctx)

        t = get_temperature(self.aneal) if self.training else 1.
        
        ctx = ctx.reshape(bs * ls, -1)

        pos_alphabet = pipe.parser_input.pos_alphabet

        obf_word = inp_word.clone()

        ori_psr_emb = self.psr_weight[inp_word] # pylint: disable=unsubscriptable-object
        # ori_atk_emb = self.atk_weight[inp_word] # pylint: disable=unsubscriptable-object

        obf_psr_emb = ori_psr_emb.clone()
        # obf_atk_emb = ori_atk_emb.clone()

        obf_mask = zeros_like(inp_word).byte()
        pri_mask = zeros_like(inp_word).byte() # NN words are those words we care in preserving
        entropy = 0
        for pos, words in self.tgtwords.items():
            pid = pos_alphabet.get_index(pos)
            pos_mask = inp_pos == pid
            obf_mask = obf_mask | pos_mask
            if is_privacy_term(pos):
                pri_mask = pri_mask | pos_mask
            N = int(pos_mask.long().sum().item())
            M = len(words)
            if N == 0:
                continue

            masked_ctx = ctx[pos_mask.reshape(-1)]
            dec = self.dec[self.pos_lut[pos]]
            pspt = dec(masked_ctx)

            entropy += (-pspt * torch.exp(pspt)).mean()
            # print("Entropy: {}".format(entropy.item()))

            spt = gumbel_softmax(pspt, t)
            spt_idx = spt.argmax(dim=-1)
            spt_word = tensor_map(spt_idx, lambda wid: words[wid])
            obf_word.masked_scatter_(pos_mask, spt_word)
            psr_emb = matmul(spt, self.psr_emb_lut[pos])
            # atk_emb = matmul(spt, self.atk_emb_lut[pos])
            obf_psr_emb.masked_scatter_(pos_mask.unsqueeze(-1), psr_emb)
            # obf_atk_emb.masked_scatter_(pos_mask.unsqueeze(-1), atk_emb)

        avoid_word_mask = (inp_word == obf_word) & obf_mask
        num_avoid = avoid_word_mask.long().sum()
        # print("num avoid: ", num_avoid)
        avoid_word_index = inp_word.new_full([num_avoid], UNK_ID)
        avoid_word_emb = self.psr_weight[avoid_word_index]

        obf_psr_emb.masked_scatter_(avoid_word_mask.unsqueeze(-1), avoid_word_emb)
        obf_word.masked_scatter_(avoid_word_mask, avoid_word_index)

        obf_word_psr_emb = obf_psr_emb

        # obf_word_atk_emb = obf_atk_emb

        ret = MrDict(fixed=False, blob=True)
        ret.obf_word = obf_word
        ret.obf_char = word_to_char(obf_word, self.lut)
        ret.obf_pos = inp_pos
        ret.obf_word_psr_emb = obf_word_psr_emb
        # ret.obf_word_atk_emb = obf_word_atk_emb
        ret.obf_mask = obf_mask
        # ret.cpy_mask = cpy_mask
        ret.pri_mask = pri_mask
        # ret.cpy_loss = cpy_loss * self.cfg.cpy_penalty
        # ret.cpy_full_loss = cpy_full_loss
        ret.ent_loss = -entropy * self.cfg.ent_penalty
        ret.t = t
        ret.fix()
        return ret


class TagSpecUnkGenerator(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = AttrDict({
                "unk_rate": 1
            })

        inputs = pipe.parser_input
        tgtwords = get_word_alltag(inputs.word_alphabet)
        self.tgtwords = tgtwords
        self.lut = get_lut(inputs.word_alphabet, inputs.char_alphabet).long().to(config.cfg.device)
        self.cfg = cfg

    def update_emb_weight(self, _, __=None):
        pass

    def forward(self, inp_word, inp_char, inp_pos, inp_mask):
        obf_word = inp_word.clone()
        pos_alphabet = pipe.parser_input.pos_alphabet
        pri_mask = torch.zeros_like(inp_word).byte()
        for pos, words in self.tgtwords.items():
            pid = pos_alphabet.get_index(pos)
            pos_mask = inp_pos == pid
            if is_privacy_term(pos):
                pri_mask |= pos_mask
        unk_mask = (torch.rand(pri_mask.shape) < self.cfg.unk_rate).type_as(pri_mask)
        obf_word[unk_mask & pos_mask] = UNK_ID
        ret = MrDict(fixed=False, blob=True)
        ret.obf_word = obf_word
        ret.pri_mask = pri_mask
        ret.obf_mask = pri_mask & unk_mask
        ret.obf_pos = inp_pos
        ret.obf_char = word_to_char(obf_word, self.lut).type_as(inp_char)
        ret.cpy_mask = inp_mask.byte() ^ ret.obf_mask

        ret.fix()
        return ret


class TagSpecRandomGenerator(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = AttrDict({
                "ctx_obf_rate": 1,
                "pri_obf_rate": 1
            })
        self.cfg = cfg

        tgtwords = get_word_alltag(pipe.parser_input.word_alphabet)
        print("tgtwords initiated")
        for pos in tgtwords.keys():
            print("{}-{}".format(pos, len(tgtwords[pos])), end=" ")
        print()
 
        self.tgtwords = tgtwords
        self.lut = get_lut(pipe.parser_input.word_alphabet, pipe.parser_input.char_alphabet).long().to(config.cfg.device)

    def update_emb_weight(self, _, __=None):
        pass

    def forward(self, inp_word, inp_char, inp_pos, inp_mask=None, reinforce=None, **kwargs):
        pos_alphabet = pipe.parser_input.pos_alphabet
        obf_word = inp_word.clone()
        pri_mask = torch.zeros_like(inp_word).byte()
        obf_mask = torch.zeros_like(inp_word).byte()

        ctx_obf_mask = (torch.rand(pri_mask.shape) < self.cfg.ctx_obf_rate).type_as(pri_mask)
        pri_obf_mask = (torch.rand(pri_mask.shape) < self.cfg.pri_obf_rate).type_as(pri_mask)
        
        for pos in self.tgtwords.keys():
            pid = pos_alphabet.get_index(pos)
            pos_mask = inp_pos == pid

            if is_privacy_term(pos):
                pri_mask |= pos_mask
                mask = pos_mask & pri_obf_mask
            else:
                mask = pos_mask & ctx_obf_mask
            
            N = int(mask.sum().item())
            M = len(self.tgtwords[pos])
            cdt = self.tgtwords[pos][torch.randint(0, M, size=(N,)).long()]
            cdt = cdt.type_as(inp_word)
            obf_word.masked_scatter_(mask.byte(), cdt)
            obf_mask |= mask

        obf_char = word_to_char(obf_word, self.lut).type_as(inp_char)
        return MrDict({
            "obf_word": obf_word,
            "ori_word": inp_word,
            "obf_char": obf_char,
            "obf_pos": inp_pos,
            "obf_mask": obf_mask,
            "pri_mask": pri_mask,
            "cpy_mask": inp_mask.byte() & (inp_word == obf_word)},
            fixed=True, blob=True)

