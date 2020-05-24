import torch
from torch import nn
from data.fileio import NUM_SYMBOLIC_TAGS, PAD_ID_WORD
from torch.nn import CrossEntropyLoss
from data import pipe
from net.encoder import InpEncoder
from model.parser.model.nn.variational_rnn import VarMaskedFastLSTM

from config.utils import MrDict, AttrDict

import config
cfg = config.cfg


class FeedForwardAttacker(nn.Module):
    """
    Simplest feed forward back-predictor, trained on paired data
    Be realistic
    """

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config.FeedForwardAttackerCfg
        self.inp_enc = InpEncoder()
        self.inp_drop = nn.Dropout(cfg.p_inp_drop)
        self.mlp = nn.Sequential(
                nn.Linear(cfg.inp_dim, cfg.hidden_size),
                nn.ReLU(),
                nn.Linear(cfg.hidden_size, pipe.parser_input.num_words)
        )
        self.loss = CrossEntropyLoss()

    def _forward(self, src, tgt=None):
        bs, ls = src.shape
        src = src.reshape(-1)
        # src of (bs, ls)
        ex = self.word_emb(src)
        # ex of (bs, ls, word_dim)
        hx = self.mlp(ex)
        # hx of (bs, ls, num_words)
        ret = {}
        if tgt is not None:
            tgt = tgt.reshape(-1)
            ret["loss"] = self.loss(hx, tgt)
        else:
            ret["tgt"] = hx.argmax(dim=-1).reshape(bs, ls)
        return ret
    
    def forward(self, inp_word, inp_char, inp_pos, inp_mask, tgt_mask=None, tgt_word=None):
        inp = self.inp_enc(inp_word, inp_char, inp_pos)  # inp_pos might be None
        inp = self.inp_drop(inp)
        oup = self.mlp(inp)

        bs, ls, nw = oup.shape

        idx = tgt_mask.reshape(-1).byte()
        prcv = oup.reshape(bs * ls, -1)[idx]  # rcv for recover =)

        ret = MrDict(fixed=False, blob=True)
        if tgt_word is not None:
            tgt_word = tgt_word.reshape(-1)[idx]
            ret.loss = self.loss(prcv, tgt_word)
        else:
            rcv_word = inp_word.clone()
            rcv = prcv.argmax(dim=-1)
            rcv_word.masked_scatter_(tgt_mask.byte(), rcv)
            ret.rcv_word = rcv_word

        return ret


class CtxSeqAttacker(nn.Module):
    # @CtxSeqAttackerCfg
    # TODO: sss
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg_p = config.cfg.BiaffineAttnParser
            cfg = AttrDict({
                "p_inp_drop": 0.33,
                "inp_dim": sum([cfg_p.char_dim if cfg_p.use_char else 0,
                    cfg_p.word_dim if cfg_p.use_word else 0,
                    cfg_p.pos_dim if cfg_p.use_pos else 0]),
                "ctx_hs": 512,
                "num_layers": 3,
                "p_rnn": (0.33, 0.33),
                "p_ctx_drop": 0.33,
                "log_minus_d_trick": True,
                }, fixed=True, blob=False)
        self.cfg = cfg

        self.inp_enc = InpEncoder()
        self.inp_drop = nn.Dropout(cfg.p_inp_drop)
        self.ctx_rnn = VarMaskedFastLSTM(cfg.inp_dim, cfg.ctx_hs, num_layers=cfg.num_layers, batch_first=True, bidirectional=True, dropout=cfg.p_rnn)
        self.dec = nn.Linear(cfg.ctx_hs * 2, pipe.parser_input.num_words)
        self.ctx_drop = nn.Dropout(cfg.p_ctx_drop)
        self.loss = CrossEntropyLoss(ignore_index=PAD_ID_WORD)
        self.ctx_word_drop = cfg.ctx_word_drop or 0
        self.relaxed_word_emb = True

    def embedding_weight(self):
        return self.inp_enc.word_embedd.weight

    def forward(self, inp_word, inp_char, inp_pos, inp_mask, pri_mask=None, tgt_word=None, inp_word_emb=None):
        """
        If tgt_word is not None, then the model is under training process, which returns a loss, otherwise returns pred_word as inference result
        Question: do we need tgt_mask that tells the model the spot? Probabily not, you don't want this mask during inference right?
        """
        # from utils import inspect_chr, inspect_sent
        # import ipdb
        # ipdb.set_trace()
        if self.ctx_word_drop > 0:
            drop_mask = (torch.rand(inp_mask.shape) < self.ctx_word_drop).type_as(pri_mask)
            drop_mask &= pri_mask
            if inp_word is not None:
                inp_word[drop_mask] = 0 # 0 is the default mask for unk word
            if inp_word_emb is not None:
                unk_emb = self.inp_enc.word_embedd.weight[0]
                inp_word_emb[drop_mask] = unk_emb
            # print("dropped {}/{}".format(drop_mask.sum().item(), (inp_mask.byte() ^ pri_mask).sum().item()))

        if inp_word_emb is not None and self.relaxed_word_emb:
            # relaxation, don't relax if not told to
            inp = self.inp_enc(inp_word_emb, inp_char, inp_pos)
        else:
            inp = self.inp_enc(inp_word, inp_char, inp_pos)
        inp = self.inp_drop(inp)
        ctx, _ = self.ctx_rnn(inp, inp_mask)
        ctx = self.ctx_drop(ctx)
        oup = self.dec(ctx)
        bs, ls, nw = oup.shape

        idx = pri_mask.reshape(-1).byte()
        has_pri = True
        if idx.sum() == 0:
            has_pri = False
        prcv = oup.reshape(bs * ls, -1)[idx]  # rcv for recover =)

        ret = MrDict(fixed=False, blob=True)
        if tgt_word is not None:
            tgt = tgt_word.reshape(-1)[idx]
            if self.cfg.log_minus_d_trick:
                ret.loss = -self.loss(1 - prcv, tgt)
            else:
                if has_pri:
                    ret.loss = self.loss(prcv, tgt)
                else:
                    ret.loss = torch.tensor(0., requires_grad=True)

        rcv_word = inp_word.clone()
        if has_pri:
            rcv = prcv.argmax(dim=-1)
            rcv_word.masked_scatter_(pri_mask.byte(), rcv)

        ret.rcv_word = rcv_word
        ret.rcv_mask = (rcv_word == tgt_word.reshape(bs, ls)) & pri_mask

        return ret


class WeightedAndFocusedCtxAttacker(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config.CtxSeqAttackerCfg  # pylint: disable=no-member
        self.inp_enc = InpEncoder()
        self.inp_drop = nn.Dropout(cfg.p_inp_drop)
        self.ctx_rnn = VarMaskedFastLSTM(cfg.inp_dim, cfg.ctx_hs, num_layers=cfg.num_layers, batch_first=True, bidirectional=True, dropout=cfg.p_rnn)
        self.dec = nn.Linear(cfg.ctx_hs * 2, pipe.parser_input.num_words)
        self.ctx_drop = nn.Dropout(cfg.p_ctx_drop)
        self.loss = CrossEntropyLoss(ignore_index=PAD_ID_WORD)
        self.ctx_word_drop = cfg.ctx_word_drop or 0
        self.relaxed_word_emb = True

    def embedding_weight(self):
        return self.inp_enc.word_embedd.weight

    def forward(self, inp_word, inp_char, inp_pos, inp_mask, pri_mask=None, tgt_word=None, inp_word_emb=None):
        """
        If tgt_word is not None, then the model is under training process, which returns a loss, otherwise returns pred_word as inference result
        Question: do we need tgt_mask that tells the model the spot? Probabily not, you don't want this mask during inference right?
        """
        # from utils import inspect_chr, inspect_sent
        # import ipdb
        # ipdb.set_trace()
        if self.ctx_word_drop > 0:
            drop_mask = (torch.rand(inp_mask.shape) < self.ctx_word_drop).type_as(pri_mask)
            drop_mask &= pri_mask
            if inp_word is not None:
                inp_word[drop_mask] = 0 # 0 is the default mask for unk word
            if inp_word_emb is not None:
                unk_emb = self.inp_enc.word_embedd.weight[0]
                inp_word_emb[drop_mask] = unk_emb
            # print("dropped {}/{}".format(drop_mask.sum().item(), (inp_mask.byte() ^ pri_mask).sum().item()))

        if inp_word_emb is not None and self.relaxed_word_emb:
            # relaxation, don't relax if not told to
            inp = self.inp_enc(inp_word_emb, inp_char, inp_pos)
        else:
            inp = self.inp_enc(inp_word, inp_char, inp_pos)
        inp = self.inp_drop(inp)
        ctx, _ = self.ctx_rnn(inp, inp_mask)
        ctx = self.ctx_drop(ctx)
        oup = self.dec(ctx)
        bs, ls, nw = oup.shape

        idx = pri_mask.reshape(-1).byte()
        prcv = oup.reshape(bs * ls, -1)[idx]  # rcv for recover =)

        ret = MrDict(fixed=False, blob=True)
        if tgt_word is not None:
            tgt = tgt_word.reshape(-1)[idx]
            ret.loss = self.loss(prcv, tgt)

        rcv_word = inp_word.clone()
        rcv = prcv.argmax(dim=-1)
        rcv_word.masked_scatter_(pri_mask.byte(), rcv)

        ret.rcv_word = rcv_word
        ret.rcv_mask = (rcv_word == tgt_word.reshape(bs, ls)) & pri_mask

        return ret


