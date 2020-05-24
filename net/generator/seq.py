# pylint: disable=invalid-name
import torch
from torch import nn
from data import pipe
from data.fileio import PAD_ID_WORD
from data.input import get_word_alltag

from utils import tensor_map, get_word_by_ids, get_char_tensor, dynamic_char_scatter, get_lut, word_to_char, is_privacy_term

from model.parser.model.nn.variational_rnn import VarMaskedFastLSTM
from net.functional.gumbel import gumbel_sample, gumbel_softmax, gumbel_sample2
from net.functional.loss import hloss
from net.encoder import InpEncoder, InpWithDistEncoder
from net.encoder import ObfEncoder

from utils import get_temperature

import config
from config.utils import MrDict
cfg = config.cfg


class Seq2SeqGenerator(nn.Module):
    """
    你怕不是失了智了。。不就是把Encode套了一下吗。。
    """
    def __init__(self, word_embedd):
        super(Seq2SeqGenerator, self).__init__()
        # input.emb_weight = torch.index_select(word_embedd.weight, 0, torch.tensor(input.word_ids, device=cfg.device))
        pipe.parser_input.emb_weight = word_embedd.weight.detach()
        self.enc = Seq2SeqEncoder(word_embedd)

    def forward(self, *input, **kwargs):
        # suppose pos taggin is not changed
        word_emb, word, char = self.enc(*input, **kwargs)
        return word_emb, word, char

class Seq2SeqPGGenerator(nn.Module):
    """
    Refined structure
    """

    def __init__(self):
        super(Seq2SeqPGGenerator, self).__init__()
        input = pipe.parser_input

        cfg_g = config.cfg_g
        input_dim = 0
        kernel_size = 3

        if cfg_g.use_word:
            input_dim += cfg_g.word_dim
            self.word_embedd = nn.Embedding(input.num_words, cfg_g.word_dim, _weight=input.word_table)
        else:
            self.word_embedd = None

        if cfg_g.use_pos:
            input_dim += cfg_g.pos_dim
            self.pos_embedd = nn.Embedding(input.num_pos, cfg_g.pos_dim, _weight=input.char_table)
        else:
            self.pos_embedd = None

        if cfg_g.use_char:
            input_dim += cfg_g.num_filters
            self.char_embedd = nn.Embedding(input.num_chars, cfg_g.char_dim, _weight=input.embedd_char)
            self.conv1d = nn.Conv1d(cfg_g.char_dim, cfg_g.num_filters, kernel_size, padding=kernel_size - 1)
        else:
            self.char_embedd = None

        self.drop = nn.Dropout(p=cfg_g.p_in)

        self.rnn_enc = VarMaskedFastLSTM(input_dim, cfg_g.hidden_size, cfg_g.num_layers, batch_first=True,
                                     bidirectional=cfg_g.bidirectional, dropout=cfg_g.p_rnn)

        dim_inp = cfg_g.word_dim + (cfg_g.hidden_size * 2 if cfg_g.bidirectional else cfg_g.hidden_size)
        # Seq2Seq, feed together the sampled decoded result of previous step
        dim_oup = cfg_g.hidden_size * 2 if cfg_g.bidirectional else cfg_g.hidden_size
        self.rnn_dec = VarMaskedFastLSTM(
            dim_inp, cfg_g.hidden_size, cfg_g.num_layers,
            batch_first=True, bidirectional=cfg_g.bidirectional, dropout=cfg_g.p_rnn
        )
        self.drop = nn.Dropout(p=cfg_g.p_out)
        self.dec = nn.Linear(dim_oup, cfg_g.num_out_words)
        # import ipdb
        # ipdb.set_trace()
        self.out_emb_weight = pipe.parser_input.emb_weight
        self.lut = get_lut(input.word_alphabet, input.char_alphabet).long().to(cfg.device)
        self.to(cfg.device)

    def forward(self, input_word, input_char, input_pos, mask=None, hx=None):
        inputs = []

        if self.word_embedd:
            word = self.word_embedd(input_word)
            # apply dropout on input
            word = self.drop(word)
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
            char = self.drop(char)
            # concatenate word and char [batch, length, word_dim+char_filter]
            inputs.append(char)

        if self.pos_embedd:
            # [batch, length, pos_dim]
            pos = self.pos_embedd(input_pos)
            # apply dropout on input
            pos = self.drop(pos)
            inputs.append(pos)

        inp = torch.cat(inputs, dim=2)

        inp, hx = self.rnn_enc(inp, mask=mask, hx=hx)

        # inp (bs, len, fs), hx [h, c], mask, (bs, len)
        bs, tot, fs = inp.shape
        # inp = torch.cat([inp, inp.new_full([bs, tot, cfg_g.word_dim], fill_value=0)], dim=-1)
        sample = inp.new_full([bs, tot], fill_value=PAD_ID_WORD)
        critic = inp.new_full([bs, tot], fill_value=PAD_ID_WORD)
        last = inp.new_full([bs, 1, cfg_g.word_dim], fill_value=0)
        sum = inp.new_full([bs], fill_value=0)
        for i in range(tot):
            fx = torch.cat([inp[:, i:i+1], last], dim=-1)
            # fx of (bs, 1, fs + wd)
            mx = mask[:, i:i+1]
            # mx of (bs, 1)
            oup, hx = self.rnn_dec(fx, mask=mx, hx=hx)
            # oup of (bs, 1, fs)
            oup = self.drop(oup)
            oup = self.dec(oup)
            # oup of (bs, 1, nw)
            oup = oup.squeeze(1)
            # oup of (bs, nw)
            oup_max, oup_arg = oup.max(dim=-1)
            sum += mx.squeeze(-1) * oup_max
            critic[:, i:i+1] = oup_arg.unsqueeze(-1)
            sam = gumbel_sample(oup).unsqueeze(-1)
            # sam of (bs, 1)
            sample[:, i:i+1] = sam
            last = self.word_embedd(sam)
            # last of (bs, 1, wd)
        word_s = (sample * mask + (1 - mask) * PAD_ID_WORD).long()
        word_c = (critic * mask + (1 - mask) * PAD_ID_WORD).long()
        char_s = word_to_char(word_s, self.lut)
        char_c = word_to_char(word_c, self.lut)
        pos_s = input_pos
        pos_c = input_pos

        return (word_s, char_s, pos_s), (word_c, char_c, pos_c), sum


class SeqLabelGenerator(nn.Module):
    def __init__(self, keywords, tgtwords, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config.SeqLabelGeneratorCfg
        inputs = pipe.parser_input

        cfg_g = config.cfg_g
        # self.keywords = sorted(keywords)
        # We will only consider word with NNP(S) tag to be obfuscated in tgtwords
        self.tgtwords = sorted(tgtwords)
        # self.keywords_set = set(keywords)
        # self.nkey = len(keywords)
        self.ntgt = len(tgtwords)
        self.nwrd = inputs.num_words
        self.word_emb = nn.Embedding(self.nwrd, cfg_g.word_dim, _weight=inputs.word_table)
        self.inp_enc = InpEncoder()
        self.ctx_rnn = VarMaskedFastLSTM(cfg.inp_dim, cfg.ctx_fs, cfg.num_layers, batch_first=True, bidirectional=cfg.bidirectional, dropout=cfg.p_rnn_drop)
        self.inp_drop = nn.Dropout(cfg.p_inp_drop)
        self.ctx_drop = nn.Dropout(cfg.p_ctx_drop)
        hs = cfg.ctx_fs * 2 if cfg.bidirectional else cfg.ctx_fs

        self.dec = nn.Sequential(
            nn.Linear(hs, self.ntgt),
            nn.Softmax(dim=-1)
        )
        self.lut = get_lut(inputs.word_alphabet, inputs.char_alphabet).long().to(cfg.device)

        self.word_emb_weight = None
        self.word_emb_tgt = None
        self.aneal = MrDict({"step": 0, "t": 1, "nstep": 500, "r": -1e-5}, fixed=True)


    def forward(self, inp_word, inp_char, inp_pos, masks, reinforce=True):
        inp = self.inp_enc(inp_word, inp_char, inp_pos)
        inp = self.inp_drop(inp)
        ctx, _ = self.ctx_rnn(inp, mask=masks)
        ctx = self.ctx_drop(ctx)

        dec = self.dec(ctx) + 1e-12
        dec = torch.log(dec)

        # dec of (bs, ls, nw)

        idx = tensor_map(inp_pos, lambda p: pipe.parser_input.pos_alphabet.get_instance(p) in cfg_g.xpos)

        bs, ls = inp_word.shape
        pspots = dec.reshape(bs * ls, -1)[idx.byte().reshape(-1)]
        # pspots of (msk.sum().item(), ntgt)

        ret = MrDict(fixed=False, blob=True)
        if reinforce:
            plog = torch.zeros([bs, ls]).float().to(inp_word.device)
            psmp, smp = gumbel_sample2(pspots)
            # smp of (msk.sum().item(), ), take the sample
            ctc = pspots.argmax(dim=-1)
            # critic by take the argmax

            # = =, bro you've written a bug
            smp = tensor_map(smp, lambda wid: self.tgtwords[wid])
            ctc = tensor_map(ctc, lambda wid: self.tgtwords[wid])

            plog.masked_scatter_(idx.byte(), psmp)
            obf_word = inp_word.clone()
            ctc_word = inp_word.clone()
            obf_word.masked_scatter_(idx.byte(), smp)
            ctc_word.masked_scatter_(idx.byte(), ctc)
            
            obf_char = word_to_char(obf_word, self.lut)
            ctc_char = word_to_char(ctc_word, self.lut)

            ret.obf_word = obf_word
            ret.obf_char = obf_char
            ret.obf_pos = inp_pos
            ret.ctc_word = ctc_word
            ret.ctc_char = ctc_char
            ret.ctc_pos = inp_pos
            ret.plog = plog
        else:
            if self.training:
                t = get_temperature(self.aneal)
            else:
                t = 1.0
                
            sft_x = gumbel_softmax(pspots, t)
            y = sft_x.argmax(dim=-1)
            y = tensor_map(y, lambda wid: self.tgtwords[wid])
            x_emb = torch.matmul(sft_x, self.word_emb_tgt)  # pylint: disable=no-member
            # y of (bs, ls)
            obf_word = inp_word.clone()
            obf_word.masked_scatter_(idx.byte(), y)
            obf_char = word_to_char(obf_word, self.lut)
            obf_word_emb = self.word_emb_weight[obf_word]
            obf_word_emb.masked_scatter_(idx.unsqueeze(-1).byte(), x_emb)

            ret.obf_word = obf_word
            ret.obf_char = obf_char
            ret.obf_pos = inp_pos
            ret.obf_word_emb = obf_word_emb
            ret.t = t


        ret.obf_mask = idx.float()
        ret.fix()
        return ret


class SeqAllGenerator(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config.SeqAllGeneratorCfg
        inputs = pipe.parser_input

        cfg_g = config.cfg_g

        self.nwrd = inputs.num_words
        self.word_emb = nn.Embedding(self.nwrd, cfg_g.word_dim, _weight=inputs.word_table)
        self.inp_enc = InpEncoder()
        self.ctx_rnn = VarMaskedFastLSTM(cfg.inp_dim, cfg.ctx_fs, cfg.num_layers, batch_first=True, bidirectional=cfg.bidirectional, dropout=cfg.p_rnn_drop)
        self.inp_drop = nn.Dropout(cfg.p_inp_drop)
        self.ctx_drop = nn.Dropout(cfg.p_ctx_drop)
        hs = cfg.ctx_fs * 2 if cfg.bidirectional else cfg.ctx_fs

        if cfg.mlp_decoding:
            self.dec = nn.Sequential(
                    nn.Linear(hs, cfg.dec_hs),
                    nn.ReLU(),
                    nn.Linear(cfg.dec_hs, self.nwrd),
                    nn.Softmax(dim=-1)
            )
        else:
            self.dec = nn.Linear(hs, self.nwrd)

        self.lut = get_lut(inputs.word_alphabet, inputs.char_alphabet).long().to(cfg.device)

        self.word_emb_weight = None
        self.word_emb_tgt = None
        self.aneal = MrDict({"step": 0, "t": 1, "nstep": 700, "r": -1e-5}, fixed=True)

    def forward(self, inp_word, inp_char, inp_pos, masks, reinforce=True):
        inp = self.inp_enc(inp_word, inp_char, inp_pos)
        inp = self.inp_drop(inp)
        ctx, _ = self.ctx_rnn(inp, mask=masks)
        ctx = self.ctx_drop(ctx)
        dec = self.dec(ctx) + 1e-12
        dec = torch.log(dec)
        # dec of (bs, ls, nw)
        bs, ls = inp_word.shape
        pspots = dec.reshape(bs * ls, -1)
        # pspots of (msk.sum().item(), ntgt)

        ret = MrDict(fixed=False, blob=True)
        if reinforce:
            psmp, smp = gumbel_sample2(pspots)
            # smp of (msk.sum().item(), ), take the sample
            ctc = pspots.argmax(dim=-1)
            # critic by take the argmax

            plog = psmp.reshape(bs, ls) * masks.float()
            obf_word = smp.reshape(bs, ls)
            ctc_word = ctc.reshape(bs, ls)
            obf_char = word_to_char(obf_word, self.lut)
            ctc_char = word_to_char(ctc_word, self.lut)

            ret.obf_word = obf_word
            ret.obf_char = obf_char
            ret.obf_pos = inp_pos
            ret.ctc_word = ctc_word
            ret.ctc_char = ctc_char
            ret.ctc_pos = inp_pos
            ret.plog = plog
        else:
            if self.training:
                t = get_temperature(self.aneal)
            else:
                t = 1.0
                
            sft_x = gumbel_softmax(pspots, t)
            y = sft_x.argmax(dim=-1)
            x_emb = torch.matmul(sft_x, self.word_emb_tgt)  # pylint: disable=no-member
            # y of (bs, ls)
            obf_word = y.reshape(bs, ls)
            obf_char = word_to_char(obf_word, self.lut)
            obf_word_emb = self.word_emb_weight[obf_word]
            obf_word_emb = x_emb.reshape(bs, ls, -1)
            ret.obf_word = obf_word
            ret.obf_char = obf_char
            ret.obf_pos = inp_pos
            ret.obf_word_emb = obf_word_emb
            ret.t = t


        ret.obf_mask = masks
        ret.fix()
        return ret


class SeqCopyGenerator(nn.Module):
    def __init__(self, keywords, tgtwords, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config.SeqCopyGeneratorCfg
        inputs = pipe.parser_input
        cfg_g = config.cfg_g

        self.keywords = sorted(keywords)
        self.tgtwords = sorted(tgtwords)
        self.keywords_set = set(keywords)
        self.nkey = len(keywords)
        self.ntgt = len(tgtwords)
        self.nwrd = inputs.num_words
        self.word_emb = nn.Embedding(self.nwrd, cfg_g.word_dim, _weight=inputs.word_table)
        self.inp_enc = InpEncoder()
        self.ctx_rnn = VarMaskedFastLSTM(cfg.inp_dim, cfg.ctx_fs, cfg.num_layers, batch_first=True, bidirectional=cfg.bidirectional, dropout=cfg.p_rnn_drop)
        self.inp_drop = nn.Dropout(cfg.p_inp_drop)
        self.ctx_drop = nn.Dropout(cfg.p_ctx_drop)
        hs = cfg.ctx_fs * 2 if cfg.bidirectional else cfg.ctx_fs

        self.spt_dec = nn.Linear(hs, self.ntgt)
        self.nrm_dec = nn.Linear(hs, self.nwrd)
        self.nrm_cpy = nn.Linear(hs, hs)

        self.lut = get_lut(inputs.word_alphabet, inputs.char_alphabet).long().to(cfg.device)

        self.word_emb_weight = None
        self.word_emb_tgt = None
        self.aneal = MrDict({"step": 0, "t": 1, "nstep": 500, "r": -1e-5}, fixed=True)


    def forward(self, inp_word, inp_char, inp_pos, masks, reinforce=True):
        bs, ls = inp_word.shape

        inp = self.inp_enc(inp_word, inp_char, inp_pos)  # TODO: append the idx mask to input
        inp = self.inp_drop(inp)  # TODO: perhaps not useful?
        ctx, _ = self.ctx_rnn(inp, mask=masks)
        ctx = self.ctx_drop(ctx)

        spt_idx = tensor_map(inp_word, lambda wid: wid in self.keywords_set).byte()
        nrm_idx = ((1 - spt_idx.float()) * masks.float()).byte()
        spt_hx = ctx.reshape(bs * ls, -1)[spt_idx.reshape(-1)]
        nrm_hx = ctx.reshape(bs * ls, -1)[nrm_idx.reshape(-1)]

        pspt = self.spt_dec(spt_hx)

        pnrm = self.nrm_dec(nrm_hx)
        pcpy = torch.matmul(self.nrm_cpy(nrm_hx).unsqueeze(-2), nrm_hx.unsqueeze(-1)).squeeze(-1)
        pnrm = torch.cat([pnrm, pcpy], dim=-1)

        pspt = torch.log(torch.softmax(pspt, dim=-1) + 1e-12)
        pnrm = torch.log(torch.softmax(pnrm, dim=-1) + 1e-12)
        nrm_id = inp_word.reshape(-1)[nrm_idx.reshape(-1)]

        # pspots of (msk.sum().item(), ntgt)
        # pnorms of (inp_word.numel() - msk.sum().item, ntgt)

        ret = MrDict(fixed=False, blob=True)
        if reinforce:
            # FIXME: not implemented
            plog = torch.zeros([bs, ls]).float().to(inp_word.device)
            psmp, smp = gumbel_sample2(pspots)
            # smp of (msk.sum().item(), ), take the sample
            ctc = pspots.argmax(dim=-1)
            # critic by take the argmax

            # = =, bro you've written a bug
            smp = tensor_map(smp, lambda wid: self.tgtwords[wid])
            ctc = tensor_map(ctc, lambda wid: self.tgtwords[wid])

            plog.masked_scatter_(idx.byte(), psmp)
            obf_word = inp_word.clone()
            ctc_word = inp_word.clone()
            obf_word.masked_scatter_(idx.byte(), smp)
            ctc_word.masked_scatter_(idx.byte(), ctc)
            
            obf_char = word_to_char(obf_word, self.lut)
            ctc_char = word_to_char(ctc_word, self.lut)

            ret.obf_word = obf_word
            ret.obf_char = obf_char
            ret.obf_pos = inp_pos
            ret.ctc_word = ctc_word
            ret.ctc_char = ctc_char
            ret.ctc_pos = inp_pos
            ret.plog = plog
        else:
            if self.training:
                t = get_temperature(self.aneal)
            else:
                t = 1.0
                
            sft_spt = gumbel_softmax(pspt, t)
            yspt = sft_spt.argmax(dim=-1)
            yspt = tensor_map(yspt, lambda wid: self.tgtwords[wid])
            spt_emb = torch.matmul(sft_spt, self.word_emb_tgt)  # pylint: disable=no-member

            sft_nrm = gumbel_softmax(pnrm, t)
            ynrm = sft_nrm.argmax(dim=-1)
            nrm_msk = ynrm == self.nwrd
            ret.nrm_msk = nrm_msk.float()
            
            tmp = torch.where(nrm_msk, nrm_id, ynrm)
            ynrm = tmp

            nrm_emb = self.word_emb_weight[nrm_id]
            nrm_emb *= sft_nrm[:, -1:]
            nrm_emb += torch.matmul(sft_nrm[:, :-1], self.word_emb_weight)

            obf_word = inp_word.clone()
            obf_word.masked_scatter_(spt_idx, yspt)
            obf_word.masked_scatter_(nrm_idx, ynrm)

            try:
                obf_char = word_to_char(obf_word, self.lut)
            except:
                import ipdb
                ipdb.set_trace()
                print(233)
            obf_word_emb = self.word_emb_weight[obf_word]
            obf_word_emb.masked_scatter_(spt_idx.unsqueeze(-1), spt_emb)
            obf_word_emb.masked_scatter_(nrm_idx.unsqueeze(-1), nrm_emb)

            ret.obf_word = obf_word
            ret.obf_char = obf_char
            ret.obf_pos = inp_pos
            ret.obf_word_emb = obf_word_emb
            ret.t = t

        ret.obf_mask = spt_idx.float()
        ret.fix()
        return ret


class AlltagCtxGenerator(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config.AlltagCopyCtxGeneratorCfg

        cfg_g = config.cfg_g
        self.cfg = cfg

        inputs = pipe.parser_input
        tgtwords = get_word_alltag(inputs.word_alphabet)
        self.tgtwords = tgtwords
        self.lut = get_lut(inputs.word_alphabet, inputs.char_alphabet).long().to(config.cfg.device)

        self.inp_enc = InpWithDistEncoder()
        # self.inp_enc = InpEncoder()
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
        self.psr_emb_lut = {}
        self.atk_emb_lut = {}
        self.psr_weight = None
        self.atk_weight = None
        self.aneal = MrDict({"step": 0, "t": 1, "nstep": 500, "r": -1e-5}, fixed=True)

    def update_emb_weight(self, psr_weight, atk_weight):
        self.psr_emb_lut = {}
        self.atk_emb_lut = {}
        self.psr_weight = psr_weight.clone()
        self.atk_weight = atk_weight.clone()
        for pos, words in self.tgtwords.items():
            self.psr_emb_lut[pos] = psr_weight[words]
            self.atk_emb_lut[pos] = atk_weight[words]

    def forward(self, inp_word, inp_char, inp_pos, inp_mask):
        inp = self.inp_enc(inp_word, inp_char, inp_pos)
        inp = self.inp_drop(inp)
        ctx, _ = self.ctx_rnn(inp, mask=inp_mask)
        ctx = self.ctx_drop(ctx)

        zeros_like = torch.zeros_like  # pylint: disable=no-member
        cat = torch.cat  # pylint: disable=no-member
        matmul = torch.matmul  # pylint: disable=no-member

        bs, ls = inp_word.shape
        ctx = ctx.reshape(bs * ls, -1)

        pos_alphabet = pipe.parser_input.pos_alphabet

        obf_word = inp_word.clone()
        obf_psr_emb = self.psr_weight[inp_word]
        obf_atk_emb = self.atk_weight[inp_word]

        obf_mask = torch.zeros_like(inp_word).byte()
        pri_mask = torch.zeros_like(inp_word).byte()

        t = get_temperature(self.aneal) if self.training else 1.

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
            atk_emb = matmul(spt, self.atk_emb_lut[pos])
            obf_psr_emb.masked_scatter_(pos_mask.unsqueeze(-1), psr_emb)
            obf_atk_emb.masked_scatter_(pos_mask.unsqueeze(-1), atk_emb)

        ret = MrDict(fixed=False, blob=True)
        ret.obf_word = obf_word
        ret.obf_char = word_to_char(obf_word, self.lut)
        ret.obf_pos = inp_pos
        ret.obf_word_psr_emb = obf_psr_emb
        ret.obf_word_atk_emb = obf_atk_emb
        ret.cpy_mask = (obf_word == inp_word) & inp_mask.byte()
        ret.obf_mask = obf_mask
        ret.pri_mask = pri_mask
        ret.ent_loss = -entropy

        ret.cpy_loss = 0
        ret.cpy_full_loss = 0

        ret.fix()
        return ret


class AlltagCopyCtxGenerator(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config.cfg.AlltagCopyCtxGenerator

        from data.input import get_word_alltag
        tgtwords = get_word_alltag(pipe.parser_input.word_alphabet)
        print("tgtwords initiated")
        for pos in tgtwords.keys():
            print("{}-{}".format(pos, len(tgtwords[pos])), end=" ")
        print()
 
        self.tgtwords = tgtwords
        self.lut = get_lut(pipe.parser_input.word_alphabet, pipe.parser_input.char_alphabet).long().to(config.cfg.device)

        inputs = pipe.parser_input
        self.nwrd = inputs.num_words

        self.inp_enc = InpEncoder()
        self.ctx_rnn = VarMaskedFastLSTM(cfg.inp_dim, cfg.ctx_fs, cfg.num_layers, batch_first=True, bidirectional=cfg.bidirectional, dropout=cfg.p_rnn_drop)
        self.inp_drop = nn.Dropout(cfg.p_inp_drop)
        self.ctx_drop = nn.Dropout(cfg.p_ctx_drop)
        hs = cfg.ctx_fs * 2 if cfg.bidirectional else cfg.ctx_fs

        pos_lut = { pos: "pos_" + str(i) for i, pos in enumerate(tgtwords.keys()) }
        self.pos_lut = pos_lut
        self.dec = nn.ModuleDict({
            pos_lut[pos]: nn.Sequential(
                nn.Linear(hs, len(words)),
                nn.Softmax(dim=-1)
            ) for pos, words in self.tgtwords.items()
        })
        self.word_emb_weight = None
        self.tgt_emb = None
        self.aneal = MrDict({"step": 0, "t": 1, "nstep": 500, "r": -1e-5}, fixed=True)

    def update_emb_weight(self, weight):
        self.word_emb_weight = weight
        self.tgt_emb = {}
        for pos, words in self.tgtwords.items():
            self.tgt_emb[pos] = weight[words]

    def forward(self, inp_word, inp_char, inp_pos, masks, reinforce=False):
        inp = self.inp_enc(inp_word, inp_char, inp_pos)
        inp = self.inp_drop(inp)
        ctx, _ = self.ctx_rnn(inp, mask=masks)
        ctx = self.ctx_drop(ctx)

        bs, ls = inp_word.shape
        ctx = ctx.reshape(bs * ls, -1)

        pos_alphabet = pipe.parser_input.pos_alphabet

        obf_word = inp_word.clone()
        obf_word_emb = self.word_emb_weight[obf_word]

        msk = torch.zeros_like(inp_word).byte()
        nn_msk = torch.zeros_like(inp_word).byte()

        t = get_temperature(self.aneal) if self.training else 1.

        for pos, words in self.tgtwords.items():
            pid = pos_alphabet.get_index(pos)
            pos_msk = inp_pos == pid
            msk = msk | pos_msk
            if pos.startswith('NN'):
                nn_msk = nn_msk | pos_msk
            N = int(pos_msk.long().sum().item())
            M = len(words)
            if N == 0:
                continue

            msk_ctx = ctx[pos_msk.reshape(-1)]
            dec = self.dec[self.pos_lut[pos]]
            pspt = dec(msk_ctx) + 1e-12
            pspt = torch.log(pspt)
            assert pspt.shape == (N, M)

            sft_x = gumbel_softmax(pspt, t)
            y = sft_x.argmax(dim=-1)
            y = tensor_map(y, lambda wid: words[wid])
            x_emb = torch.matmul(sft_x, self.tgt_emb[pos])  # pylint: disable=no-member
            obf_word.masked_scatter_(pos_msk, y)
            obf_word_emb.masked_scatter_(pos_msk.unsqueeze(-1), x_emb)

        ret = MrDict(fixed=False, blob=True)
        ret.obf_word = obf_word
        ret.obf_char = word_to_char(obf_word, self.lut)
        ret.obf_pos = inp_pos
        ret.obf_word_emb = obf_word_emb
        ret.t = t
        ret.obf_mask = msk.float()
        ret.nn_msk = nn_msk
        ret.fix()
        return ret



