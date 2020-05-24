# pylint: disable=invalid-name
import torch
from torch import nn
from torch.nn import functional as F
from data import pipe
from data.fileio import PAD_ID_WORD, read_inputs, UNK_ID
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

def is_privacy_term(word, pos):
    obf_term = config.cfg.Generator.obf_term
    pos_alphabet = read_inputs().pos_alphabet
    pos_tag = pos_alphabet.get_instance(pos)
    return 1 if pos_tag in obf_term else 0

class TagAgnosticGenerator(nn.Module):
    def __init__(self, cfg=None, use_recovery=False):
        super().__init__()
        if cfg is None:
            cfg = config.cfg.AlltagCopyCtxGenerator

        self.cfg = cfg

        inputs = read_inputs()
        self.lut = get_lut(inputs.word_alphabet, inputs.char_alphabet).long().to(config.cfg.device)

        # self.inp_enc = InpEncoder()
        self.inp_enc = InpWithDistEncoder()

        self.ctx_rnn = VarMaskedFastLSTM(cfg.inp_dim, cfg.ctx_fs, cfg.num_layers,
                batch_first=True, bidirectional=cfg.bidirectional, dropout=cfg.p_rnn_drop)

        self.inp_drop = nn.Dropout(cfg.p_inp_drop)
        self.ctx_drop = nn.Dropout(cfg.p_ctx_drop)

        hs = cfg.ctx_fs * 2 if cfg.bidirectional else cfg.ctx_fs
        self.dec = nn.Sequential(
            nn.Linear(hs, 256),
            nn.ReLU(),
            nn.Linear(256, inputs.num_words),
            nn.LogSoftmax(dim=-1)
        )
        self.aneal = MrDict({"step": 0, "t": 1, "nstep": 500, "r": -1e-5}, fixed=True)
        self.copy_classifier = nn.Sequential(
                nn.Linear(hs, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.LogSoftmax(dim=-1)
            )
        self.parser_emb_weight = None

        self.use_recovery = use_recovery
        if use_recovery:
            self.recovery_embedding = nn.Embedding(inputs.num_words, 100, _weight=inputs.word_table.clone())
            self.recovery_lstm = VarMaskedFastLSTM(100, 128, 1, batch_first=True, bidirectional=True, dropout=cfg.p_rnn_drop)
            self.recovery_classifier = nn.Sequential(
                nn.Linear(256, inputs.pos_alphabet.size()),
                nn.LogSoftmax(dim=-1)
            )

    def update_emb_weight(self, psr_weight=None, atk_weight=None):
        self.parser_emb_weight = psr_weight.clone()

    def forward(self, inp_word, inp_char, inp_pos, inp_mask, obf_mask=None, **kwargs):
        bs, ls = inp_word.shape

        zeros_like = torch.zeros_like  # pylint: disable=no-member
        cat = torch.cat  # pylint: disable=no-member
        matmul = torch.matmul  # pylint: disable=no-member

        inp = self.inp_enc(inp_word, inp_char, inp_pos)
        inp = self.inp_drop(inp)
        ctx, _ = self.ctx_rnn(inp, mask=inp_mask)
        ctx = self.ctx_drop(ctx)

        gen_logits = self.dec(ctx)

        temperature = get_temperature(self.aneal) if self.training else 1.
        batch_size, seq_length, vocab_size = gen_logits.shape

        if obf_mask is None:
            obf_mask = inp_pos.new_zeros(inp_pos.shape)
            for i in range(batch_size):
                for j in range(seq_length):
                    if inp_mask[i, j] > 0.5:
                        obf_mask[i, j] = is_privacy_term(inp_word[i, j], inp_pos[i, j])
                    else:
                        obf_mask[i, j] = 0

        # shape (batch_size * sequence_length, vocab_size)

        gen_logits = gen_logits.reshape(batch_size * seq_length, -1)
        targeted_logits = gen_logits[obf_mask.reshape(-1).byte()]

        ent_loss = torch.mean(targeted_logits * torch.exp(targeted_logits))
        
        original_target_word = inp_word.reshape(batch_size * seq_length)[obf_mask.reshape(-1).byte()]

        relaxed_dist = gumbel_softmax(targeted_logits, temperature)
        sampled_word_index = relaxed_dist.argmax(dim=-1)
        sampled_word_emb = torch.matmul(relaxed_dist, self.parser_emb_weight)

        safe_word_mask = ~(sampled_word_index == original_target_word)
        avoid_word_index = inp_word.new_full([obf_mask.sum().item()], UNK_ID)
        avoid_word_emb = self.parser_emb_weight[avoid_word_index]

        sampled_word_emb = avoid_word_emb.masked_scatter_(safe_word_mask.unsqueeze(-1), sampled_word_emb)
        sampled_word_index = avoid_word_index.masked_scatter_(safe_word_mask, sampled_word_index)

        obf_word = inp_word.clone()
        obf_word.masked_scatter_(obf_mask.byte(), sampled_word_index)

        obf_word_emb = self.parser_emb_weight[obf_word]
        obf_word_emb.masked_scatter_(obf_mask.unsqueeze(-1).byte(), sampled_word_emb)

        if self.use_recovery:
            rec_embedding = self.recovery_embedding.weight[sampled_word_index]
            avg_weight = F.softmax(targeted_logits, dim=-1)
            avoid_word_emb = self.recovery_embedding.weight[avoid_word_index]
            aggregated_embedding = torch.matmul(avg_weight, self.recovery_embedding.weight)
            aggregated_embedding = avoid_word_emb.masked_scatter_(safe_word_mask.unsqueeze(-1))

        output_dict = AttrDict({
            "obf_word": obf_word,
            "obf_word_psr_emb": obf_word_emb,
            "obf_char": word_to_char(obf_word, self.lut),
            "obf_pos": inp_pos,
            "obf_mask": obf_mask.byte(),
            "ent_loss": ent_loss
        }, blob=True)

        return output_dict


class AlltagCopyCtxGenerator(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config.cfg.AlltagCopyCtxGenerator

        self.cfg = cfg

        inputs = pipe.parser_input
        tgtwords = get_word_alltag(inputs.word_alphabet)
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
        self.atk_emb_lut = {}
        self.psr_weight = None
        self.atk_weight = None

    def update_emb_weight(self, psr_weight, atk_weight):
        self.psr_emb_lut = {}
        self.atk_emb_lut = {}
        self.psr_weight = psr_weight.clone()
        self.atk_weight = atk_weight.clone()
        for pos, words in self.tgtwords.items():
            self.psr_emb_lut[pos] = psr_weight[words]
            self.atk_emb_lut[pos] = atk_weight[words]

    def forward(self, inp_word, inp_char, inp_pos, inp_mask):
        bs, ls = inp_word.shape

        zeros_like = torch.zeros_like  # pylint: disable=no-member
        cat = torch.cat  # pylint: disable=no-member
        matmul = torch.matmul  # pylint: disable=no-member

        inp = self.inp_enc(inp_word, inp_char, inp_pos)
        inp = self.inp_drop(inp)
        ctx, _ = self.ctx_rnn(inp, mask=inp_mask)
        ctx = self.ctx_drop(ctx)

        t = get_temperature(self.aneal) if self.training else 1.
        pcpy = self.copy_classifier(ctx)
        
        ctx = ctx.reshape(bs * ls, -1)

        pos_alphabet = pipe.parser_input.pos_alphabet

        obf_word = inp_word.clone()

        ori_psr_emb = self.psr_weight[inp_word] # pylint: disable=unsubscriptable-object
        ori_atk_emb = self.atk_weight[inp_word] # pylint: disable=unsubscriptable-object

        obf_psr_emb = ori_psr_emb.clone()
        obf_atk_emb = ori_atk_emb.clone()

        obf_mask = zeros_like(inp_word).byte()
        pri_mask = zeros_like(inp_word).byte() # NN words are those words we care in preserving
        entropy = 0
        for pos, words in self.tgtwords.items():
            pid = pos_alphabet.get_index(pos)
            pos_mask = inp_pos == pid
            obf_mask = obf_mask | pos_mask
            if is_privacy_term(pos):
            # if pos.startswith(self.cfg.privacy_term): # either NN, NNP or ''
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

        if self.cfg.do_not_copy_pri:
            num_pri = pri_mask.sum().item()
            zeros = pcpy.new_zeros([num_pri])
            ones = pcpy.new_ones([num_pri])
            ano_mask = pri_mask.new_zeros(pri_mask.shape)
            idx_mask = torch.cat([pri_mask.unsqueeze(-1), ano_mask.unsqueeze(-1)], dim=-1)
            pcpy.masked_scatter_(idx_mask, zeros)
            idx_mask = torch.cat([ano_mask.unsqueeze(-1), pri_mask.unsqueeze(-1)], dim=-1)
            pcpy.masked_scatter_(idx_mask, ones)


        # TODO: it is questionable that should we use gumbel softmax reparameterization here \
        # or just a simple relaxation, since we don't really want the copy mechanism to be \
        # stochastic
        cpy = gumbel_softmax(pcpy, t)
        # cpy of (bs * ls, 2), if cpy[n][0] == 1, then we copy, otherwise we don't copy
        
        cpy_mask = (cpy[:, 0] == 1).reshape(bs, ls)
        cpy_mask = cpy_mask & inp_mask.byte()


        noroot_mask = inp_mask.byte().clone()
        noroot_mask[:, 0] = 0

        # cpy_pri_mask = cpy_mask[pri_mask]
        # cpy_pri_prob = pcpy[pri_mask]
        # try:
        #     cpy_pri_loss = nn.functional.cross_entropy(cpy_pri_prob, torch.ones_like(cpy_pri_mask).long())
        # except:
        #     import ipdb
        #     ipdb.set_trace()
        #     print("Is the mask null?")
        cpy_loss = nn.functional.cross_entropy(pcpy[noroot_mask], pcpy.new_ones([noroot_mask.long().sum().item()]).long()) * self.cfg.cpy_penalty
        # cpy_pri_mask is for those privacy words being copied, and the logits for these prediction is cpy_pri_prob[0], the thing is we don't want privcay words to be copied, so we put a penalty on it, another thing is that we can directly optimize it without noticing these words are copied or not

        if self.cfg.use_copy:
            psr_emb = cat([ori_psr_emb.unsqueeze(-1), obf_psr_emb.unsqueeze(-1)], dim=-1)
            atk_emb = cat([ori_atk_emb.unsqueeze(-1), obf_atk_emb.unsqueeze(-1)], dim=-1)

            cpy = cpy.reshape(bs, ls, 2, 1)
            obf_word_psr_emb = matmul(psr_emb, cpy).squeeze(-1)
            obf_word_atk_emb = matmul(atk_emb, cpy).squeeze(-1)
            # each of (bs, ls, dim)
            
            obf_mask = obf_mask & (~cpy_mask)

            word = cat([inp_word.unsqueeze(-1), obf_word.unsqueeze(-1)], dim=-1)
            word = word.reshape(bs, ls, 1, 2).float()
            obf_word = matmul(word, cpy).squeeze(-1).squeeze(-1).long()
            # of (bs, ls)

        else:
            obf_word_psr_emb = obf_psr_emb
            obf_word_atk_emb = obf_atk_emb       

        ret = MrDict(fixed=False, blob=True)
        ret.obf_word = obf_word
        ret.obf_char = word_to_char(obf_word, self.lut)
        ret.obf_pos = inp_pos
        ret.obf_word_psr_emb = obf_word_psr_emb
        ret.obf_word_atk_emb = obf_word_atk_emb
        ret.obf_mask = obf_mask
        ret.cpy_mask = cpy_mask
        ret.pri_mask = pri_mask
        ret.cpy_loss = cpy_loss * self.cfg.cpy_penalty
        # ret.cpy_full_loss = cpy_full_loss
        ret.ent_loss = -entropy * self.cfg.ent_penalty
        ret.t = t
        ret.fix()
        return ret

