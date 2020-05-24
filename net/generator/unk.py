# pylint: disable=invalid-name
import torch
from collections import defaultdict
from torch import nn
from data import pipe
from data.fileio import PAD_ID_WORD, UNK_ID
from utils import tensor_map, get_word_by_ids, get_char_tensor, dynamic_char_scatter, get_lut, word_to_char, is_privacy_term
from data.input import get_word_alltag

import config
from config.utils import MrDict, AttrDict
cfg = config.cfg
device = cfg.device

class UnkGenerator(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config.UnkGeneratorCfg

        inputs = pipe.parser_input
        tgtwords = get_word_alltag(inputs.word_alphabet)
        self.tgtwords = tgtwords
        self.lut = get_lut(inputs.word_alphabet, inputs.char_alphabet).long().to(config.cfg.device)
        self.cfg = cfg

    def update_emb_weight(self, _, __):
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
        obf_word[unk_mask & pri_mask] = UNK_ID
        ret = MrDict(fixed=False, blob=True)
        ret.obf_word = obf_word
        ret.pri_mask = pri_mask
        ret.obf_mask = pri_mask & unk_mask
        ret.obf_pos = inp_pos
        ret.obf_char = word_to_char(obf_word, self.lut).type_as(inp_char)
        ret.cpy_mask = inp_mask.byte() ^ ret.obf_mask

        ret.fix()
        return ret

class _UnkGenerator(nn.Module):
    def __init__(self, keywords, tgtwords, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config.UnkGeneratorCfg
        self.cfg = cfg
        self.keywords = keywords
        self.keywords_set = set(keywords)
        self.tgtwords = torch.LongTensor(tgtwords)
        inputs = pipe.parser_input
        self.lut = get_lut(inputs.word_alphabet, inputs.char_alphabet).long().to(cfg.device)

    def update_emb_weight(self, _):
        pass

    def forward(self, word, char, pos):
        idx = tensor_map(pos, lambda p: pipe.parser_input.pos_alphabet.get_instance(p) in cfg_g.xpos)
        # print(idx.float().sum().item())
        if self.cfg.random:
            N = int(idx.sum().item())
            M = len(self.tgtwords)
            cdt = self.tgtwords[torch.randint(0, M, size=(N,)).long()]
            cdt = cdt.type_as(word)
            obf_word = word.clone()
            obf_word.masked_scatter_(idx.byte(), cdt)
        else:
            obf_word = word.masked_fill(idx.byte(), UNK_ID)
        obf_char = word_to_char(obf_word, self.lut).type_as(char)
        return MrDict({"obf_word": obf_word, "obf_char": obf_char, "obf_pos": pos, "obf_mask": idx}, fixed=True, blob=True)


class AlltagRandomGenerator(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = AttrDict({
                "ctx_obf_rate": 0,
                "pri_obf_rate": 1
            })
        self.cfg = cfg

        from data.input import get_word_alltag
        tgtwords = get_word_alltag(pipe.parser_input.word_alphabet)
        print("tgtwords initiated")
        for pos in tgtwords.keys():
            print("{}-{}".format(pos, len(tgtwords[pos])), end=" ")
        print()
 
        self.tgtwords = tgtwords
        self.lut = get_lut(pipe.parser_input.word_alphabet, pipe.parser_input.char_alphabet).long().to(config.cfg.device)

    def update_emb_weight(self, _, __):
        pass

    def forward(self, inp_word, inp_char, inp_pos, inp_mask=None, reinforce=None):
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


class SearchBasedGenerator(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        from data.input import get_word_alltag
        tgtwords = get_word_alltag(pipe.parser_input.word_alphabet)
        self.tgtwords = tgtwords
        self.lut = get_lut(pipe.parser_input.word_alphabet, pipe.parser_input.char_alphabet).long().to(config.cfg.device)
        self.psr_emb_lut = {}
        self.psr_weight = None

    def update_emb_weight(self, psr_weight):
        self.psr_emb_lut = {}
        self.psr_weight = psr_weight.clone()
        for pos, words in self.tgtwords.items():
            self.psr_emb_lut[pos] = psr_weight[words]

    def closest_words(self, word, pos, num=20):
        pos = pipe.parser_input.pos_alphabet.get_instance(pos.item())
        embedd = self.psr_weight[word]
        others = self.psr_emb_lut[pos]
        dists = torch.nn.functional.cosine_similarity(embedd.unsqueeze(0), others)
        _, sorted_idx = torch.sort(dists)

        return [w.item() for w in self.tgtwords[pos][sorted_idx[:num]]]

    def search(self, word, lens, candid, top_n=40000, obf_rate=0.6):
        # god bless me
        
        padded_to = word.size(0)

        def too_close(sent):
            cnt = 0
            for i, w in enumerate(sent):
                if w == word[i]:
                    cnt += 1
            return (1 - cnt / lens) < obf_rate

        f = defaultdict(list)
        f[0] = [[2]]
        from tqdm import tqdm
        for i in tqdm(range(1, lens)):
            g = defaultdict(list)
            c = candid[i]
            for d, w in enumerate(c):
                for j in sorted(f.keys()):
                    for s in f[j]:
                        g[j + d].append(s + [w])
            cnt = 0
            f = defaultdict(list)
            for j in sorted(g.keys()):
                for s in g[j]:
                    if not too_close(s) and cnt < top_n:
                        f[j].append(s)
                        cnt += 1
                    if cnt == top_n:
                        break
                if cnt == top_n:
                    break
        ret = []
        for j in sorted(f.keys()):
            for s in f[j]:
                ret.append(s + [PAD_ID_WORD] * (padded_to - lens))
        return ret

    def forward(self, inp_word, inp_char, inp_pos, inp_mask):
        bs, ls = inp_word.shape
        assert bs == 1
        # it is better that bs == 1..
        # 我实在是不知道怎么用python写算法题（捂脸

        i = 0
        lens = int(inp_mask[i].sum().item())
        candid = [0] + [self.closest_words(inp_word[i][j], inp_pos[i][j]) for j in range(1, lens)]
        obfuscated = self.search(inp_word[i], lens, candid)
        return MrDict({
            "obf_word": torch.tensor(obfuscated).type_as(inp_word)
            }, blob=True, fixed=False)


