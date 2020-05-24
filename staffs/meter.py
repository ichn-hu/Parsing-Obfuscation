# pylint: disable=invalid-name, missing-docstring
import sys
import torch
import numpy as np
from data import pipe
from data.fileio import PAD_ID_WORD, ROOT, Alphabet, read_inputs
from utils import duplicate_tensor_to_cpu, dynamic_cat_sent
from collections import defaultdict
import nltk
from nltk import FreqDist
from config.utils import MrDict

class Meter(object):
    width = 80
    def __init__(self):
        self.printed = False

    def report(self):
        print(self)

    def is_better_than(self, other):
        pass

    def state_dict(self):
        return self.__dict__
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def combine_2dattr(self, attr, rhs, pad=PAD_ID_WORD):
        rhs = rhs.clone()
        if self.__dict__[attr] is None:
            self.__dict__[attr] = rhs
        else:
            lhs = self.__dict__[attr]
            if lhs.size(1) < rhs.size(1): # ls should be padded
                lhs, rhs = rhs, lhs
            if lhs.size(1) != rhs.size(1):
                pad_rhs = rhs.new_full([rhs.size(0), lhs.size(1) - rhs.size(1)], fill_value=pad)
                rhs = torch.cat([rhs, pad_rhs], dim=1)
            self.__dict__[attr] = torch.cat([lhs, rhs], dim=0)


class ParsingMeter(Meter):
    def __init__(self):
        super().__init__()
        self.tgt_arcs = None
        self.tgt_rels = None
        self.masks = None
        self.pred_arcs = None
        self.pred_rels = None
        self.uas = 0
        self.las = 0

    def measure(self, inp, tgt, oup):
        pred_arcs = oup["arcs"].cpu()
        pred_rels = oup["rels"].cpu()
        tgt_arcs = tgt["arcs"].cpu()
        tgt_rels = tgt["rels"].cpu()
        masks = inp[3].cpu()
        if self.pred_arcs is None:
            self.pred_arcs = pred_arcs
            self.pred_rels = pred_rels
            self.tgt_arcs = tgt_arcs
            self.tgt_rels = tgt_rels
            self.masks = masks
        else:
            self.pred_rels = dynamic_cat_sent(self.pred_rels, pred_rels)
            self.pred_arcs = dynamic_cat_sent(self.pred_arcs, pred_arcs)
            self.tgt_arcs = dynamic_cat_sent(self.tgt_arcs, tgt_arcs)
            self.tgt_rels = dynamic_cat_sent(self.tgt_rels, tgt_rels)
            self.masks = dynamic_cat_sent(self.masks, masks, 0.0)

    def report(self):
        # import ipdb
        # ipdb.set_trace()
        self.masks[:, 0] = 0.0
        correct_arcs = (self.pred_arcs == self.tgt_arcs).float() * self.masks
        correct_rels = (self.pred_rels == self.tgt_rels).float() * self.masks
        uas = (correct_arcs).sum() / self.masks.sum()
        las = (correct_arcs * correct_rels).sum() / self.masks.sum()
        self.uas = uas
        self.las = las
        print("Meter result: uas {:.4f}% las {:.4f}%".format(uas.item() * 100, las.item() * 100))

    def is_better_than(self, other):
        if other is None:
            return True
        return self.uas > other.uas or (self.uas == other.uas and self.las > other.las)


class AdvMeter(Meter):
    def __init__(self):
        super().__init__()
        self.ori_word = None
        self.rcv_word = None
        self.obf_word = None
        self.inp_mask = None
        self.obf_mask = None
        self.pri_mask = None
        self.atk_acc = 0

    def measure(self, inp, tgt, oup):
        self.combine_2dattr("obf_word", oup.obf_word)
        self.combine_2dattr("rcv_word", oup.rcv_word)
        self.combine_2dattr("ori_word", oup.ori_word)
        self.combine_2dattr("obf_mask", oup.obf_mask, pad=0)
        self.combine_2dattr("pri_mask", oup.pri_mask, pad=0)
        self.combine_2dattr("inp_mask", oup.inp_mask, pad=0)

    def analysis(self):
        inputs = pipe.parser_input
        match_mask = (self.ori_word == self.rcv_word)
        rcv_mask = match_mask & self.pri_mask
        rcv = self.rcv_word[rcv_mask]
        fq_rcv = FreqDist([t.item() for t in rcv])
        to_word = inputs.word_alphabet.get_instance
        to_char = inputs.char_alphabet.get_instance

        att_scr = 0
        bs, ls = rcv_mask.shape
        for i in range(bs):
            for j in range(ls):
                if rcv_mask[i][j] > 0:
                    att_scr += pipe.NNP_vocab.get_score(self.rcv_word[i][j].item())

        print("Attack Score: {:.4f} norm: {:.4f}".format(att_scr, att_scr / self.pri_mask.sum().item()))
        
        rcv_cnt = rcv_mask.sum().item()
        pri_cnt = self.pri_mask.sum().item()
        rcv_rate = rcv_cnt / pri_cnt
        def show_most_common():
            rcv_info = []
            for i, j in fq_rcv.most_common():
                rcv_info.append("%10s %4d %.2f%% |" % (to_word(i), j, j / rcv_cnt * 100))
            for j, i in enumerate(rcv_info):
                print(i, end=' ')
                if (j + 1) % 8 == 0:
                    print()

        show_most_common()
        import ipdb
        ipdb.set_trace()

        # rcv_word_list = list(torch.masked_select(self.ori_word, rcv_mask))
        # rcv_fd = FreqDist(rcv_word_list)
        # word_alphabet = pipe.parser_input.word_alphabet

        print("rcv/pri: {}/{}={:.4f}".format(rcv_cnt, pri_cnt, rcv_rate))
        self.atk_acc = rcv_rate

    def report(self):
        match_mask = (self.ori_word == self.rcv_word)
        rcv_mask = match_mask & self.pri_mask
        rcv_cnt = rcv_mask.sum().item()
        pri_cnt = self.pri_mask.sum().item()
        rcv_rate = rcv_cnt / pri_cnt

        # rcv_word_list = list(torch.masked_select(self.ori_word, rcv_mask))
        # rcv_fd = FreqDist(rcv_word_list)
        # word_alphabet = pipe.parser_input.word_alphabet

        print("rcv/pri: {}/{}={:.4f}".format(rcv_cnt, pri_cnt, rcv_rate))
        self.atk_acc = rcv_rate


    def is_better_than(self, other):
        if other is None:
            return True
        return self.atk_acc > other.atk_acc


class ObfGanMeter(Meter):
    def __init__(self):
        super().__init__()
        self.uas = 0
        self.rcv_rate = 1
        self.uas_bar = 0.7
        # maximize UAS and minimize rcv_rate
        # if uas > uas_bar, minimize the rcv_rate

        self.ori_word = None
        self.obf_word = None
        self.rcv_word = None

        self.inp_mask = None
        self.obf_mask = None
        self.pri_mask = None
        self.rcv_mask = None
        self.cpy_mask = None

        self.ori_rels = None
        self.ori_arcs = None
        self.obf_rels = None
        self.obf_arcs = None
        self.gld_rels = None
        self.gld_arcs = None

    def measure(self, inp, tgt, oup):
        self.combine_2dattr("ori_word", inp[0])
        self.combine_2dattr("obf_word", oup["obf_word"])
        self.combine_2dattr("rcv_word", oup["rcv_word"])

        self.combine_2dattr("inp_mask", inp[3].byte(), pad=0)
        self.combine_2dattr("obf_mask", oup["obf_mask"], pad=0)
        self.combine_2dattr("pri_mask", oup["pri_mask"], pad=0)
        self.combine_2dattr("rcv_mask", oup["rcv_mask"], pad=0)
        self.combine_2dattr("cpy_mask", oup["cpy_mask"], pad=0)

        self.combine_2dattr("ori_rels", oup["ori_rels"])
        self.combine_2dattr("ori_arcs", oup["ori_arcs"])
        self.combine_2dattr("obf_rels", oup["obf_rels"])
        self.combine_2dattr("obf_arcs", oup["obf_arcs"])
        self.combine_2dattr("gld_rels", tgt["rels"])
        self.combine_2dattr("gld_arcs", tgt["arcs"])

    def parsing_score(self, pred_arcs, pred_rels):
        inp_mask = self.inp_mask.clone()
        inp_mask[:, 0] = 0 # don't count root
        matched_arcs = (self.gld_arcs == pred_arcs) & inp_mask
        matched_rels = (self.gld_rels == pred_rels) & inp_mask
        matched = matched_arcs & matched_rels
        UAS = matched_arcs.sum().item() / inp_mask.sum().item()
        LAS = matched.sum().item() / inp_mask.sum().item()
        matched_sent = matched_arcs.sum(dim=1) == inp_mask.sum(dim=1)
        UEM = matched_sent.sum().item() / inp_mask.shape[0]
        LEM = (matched_sent & (matched_rels.sum(dim=1) == inp_mask.sum(dim=1))).sum().item() / inp_mask.shape[0]
        return UAS, LAS, UEM, LEM


    def report(self):
        ori_score = self.parsing_score(self.ori_arcs, self.ori_rels)
        obf_score = self.parsing_score(self.obf_arcs, self.obf_rels)

        info = MrDict(fixed=False)
        info.rcv_num = self.rcv_mask.sum().item()
        info.rcv_rate = info.rcv_num / self.pri_mask.sum().item()

        info.cpy_num = self.cpy_mask.sum().item()
        info.cpy_rate = info.cpy_num / self.inp_mask.sum().item()

        info.pri_cpy_num = (self.cpy_mask & self.pri_mask).sum().item()
        info.pri_cpy_rate = info.pri_cpy_num / self.pri_mask.sum().item()

        info.obf_num = (self.obf_mask & self.inp_mask).sum().item()
        info.obf_rate = info.obf_num / self.inp_mask.sum().item()

        info.pri_obf_num = (self.obf_mask & self.pri_mask).sum().item()
        info.pri_obf_rate = info.pri_obf_num / self.pri_mask.sum().item()

        self.uas = obf_score[0]
        self.rcv_rate = info.rcv_rate
        def print_score(flag, score):
            score = [s * 100 for s in score]
            print(flag + "UAS {:.2f}% LAS {:.2f}% UEM {:.2f}% LEM {:.2f}%".format(*score))

        def show_rate(flag):
            return flag + " {}/{:.2f}%".format(
                info[flag + '_num'],
                info[flag + '_rate'] * 100)

        print_score("ori ", ori_score)
        print_score("obf ", obf_score)
        print("num/rate", " ".join([show_rate(flag) for flag in ('rcv', 'cpy', 'pri_cpy', 'obf', 'pri_obf')]))

    def is_better_than(self, other):
        if other is None:
            return True
        if self.uas > self.uas_bar and other.uas > self.uas_bar:
            return self.rcv_rate < other.rcv_rate
        return self.uas > other.uas


class ObfMeter(Meter):
    def __init__(self):
        super().__init__()
        self.uas = 0
        self.rcv_rate = 1
        self.uas_bar = 0.7
        # maximize UAS and minimize rcv_rate
        # if uas > uas_bar, minimize the rcv_rate

        self.ori_word = None
        self.obf_word = None
        self.rcv_word = None

        self.inp_mask = None
        self.obf_mask = None
        self.pri_mask = None
        self.rcv_mask = None
        self.cpy_mask = None

        self.ori_rels = None
        self.ori_arcs = None
        self.obf_rels = None
        self.obf_arcs = None
        self.gld_rels = None
        self.gld_arcs = None

    def measure(self, inp, tgt, oup):
        self.combine_2dattr("ori_word", inp[0])
        self.combine_2dattr("obf_word", oup["obf_word"])

        self.combine_2dattr("inp_mask", inp[3].byte(), pad=0)
        self.combine_2dattr("obf_mask", oup["obf_mask"], pad=0)
        self.combine_2dattr("pri_mask", oup["pri_mask"], pad=0)
        self.combine_2dattr("cpy_mask", oup["cpy_mask"], pad=0)

        self.combine_2dattr("ori_rels", oup["ori_rels"])
        self.combine_2dattr("ori_arcs", oup["ori_arcs"])
        self.combine_2dattr("obf_rels", oup["obf_rels"])
        self.combine_2dattr("obf_arcs", oup["obf_arcs"])
        self.combine_2dattr("gld_rels", tgt["rels"])
        self.combine_2dattr("gld_arcs", tgt["arcs"])

    def parsing_score(self, pred_arcs, pred_rels):
        inp_mask = self.inp_mask.clone()
        inp_mask[:, 0] = 0 # don't count root
        matched_arcs = (self.gld_arcs == pred_arcs) & inp_mask
        matched_rels = (self.gld_rels == pred_rels) & inp_mask
        matched = matched_arcs & matched_rels
        UAS = matched_arcs.sum().item() / inp_mask.sum().item()
        LAS = matched.sum().item() / inp_mask.sum().item()
        matched_sent = matched_arcs.sum(dim=1) == inp_mask.sum(dim=1)
        UEM = matched_sent.sum().item() / inp_mask.shape[0]
        LEM = (matched_sent & (matched_rels.sum(dim=1) == inp_mask.sum(dim=1))).sum().item() / inp_mask.shape[0]
        return UAS, LAS, UEM, LEM

    def analysis(self):
        pred_arcs = self.obf_arcs
        pred_rels = self.obf_rels
        inputs = pipe.parser_input
        to_word = inputs.word_alphabet.get_instance
        to_char = inputs.char_alphabet.get_instance
 
        mch_mask = (self.obf_word == self.ori_word) & self.inp_mask
        mch_words = self.ori_word[mch_mask]
        mch_word_fd = FreqDist()
        for word in mch_words:
            mch_word_fd[to_word(word.item())] += 1
 
        import ipdb
        ipdb.set_trace()       

        inp_mask = self.inp_mask.clone()
        inp_mask[:, 0] = 0 # don't count root
        matched_arcs = (self.gld_arcs == pred_arcs) & inp_mask
        matched_rels = (self.gld_rels == pred_rels) & inp_mask
        matched = matched_arcs & matched_rels
        matched_sent_mask = matched_arcs.sum(dim=1) == inp_mask.sum(dim=1)

        def print_parallel(ori, obf):
            ori_tokens = []
            obf_tokens = []
            for i, wid in enumerate(ori):
                if wid.item() == PAD_ID_WORD:
                    break
                ori_tokens.append(to_word(wid.item()))
                obf_tokens.append(to_word(obf[i].item()))
            if len(ori_tokens) != len(obf_tokens):
                import ipdb
                ipdb.set_trace()
                print(ori_tokens)
                print(obf_tokens)
            spaces = [max(len(ori_tokens[i]), len(obf_tokens[i])) for i in range(len(ori_tokens))]
            ori_str = ""
            obf_str = ""
            obf_cnt = 0
            for i in range(len(ori_tokens)):
                if ori_tokens[i] != obf_tokens[i]:
                    obf_cnt += 1
                ori_str += ori_tokens[i] + (spaces[i] - len(ori_tokens[i]) + 1) * ' '
                obf_str += obf_tokens[i] + (spaces[i] - len(obf_tokens[i]) + 1) * ' '
            print("obf_cnt: {} obf_rate: {:.2f}%".format(obf_cnt, obf_cnt / len(obf_tokens) * 100))
            print("ori: {}".format(ori_str))
            print("obf: {}".format(obf_str))

        bs, ls = self.obf_word.shape
        for i in range(bs):
            if matched_sent_mask[i] > 0:
                ori_sent = self.ori_word[i]
                obf_sent = self.obf_word[i]
                print_parallel(ori_sent, obf_sent)
        import ipdb
        ipdb.set_trace()
        print("GG")

    def report(self):
        ori_score = self.parsing_score(self.ori_arcs, self.ori_rels)
        obf_score = self.parsing_score(self.obf_arcs, self.obf_rels)

        info = MrDict(fixed=False)
        
#        self.analysis()

        info.mch_num = ((self.obf_word == self.ori_word) & self.inp_mask).sum().item()
        info.mch_rate = info.mch_num / self.inp_mask.sum().item()

        info.cpy_num = self.cpy_mask.sum().item()
        info.cpy_rate = info.cpy_num / self.inp_mask.sum().item()

        info.pri_cpy_num = (self.cpy_mask & self.pri_mask).sum().item()
        info.pri_cpy_rate = info.pri_cpy_num / self.pri_mask.sum().item()

        info.obf_num = (self.obf_mask & self.inp_mask).sum().item()
        info.obf_rate = info.obf_num / self.inp_mask.sum().item()

        info.pri_obf_num = (self.obf_mask & self.pri_mask).sum().item()
        info.pri_obf_rate = info.pri_obf_num / self.pri_mask.sum().item()

        self.uas = obf_score[0]
        def print_score(flag, score):
            score = [s * 100 for s in score]
            print(flag + "UAS {:.2f}% LAS {:.2f}% UEM {:.2f}% LEM {:.2f}%".format(*score))

        def show_rate(flag):
            return flag + " {}/{:.2f}%".format(
                info[flag + '_num'],
                info[flag + '_rate'] * 100)

        print_score("ori ", ori_score)
        print_score("obf ", obf_score)
        print("num/rate", " ".join([show_rate(flag) for flag in ('mch', 'cpy', 'pri_cpy', 'obf', 'pri_obf')]))

    def is_better_than(self, other):
        if other is None:
            return True
        return self.uas > other.uas


class AttackingMeter(Meter):
    def __init__(self):
        super().__init__()
        self.ori_word = None
        self.obf_word = None
        self.rcv_word = None
        self.masks = None
        self.obf_mask = None
        self.acc = 0
        self.obf_acc = 0

    def measure(self, inp, tgt, oup):
        self.combine_2dattr("obf_word", inp[0])  # inp[0] is the obf_word by generator
        self.combine_2dattr("ori_word", tgt["ori_word"])
        self.combine_2dattr("rcv_word", oup["rcv_word"])
        self.combine_2dattr("masks", inp[3], 0)
        self.combine_2dattr("obf_mask", inp[4], 0)

    def report(self):
        match = (self.ori_word == self.rcv_word).float() * self.obf_mask.float()
        tot_recovered = self.obf_mask.float().sum()

        obf_acc = match.sum() / tot_recovered
        recovered_words = list(torch.masked_select(self.ori_word, match.byte()))
        freq_dist = FreqDist(recovered_words)
        freq_info = ""
        word_alphabet = pipe.parser_input.word_alphabet
        for word_id, freq in freq_dist.most_common(10):
            freq_info += "{}/{}/{:.4f} ".format(word_alphabet.get_instance(word_id), freq, freq / tot_recovered)
        freq_info += "\n"
        print(freq_info)
        print("rcv/obf: {}/{} Obf Recovery acc: {:.2f}%".format(
            match.sum().item(), self.obf_mask.sum().item(),
            obf_acc * 100
        ))
        # import ipdb
        # ipdb.set_trace()
        self.obf_acc = obf_acc

    def is_better_than(self, other):
        if other is None:
            return True
        return self.obf_acc > other.obf_acc


class KpffMeter(Meter):
    def __init__(self):
        self.uas = -1
        self.ho, self.h, self.wo, self.w, self.ro, self.r, self.hg, self.rg, self.msk = \
            None, None, None, None, None, None, None, None, None

    def measure(self, inp, tgt, oup):
        """
        :param ho: head obfuscated, (bs, len)
        :param ro: rels obfuscated, (bs, len)
        :param wo: word obfuscated, (bs, len)
        :param h:
        :param r:
        :param w:
        :param hg: head gold
        :param rg: rels gold
        :return:
        """
        self.combine_2dattr("ho", oup["obf_arcs"], pad=-1)
        self.combine_2dattr("ro", oup["obf_rels"], pad=-1)
        self.combine_2dattr("hg", tgt["arcs"], pad=-1)
        self.combine_2dattr("rg", tgt["rels"], pad=-1)
        self.combine_2dattr("h", oup["ori_arcs"], pad=-1)
        self.combine_2dattr("r", oup["ori_rels"], pad=-1)
        self.combine_2dattr("w", inp[0])
        self.combine_2dattr("wo", oup["obf_word"])
        self.combine_2dattr("msk", oup["obf_mask"], pad=0)

    def is_better_than(self, other, criterion="err"):
        """
        return True if other is better than self
        """
        if other is None:
            return True
        if self.uas > other.uas:
            return True
        return False

    def neighbour_analysis(self, obf_arc, obf_rel):
        arc = self.hg
        rel = self.rg
        # obf_arc = self.ho
        # obf_rel = self.ro
        obf_msk = self.msk.byte()
        # msk = (self.w != PAD_ID_WORD) & (self.w != pipe.parser_input.word_alphabet.get_index(ROOT))

        bs, ls = obf_msk.shape
        
        gld_arc_set = defaultdict(set)
        obf_arc_set = defaultdict(set)
        cor_cnt = 0
        pre_cnt = 0
        rec_cnt = 0
        for b in range(bs):
            for i in range(ls):
                j = arc[b, i].item()
                gld_arc_set[(b, i)].add(j)
                gld_arc_set[(b, j)].add(i)

                j = obf_arc[b, i].item()
                obf_arc_set[(b, i)].add(j)
                obf_arc_set[(b, j)].add(i)
            for i in range(ls):
                if obf_msk[b, i].item():
                    gset = gld_arc_set[(b, i)]
                    oset = obf_arc_set[(b, i)]
                    cor = oset.intersection(gset)
                    cor_cnt += len(cor)
                    pre_cnt += len(oset)
                    rec_cnt += len(gset)

        precision = cor_cnt / (pre_cnt + 1e-13)
        recall = cor_cnt / (rec_cnt + 1e-13)
        f1 = 2 * precision * recall / (precision + recall + 1e-13)
        #import ipdb
        #ipdb.set_trace()
        print("cor_cnt {} pre_cnt {} rec_cnt {}".format(cor_cnt, pre_cnt, rec_cnt))
        print("prec {:.4f} recl {:.4f} f1 {:.4f}".format(precision, recall, f1))

    def report(self):
        # self.neighbour_analysis(self.ho, self.ro)
        # self.neighbour_analysis(self.h, self.r)
        h, ho, hg, r, ro, rg, w, wo, msk = self.h, self.ho, self.hg, self.r, self.ro, self.rg, self.w, self.wo, self.msk
        #import ipdb
        #ipdb.set_trace()
        word_mask = (w != PAD_ID_WORD) & (w != pipe.parser_input.word_alphabet.get_index(ROOT))
        mask = word_mask.type_as(h)
        h = h * mask
        ho = ho * mask
        hg = hg * mask
        r = r * mask
        ro = ro * mask
        rg = rg * mask
        w = w * mask
        wo = wo * mask

        tot_sent = w.shape[0]
        tot_word = word_mask.sum().item()
        obf_word_mask = (w != wo) * word_mask
        num_word = obf_word_mask.sum().item()
        obf_sent_mask = obf_word_mask.sum(dim=1) > 0
        num_sent = obf_sent_mask.sum().item()

        def calc(ho, hg, ro, rg, tw, ts):
            err_head_mask = ho != hg
            err_rels_mask = ro != rg
            # skip the symbolic head
            err_mask = (err_head_mask + err_rels_mask) > 0

            uas = 1 - err_head_mask.sum().item() / tw
            las = 1 - err_mask.sum().item() / tw
            uem = (err_head_mask.sum(dim=1) == 0).sum().item() / ts
            lem = (err_mask.sum(dim=1) == 0).sum().item() / ts
            return uas, las, uem, lem

        def collect(w):
            return torch.masked_select(w, obf_sent_mask.reshape(-1, 1)).reshape(num_sent, -1)

        cho, cro, cw, cwo = collect(ho), collect(ro), collect(w), collect(wo)
        tot_sent_obf = num_sent
        tot_word_obf = (cw > 0).sum().item()

        buas, blas, buem, blem = calc(h, hg, r, rg, tot_word, tot_sent)
        uas, las, uem, lem = calc(ho, hg, ro, rg, tot_word, tot_sent)
        puas, plas, puem, plem = calc(ho, h, ro, r, tot_word, tot_sent)
        ouas, olas, ouem, olem = calc(cho, collect(hg), cro, collect(rg), tot_word_obf, tot_sent_obf)
        opuas, oplas, opuem, oplem = calc(cho, collect(h), cro, collect(r), tot_word_obf, tot_sent_obf)
        num_spot = (msk.float() * mask.float()).sum().item()
        num_spot_sent = ((msk.float() * mask.float()).sum(dim=-1) > 0).sum().item()
        wc = num_word / num_spot
        sc = num_sent / num_spot_sent

        def p_ratio(t):
            return "{:4.2f}%".format(t * 100)

        res = ""
        res += "UAS: {} LAS: {} UEM: {} LEM: {}\n".format(p_ratio(buas), p_ratio(blas), p_ratio(buem), p_ratio(blem))
        res += "uas: {} las: {} uem: {} lem: {}\n".format(p_ratio(uas), p_ratio(las), p_ratio(uem), p_ratio(lem))
        res += "puas: {} las: {} uem: {} lem: {}\n".format(p_ratio(puas), p_ratio(plas), p_ratio(puem), p_ratio(plem))
        res += "ouas: {} olas: {} ouem: {} olem: {}\n".format(p_ratio(ouas), p_ratio(olas), p_ratio(ouem), p_ratio(olem))
        res += "opuas: {} oplas: {} opuem: {} oplem: {}\n".format(p_ratio(opuas), p_ratio(oplas), p_ratio(opuem), p_ratio(oplem))
        res += "wc: {}/{}={} sc: {}/{}={}\n".format(num_word, num_spot, p_ratio(wc), num_sent, num_spot_sent, p_ratio(sc))
        self.uas = uas
        print(res)
        return uas

 
class AttMeter(Meter):
    def __init__(self, name="Yet Another Meter"):
        super(AttMeter, self).__init__()
        self.name = name
        self.acc = -1
        self.pred_sent = None
        self.gold_sent = None
        inp = read_inputs()
        self.word_alphabet = inp.word_alphabet
        self.char_alphabet = inp.char_alphabet

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k not in ["word_alphabet", "char_alphabet"]}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        inp = read_inputs()
        self.word_alphabet = inp.word_alphabet
        self.char_alphabet = inp.char_alphabet

    def measure(self, gold_sent, pred_sent):
        if self.gold_sent is None:
            self.gold_sent = gold_sent
            self.pred_sent = pred_sent
        else:
            self.gold_sent = dynamic_cat_sent(self.gold_sent, gold_sent)
            self.pred_sent = dynamic_cat_sent(self.pred_sent, pred_sent)

    def is_better_than(self, other, criterion="err"):
        """
        return True if self is better than other, always true if other is None
        """
        if other is None:
            return True
        if self.acc < 0:
            self.analysis()
        if other.acc < 0:
            other.analysis()
        if self.acc > other.acc:
            return True
        return False

    def analysis(self):
        self.acc = (self.pred_sent == self.gold_sent).float().mean()

        def p_ratio(t):
            return "{:4.2f}%".format(t * 100)

        res = p_ratio(self.acc)
        return res


    def __repr__(self):
        title = " AttMeter: {} ".format(self.name)
        if len(title) % 2 == 1: title += ' '
        margin_width = (self.width - len(title)) // 2
        repr = ""
        repr += "*" * margin_width +  title + "*" * margin_width + "\n"
        repr += self.analysis()
        repr += "*" * self.width
        return repr


class HybridMeter(Meter):
    def __init__(self):
        super().__init__()
        self.uas = 0
        self.rcv_rate = 1
        self.uas_bar = 0.90
        self.atk_acc = 0
        # maximize UAS and minimize rcv_rate
        # if uas > uas_bar, minimize the rcv_rate

        self.ori_word = None
        self.obf_word = None
        self.rcv_word = None

        self.inp_mask = None
        self.obf_mask = None
        self.pri_mask = None
        self.rcv_mask = None
        self.cpy_mask = None

        self.ori_rels = None
        self.ori_arcs = None
        self.obf_rels = None
        self.obf_arcs = None
        self.gld_rels = None
        self.gld_arcs = None

    def measure(self, inp, tgt, oup):
        self.combine_2dattr("ori_word", inp[0])
        self.combine_2dattr("obf_word", oup["obf_word"])
        self.combine_2dattr("rcv_word", oup["rcv_word"])

        self.combine_2dattr("inp_mask", inp[3].byte(), pad=0)
        self.combine_2dattr("obf_mask", oup["obf_mask"], pad=0)
        self.combine_2dattr("pri_mask", oup["pri_mask"], pad=0)
        self.combine_2dattr("cpy_mask", oup["cpy_mask"], pad=0)

        self.combine_2dattr("ori_rels", oup["ori_rels"])
        self.combine_2dattr("ori_arcs", oup["ori_arcs"])
        self.combine_2dattr("obf_rels", oup["obf_rels"])
        self.combine_2dattr("obf_arcs", oup["obf_arcs"])
        self.combine_2dattr("gld_rels", tgt["rels"])
        self.combine_2dattr("gld_arcs", tgt["arcs"])

    def parsing_score(self, pred_arcs, pred_rels):
        inp_mask = self.inp_mask.clone()
        inp_mask[:, 0] = 0 # don't count root
        matched_arcs = (self.gld_arcs == pred_arcs) & inp_mask
        matched_rels = (self.gld_rels == pred_rels) & inp_mask
        matched = matched_arcs & matched_rels
        UAS = matched_arcs.sum().item() / inp_mask.sum().item()
        LAS = matched.sum().item() / inp_mask.sum().item()
        matched_sent = matched_arcs.sum(dim=1) == inp_mask.sum(dim=1)
        UEM = matched_sent.sum().item() / inp_mask.shape[0]
        LEM = (matched_sent & (matched_rels.sum(dim=1) == inp_mask.sum(dim=1))).sum().item() / inp_mask.shape[0]
        return UAS, LAS, UEM, LEM

    def analysis(self):
        pred_arcs = self.obf_arcs
        pred_rels = self.obf_rels
        inputs = pipe.parser_input
        to_word = inputs.word_alphabet.get_instance
        to_char = inputs.char_alphabet.get_instance
 
        mch_mask = (self.obf_word == self.ori_word) & self.inp_mask
        mch_words = self.ori_word[mch_mask]
        mch_word_fd = FreqDist()
        for word in mch_words:
            mch_word_fd[to_word(word.item())] += 1
 
        import ipdb
        ipdb.set_trace()       

        inp_mask = self.inp_mask.clone()
        inp_mask[:, 0] = 0 # don't count root
        matched_arcs = (self.gld_arcs == pred_arcs) & inp_mask
        matched_rels = (self.gld_rels == pred_rels) & inp_mask
        matched = matched_arcs & matched_rels
        matched_sent_mask = matched_arcs.sum(dim=1) == inp_mask.sum(dim=1)

        def print_parallel(ori, obf):
            ori_tokens = []
            obf_tokens = []
            for i, wid in enumerate(ori):
                if wid.item() == PAD_ID_WORD:
                    break
                ori_tokens.append(to_word(wid.item()))
                obf_tokens.append(to_word(obf[i].item()))
            if len(ori_tokens) != len(obf_tokens):
                import ipdb
                ipdb.set_trace()
                print(ori_tokens)
                print(obf_tokens)
            spaces = [max(len(ori_tokens[i]), len(obf_tokens[i])) for i in range(len(ori_tokens))]
            ori_str = ""
            obf_str = ""
            obf_cnt = 0
            for i in range(len(ori_tokens)):
                if ori_tokens[i] != obf_tokens[i]:
                    obf_cnt += 1
                ori_str += ori_tokens[i] + (spaces[i] - len(ori_tokens[i]) + 1) * ' '
                obf_str += obf_tokens[i] + (spaces[i] - len(obf_tokens[i]) + 1) * ' '
            print("obf_cnt: {} obf_rate: {:.2f}%".format(obf_cnt, obf_cnt / len(obf_tokens) * 100))
            print("ori: {}".format(ori_str))
            print("obf: {}".format(obf_str))

        bs, ls = self.obf_word.shape
        for i in range(bs):
            if matched_sent_mask[i] > 0:
                ori_sent = self.ori_word[i]
                obf_sent = self.obf_word[i]
                print_parallel(ori_sent, obf_sent)
        import ipdb
        ipdb.set_trace()
        print("GG")

    def report(self):
        ori_score = self.parsing_score(self.ori_arcs, self.ori_rels)
        obf_score = self.parsing_score(self.obf_arcs, self.obf_rels)

        info = MrDict(fixed=False)
        
#        self.analysis()

        info.mch_num = ((self.obf_word == self.ori_word) & self.inp_mask).sum().item()
        info.mch_rate = info.mch_num / self.inp_mask.sum().item()

        info.cpy_num = self.cpy_mask.sum().item()
        info.cpy_rate = info.cpy_num / self.inp_mask.sum().item()

        info.pri_cpy_num = (self.cpy_mask & self.pri_mask).sum().item()
        info.pri_cpy_rate = info.pri_cpy_num / self.pri_mask.sum().item()

        info.obf_num = (self.obf_mask & self.inp_mask).sum().item()
        info.obf_rate = info.obf_num / self.inp_mask.sum().item()

        info.pri_obf_num = (self.obf_mask & self.pri_mask).sum().item()
        info.pri_obf_rate = info.pri_obf_num / self.pri_mask.sum().item()

        self.uas = obf_score[0]
        def print_score(flag, score):
            score = [s * 100 for s in score]
            print(flag + "UAS {:.2f}% LAS {:.2f}% UEM {:.2f}% LEM {:.2f}%".format(*score))

        def show_rate(flag):
            return flag + " {}/{:.2f}%".format(
                info[flag + '_num'],
                info[flag + '_rate'] * 100)

        print_score("ori ", ori_score)
        print_score("obf ", obf_score)
        print("num/rate", " ".join([show_rate(flag) for flag in ('mch', 'cpy', 'pri_cpy', 'obf', 'pri_obf')]))

        match_mask = (self.ori_word == self.rcv_word)
        rcv_mask = match_mask & self.pri_mask
        rcv_cnt = rcv_mask.sum().item()
        pri_cnt = self.pri_mask.sum().item()
        rcv_rate = rcv_cnt / pri_cnt

        # rcv_word_list = list(torch.masked_select(self.ori_word, rcv_mask))
        # rcv_fd = FreqDist(rcv_word_list)
        # word_alphabet = pipe.parser_input.word_alphabet

        print("rcv/pri: {}/{}={:.4f}".format(rcv_cnt, pri_cnt, rcv_rate))
        self.rcv_rate = rcv_rate
        self.atk_acc = rcv_rate

        n_inst = self.ori_word.size(0)

        def show(ith):
            show_example(self.ori_word[ith], self.obf_word[ith], self.rcv_word[ith])
        
        show(np.random.randint(n_inst))

        # import ipdb
        # ipdb.set_trace()
        # print("GG")

    def is_better_than(self, other):
        if other is None:
            return True
        if self.uas > self.uas_bar and other.uas > self.uas_bar:
            return self.rcv_rate < other.rcv_rate
        return self.uas > other.uas

def show_example(ori, obf, rcv):
    inputs = pipe.parser_input
    to_word = inputs.word_alphabet.get_instance
    to_char = inputs.char_alphabet.get_instance

    ori_tokens = []
    obf_tokens = []
    rcv_tokens = []
    for i, wid in enumerate(ori):
        if wid.item() == PAD_ID_WORD:
            break
        ori_tokens.append(to_word(wid.item()))
        obf_tokens.append(to_word(obf[i].item()))
        if rcv is not None:
            rcv_tokens.append(to_word(rcv[i].item()))
        else:
            rcv_tokens.append("_")
    
    n = len(ori_tokens)
    spaces = [max(len(ori_tokens[i]), len(obf_tokens[i]), len(rcv_tokens[i])) for i in range(n)]
    place_holders = ["{:%d}" % i for i in spaces]
    import termcolor
    print("=" * 80)
    print("[ori] ", end="")
    for i in range(n):
        print(place_holders[i].format(ori_tokens[i]), end=" ")
    print("\n[obf] ", end="")
    for i in range(n):
        to_print = place_holders[i].format(obf_tokens[i])
        if obf_tokens[i] != ori_tokens[i]:
            to_print = termcolor.colored(to_print, "red")
        print(to_print, end=" ")
    if rcv is not None:
        print("\n[rcv] ", end="")
        for i in range(n):
            # token = rcv_tokens[i] if rcv_tokens[i] == ori_tokens[i] else "_"
            to_print = place_holders[i].format(rcv_tokens[i])
            if rcv_tokens[i] == ori_tokens[i] and obf_tokens[i] != ori_tokens[i]:
                to_print = termcolor.colored(to_print, "green")
            print(to_print, end=" ")
    print()

class TagSpecMeter(Meter):
    def __init__(self):
        super().__init__()
        self.uas = 0
        # self.rcv_rate = 1
        # self.uas_bar = 0.7
        # maximize UAS and minimize rcv_rate
        # if uas > uas_bar, minimize the rcv_rate

        self.ori_word = None
        self.obf_word = None
        # self.rcv_word = None

        self.inp_mask = None
        self.obf_mask = None
        # self.pri_mask = None
        # self.rcv_mask = None
        # self.cpy_mask = None

        self.ori_rels = None
        self.ori_arcs = None
        self.obf_rels = None
        self.obf_arcs = None
        self.gld_rels = None
        self.gld_arcs = None

    def measure(self, inp, tgt, oup):
        self.combine_2dattr("ori_word", inp[0])
        self.combine_2dattr("obf_word", oup["obf_word"])

        self.combine_2dattr("inp_mask", inp[3].byte(), pad=0)
        self.combine_2dattr("obf_mask", oup["obf_mask"], pad=0)
        # self.combine_2dattr("pri_mask", oup["pri_mask"], pad=0)
        # self.combine_2dattr("cpy_mask", oup["cpy_mask"], pad=0)

        self.combine_2dattr("ori_rels", oup["ori_rels"])
        self.combine_2dattr("ori_arcs", oup["ori_arcs"])
        self.combine_2dattr("obf_rels", oup["obf_rels"])
        self.combine_2dattr("obf_arcs", oup["obf_arcs"])
        self.combine_2dattr("gld_rels", tgt["rels"])
        self.combine_2dattr("gld_arcs", tgt["arcs"])

    def parsing_score(self, pred_arcs, pred_rels):
        inp_mask = self.inp_mask.clone()
        inp_mask[:, 0] = 0 # don't count root
        matched_arcs = (self.gld_arcs == pred_arcs) & inp_mask
        matched_rels = (self.gld_rels == pred_rels) & inp_mask
        matched = matched_arcs & matched_rels
        UAS = matched_arcs.sum().item() / inp_mask.sum().item()
        LAS = matched.sum().item() / inp_mask.sum().item()
        matched_sent = matched_arcs.sum(dim=1) == inp_mask.sum(dim=1)
        UEM = matched_sent.sum().item() / inp_mask.shape[0]
        LEM = (matched_sent & (matched_rels.sum(dim=1) == inp_mask.sum(dim=1))).sum().item() / inp_mask.shape[0]
        return UAS, LAS, UEM, LEM

    def analysis(self):
        pred_arcs = self.obf_arcs
        pred_rels = self.obf_rels
        inputs = pipe.parser_input
        to_word = inputs.word_alphabet.get_instance
        to_char = inputs.char_alphabet.get_instance
 
        mch_mask = (self.obf_word == self.ori_word) & self.inp_mask
        mch_words = self.ori_word[mch_mask]
        mch_word_fd = FreqDist()
        for word in mch_words:
            mch_word_fd[to_word(word.item())] += 1
 
        import ipdb
        ipdb.set_trace()       

        inp_mask = self.inp_mask.clone()
        inp_mask[:, 0] = 0 # don't count root
        matched_arcs = (self.gld_arcs == pred_arcs) & inp_mask
        matched_rels = (self.gld_rels == pred_rels) & inp_mask
        matched = matched_arcs & matched_rels
        matched_sent_mask = matched_arcs.sum(dim=1) == inp_mask.sum(dim=1)

        def print_parallel(ori, obf):
            ori_tokens = []
            obf_tokens = []
            for i, wid in enumerate(ori):
                if wid.item() == PAD_ID_WORD:
                    break
                ori_tokens.append(to_word(wid.item()))
                obf_tokens.append(to_word(obf[i].item()))
            if len(ori_tokens) != len(obf_tokens):
                import ipdb
                ipdb.set_trace()
                print(ori_tokens)
                print(obf_tokens)
            spaces = [max(len(ori_tokens[i]), len(obf_tokens[i])) for i in range(len(ori_tokens))]
            ori_str = ""
            obf_str = ""
            obf_cnt = 0
            for i in range(len(ori_tokens)):
                if ori_tokens[i] != obf_tokens[i]:
                    obf_cnt += 1
                ori_str += ori_tokens[i] + (spaces[i] - len(ori_tokens[i]) + 1) * ' '
                obf_str += obf_tokens[i] + (spaces[i] - len(obf_tokens[i]) + 1) * ' '
            print("obf_cnt: {} obf_rate: {:.2f}%".format(obf_cnt, obf_cnt / len(obf_tokens) * 100))
            print("ori: {}".format(ori_str))
            print("obf: {}".format(obf_str))

        bs, ls = self.obf_word.shape
        for i in range(bs):
            if matched_sent_mask[i] > 0:
                ori_sent = self.ori_word[i]
                obf_sent = self.obf_word[i]
                print_parallel(ori_sent, obf_sent)
        import ipdb
        ipdb.set_trace()
        print("GG")

    def report(self):
        ori_score = self.parsing_score(self.ori_arcs, self.ori_rels)
        obf_score = self.parsing_score(self.obf_arcs, self.obf_rels)

        info = MrDict(fixed=False)
        
#        self.analysis()

        info.mch_num = ((self.obf_word == self.ori_word) & self.inp_mask).sum().item()
        info.mch_rate = info.mch_num / self.inp_mask.sum().item()

        # info.cpy_num = self.cpy_mask.sum().item()
        # info.cpy_rate = info.cpy_num / self.inp_mask.sum().item()

        # info.pri_cpy_num = (self.cpy_mask & self.pri_mask).sum().item()
        # info.pri_cpy_rate = info.pri_cpy_num / self.pri_mask.sum().item()

        info.obf_num = (self.obf_mask & self.inp_mask).sum().item()
        info.obf_rate = info.obf_num / self.inp_mask.sum().item()

        # info.pri_obf_num = (self.obf_mask & self.pri_mask).sum().item()
        # info.pri_obf_rate = info.pri_obf_num / self.pri_mask.sum().item()

        self.uas = obf_score[0]
        def print_score(flag, score):
            score = [s * 100 for s in score]
            print(flag + "UAS {:.2f}% LAS {:.2f}% UEM {:.2f}% LEM {:.2f}%".format(*score))

        def show_rate(flag):
            return flag + " {}/{:.2f}%".format(
                info[flag + '_num'],
                info[flag + '_rate'] * 100)

        print_score("ori ", ori_score)
        print_score("obf ", obf_score)
        print("num/rate", " ".join([show_rate(flag) for flag in ('mch', 'obf')]))
        n_inst = self.ori_word.size(0)

        def show(ith):
            show_example(self.ori_word[ith], self.obf_word[ith], None)
        
        show(np.random.randint(n_inst))
        show(np.random.randint(n_inst))
        show(np.random.randint(n_inst))
        print("=" * 80)


    def is_better_than(self, other):
        if other is None:
            return True
        return self.uas > other.uas

