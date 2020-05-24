import sys
sys.path.append('.')
sys.path.append('..')
import torch
import torch.nn as nn
import infra
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, SGD, Adamax
import numpy as np
import pickle
import time
import re
from task.parser import decode_MST
import iomodule.maxio as conllx_data


def freeze_embedding(embedding):
    assert isinstance(embedding, nn.Embedding), "input should be an Embedding module."
    embedding.weight.detach_()


def generate_optimizer(args, params):
    opt = args.optimizer
    lr = args.learning_rate

    params = filter(lambda param: param.requires_grad, params)
    if opt == 'adam':
        return Adam(params, lr=lr, betas=args.betas, weight_decay=args.gamma, eps=args.eps)
    elif opt == 'sgd':
        return SGD(params, lr=lr, momentum=args.momentum, weight_decay=args.gamma, nesterov=True)
    elif opt == 'adamax':
        return Adamax(params, lr=lr, betas=args.betas, weight_decay=args.gamma, eps=args.eps)
    else:
        raise ValueError('Unknown optimization algorithm: %s' % opt)


def read_data():
    """
    TODO: read_data again, make data part reusable! this might be a future feature
    :return:
    """
    pass


class PerformanceMeter(object):
    def __init__(self, which_epoch=0):
        self.ucorr = 0.
        self.lcorr = 0.
        self.total = 0.
        self.ucomplete_match = 0.
        self.lcomplete_match = 0.
        self.ucorr_nopunc = 0.
        self.lcorr_nopunc = 0.
        self.total_nopunc = 0.
        self.ucomplete_match_nopunc = 0.
        self.lcomplete_match_nopunc = 0.
        self.rcorr = 0.
        self.total_root = 0.
        self.total_inst = 0.
        self.epoch = which_epoch

    def eval(self, words, postags, heads_pred, rels_pred, heads, rels, word_alphabet, pos_alphabet, lengths,
             punct_set=None, symbolic_root=False, symbolic_end=False):

        def is_uni_punctuation(word):
            match = re.match("^[^\w\s]+$]", word, flags=re.UNICODE)
            return match is not None

        def is_punctuation(word, pos, punct_set=None):
            if punct_set is None:
                return is_uni_punctuation(word)
            else:
                return pos in punct_set

        batch_size, _ = words.shape
        ucorr = 0.
        lcorr = 0.
        total = 0.
        ucomplete_match = 0.
        lcomplete_match = 0.

        ucorr_nopunc = 0.
        lcorr_nopunc = 0.
        total_nopunc = 0.
        ucomplete_match_nopunc = 0.
        lcomplete_match_nopunc = 0.

        corr_root = 0.
        total_root = 0.
        start = 1 if symbolic_root else 0
        end = 1 if symbolic_end else 0
        for i in range(batch_size):
            ucm = 1.
            lcm = 1.
            ucm_nopunc = 1.
            lcm_nopunc = 1.
            for j in range(start, lengths[i] - end):
                word = word_alphabet.get_instance(words[i, j])
                word = word.encode('utf8')

                pos = pos_alphabet.get_instance(postags[i, j])
                pos = pos.encode('utf8')

                total += 1
                if heads[i, j] == heads_pred[i, j]:
                    ucorr += 1
                    if rels[i, j] == rels_pred[i, j]:
                        lcorr += 1
                    else:
                        lcm = 0
                else:
                    ucm = 0
                    lcm = 0

                if not is_punctuation(word, pos, punct_set):
                    total_nopunc += 1
                    if heads[i, j] == heads_pred[i, j]:
                        ucorr_nopunc += 1
                        if rels[i, j] == rels_pred[i, j]:
                            lcorr_nopunc += 1
                        else:
                            lcm_nopunc = 0
                    else:
                        ucm_nopunc = 0
                        lcm_nopunc = 0

                if heads[i, j] == 0:
                    total_root += 1
                    corr_root += 1 if heads_pred[i, j] == 0 else 0

            ucomplete_match += ucm
            lcomplete_match += lcm
            ucomplete_match_nopunc += ucm_nopunc
            lcomplete_match_nopunc += lcm_nopunc

        self.ucorr += ucorr
        self.lcorr += lcorr
        self.total += total
        self.ucomplete_match += ucomplete_match
        self.lcomplete_match += lcomplete_match
        self.ucorr_nopunc += ucorr_nopunc
        self.lcorr_nopunc += lcorr_nopunc
        self.total_nopunc += total_nopunc
        self.ucomplete_match_nopunc += ucomplete_match_nopunc
        self.lcomplete_match_nopunc += lcomplete_match_nopunc
        self.rcorr += corr_root
        self.total_root += total_root
        self.total_inst += batch_size

    def print(self, prompt=""):
        print('%sW. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
            prompt, self.ucorr, self.lcorr, self.total, self.ucorr * 100 / self.total, self.lcorr * 100 / self.total,
            self.ucomplete_match * 100 / self.total_inst, self.lcomplete_match * 100 / self.total_inst,
            self.epoch))
        print('%sWo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
            prompt, self.ucorr_nopunc, self.lcorr_nopunc, self.total_nopunc,
            self.ucorr_nopunc * 100 / self.total_nopunc, self.lcorr_nopunc * 100 / self.total_nopunc,
            self.ucomplete_match_nopunc * 100 / self.total_inst, self.lcomplete_match_nopunc * 100 / self.total_inst,
            self.epoch))
        print('%sRoot: corr: %d, total: %d, acc: %.2f%% (epoch: %d)' % (
            prompt, self.rcorr, self.total_root, self.rcorr * 100 / self.total_root, self.epoch))


class TrainingWatch(object):
    def __init__(self, args, epoch, total, patience, decay, print_every=10):

        self.epoch = epoch
        self.total = total
        self.print_every = print_every
        self.loss = 0.
        self.loss_arc = 0.
        self.loss_type = 0.
        self.num_inst = 0.
        self.num_back = 0
        self.init_time = time.time()
        self.cnt_batch = 0

    def update(self, loss, loss_arc, loss_type, num_inst):
        self.loss += loss * num_inst
        self.loss_arc += loss_arc * num_inst
        self.loss_type += loss_type * num_inst
        self.num_inst += num_inst
        self.cnt_batch += 1
        if self.cnt_batch % self.print_every == 0:
            self.bark()

    def notify(self):
        try:
            spend = time.time() - self.init_time
            ave_time = spend / self.cnt_batch
            remain = (self.total - self.cnt_batch) * ave_time
            return "train {}/{} loss: total {:.4f} arc {:.4f} type {:.4f}" \
                   " time: spend {:.4f} average {:.4f} remain {:.4f}".format(
                self.cnt_batch, self.total, self.loss / self.num_inst, self.loss_arc / self.num_inst,
                self.loss_type / self.num_inst, spend, ave_time, remain)
        except Exception as e:
            return "notify failed @ " + str(e)

    def bark(self):
        sys.stdout.write("\b" * self.num_back)
        sys.stdout.write(" " * self.num_back)
        sys.stdout.write("\b" * self.num_back)
        notification = self.notify()
        sys.stdout.write(notification)
        sys.stdout.flush()
        self.num_back = len(notification)

    def kill(self):
        self.bark()
        sys.stdout.write("\n")
        sys.stdout.flush()


def loss_arc_from_graph(graph, heads, masks):
    # graph of [batch_size, length, length], heads of [batch_size, length], masks of [batch_size, length]
    batch_size, length = masks.shape

    # graph[masks.unsqueeze(1) * masks.unsqueeze(0) == 0] = -233
    heads[masks == 0] = -233
    heads[:, 0] = -233
    return F.nll_loss(graph.transpose(1, 2).reshape(batch_size * length, length), heads.reshape(-1),
                      size_average=True, ignore_index=-233)


def loss_rels_from_pred(rels_pred, rels, masks):
    # rels_pred of shape [batch, length, num_lable], and rels of shape [batch_size, length]
    batch_size, length, num_label = rels_pred.shape
    # rels_pred[masks.expand(batch_size, length, num_label) == 0] = -233
    rels[masks == 0] = -233
    rels[:, 0] = -233
    return F.nll_loss(rels_pred.reshape(batch_size * length, num_label), rels.reshape(-1),
                      size_average=True, ignore_index=-233)


class Decoder(object):
    def __init__(self, method="naive"):
        self.method = method
        if method not in ["naive", "mst"]:
            raise NotImplemented

    @staticmethod
    def find_cycle(fa):
        n = len(fa)
        vis = np.ndarray([n], dtype=np.bool)
        vis.fill(False)
        vis[0] = True
        cycles = []
        # 0 is always the root hence has no father
        # cycle finding algorithm: for any unseen node, walk through its unvisited parent link, if it stops at a node
        # visited during this walk, then it must be a cycle.
        # for any cycle in this graph, it must be like a rho-shaped graph, whichever node being the start point, the
        # cycle can be detected using this algorithm
        for i in range(1, n):
            if vis[i]:
                continue
            cyc = set()
            cyc.add(i)
            vis[i] = True
            j = fa[i]
            while not vis[j]:
                cyc.add(j)
                vis[j] = True
                j = fa[j]
            if j in cyc:
                new_cycle = [j]
                t = fa[j]
                while new_cycle[0] != t:
                    new_cycle.append(t)
                    t = fa[t]
                cycles.append(new_cycle)
        return cycles

    @staticmethod
    def mst(g):
        # given g of [n, n], we want the mst rooted at 0
        n, _ = g.shape
        # setting axis to 0 so that we get the position of the maximum value vertically
        np.fill_diagonal(g, -np.inf)
        fa = g.argmax(axis=0)
        # print(fa)
        cycles = Decoder.find_cycle(fa)
        c = len(cycles)
        if c == 0:
            return fa
        cid = np.ndarray([n], dtype=np.int)
        cid.fill(-1)
        for i in range(c):
            for j in cycles[i]:
                cid[j] = i
        m = len(np.where(cid == -1)[0]) + c
        # m is the number of nodes after contraction
        w = np.ndarray([m, m], dtype=np.float)
        w.fill(-np.inf)
        j = 1
        idx = np.ndarray([n], dtype=np.int)
        idx[0] = 0
        for cyc in cycles:
            for i in cyc:
                idx[i] = j
            j += 1
        for i in range(1, n):
            if cid[i] == -1:
                idx[i] = j
                j += 1
            else:
                g[:, j] -= max(g[:, j])
        # print("cycles:", cycles)
        # print(j, m)

        arc = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if w[idx[i]][idx[j]] < g[i][j]:
                    w[idx[i]][idx[j]] = g[i][j]
                    arc[(idx[i], idx[j])] = (i, j)

        nfa = Decoder.mst(w)
        for i in range(1, m):
            a, b = arc[(nfa[i], i)]
            fa[b] = a
        return fa

    def __call__(self, graphs, mask):
        # graph of [bs, length, length]
        # mask of [bs, length]
        if type(graphs) is not np.ndarray:
            graphs = graphs.cpu().numpy()
        if self.method == "naive":
            return graphs.argmax(axis=1)
        if self.method == "mst":
            bs, length, _ = graphs.shape
            res = np.ndarray([bs, length])
            for i in range(bs):
                n = int(mask[i].sum())
                g = graphs[i, :n, :n]
                res[i, :n] = Decoder.mst(g)
            return res


def save_state(args: infra.Arguments, trainer, net, opt, name=None):
    print("!!! warning !!!")
    if name is None:
        torch.save((net, opt), open(args.save_prefix + '.save.ptr', "wb"))
    else:
        torch.save((net, opt), open(name, "wb"))


def load_state(path):
    net, opt = torch.load(path)
    def rename_key(state_dict):
        from collections import OrderedDict
        ret = OrderedDict()
        for key in state_dict:
            if key.startswith("module."):
                ret[key[7:]] = state_dict[key]
            else:
                ret[key] = state_dict[key]
        return ret
    return rename_key(net), rename_key(opt)

