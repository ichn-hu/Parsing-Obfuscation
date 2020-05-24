import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class BiaffineAttentionObsolete(nn.Module):
    def __init__(self, out_dim, inp_dim, inn_dim=0, rec_num=1, bias=True):
        """
        :param out_dim:
        :param inp_dim:
        :param k_dim: inner dim
        :param a_num: times of attention, stacking output into a 3-dim tensor
        :param bias: use bias or not
        """
        super(BiaffineAttentionObsolete, self).__init__()
        self.out_dim = out_dim
        self.inp_dim = inp_dim
        self.inn_dim = inn_dim
        self.rec_num = rec_num
        self.bias = bias

        self.xform_key = nn.Linear(inp_dim, inn_dim, bias=bias)
        self.xform_query = nn.Linear(inp_dim, inn_dim, bias=bias)
        self.b = nn.Parameter(torch.Tensor(rec_num, inn_dim, 1))
        self.W = nn.Parameter(torch.Tensor(rec_num, inn_dim, inn_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # another way of initialization
        # https://github.com/chantera/teras/blob/f25895b2cc1ecb6bf41a5d3da2e582014d23468d/teras/framework/pytorch/model.py#L226
        nn.init.xavier_uniform_(self.xform_key.weight)
        nn.init.xavier_uniform_(self.xform_query.weight)
        nn.init.constant_(self.b, 0)
        nn.init.xavier_uniform_(self.W)

    def forward(self, key, query, mask_key=None, mask_query=None):
        """
        :param key: batch_size * out_dim * inp_dim
        :param query: batch_size * out_dim * inp_dim
        :param mask_key: batch_size * out_dim
        :param mask_query: batch_size * out_dim
        :return:
        """
        batch_size = key.size(0)
        # optional if inn_dim == inp_dim
        key = self.xform_key(key)
        query = self.xform_query(query)
        scr = torch.matmul(key.unsqueeze(1), self.W).matmul(query.unsqueeze(1)) +\
              torch.matmul(key.unsqueeze(0), self.b)
        return scr


class BiaffineAttention(nn.Module):
    def __init__(self, inp_dim, inn_dim, non_linear=F.elu):
        super(BiaffineAttention, self).__init__()
        self.arc_head = nn.Linear(inp_dim, inn_dim)
        self.arc_dep = nn.Linear(inp_dim, inn_dim)
        self.non_linear = non_linear
        self.W = nn.Parameter(torch.Tensor(inn_dim, inn_dim))
        self.b = nn.Parameter(torch.Tensor(inn_dim, 1))

        nn.init.xavier_uniform_(self.W)
        nn.init.constant_(self.b, 0)

    def forward(self, inp, masks):
        # inp of [bs, length (n), inp_dim]
        # masks of [bs, n]
        H = self.non_linear(self.arc_head(inp))
        D = self.non_linear(self.arc_dep(inp))
        W = self.W
        b = self.b
        # H, D of [bs, n, inn_dim (k)]
        bs, n, k = H.shape
        H = H * masks.reshape(bs, n, 1)
        D = D * masks.reshape(bs, n, 1)

        return H.matmul(W).matmul(D.transpose(1, 2)) + H.matmul(b)


class BilinearAttention(nn.Module):
    def __init__(self, inp_dim, inn_dim, rel_num, non_linear=F.elu):
        super(BilinearAttention, self).__init__()
        self.rel_dep = nn.Linear(inp_dim, inn_dim)
        self.rel_head = nn.Linear(inp_dim, inn_dim)
        self.non_linear = non_linear
        self.m = rel_num
        self.U = nn.Parameter(torch.Tensor(rel_num, inn_dim, inn_dim))
        self.W = nn.Parameter(torch.Tensor(rel_num, inn_dim * 2))
        self.b = nn.Parameter(torch.Tensor(rel_num, 1))

        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.W)
        nn.init.constant_(self.b, 0)

    def forward(self, inp, heads, masks):
        # inp of [bs, length (n), inp_dim]
        # heads of [bs, n]
        # masks of [bs, n]
        H = self.non_linear(self.rel_head(inp))
        D = self.non_linear(self.rel_dep(inp))
        W = self.W
        U = self.U
        b = self.b
        m = self.m
        # H, D of [bs, n, inn_dim (d)]
        # head of [bs, n], head[0] always dose not matter
        # W of [m, d * 2]
        bs, n, d = H.shape

        H = H * masks.reshape(bs, n, 1)
        D = D * masks.reshape(bs, n, 1)

        # see here https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays
        # to extract the vectors referred by head
        h = H[np.arange(bs), heads.data.t()].transpose(0, 1)
        # h of [bs, n, d]
        out = h.reshape(bs, n, 1, 1, d).matmul(U)
        # out of [bs, n, m, 1, d]
        out = out.transpose(1, 2).squeeze()
        # out of [bs, m, n, d]
        out = (out * D.reshape(bs, 1, n, d)).sum(dim=3)
        # out of [bs, m, n]
        out = out.transpose(1, 2)
        # out of [bs, n, m]
        h = torch.cat([h, D], dim=2)
        # h of [bs, n, d * 2]
        out = out + h.matmul(W.t()) + b.reshape(1, 1, m)
        return out

