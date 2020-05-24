import torch.nn as nn
import torch
from .emd import WlossLayer
import pdb
import torch.nn.functional as F
from .custom_modules import BiLinear, BiAffine, Transpose
from ..utils import outer_product, outer_concat


class MorphPredictor(nn.Module):
    def __init__(self, n_morph_word, n_feat_input, type='simple'):
        super(MorphPredictor, self).__init__()

        if type == 'word':
            self.predictor = nn.Sequential(
                nn.Linear(n_feat_input, n_morph_word)
            )
        elif type == "multi_label":
            raise NotImplementedError

    def forward(self, x):
        return self.predictor(x)


class ArcPredictor(nn.Module):
    """ArcPredictor: predict the head-dependent arc matrix
    Inputs:
        features: shape of (n, l, l, h)
    Outputs:
        pred: shape of (n, l, l)
    """

    def __init__(self, n_feat, n_mlp_hidden, biaffine=False, args=None):
        super(ArcPredictor, self).__init__()
        self.biaffine = biaffine
        if biaffine:
            self.lin1 = nn.Sequential(nn.Linear(n_feat, n_mlp_hidden),
                                      nn.Dropout(0.33),
                                      nn.ELU(inplace=True))
            self.lin2 = nn.Sequential(nn.Linear(n_feat, n_mlp_hidden),
                                      nn.Dropout(0.33),
                                      nn.ELU(inplace=True))
            self.predictor = BiAffine(n_mlp_hidden, n_mlp_hidden, 1)
            self.dropout = nn.Sequential( # drop along the word dim
                Transpose(1, 2),
                nn.Dropout(0.33),
                Transpose(1, 2))
        else:
            self.predictor = nn.Sequential(nn.Linear(n_feat * 2, n_mlp_hidden),
                                           nn.Tanh(),
                                           nn.Linear(n_mlp_hidden, 1, bias=False))
            for t in self.predictor.children():
                 if type(t) is nn.Linear:
                     torch.nn.init.xavier_normal_(t.weight)

    def forward(self, feat, mask=None):
        n, l, h = feat.shape
        # pdb.set_trace()
        if self.biaffine:
            feat_head = self.lin1(feat)
            feat_child = self.lin2(feat)
            feat = torch.cat([feat_head, feat_child], dim=1)
            # feat = self.dropout(feat)
            feat_head, feat_child = feat.chunk(2, 1)
            feat = self.predictor(feat_head.contiguous(), feat_child.contiguous())
        else:
            feat = outer_concat(feat).view(n, -1, h*2)
            feat = self.predictor(feat)
        feat = feat.view(n, l, l)
        return feat


class ArcPredictor2D(nn.Module):
    """ArcPredictor: predict the head-dependent arc matrix
    Inputs:
        features: shape of (n, l, l, h)
    Outputs:
        pred: shape of (n, l, l)
    """

    def __init__(self, n_feat, n_mlp_hidden, args=None):
        super(ArcPredictor2D, self).__init__()
        self.predictor = nn.Sequential(nn.Linear(n_feat, n_mlp_hidden),
                                       nn.Dropout(args.drop_mlp),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(n_mlp_hidden, 1, bias=None))

    def forward(self, feat, mask=None):
        n, l, l, h = feat.shape
        # pdb.set_trace()
        feat = feat.view(n, -1, h)
        feat = self.predictor(feat)
        feat = feat.view(n, l, l)
        return feat


class LabelPredictor(nn.Module):
    """LabelPredictor: predict the label of the arc
    Inputs:
        features: shape of (n, l, h)
        assignment: head assignment in shape of (n, l)
    Outputs:
        pred: shape of (n, l, k), in which $k = #{label type}$
    """

    def __init__(self, feat_dim, hidden_dim, n_labels,
                 biaffine=False, args=None):
        super(LabelPredictor, self).__init__()
        self.biaffine = biaffine
        if biaffine:
            self.predictor = BiLinear(n_left=feat_dim, n_right=feat_dim,
                                      n_out=n_labels)
            self.dropout = nn.Sequential( # drop along the word dim
                Transpose(1, 2),
                nn.Dropout(0.33),
                Transpose(1, 2))
        else:
            self.predictor = nn.Sequential(nn.Linear(feat_dim * 2, hidden_dim),
                                           nn.Dropout(0.33),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, n_labels))
            for t in self.predictor.children():
                 if type(t) is nn.Linear:
                     torch.nn.init.xavier_normal_(t.weight)

    def forward(self, feat, heads, masks):
        # pdb.set_trace()
        dependent_f = feat[:, 1:, :]
        n, l, h = dependent_f.shape
        # create batch index [batch]
        batch_index = torch.arange(0, n).type_as(feat.data).long()
        # get vector for heads [batch, length, label_spaces]
        # import pdb
        # pdb.set_trace()
        head_f = feat[batch_index, heads.data.t()].transpose(0, 1).contiguous()
        # print(batch_index.size(), feat.size(), feat, heads)
        if self.biaffine:
            # joint dropout
            feat = torch.cat((dependent_f, head_f), dim=1)
            feat = self.dropout(feat)
            dependent_f, head_f = feat.chunk(2, 1)
            label_f = self.predictor(dependent_f.contiguous(), head_f.contiguous())
        else:
            cat_f = torch.cat((dependent_f, head_f), dim=2)
            cat_f = cat_f.view(n * l, h * 2)
            feat = self.predictor(cat_f)
            label_f = feat.view(n, l, -1)

        return label_f


class ArcPredictorWloss(nn.Module):
    """ArcPredictor: predict the head-dependent arc matrix
    Inputs:
        features: shape of (n, l, l, h)
    Outputs:
        pred: shape of (n, l, l)
    """

    def __init__(self, features_dim, hidden_dim, nonlinearity=nn.functional.tanh):
        super(ArcPredictorWloss, self).__init__()
        self.lin1 = nn.Linear(features_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)
        self.nonlinearity = nonlinearity
        self.wloss = WlossLayer()

    def forward(self, f, head):
        # import pdb
        # pdb.set_trace()
        n, l, _, h = f.shape
        f_cost = f.view(n, l*l, h).contiguous()
        f = f.view(n * l * l, h)
        f = self.lin1(f)
        f = self.nonlinearity(f)
        f = self.lin2(f)
        f = f.view(n, l, l)
        wloss = None
        for i in range(n):
            f_cost = (torch.bmm(f_cost[i] * f_cost[i].transpose(1, 2))).view(l, l)
            preds = f[i] + 1e-8
            targets = head[i] + 1e-8
            loss, _ = self.wloss(preds, targets, f_cost)
            if wloss is None:
                wloss = loss
            else:
                wloss += loss

        return f, wloss


class LabelPredictor2D(nn.Module):
    """LabelPredictor: predict the label of the arc
    Inputs:
        features: shape of (n, l, l, h)
        assignment: head assignment in shape of (n, l)
    Outputs:
        pred: shape of (n, l, k), in which $k = #{label type}$
    """

    def __init__(self, features_dim, hidden_dim, n_label, args=None):
        super(LabelPredictor2D, self).__init__()
        self.lin1 = nn.Linear(features_dim, hidden_dim)  # the concatenation of the features are used
        self.drop = nn.Dropout(0.33)
        self.lin2 = nn.Linear(hidden_dim, n_label)
        self.n_label = n_label
        self.nonlinearity = nn.Tanh()

    def forward(self, feat, heads):
        # pdb.set_trace()
        feat = feat[:, 1:, :, :]
        n, l, _, h = feat.shape
        # create batch index [batch]
        batch_index = torch.arange(0, n).long().\
            view(1, -1).repeat(l, 1).t_().contiguous().view(-1).tolist()
        # create sent_length index
        length_index = torch.arange(0, l).long().repeat(n).view(-1).tolist()
        # get vector for heads [batch, length, type_space],
        feat = feat[batch_index, length_index, heads.data.view(-1).cpu().tolist()].view(n, l, -1)

        feat = self.lin1(feat)
        feat = self.drop(feat)
        feat = self.nonlinearity(feat)
        feat = self.lin2(feat)
        label_f = feat.view(n, l, self.n_label)

        return label_f
