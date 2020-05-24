import torch
from torch import nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, n_inp, n_step=3):
        super(GCN, self).__init__()
        self.aggs = nn.ModuleList([nn.Linear(n_inp, n_inp) for i in range(n_step)])
        # self.gates = nn.ModuleList([nn.Linear(n_inp, n_inp) for i in range(n_step)])
        self.upds = nn.ModuleList([nn.Linear(n_inp * 2, n_inp) for i in range(n_step)])
        self.attns = nn.ModuleList([nn.Linear(n_inp * 2, n_inp) for i in range(n_step)])
        self.attn_drop = nn.ModuleList([nn.Dropout(0.33) for i in range(n_step)])
        self.n_step = n_step

    def forward(self, feat: torch.Tensor, mask: torch.Tensor):
        """
        The feat is a (bs * l_sent * n_inp) tensor
        :param feat:
        :return: the same size as input
        """
        hid = feat
        # hid = feat.clone()
        bs, l_sent, n_inp = feat.shape
        for i in range(self.n_step):
            hid = hid * mask.view([bs, l_sent, 1])
            xhid = self.aggs[i](hid)
            agg_res = []
            for j in range(l_sent):
                attn = self.attns[i](torch.cat((xhid, xhid[:, j:j+1, :].expand(-1, l_sent, -1)), 2))
                attn_weight = F.softmax(attn, dim=1)
                # of size (bs, l_sent, b_inp)
                attn_xhid = attn_weight * xhid
                # of size (bs, l_sent, b_inp)
                sum_xhid = torch.sum(attn_xhid, dim=1, keepdim=True)
                agg_res.append(F.sigmoid(sum_xhid))
            agg = torch.cat(agg_res, dim=1)
            xhid = torch.cat([xhid, agg], 2)
            xhid = self.attn_drop[i](xhid)
            hid = self.upds[i](xhid)

        return hid


class GCNShared(nn.Module):
    def __init__(self, n_inp, n_step=3):
        super(GCNShared, self).__init__()
        self.agg = nn.Linear(n_inp, n_inp)
        self.upd = nn.Linear(n_inp * 2, n_inp)
        self.attn = nn.Linear(n_inp * 2, n_inp)
        self.attn_drop = nn.Dropout(0.33)
        self.n_step = n_step

    def forward(self, feat: torch.Tensor, mask: torch.Tensor):
        """
        The feat is a (bs * l_sent * n_inp) tensor
        :param feat:
        :return: the same size as input
        """
        hid = feat
        # hid = feat.clone()
        bs, l_sent, n_inp = feat.shape
        for i in range(self.n_step):
            hid = hid * mask.view([bs, l_sent, 1])
            xhid = self.agg(hid)
            agg_res = []
            for j in range(l_sent):
                attn = self.attn(torch.cat((xhid, xhid[:, j:j+1, :].expand(-1, l_sent, -1)), 2))
                attn_weight = F.softmax(attn, dim=1)
                # of size (bs, l_sent, b_inp)
                attn_xhid = attn_weight * xhid
                # of size (bs, l_sent, b_inp)
                sum_xhid = torch.sum(attn_xhid, dim=1, keepdim=True)
                agg_res.append(F.sigmoid(sum_xhid))
            agg = torch.cat(agg_res, dim=1)
            xhid = torch.cat([xhid, agg], 2)
            xhid = self.attn_drop(xhid)
            hid = self.upd(xhid)

        return hid


def outer_concat(feat):
    """InterProduct: Get inter sequence concatenation of features
    Arguments:
        feat: feature vectors of sequence in the shape of (n, l, h)
    Return:
        f: product result in (n, l, l, h) shape
    """
    n, l, c = feat.shape
    x = feat.contiguous().view(n, l, 1, c)
    x = x.expand(n, l, l, c)
    y = feat.view(n, 1, l, c)
    y = y.expand(n, l, l, c)
    return torch.cat((x, y), dim=3)


class EdgeFocusedGraphNetwork(nn.Module):
    """
    in the simplest manner, we use outer-concatenate to get the initial edge information (like emma)
    """
    def __init__(self, n_inp, n_inn=256, n_step=3, p_attn=0.33, p_xform=0.33):
        super(EdgeFocusedGraphNetwork, self).__init__()
        self.n_inp = n_inp
        self.n_inn = n_inn
        self.n_step = n_step

        self.xform_inp = nn.Linear(n_inp, n_inn)
        self.xform_oup = nn.Linear(n_inn, n_inp)
        self.xform_feat_e = nn.Linear(n_inn * 2, n_inn)
        self.upd_e = nn.Linear(n_inn * 2, n_inn)

        self.agg_v = nn.Linear(n_inn, n_inn)
        self.upd_v = nn.Linear(n_inn * 2, n_inn)
        self.attn_v = nn.Linear(n_inn, n_inn)
        self.attn_v_drop = nn.Dropout(0.33)

    def edge_update(self, feat_v, feat_e, mask):
        _feat_e = outer_concat(feat_v)
        _feat_e = self.xform_feat_e(_feat_e)
        _feat_e = torch.cat((_feat_e, feat_e), 3)
        _feat_e = self.upd_e(_feat_e)
        return _feat_e

    def vertex_update(self, feat_v, feat_e, mask):
        hid = feat_v
        # hid = feat.clone()
        bs, l_sent, n_inn = feat_v.shape
        hid = hid * mask.view([bs, l_sent, 1])
        # TODO: analyze the padding updates in the network
        xhid = self.agg_v(hid)
        agg_res = []
        for j in range(l_sent):
            attn = self.attn_v(feat_e[:, :, j, :])
            attn_weight = F.softmax(attn, dim=1)
            # of size (bs, l_sent, b_inp)
            attn_xhid = attn_weight * xhid
            # of size (bs, l_sent, b_inp)
            sum_xhid = torch.sum(attn_xhid, dim=1, keepdim=True)
            agg_res.append(F.sigmoid(sum_xhid))
        agg = torch.cat(agg_res, dim=1)
        xhid = torch.cat([xhid, agg], 2)
        xhid = self.attn_v_drop(xhid)
        feat_v = self.upd_v(xhid)
        return feat_v

    def message_passing(self, feat_v, feat_e, mask):
        _feat_e = self.edge_update(feat_v, feat_e, mask)
        _feat_v = self.vertex_update(feat_v, _feat_e, mask)
        # global update is ignored
        return _feat_v, _feat_e

    def pred_head(self, feat_v, feat_e):
        pass

    def pred_label(self, feat_v, feat_e):
        pass

    def forward(self, feat: torch.Tensor, mask: torch.Tensor):
        feat = self.xform_inp(feat)

        feat_e = outer_concat(feat)
        feat_e = self.xform_feat_e(feat_e)

        feat_v = feat
        for step in range(self.n_step):
            feat_v, feat_e = self.message_passing(feat_v, feat_e, mask)

        feat_v = self.xform_oup(feat_v)
        return feat_v


class EdgeFocusedGraphNetworkWithEdge(nn.Module):
    """
    in the simplest manner, we use outer-concatenate to get the initial edge information (like emma)
    """
    def __init__(self, n_inp, n_inn=256, n_step=3, p_attn=0.33, p_xform=0.33):
        super(EdgeFocusedGraphNetworkWithEdge, self).__init__()
        self.n_inp = n_inp
        self.n_inn = n_inn
        self.n_step = n_step

        self.xform_inp = nn.Linear(n_inp, n_inn)
        self.xform_oup = nn.Linear(n_inn, n_inp)
        self.xform_feat_e = nn.Linear(n_inn * 2, n_inn)
        self.upd_e = nn.Linear(n_inn * 2, n_inn)

        self.agg_v = nn.Linear(n_inn, n_inn)
        self.upd_v = nn.Linear(n_inn * 2, n_inn)
        self.attn_v = nn.Linear(n_inn, n_inn)
        self.attn_v_drop = nn.Dropout(p_attn)

    def edge_update(self, feat_v, feat_e, mask):
        _feat_e = outer_concat(feat_v)
        _feat_e = self.xform_feat_e(_feat_e)
        _feat_e = torch.cat((_feat_e, feat_e), 3)
        _feat_e = self.upd_e(_feat_e)
        _feat_e *= (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(3)
        return _feat_e

    def vertex_update(self, feat_v, feat_e, mask):
        hid = feat_v
        # hid = feat.clone()
        bs, l_sent, n_inn = feat_v.shape
        hid = hid * mask.view([bs, l_sent, 1])
        # TODO: analyze the padding updates in the network
        xhid = self.agg_v(hid)
        agg_res = []
        for j in range(l_sent):
            attn = self.attn_v(feat_e[:, :, j, :])
            attn_weight = F.softmax(attn, dim=1)
            # of size (bs, l_sent, b_inp)
            attn_xhid = attn_weight * xhid
            # of size (bs, l_sent, b_inp)
            sum_xhid = torch.sum(attn_xhid, dim=1, keepdim=True)
            agg_res.append(F.sigmoid(sum_xhid))
        agg = torch.cat(agg_res, dim=1)
        xhid = torch.cat([xhid, agg], 2)
        xhid = self.attn_v_drop(xhid)
        feat_v = self.upd_v(xhid)
        return feat_v

    def message_passing(self, feat_v, feat_e, mask):
        _feat_e = self.edge_update(feat_v, feat_e, mask)
        _feat_v = self.vertex_update(feat_v, _feat_e, mask)
        # global update is ignored
        return _feat_v, _feat_e

    def pred_head(self, feat_v, feat_e):
        pass

    def pred_label(self, feat_v, feat_e):
        pass

    def forward(self, feat: torch.Tensor, masks: torch.Tensor):
        feat = self.xform_inp(feat)

        feat_e = outer_concat(feat)
        feat_e = self.xform_feat_e(feat_e)

        feat_v = feat
        for step in range(self.n_step):
            feat_v, feat_e = self.message_passing(feat_v, feat_e, masks)

        feat_v = self.xform_oup(feat_v)
        return feat_v, feat_e

