import torch
import torch.nn as nn
import infra.config
import numpy as np
import torch.nn.functional as F
from .nn.variational_rnn import VarMaskedFastLSTM
from .nn.custom_modules import BiAffine, BiLinear
from task import parser
from .nn.graph_network import outer_concat, EdgeFocusedGraphNetwork, EdgeFocusedGraphNetworkWithEdge
from .nn.encoder import InputEncoder
from .nn.attention import BilinearAttention, BiaffineAttention


class ConstrainedEdgeFocusedGraphNetwork(nn.Module):
    def __init__(self, args: infra.config.Arguments):
        super(ConstrainedEdgeFocusedGraphNetwork, self).__init__()
        self.encoder = InputEncoder(args, args.inputs)
        lstm_feat_dim = args.hidden_size * 2
        self.mlp_arc_dep = nn.Linear(lstm_feat_dim, args.arc_space)
        self.mlp_arc_head = nn.Linear(lstm_feat_dim, args.arc_space)
        self.mlp_rel_dep = nn.Linear(lstm_feat_dim, args.rel_space)
        self.mlp_rel_head = nn.Linear(lstm_feat_dim, args.rel_space)

    def forward(self, inp_word, inp_char, inp_pos, heads=None, masks=None, lengths=None, decoder=None, hx=None):
        # use decoder: graph[bs, length, length], masks[bs, length] -> heads_pred[bs, length]
        # in inference
        output, hn = self.encoder.forward(inp_word, inp_char, inp_pos, masks, hx)
        arc_dep = F.elu(self.mlp_arc_dep(output))
        arc_head = F.elu(self.mlp_arc_head(output))
        rel_dep = F.elu(self.mlp_rel_dep(output))
        rel_head = F.elu(self.mlp_rel_head(output))
        pass


class DirectlyDecodingFromEdge(nn.Module):
    def __init__(self, args: infra.config.Arguments):
        super(DirectlyDecodingFromEdge, self).__init__()
        self.encoder = InputEncoder(args, args.inputs)
        lstm_feat_dim = args.hidden_size * 2
        self.efgn = EdgeFocusedGraphNetworkWithEdge(lstm_feat_dim, args.efgn_inn_dim)
        self.head_readout = nn.Linear(args.efgn_inn_dim, 1)
        self.rels_readout = nn.Linear(args.efgn_inn_dim, args.inputs.num_rels)

    def forward(self, inp_word, inp_char, inp_pos, heads=None, masks=None, decoder=None, hx=None):
        output, hn = self.encoder.forward(inp_word, inp_char, inp_pos, masks, hx)
        vertex, edge = self.efgn.forward(output, masks)
        graph = self.head_readout(edge).squeeze(-1)
        if heads is None:
            heads = torch.Tensor(decoder(graph, masks)).type(torch.LongTensor)

        rels_pred = self.rels_readout(edge)
        bs, l, _, d = rels_pred.shape
        rels_pred = rels_pred.transpose(1, 2)
        rels_pred = rels_pred.reshape(bs * l, l, d)
        heads = heads.reshape(bs * l)
        rels_pred = rels_pred[np.arange(bs * l), heads].reshape(bs, l, d)

        # 不要把算loss放到模型里面来
        # heads在inference的时候是必须的信息，因此应该传入，而loss还是在外面算比较好
        ninf = 1e-8
        masks = (1 - masks) * ninf
        heads_pred = torch.log_softmax(graph + masks.unsqueeze(1) + masks.unsqueeze(2), dim=1)
        rels_pred = torch.log_softmax(rels_pred, dim=2)
        # we want the cross entropy loss, e.g. negative log of softmax
        # to mask out the paddings, we have to make them small
        return heads_pred, rels_pred


class DozatBiaffine(nn.Module):
    def __init__(self, args: infra.config.Arguments):
        super(DozatBiaffine, self).__init__()
        self.encoder = InputEncoder(args, args.inputs)
        lstm_feat_dim = args.hidden_size * 2
        self.efgn = EdgeFocusedGraphNetworkWithEdge(lstm_feat_dim, args.efgn_inn_dim)
        self.biaffine = BiaffineAttention(lstm_feat_dim, args.arc_space)
        self.bilinear = BilinearAttention(lstm_feat_dim, args.rel_space, args.inputs.num_rels)

    def forward(self, inp_word, inp_char, inp_pos, heads=None, masks=None, decoder=None, hx=None):
        output, hn = self.encoder.forward(inp_word, inp_char, inp_pos, masks, hx)
        vertex, edge = self.efgn.forward(output, masks)
        graph = self.biaffine.forward(output, masks)

        if heads is None:
            heads = torch.Tensor(decoder(graph, masks)).type(torch.LongTensor)

        rels = self.bilinear.forward(output, heads, masks)
        # rels of [bs, length (n), num_rels (m)]
        bs, n, m = rels.shape

        # 不要把算loss放到模型里面来
        # heads在inference的时候是必须的信息，因此应该传入，而loss还是在外面算比较好
        ninf = 1e-8
        masks = (1 - masks) * ninf
        graph = torch.log_softmax(graph + masks.unsqueeze(1) + masks.unsqueeze(2), dim=1)
        rels = torch.log_softmax(rels, dim=2)
        # we want the cross entropy loss, e.g. negative log of softmax
        # to mask out the paddings, we have to make them small
        return graph, rels


class BiRecurrentConvBiAffineGraph(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, kernel_size, rnn_mode, hidden_size, num_layers, num_labels, arc_space, rel_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33), biaffine=True, pos=True, char=True, args=None):
        super(BiRecurrentConvBiAffineGraph, self).__init__()

        self.word_embedd = nn.Embedding(num_words, word_dim, _weight=embedd_word)
        self.pos_embedd = nn.Embedding(num_pos, pos_dim, _weight=embedd_pos) if pos else None
        self.char_embedd = nn.Embedding(num_chars, char_dim, _weight=embedd_char) if char else None

        self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1) if char else None
        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)
        self.num_labels = num_labels
        self.pos = pos
        self.char = char

        if rnn_mode == 'RNN':
            RNN = VarMaskedRNN
        elif rnn_mode == 'LSTM':
            RNN = VarMaskedLSTM
        elif rnn_mode == 'FastLSTM':
            RNN = VarMaskedFastLSTM
        elif rnn_mode == 'GRU':
            RNN = VarMaskedGRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        dim_enc = word_dim
        if pos:
            dim_enc += pos_dim
        if char:
            dim_enc += num_filters

        self.rnn = RNN(dim_enc, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn)

        self.use_efgn = args.use_efgn
        if args.use_efgn:
            self.graph_network = EdgeFocusedGraphNetwork(hidden_size * 2, n_inn=args.efgn_inn_dim)
        else:
            self.graph_network = None

        out_dim = hidden_size * 2
        self.arc_h = nn.Linear(out_dim, arc_space)
        self.arc_c = nn.Linear(out_dim, arc_space)
        self.attention = BiAffine(arc_space, arc_space, 1, biaffine=biaffine)

        self.type_h = nn.Linear(out_dim, rel_space)
        self.type_c = nn.Linear(out_dim, rel_space)
        self.bilinear = BiLinear(rel_space, rel_space, self.num_labels)

    def forward(self, input_word, input_char, input_pos, heads=None, types=None, mask=None, length=None, hx=None, task=None, leading_symbolic=None):
        def _get_rnn_output(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
            # [batch, length, word_dim]
            word = self.word_embedd(input_word)
            # apply dropout on input
            word = self.dropout_in(word)

            input = word

            if self.char:
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
                char = self.dropout_in(char)
                # concatenate word and char [batch, length, word_dim+char_filter]
                input = torch.cat([input, char], dim=2)

            if self.pos:
                # [batch, length, pos_dim]
                pos = self.pos_embedd(input_pos)
                # apply dropout on input
                pos = self.dropout_in(pos)
                input = torch.cat([input, pos], dim=2)

            # output from rnn [batch, length, hidden_size]
            output, hn = self.rnn(input, mask, hx=hx)


            # apply dropout for output
            # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
            output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

            # use edge focused graph_network
            if self.use_efgn:
                output = self.graph_network(output, mask)

            # output size [batch, length, arc_space]
            arc_h = F.elu(self.arc_h(output))
            arc_c = F.elu(self.arc_c(output))

            # output size [batch, length, rel_space]
            type_h = F.elu(self.type_h(output))
            type_c = F.elu(self.type_c(output))

            # apply dropout
            # [batch, length, dim] --> [batch, 2 * length, dim]
            arc = torch.cat([arc_h, arc_c], dim=1)
            type = torch.cat([type_h, type_c], dim=1)

            arc = self.dropout_out(arc.transpose(1, 2)).transpose(1, 2)
            arc_h, arc_c = arc.chunk(2, 1)

            type = self.dropout_out(type.transpose(1, 2)).transpose(1, 2)
            type_h, type_c = type.chunk(2, 1)
            type_h = type_h.contiguous()
            type_c = type_c.contiguous()

            return (arc_h, arc_c), (type_h, type_c), hn, mask, length
        # output from rnn [batch, length, tag_space]
        arc, out_type, _, mask, length = _get_rnn_output(self, input_word, input_char, input_pos, mask=mask, length=length, hx=hx)
        # [batch, length, length]
        out_arc = self.attention(arc[0], arc[1], mask_d=mask, mask_e=mask).squeeze(dim=1)

        # loss
        if task == "loss":
            # out_arc shape [batch, length, length]
            batch, max_len, _ = out_arc.size()

            if length is not None and heads.size(1) != mask.size(1):
                heads = heads[:, :max_len]
                types = types[:, :max_len]

            # out_type shape [batch, length, rel_space]
            type_h, type_c = out_type

            # create batch index [batch]
            batch_index = torch.arange(0, batch).type_as(out_arc.data).long()
            # get vector for heads [batch, length, rel_space],
            type_h = type_h[batch_index, heads.data.t()].transpose(0, 1).contiguous()
            # compute output for type [batch, length, num_labels]
            out_type = self.bilinear(type_h, type_c)

            # mask invalid position to -inf for log_softmax
            if mask is not None:
                minus_inf = -1e8
                minus_mask = (1 - mask) * minus_inf
                out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

            # loss_arc shape [batch, length, length]
            loss_arc = F.log_softmax(out_arc, dim=1)
            # loss_type shape [batch, length, num_labels]
            loss_type = F.log_softmax(out_type, dim=2)

            # mask invalid position to 0 for sum loss
            if mask is not None:
                loss_arc = loss_arc * mask.unsqueeze(2) * mask.unsqueeze(1)
                loss_type = loss_type * mask.unsqueeze(2)
                # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
                num = mask.sum() - batch
            else:
                # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
                num = float(max_len - 1) * batch

            # first create index matrix [length, batch]
            child_index = torch.arange(0, max_len).view(max_len, 1).expand(max_len, batch)
            child_index = child_index.type_as(out_arc.data).long()
            # [length-1, batch]
            loss_arc = loss_arc[batch_index, heads.data.t(), child_index][1:]
            loss_type = loss_type[batch_index, child_index, types.data.t()][1:]

            return loss_arc, loss_type
        elif task == "decode_mst":
            # out_type shape [batch, length, rel_space]
            type_h, type_c = out_type
            batch, max_len, rel_space = type_h.size()

            # compute lengths
            if length is None:
                if mask is None:
                    length = [max_len for _ in range(batch)]
                else:
                    length = mask.data.sum(dim=1).long().cpu().numpy()

            type_h = type_h.unsqueeze(2).expand(batch, max_len, max_len, rel_space).contiguous()
            type_c = type_c.unsqueeze(1).expand(batch, max_len, max_len, rel_space).contiguous()
            # compute output for type [batch, length, length, num_labels]
            out_type = self.bilinear(type_h, type_c)

            # mask invalid position to -inf for log_softmax
            if mask is not None:
                minus_inf = -1e8
                minus_mask = (1 - mask) * minus_inf
                out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

            # loss_arc shape [batch, length, length]
            loss_arc = F.log_softmax(out_arc, dim=1)
            # loss_type shape [batch, length, length, num_labels]
            loss_type = F.log_softmax(out_type, dim=3).permute(0, 3, 1, 2)
            # [batch, num_labels, length, length]
            energy = torch.exp(loss_arc.unsqueeze(1) + loss_type)

            # TODO: remove decode_MST, we only need energy and length! not necessary!
            return energy, length
            # return parser.decode_MST(energy.data.cpu().numpy(), length, leading_symbolic=leading_symbolic, labeled=True)
        else:
            return out_arc, type, mask, length


