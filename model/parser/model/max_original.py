__author__ = 'max'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn.variational_rnn import VarMaskedFastLSTM
# from ..nn import BiAAttention, BiLinear
from .nn.custom_modules import BiAffine, BiLinear

from task import parser
from config import Global
cfg = Global()


class BiRecurrentConvBiAffine(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, kernel_size, rnn_mode, hidden_size, num_layers, num_labels, arc_space, type_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33), biaffine=True, pos=True, char=True):
        super(BiRecurrentConvBiAffine, self).__init__()

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
            # RNN = VarMaskedRNN
            pass
        elif rnn_mode == 'LSTM':
            # RNN = VarMaskedLSTM
            pass
        elif rnn_mode == 'FastLSTM':
            RNN = VarMaskedFastLSTM
        elif rnn_mode == 'GRU':
            # RNN = VarMaskedGRU
            pass
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        dim_enc = word_dim
        if pos:
            dim_enc += pos_dim
        if char:
            dim_enc += num_filters

        self.rnn = RNN(dim_enc, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn)

        out_dim = hidden_size * 2
        self.arc_h = nn.Linear(out_dim, arc_space)
        self.arc_c = nn.Linear(out_dim, arc_space)
        self.attention = BiAffine(arc_space, arc_space, 1, biaffine=biaffine)

        self.type_h = nn.Linear(out_dim, type_space)
        self.type_c = nn.Linear(out_dim, type_space)
        self.bilinear = BiLinear(type_space, type_space, self.num_labels)

    def _get_rnn_output(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, activate_obf=False):
        word_obf = None
        if cfg.model in ["ffobf"]:
            word, word_obf, input_char = self.word_embedd(input_word, input_char, activate_obf)
        elif cfg.model == "seq2seq":
            word, word_obf, input_char = self.word_embedd(input_word, input_char, input_pos, mask=mask, hx=hx)
        else:
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

        # output size [batch, length, arc_space]
        arc_h = F.elu(self.arc_h(output))
        arc_c = F.elu(self.arc_c(output))

        # output size [batch, length, type_space]
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

        if cfg.model in ["ffobf", "seq2seq"]:
            return (arc_h, arc_c), (type_h, type_c), hn, mask, length, word_obf
        return (arc_h, arc_c), (type_h, type_c), hn, mask, length

    def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None,
                activate_obf=False):
        # output from rnn [batch, length, tag_space]
        word_obf = None
        if cfg.model in ["ffobf", "seq2seq"]:
            arc, type, _, mask, length, word_obf = self._get_rnn_output(input_word, input_char, input_pos, mask=mask,
                                                              length=length, hx=hx, activate_obf=activate_obf)
        else:
            arc, type, _, mask, length = self._get_rnn_output(input_word, input_char, input_pos, mask=mask,
                                                              length=length, hx=hx)
        # [batch, length, length]
        out_arc = self.attention(arc[0], arc[1], mask_d=mask, mask_e=mask).squeeze(dim=1)

        # TODO: change the ["ffobf"] to a cfg named obf_model_list
        if cfg.model in ["ffobf", "seq2seq"]:
            return out_arc, type, mask, length, word_obf
        return out_arc, type, mask, length

    def loss(self, input_word, input_char, input_pos, heads, types, mask=None, length=None, hx=None, activate_obf=False):
        # out_arc shape [batch, length, length]
        word_obf = None
        if cfg.model in ["ffobf", "seq2seq"]:
            out_arc, out_type, mask, length, word_obf = self.forward(input_word, input_char, input_pos, mask=mask, length=length, hx=hx, activate_obf=activate_obf)
        else:
            out_arc, out_type, mask, length = self.forward(input_word, input_char, input_pos, mask=mask, length=length, hx=hx)
        batch, max_len, _ = out_arc.size()

        if length is not None and heads.size(1) != mask.size(1):
            heads = heads[:, :max_len]
            types = types[:, :max_len]

        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type

        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(out_arc).long()
        # get vector for heads [batch, length, type_space],
        type_h = type_h[batch_index, heads.t()].transpose(0, 1).contiguous()
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
        child_index = child_index.type_as(out_arc).long()
        # [length-1, batch]
        loss_arc = loss_arc[batch_index, heads.t(), child_index][1:]
        loss_type = loss_type[batch_index, child_index, types.t()][1:]

        if cfg.model in ["ffobf", "seq2seq"]:
            return -loss_arc.sum() / num, -loss_type.sum() / num, word_obf
        return -loss_arc.sum() / num, -loss_type.sum() / num

    def _decode_types(self, out_type, heads, leading_symbolic):
        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type
        batch, max_len, _ = type_h.size()
        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(type_h).long()
        # get vector for heads [batch, length, type_space],
        type_h = type_h[batch_index, heads.t()].transpose(0, 1).contiguous()
        # compute output for type [batch, length, num_labels]
        out_type = self.bilinear(type_h, type_c)
        # remove the first #leading_symbolic types.
        out_type = out_type[:, :, leading_symbolic:]
        # compute the prediction of types [batch, length]
        _, types = out_type.max(dim=2)
        return types + leading_symbolic

    def decode(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, leading_symbolic=0):
        # out_arc shape [batch, length, length]
        out_arc, out_type, mask, length = self.forward(input_word, input_char, input_pos, mask=mask, length=length, hx=hx)
        batch, max_len, _ = out_arc.size()
        # set diagonal elements to -inf
        out_arc = out_arc + torch.diag(out_arc.new_full((max_len, ), -np.inf))
        # set invalid positions to -inf
        if mask is not None:
            # minus_mask = (1 - mask).byte().view(batch, max_len, 1)
            minus_mask = (1 - mask).byte().unsqueeze(2)
            out_arc.masked_fill_(minus_mask, -np.inf)

        # compute naive predictions.
        # predition shape = [batch, length]
        _, heads = out_arc.max(dim=1)

        types = self._decode_types(out_type, heads, leading_symbolic)

        return heads.cpu().numpy(), types.cpu().numpy()

    def decode_mst(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, leading_symbolic=0,
                   activate_obf=False):
        '''
        Args:
            input_word: Tensor
                the word input tensor with shape = [batch, length]
            input_char: Tensor
                the character input tensor with shape = [batch, length, char_length]
            input_pos: Tensor
                the pos input tensor with shape = [batch, length]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            length: Tensor or None
                the length tensor with shape = [batch]
            hx: Tensor or None
                the initial states of RNN
            leading_symbolic: int
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: (Tensor, Tensor)
                predicted heads and types.

        '''
        # out_arc shape [batch, length, length]

        word_obf = None
        if cfg.model in ["ffobf", "seq2seq"]:
            out_arc, out_type, mask, length, word_obf = self.forward(input_word, input_char, input_pos, mask=mask, length=length,
                                                           hx=hx, activate_obf=activate_obf)
        else:
            out_arc, out_type, mask, length = self.forward(input_word, input_char, input_pos, mask=mask, length=length,
                                                           hx=hx, activate_obf=activate_obf)

        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type
        batch, max_len, type_space = type_h.size()

        # compute lengths
        if length is None:
            if mask is None:
                length = [max_len for _ in range(batch)]
            else:
                length = mask.sum(dim=1).long().cpu().numpy()

        type_h = type_h.unsqueeze(2).expand(batch, max_len, max_len, type_space).contiguous()
        type_c = type_c.unsqueeze(1).expand(batch, max_len, max_len, type_space).contiguous()
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

        heads_pred, rels_pred = parser.decode_MST(energy.cpu().numpy(), length, leading_symbolic=leading_symbolic, labeled=True)

        if cfg.model in ["ffobf", "seq2seq"]:
            return heads_pred, rels_pred, word_obf
        return heads_pred, rels_pred

