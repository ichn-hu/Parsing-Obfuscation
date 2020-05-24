import torch
import torch.nn as nn
from .embedding import Conv1dCharEmbedding
import infra.config
from .variational_rnn import VarMaskedFastLSTM


class InputEncoder(nn.Module):
    def __init__(self, args: infra.config.Arguments, input: infra.config.Inputs):
        super(InputEncoder, self).__init__()

        input_dim = 0
        kernel_size = 3

        if args.use_word:
            input_dim += args.word_dim
            self.word_embedd = nn.Embedding(input.num_words, args.word_dim, _weight=input.embedd_word)
        else:
            self.word_embedd = None

        if args.use_pos:
            input_dim += args.pos_dim
            self.pos_embedd = nn.Embedding(input.num_pos, args.pos_dim, _weight=input.embedd_pos)
        else:
            self.pos_embedd = None

        if args.use_char:
            input_dim += args.num_filters
            self.char_embedd = nn.Embedding(input.num_chars, args.char_dim, _weight=input.embedd_char)
            self.conv1d = nn.Conv1d(args.char_dim, args.num_filters, kernel_size, padding=kernel_size - 1)
        else:
            self.char_embedd = None

        self.drop = nn.Dropout(p=args.p_in)

        if args.mode == 'RNN':
            raise NotImplemented
        elif args.mode == 'FastLSTM':
            self.rnn = VarMaskedFastLSTM(input_dim, args.hidden_size, args.num_layers, batch_first=True,
                                         bidirectional=True, dropout=args.p_rnn)

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

        input = torch.cat(inputs, dim=2)
        output, hn = self.rnn(input, mask=mask, hx=hx)
        return output, hn


