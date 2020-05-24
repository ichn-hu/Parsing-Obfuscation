from __future__ import unicode_literals
import os
import re
from collections import OrderedDict, defaultdict
import torch as t

DEFAULT_FIELDS = ('id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc')

deps_pattern = r"\d+:[a-z][a-z_-]*(:[a-z][a-z_-]*)?"
MULTI_DEPS_PATTERN = re.compile(r"^{}(\|{})*$".format(deps_pattern, deps_pattern))

class ParseException(Exception):
    pass

def parse(text, fields=DEFAULT_FIELDS):
    return [
        [
            parse_line(line, fields)
            for line in sentence.split("\n")
            if line and not line.strip().startswith("#")
        ]
        for sentence in text.split("\n\n")
        if sentence
    ]

def parse_line(line, fields=DEFAULT_FIELDS):
    line = re.split(r"\t| {2,}", line)

    if len(line) == 1 and " " in line[0]:
        raise ParseException("Invalid line format, line must contain either tabs or two spaces.")

    data = OrderedDict()

    for i, field in enumerate(fields):
        # Allow parsing CoNNL-U files with fewer columns
        if i >= len(line):
            break

        if field == "id":
            value = parse_int_value(line[i])

        elif field == "xpostag":
            value = parse_nullable_value(line[i])

        elif field == "feats":
            value = parse_dict_value(line[i])

        elif field == "head":
            value = parse_int_value(line[i])

        elif field == "deps":
            value = parse_paired_list_value(line[i])

        elif field == "misc":
            value = parse_dict_value(line[i])

        else:
            value = line[i]

        data[field] = value

    return data

def parse_int_value(value):
    if value == '_':
        return None
    try:
        return int(value)
    except ValueError:
        return None

def parse_paired_list_value(value):
    if re.match(MULTI_DEPS_PATTERN, value):
        return [
            (part.split(":", 1)[1], parse_int_value(part.split(":", 1)[0]))
            for part in value.split("|")
        ]

    return parse_nullable_value(value)

def parse_dict_value(value):
    if "=" in value:
        return OrderedDict([
            (part.split("=")[0], parse_nullable_value(part.split("=")[1]))
            for part in value.split("|") if len(part.split('=')) == 2
        ])

    return parse_nullable_value(value)

def parse_nullable_value(value):
    if not value or value == "_":
        return None

    return value

def serialize_field(field):
    if field is None:
        return '_'

    if isinstance(field, OrderedDict):
        serialized_fields = []
        for key_value in field.items():
            serialized_fields.append('='.join(key_value))

        return '|'.join(serialized_fields)

    return "{}".format(field)


class Subword():

    def __init__(self, char_vocab, word_vocab, bow, eow, path, train, test=None, val=None):
        text = ""
        if train:
            text += open(os.path.join(path, train), "r").read()
        if test:
            text += open(os.path.join(path, test), "r").read()
        if val:
            text += open(os.path.join(path, val), "r").read()
        parsed = parse(text) # parsed input, https://github.com/EmilStenstrom/conllu

        self.itos = []
        self.stoi = {}
        self.stos = {} # from word to subword representation (list of subword indexes)
        self.char_vocab = char_vocab
        self.word_vocab = word_vocab

        bow = self.char_vocab.stoi.get(bow, len(self.char_vocab.stoi))
        eow = self.char_vocab.stoi.get(eow, len(self.char_vocab.stoi) + 1)

        # build subword vocabulary & stos
        for sent in parsed:
            for word in sent:
                # begin with bow and end with eow, in case of word like 'a'
                form = [bow] + [self.char_vocab.stoi[c] for c in word['form']] + [eow]
                sr = []
                for i in range(0, len(form) - 2):
                    # subword get!
                    part = tuple(form[i:i + 3])
                    if part not in self.stoi:
                        self.stoi[part] = len(self.itos)
                        self.itos.append(part)
                    sr.append(self.stoi[part])
                self.stos[word['form']] = sr

    @property
    def size(self):
        return len(self.stoi)

    def extract(self, minibatch, onehot=True):
        """
        :param minibatch:
        :param onehot: whether you want a onehot-like vector, or bag of subword
        :return: for each word, it returns whether an onehot-like vector of bag of subword,
         indicating the subword it contains
        """
        nbatch, lsent = minibatch.word[0].size()
        ret = t.zeros(nbatch, lsent, self.size if onehot else minibatch.char.size(-1)).type(t.LongTensor)
        if onehot:
            for s in range(nbatch):
                for w in range(lsent):
                    if minibatch.word[2][s][w]:
                        word = self.word_vocab.itos[int(minibatch.word[0][s][w])]
                        for i in self.stos.get(word, []):
                            ret[s][w][i] += 1
        else:
            for s in range(nbatch):
                for w in range(lsent):
                    if minibatch.word[2][s][w]:
                        word = self.word_vocab.itos[int(minibatch.word[0][s][w])]
                        for i, c in enumerate(self.stos.get(word, [])):
                            ret[s][w][i] = c

        if minibatch.word[0].is_cuda:
            ret = ret.cuda()
        return ret