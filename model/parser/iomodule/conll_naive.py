from __future__ import unicode_literals
import re
from collections import OrderedDict, defaultdict
from tqdm import tqdm, trange
import torch as t
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from conllu.tree_helpers import create_tree
from .utils import pad_totensor

class Vocabulary(object):
    def __init__(self, pad="@#PAD#@", unk="@#UNK#@", root="@#ROOT#@"):
        self._stoi = {pad: 0, unk: 1, root: 2}
        self._itos = [pad, unk, root]
        self.stoe = None
        self.vec_dim = None

    def add(self, token):
        if token is not None and token not in self._stoi:
            self._stoi.update({token: len(self._itos)})
            self._itos.append(token)
        return self.stoi(token)

    def build(self, tokens):
        for token in tokens:
            self.add(token)

    @property
    def size(self):
        return len(self._itos)

    def stoi(self, token):
        if token is None or token not in self._stoi:
            return 1
        return self._stoi[token]

    def load_pretrain(self, path):
        self.stoe = {} # index to embedding
        pretrain = open(path, 'r').read()
        print("Loading pretrained word vectors")
        for line in tqdm(pretrain):
            temp = line.split(' ')
            word = temp[0]
            vec = list(map(float, temp[1:]))
            vec_dim = len(vec)
            if self.vec_dim:
                assert self.vec_dim == vec_dim
            self.vec_dim = vec_dim
            self.stoe.update({word: vec})




DEFAULT_FIELDS = ('id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc')

deps_pattern = r"\d+:[a-z][a-z_-]*(:[a-z][a-z_-]*)?"
MULTI_DEPS_PATTERN = re.compile(r"^{}(\|{})*$".format(deps_pattern, deps_pattern))

class ParseException(Exception):
    pass

class CoNLLPreprocessor(object):
    def __init__(self, fields=DEFAULT_FIELDS):
        self.fields = fields
        self.words = Vocabulary()
        self.utags = Vocabulary()
        self.xtags = Vocabulary()
        self.deprel = Vocabulary()

    def parse(self, path, is_train=False):
        """
        Parse a raw conllu format file into list of tensor
        if is_train, the word it has seen will be taken into account as visible to further evaluation,
        otherwise, the word will be assigned as OOV, <del>it will get an unique index bigger than
         the vocabulary size for you to distinguish it</del> it will get 1 for unknown word
        :param path:
        :param is_train:
        :return:
        """
        print("Parsing {} ...".format(path))
        text = open(path, 'r').read()
        raw_sents = text.split("\n\n")
        sents = []

        for i in trange(len(raw_sents)):
            sent = raw_sents[i]
            parsed_sent = []
            for line in sent.split("\n"):
                if line and not line.strip().startswith("#"):
                    parsed_sent.append(self.parse_line(line, self.fields, is_train))
            length = len(parsed_sent)
            if length == 0:
                continue
            tw = [self.words.stoi('@#ROOT#@')] + [token['form'] for token in parsed_sent]
            tu = [self.words.stoi('@#ROOT#@')] + [token['upostag'] for token in parsed_sent]
            tx = [self.words.stoi('@#ROOT#@')] + [token['xpostag'] for token in parsed_sent]
            th = [token['head'] for token in parsed_sent]
            td = [token['deprel'] for token in parsed_sent]
            tl = length + 1
            sents.append(OrderedDict(length=tl, words=tw, utags=tu,
                                     tags=tx, heads=th, labels=td))

        return sents

    def regroup(self, sents, batch_size):
        return sorted(sents, key=lambda x: x['length'].item())

    def batchify(self, sents, batch_size, batch_first=False):
        sents = self.regroup(sents, batch_size)
        tot = len(sents)
        batched = []
        print("Batching ...")
        for i in tqdm(range(0, tot, batch_size)):
            words, masks = pad_totensor([token['words'] for token in sents[i:i + batch_size]], batch_first=batch_first)
            utags, __ = pad_totensor([token['utags'] for token in sents[i:i + batch_size]], batch_first=batch_first)
            tags, __ = pad_totensor([token['tags'] for token in sents[i:i + batch_size]], batch_first=batch_first)
            heads, __ = pad_totensor([token['heads'] for token in sents[i:i + batch_size]], batch_first=batch_first)
            labels, __ = pad_totensor([token['labels'] for token in sents[i:i + batch_size]], batch_first=batch_first)
            lengths = t.tensor([token['length'] for token in sents[i:i + batch_size]])
            batched.append((words, utags, tags, lengths, masks, heads, labels))
        return batched

    def parse_line(self, line, fields, is_train):
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

            elif field == "form":
                if is_train:
                    value = self.words.add(line[i])
                else:
                    value = self.words.stoi(line[i])

            elif field == "upostag":
                if is_train:
                    value = self.utags.add(line[i])
                else:
                    value = self.utags.stoi(line[i])

            elif field == "xpostag":
                value = parse_nullable_value(line[i])
                if is_train:
                    value = self.xtags.add(line[i])
                else:
                    value = self.xtags.stoi(line[i])

            elif field == 'deprel':
                if is_train:
                    value = self.deprel.add(line[i])
                else:
                    value = self.deprel.stoi(line[i])

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




# def sent_to_tree(sentence):
#     head_indexed = defaultdict(list)
#     for token in sentence:
#         # If HEAD is negative, treat it as child of the root node
#         head = max(token["head"] or 0, 0)
#         head_indexed[head].append(token)
#
#     return create_tree(head_indexed)
#
# def parse_tree(text):
#     result = parse(text)
#
#     if "head" not in result[0][0]:
#         raise ParseException("Can't parse tree, missing 'head' field.")
#
#     trees = []
#     for sentence in result:
#         trees += sent_to_tree(sentence)
#
#     return trees



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

# def serialize_tree(root):
#     def add_subtree(root_token, token_list):
#         for child_token in root_token.children:
#             token_list = add_subtree(child_token, token_list)
#
#         token_list.append(root_token.data)
#         return token_list
#
#     tokens = []
#     add_subtree(root, tokens)
#
#     sorted_tokens = sorted(tokens, key=lambda t: t['id'])
#     lines = []
#     for token_data in sorted_tokens:
#         line = '\t'.join(serialize_field(val) for val in token_data.values())
#         lines.append(line)
#
#     text = '\n'.join(lines)
#     return text