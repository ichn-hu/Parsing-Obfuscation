import os
import sys
import numpy as np
import torch
import torch.nn as nn
sys.path.append('.')
sys.path.append('..')
import argparse
import pickle
import gzip
from infra import info
import iomodule.maxio as conllx_data
from iomodule.maxio import get_logger, CoNLLXReader, CoNLLXWriter, DIGIT_RE


def load_embedding_dict(embedding, embedding_path, normalize_digits=True):
    """
    load word embeddings from file
    :param embedding:
    :param embedding_path:
    :return: embedding dict, embedding dimention, caseless
    """
    print("loading embedding: %s from %s" % (embedding, embedding_path))
    if embedding == 'word2vec':
        # loading word2vec
        # word2vec = Word2Vec.load_word2vec_format(embedding_path, binary=True)
        # embedd_dim = word2vec.vector_size
        # return word2vec, embedd_dim
        raise NotImplemented
    elif embedding == 'glove':
        # loading GloVe
        embedd_dim = -1
        embedd_dict = dict()
        with gzip.open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                line = line.decode('utf-8')
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=np.float32)
                embedd[:] = tokens[1:]
                word = DIGIT_RE.sub("0", tokens[0]) if normalize_digits else tokens[0]
                embedd_dict[word] = embedd
        return embedd_dict, embedd_dim
    elif embedding == 'senna':
        # loading Senna
        embedd_dim = -1
        embedd_dict = dict()
        with gzip.open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                line = line.decode('utf-8')
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=np.float32)
                embedd[:] = tokens[1:]
                word = DIGIT_RE.sub("0", tokens[0]) if normalize_digits else tokens[0]
                embedd_dict[word] = embedd
        return embedd_dict, embedd_dim
    elif embedding == 'sskip':
        embedd_dim = -1
        embedd_dict = dict()
        with gzip.open(embedding_path, 'r') as file:
            # skip the first line
            file.readline()
            for line in file:
                line = line.strip()
                try:
                    line = line.decode('utf-8')
                    if len(line) == 0:
                        continue

                    tokens = line.split()
                    if len(tokens) < embedd_dim:
                        continue

                    if embedd_dim < 0:
                        embedd_dim = len(tokens) - 1

                    embedd = np.empty([1, embedd_dim], dtype=np.float32)
                    start = len(tokens) - embedd_dim
                    word = ' '.join(tokens[0:start])
                    embedd[:] = tokens[start:]
                    word = DIGIT_RE.sub("0", word) if normalize_digits else word
                    embedd_dict[word] = embedd
                except UnicodeDecodeError:
                    continue
        return embedd_dict, embedd_dim
    elif embedding == 'polyglot':
        words, embeddings = pickle.load(open(embedding_path, 'rb'))
        _, embedd_dim = embeddings.shape
        embedd_dict = dict()
        for i, word in enumerate(words):
            embedd = np.empty([1, embedd_dim], dtype=np.float32)
            embedd[:] = embeddings[i, :]
            word = DIGIT_RE.sub("0", word) if normalize_digits else word
            embedd_dict[word] = embedd
        return embedd_dict, embedd_dim

    else:
        raise ValueError("embedding should choose from [word2vec, senna, glove, sskip, polyglot]")


class Inputs(object):
    def __init__(self):
        self.word_alphabet = None
        self.num_words = 0
        self.embedd_word = None

        self.char_alphabet = None
        self.num_chars = 0
        self.embedd_char = None

        self.pos_alphabet = None
        self.num_pos = 0
        self.embedd_pos = None

        self.rel_alphabet = None
        self.num_rels = 0

        self.data_train = None
        self.data_test = None
        self.data_dev = None

        self.pred_writer = None
        self.gold_writer = None
        self.punct_set = None

        self.num_data = 0

        self.get_batch_tensor = None
        self.iterate_batch_tensor = None

        self.word_table = None
        self.char_table = None

    def init_alphabets(self, alphabets, logger=None):
        self.word_alphabet, self.char_alphabet, self.pos_alphabet, self.rel_alphabet = alphabets
        self.num_words, self.num_chars, self.num_pos, self.num_rels = tuple(map(lambda x: x.size(), alphabets))
        if logger:
            logger.info("Word Alphabet Size: %d" % self.num_words)
            logger.info("Character Alphabet Size: %d" % self.num_chars)
            logger.info("POS Alphabet Size: %d" % self.num_pos)
            logger.info("Type Alphabet Size: %d" % self.num_rels)

    def __repr__(self):
        repr = []
        for attr in self.__dict__:
            repr.append("{:10}: {:10}".format(attr, str(self.__dict__[attr])[:10]))
        repr.sort()
        tot = len(repr)
        res = ""
        s = 4
        for i in range(0, tot, s):
            res += ' '.join(repr[i: i+s]) + '\n'
        return res

    def serialize(self):
        ignore = ['pred_writer', 'gold_writer']
        attrs = {}
        for attr in self.__dict__:
            if attr in ignore:
                continue
            if attr.endswith('alphabet'):
                attrs[attr] = self.__dict__[attr].get_content()
            else:
                attrs[attr] = self.__dict__[attr]
        return attrs

    def save(self, path="/dev/shm/GN4ParsingInputs"):
        pickle.dump(self.serialize(), open(path, 'wb'))

    def load(self, args, path="/dev/shm/GN4ParsingInputs"):
        if not os.path.exists(path):
            inputs = Inputs()

            word_dict, word_dim = load_embedding_dict(args.word_embedding, args.word_path)
            char_dict, char_dim = None, 0
            if args.char_embedding != 'random':
                char_dict, char_dim = load_embedding_dict(args.char_embedding, args.char_path)

            logger = args.get_logger(__name__)
            logger.info("Creating Alphabets")
            alphabet_path = os.path.join(args.save_to, 'alphabets/')
            alphabets = conllx_data.create_alphabets(alphabet_path, args.train_path,
                    data_paths=[args.dev_path, args.test_path], max_vocabulary_size=100000, embedd_dict=word_dict)

            inputs.pred_writer = CoNLLXWriter(*alphabets)
            inputs.gold_writer = CoNLLXWriter(*alphabets)

            inputs.init_alphabets(alphabets, logger=logger)

            logger.info("Reading Data")

            data_train = conllx_data.read_data_to_tensor(args.train_path, *alphabets, symbolic_root=True, device=args.device)
            data_dev = conllx_data.read_data_to_tensor(args.dev_path, *alphabets, symbolic_root=True, device=args.device)
            data_test = conllx_data.read_data_to_tensor(args.test_path, *alphabets, symbolic_root=True, device=args.device)
            inputs.data_train, inputs.data_test, inputs.data_dev = data_train, data_test, data_dev
            inputs.num_data = sum(inputs.data_train[1])
            inputs.get_batch_tensor = conllx_data.get_batch_tensor
            inputs.iterate_batch_tensor = conllx_data.iterate_batch_tensor

            if args.punctuation is not None:
                inputs.punct_set = set(args.punctuation)
                logger.info("punctuations(%d): %s" % (len(inputs.punct_set), ' '.join(inputs.punct_set)))

            def construct_word_embedding_table():
                word_alphabet = inputs.word_alphabet

                scale = np.sqrt(3.0 / word_dim)
                table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
                table[conllx_data.UNK_ID, :] = np.zeros([1, word_dim]).astype(np.float32) if args.freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
                oov = 0
                for word, index in word_alphabet.items():
                    if word in word_dict:
                        embedding = word_dict[word]
                    elif word.lower() in word_dict:
                        embedding = word_dict[word.lower()]
                    else:
                        embedding = np.zeros([1, word_dim]).astype(np.float32) if args.freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
                        oov += 1
                    table[index, :] = embedding
                print('word OOV: %d' % oov)
                return torch.from_numpy(table)

            def construct_char_embedding_table():
                char_alphabet = inputs.char_alphabet

                if char_dict is None:
                    return None

                scale = np.sqrt(3.0 / char_dim)
                table = np.empty([char_alphabet.size(), char_dim], dtype=np.float32)
                table[conllx_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
                oov = 0
                for char, index, in char_alphabet.items():
                    if char in char_dict:
                        embedding = char_dict[char]
                    else:
                        embedding = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
                        oov += 1
                    table[index, :] = embedding
                print('character OOV: %d' % oov)
                return torch.from_numpy(table)

            inputs.word_table = construct_word_embedding_table()
            inputs.char_table = construct_char_embedding_table()
            inputs.save()
            self.__dict__.update(inputs.__dict__)
        else:
            inputs_dict = pickle.load(open(path, "rb"))
            for attr in inputs_dict:
                if attr.endswith('alphabet'):
                    singleton = False
                    default_value = False
                    if attr.startswith('word'):
                        default_value = True
                        singleton = True
                    if attr.startswith('char'):
                        default_value = True

                    self.__dict__[attr] = conllx_data.Alphabet(attr, default_value=default_value, singleton=singleton)
                    self.__dict__[attr].load_from(inputs_dict[attr])
                else:
                    self.__dict__[attr] = inputs_dict[attr]

            self.pred_writer = CoNLLXWriter(self.word_alphabet, self.char_alphabet, self.pos_alphabet, self.rel_alphabet)
            self.gold_writer = CoNLLXWriter(self.word_alphabet, self.char_alphabet, self.pos_alphabet, self.rel_alphabet)


class Arguments(object):
    """
    Arguments with auto-completion for IDEs
    argname -> default value indicates type, if _type_argname does not exist, and if None, will be ignored
    _type_argname -> explicitly point out the type of argname
    _help_argname -> help of argname
    _choices_argname -> choices for argname
    _required_argname -> required
    _nargs_argname
    TODO: inherent
    """
    def __init__(self, parse_args=True):
        # outer arguments
        try:
            # experiment setup
            self.exp_name = "YetAnotherExperiment"
            self.save_to = "save"
            self.train_path = "/home/zfhu/data/ptb/train.conll"
            self.test_path = "/home/zfhu/data/ptb/test.conll"
            self.dev_path = "/home/zfhu/data/ptb/dev.conll"
            self.save_every = False

            # training scheme configuration
            self.resume = None
            self._type_resume = str
            self._help_resume = "Path to the ptr file for training resume, " \
                                "this file should be in the same directory of args saved"

            self.cuda = True
            self.paralell = False
            self.decode, self._choices_decode = "mst", ["mst", "greedy"]
            self.optimizer = "adam"
            self.objective = "cross_entropy"
            self.learning_rate = 0.01
            self.decay_rate = 0.05
            self.clip = 5.0
            self.gamma = 0.0
            self.eps = 0.0
            self.momentum = 0.9
            self.betas = [0.9, 0.9]
            self.num_epochs = 200
            self.batch_size = 64
            self.schedule = 0
            self.unk_replace = 0.
            self.punctuation = ['.']

            self.word_embedding = 'sskip'
            self.word_path = ''
            self.char_embedding = 'random'
            self.char_path = ''
            self.freeze = False

            # model specification
            self.mode = 'FastLSTM'
            self.model = 'original'

            # EdgeFocusedGraphNetwork
            self.use_efgn = False
            self.efgn_inn_dim = 256

            self.use_word = True
            self.word_dim = 100
            self.use_pos = False
            self.pos_dim = 50
            self.use_char = False
            self.char_dim = 50
            self.p_rnn = [0.33, 0.33]
            self.p_in = 0.33
            self.p_out = 0.33

            self.hidden_size = 256
            self.num_layers = 3
            self.arc_space = 128
            self.type_space = 128
            self.rel_space = 128

            # character level embedding
            self.num_filters = 50
            self.window = 3
        except:
            pass

        if parse_args:
            self.build_args()

        # inner arguments
        # exp_setup
        self.exp_start_time = None
        self.save_prefix = None
        self.get_logger = object()
        #

        # read_data
        self.inputs = Inputs()
        self.device = None


    def build_args(self):
        parser = argparse.ArgumentParser(description="Graph Network for Parsing")
        print(' '.join(sys.argv))
        dict = self.__dict__
        for argname in dict:
            if argname.startswith('_'):
                continue

            default = dict[argname]
            nargs = dict.get('_nargs_' + argname, None)
            argtype = dict.get('_type_' + argname, type(default))

            if argtype == list:
                argtype = type(default[0])
                nargs = '+'

            help = dict.get('_help_' + argname, None)
            choices = dict.get('_choices_' + argname, None)
            required = dict.get('_required_' + argname, False)

            if default is None and argtype == type(None):
                # this is an inner argument
                continue
            elif argtype is bool:
                parser.add_argument('--' + argname, action='store_true', help=help)
            else:
                parser.add_argument('--' + argname, default=default, type=argtype, nargs=nargs,
                                    help=help, choices=choices, required=required)

        args = parser.parse_args()
        self.__dict__.update(args.__dict__)  # 这个是最骚的呜哈哈，不过既然写了这个Arguments类，为啥还要从终端传参呢（不禁陷入思考


