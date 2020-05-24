import torch
import os
import pickle
import numpy as np
import gzip
from .maxio import DIGIT_RE, UNK_ID, create_alphabets, read_data_to_tensor, get_batch_tensor,\
    iterate_batch_tensor, Alphabet
import config
cfg = config.cfg
from nltk import ConditionalFreqDist
from collections import defaultdict


def get_word_ids(alphabet: Alphabet, path):
    """
    get word_ids for top N most frequent token in alphabet
    :param alphabet:
    :param xpos:
    :param N:
    :return:
    """

    cfg_g = config.cfg_g
    if cfg_g.fully_obfuscate:
        f = open(path, "r").read()
        pos = cfg_g.xpos
        sents = [t.split("\n") for t in f.split("\n\n")]
        word_list = []
        for sent in sents:
            for token in sent:
                t = token.split('\t')
                if len(t) == 10:
                    word = t[1]
                    xpos = t[4]
                    if len(pos) == 0 or xpos in pos:
                        word_list.append(word)
        word_list = list(set(word_list))
        word_ids = []
        for w in word_list:
            try:
                wid = alphabet.get_index(w)
                if wid == UNK_ID:
                    continue
                word_ids.append(wid)
            except KeyError as e:
                oov_count += 1
        return sorted(word_ids)


    if cfg.model in ["ffobf", "kpff"]:
        f = open(path, "r").read()
        N = cfg_g.top_n
        pos = cfg_g.xpos
        sents = [t.split("\n") for t in f.split("\n\n")]
        pairs = []
        for sent in sents:
            for token in sent:
                t = token.split('\t')
                if len(t) == 10:
                    word = t[1]
                    xpos = t[4]
                    pairs.append((xpos, word))
        cfd = ConditionalFreqDist(pairs)
        word_list = sorted([(t[1], t[0]) for p in pos for t in cfd[p].most_common()])[::-1]
        word_ids = []
        oov_count = 0
        n = 0
        for w in word_list:
            try:
                wid = alphabet.get_index(w[1])
                word_ids.append(wid)
                n += 1
            except KeyError as e:
                oov_count += 1
            if n == N:
                break
        assert len(word_ids) == N
    elif cfg.model == "seq2seq":
        # N = cfg_g.num_out_words
        # word_ids = [0]
        # f = open(path, "r").read()
        # sents = [t.split("\n") for t in f.split("\n\n")]
        # pairs = []
        # for sent in sents:
        #     for token in sent:
        #         t = token.split('\t')
        #         if len(t) == 10:
        #             word = t[1]
        #             xpos = t[4]
        #             pairs.append((xpos, word))
        # cfd = ConditionalFreqDist(pairs)
        # tot = sum([cfd[k].B() for k in cfd])
        # kr = [(key, cfd[key].B() / tot) for key in cfd.keys()]
        # num_key = len(kr)
        # remain = N - 1
        # import ipdb
        # ipdb.set_trace()
        # for i, (key, ratio) in enumerate(kr):
        #     num = remain if i == num_key - 1 else round(ratio * N)
        #     remain -= num
        #     word_ids += [a[1] for a in cfd[key].most_common(num)]
        # assert len(word_ids) == N
        word_ids = alphabet.instances
        assert cfg_g.num_out_words == alphabet.size()

    return sorted(word_ids)


def get_word_alltag(alphabet, inspect=False):
    path = cfg.ext.train_path
    tgtwords = defaultdict(set)

    f = open(path, "r").read()
    sents = [t.split("\n") for t in f.split("\n\n")]
    for sent in sents:
        for token in sent:
            t = token.split('\t')
            if len(t) == 10:
                word = t[1]
                xpos = t[4]
                tgtwords[xpos].add(word)
    
    if inspect:
        tag_freq = [(k, len(v)) for k, v in tgtwords.items()]
        tag_freq = sorted(tag_freq, key=lambda x: x[1], reverse=True)
        print(tag_freq)

    def filter_word(word_iter):
        filtered = []
        for word in word_iter:
            try:
                wid = alphabet.get_index(word)
                if wid == UNK_ID:
                    continue
                filtered.append(wid)
            except KeyError as e:
                pass
        return torch.LongTensor(sorted(filtered)).to(cfg.device)

    ret = {}
    obf_term = cfg.Generator.obf_term
    for xpos in tgtwords.keys():
        if xpos in obf_term:
            ret[xpos] = filter_word(tgtwords[xpos])

    return ret

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

        self.punct_set = None

        self.num_data = 0

        self.get_batch_tensor = None
        self.iterate_batch_tensor = None

        self.word_table = None
        self.char_table = None

        self.NNP_vocab = None

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

    @staticmethod
    def get_cache_blob():
        return os.path.join(cfg.ext.cache_dir, "InputsCache")
    def save(self):
        cached_blob = self.get_cache_blob()

        pickle.dump(self.serialize(), open(cached_blob, 'wb'))

    def load(self, logger, device):
        cached_blob = self.get_cache_blob()
        cfg_p = cfg.BiaffineAttnParser

        if not os.path.exists(cached_blob):
            inputs = Inputs()

            word_dict, word_dim = load_embedding_dict(cfg_p.word_embedding, cfg.ext.pretrained_embedding_path)
            char_dict, char_dim = None, 0
            if cfg_p.char_embedding != 'random':
                char_dict, char_dim = load_embedding_dict(cfg_p.char_embedding, cfg_p.char_path)

            logger.info("Creating Alphabets")
            alphabet_path = os.path.join(cfg.ext.cache_dir, 'alphabets/')
            alphabets = create_alphabets(alphabet_path, cfg.ext.train_path,
                                                     data_paths=[cfg.ext.dev_path, cfg.ext.test_path], max_vocabulary_size=100000, embedd_dict=word_dict)
            inputs.init_alphabets(alphabets, logger=logger)

            logger.info("Reading Data")

            data_train = read_data_to_tensor(cfg.ext.train_path, *alphabets, symbolic_root=True, device=device, remove_invalid_data=True)
            data_dev = read_data_to_tensor(cfg.ext.dev_path, *alphabets, symbolic_root=True, device=device)
            data_test = read_data_to_tensor(cfg.ext.test_path, *alphabets, symbolic_root=True, device=device)
            inputs.data_train, inputs.data_test, inputs.data_dev = data_train, data_test, data_dev
            inputs.num_data = sum(inputs.data_train[1])
            inputs.get_batch_tensor = get_batch_tensor
            inputs.iterate_batch_tensor = iterate_batch_tensor

            if cfg_p.punctuation is not None:
                inputs.punct_set = set(cfg_p.punctuation)
                logger.info("punctuations(%d): %s" % (len(inputs.punct_set), ' '.join(inputs.punct_set)))

            def construct_word_embedding_table():
                word_alphabet = inputs.word_alphabet

                scale = np.sqrt(3.0 / word_dim)
                table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
                table[UNK_ID, :] = np.zeros([1, word_dim]).astype(np.float32) if cfg_p.freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
                oov = 0
                for word, index in word_alphabet.items():
                    if word in word_dict:
                        embedding = word_dict[word]
                    elif word.lower() in word_dict:
                        embedding = word_dict[word.lower()]
                    else:
                        embedding = np.zeros([1, word_dim]).astype(np.float32) if cfg_p.freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
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
                table[UNK_ID, :] = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
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
            # alphabets[0] is the word_alphabet
            inputs.save()
            self.__dict__.update(inputs.__dict__)
        else:
            inputs_dict = pickle.load(open(self.get_cache_blob(), "rb"))
            for attr in inputs_dict:
                if attr.endswith('alphabet'):
                    singleton = False
                    default_value = False
                    if attr.startswith('word'):
                        default_value = True
                        singleton = True
                    if attr.startswith('char'):
                        default_value = True

                    self.__dict__[attr] = Alphabet(attr, default_value=default_value, singleton=singleton)
                    self.__dict__[attr].load_from(inputs_dict[attr])
                else:
                    self.__dict__[attr] = inputs_dict[attr]
