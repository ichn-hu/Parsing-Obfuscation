from collections import defaultdict
# from data.fileio import read_inputs
from nltk import FreqDist

from data.input import get_word_alltag
from data import pipe


def get_tags():
    inputs = pipe.parser_input
    tgtwords = get_word_alltag(inputs.word_alphabet, inspect=True)


# TRAIN_PATH = "$ROOT/Project/data/ptb/train.gold.conll"
# DEV_PATH = "$ROOT/Project/data/ptb/dev.gold.conll"
# TEST_PATH = "$ROOT/Project/data/ptb/test.gold.conll"
is_NNP = lambda pos: pos == 'NNP' or pos == 'NNPS'

class NNP_vocab(object):
    def __init__(self, words, word_alphabet):
        self.fd = FreqDist(words)
        self.wa = word_alphabet
        oov_cnt = 0
        OOV_words = set()
        for word in self.fd.keys():
            wid = word_alphabet.get_index(word)
            if wid == 0: # for UNK_ID
                oov_cnt += 1
                OOV_words.add(word)
        oov_rate = oov_cnt / self.fd.B()
        print("OOV rate {:.2f}%".format(oov_rate * 100))

    def get_weight(self, wids):
        pass

    def get_score(self, wid):
        word = self.wa.get_instance(wid)
        try:
            scr = 1 / self.fd.freq(word)
        except:
            # import ipdb
            # ipdb.set_trace()
            print(wid, word)
            return 0
        return scr


def read_NNP_vocab(inputs, filepath=None):
    import config

    if filepath is None:
        filepath = config.cfg.ext.train_path
    f = open(filepath, "r").read()
    sents = f.split("\n\n")
    sent_inst = []
    all_words = FreqDist()
    NNP_words = FreqDist()
    num_sent = len(sents)
    num_sent_with_NNP = 0
    for sent in sents:
        tokens = sent.split("\n")
        if len(tokens) == 0:
            continue
        words = []
        pos_tags = []
        has_NNP = False
        for token in tokens:
            fields = token.split('\t')
            if len(fields) != 10:
                print(token)
                continue
            all_words[fields[1]] += 1
            words.append(fields[1])
            pos_tags.append(fields[4])
            if is_NNP(fields[4]):
                NNP_words[fields[1]] += 1
                has_NNP = True
        if has_NNP:
            num_sent_with_NNP += 1
    print("{:.2f}% out of {} has NNP words".format(num_sent_with_NNP / num_sent * 100, num_sent))
    word_alphabet = inputs.word_alphabet
    return NNP_vocab(NNP_words, word_alphabet)

# read_NNP_vocab(TRAIN_PATH)
# read_NNP_vocab(TEST_PATH)
# read_NNP_vocab(DEV_PATH)
