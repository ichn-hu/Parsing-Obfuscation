import sys
sys.path.append("/mnt/e/Projects/Homomorphic-Obfuscation/model/parser")
import numpy as np
import nltk
import config
from collections import defaultdict
from nltk import FreqDist, ConditionalFreqDist
from data.fileio import read_inputs

from infra.config import Arguments, Inputs

cfg = config.cfg_p
inp = read_inputs()


def read_conll(path):
    f = open(path, "r").read()
    sents = [t.split("\n") for t in f.split("\n\n")]
    return sents


def get_pair(sents, omit_first=False):
    pairs = []
    for sent in sents:
        is_first = True
        for token in sent:
            if omit_first and is_first:
                is_first = False
                continue

            t = token.split('\t')
            if len(t) == 10:
                word = t[1]
                xpos = t[4]
                pairs.append((xpos, word))
            else:
                print(token)
    return pairs


def collect(pairs):
    dict = defaultdict(list)
    for xpos, word in pairs:
        dict[xpos].append(word)
    return dict


def analysis(path):
    print(path)
    sents = read_conll(path)
    pairs = filter(lambda X: X[1].istitle(), get_pair(sents, True))
    cfd = ConditionalFreqDist(pairs)
    return cfd


# for path in [cfg.train_path, cfg.dev_path, cfg.test_path]:
#     analysis(path)

train_cfd = analysis(cfg.train_path)
dev_cfd = analysis(cfg.dev_path)
N = 100

def get_words(cfd, xpos, N):
    return {t[0] for t in cfd[xpos].most_common(N)}

def oov_ratio(N):
    alphabet = {t[0] for t in train_cfd['NN'].most_common(N)}
    alphabet2 = {t[0] for t in dev_cfd['NN'].most_common(N)}
    print(1 - len(alphabet2.intersection(alphabet)) / N)


def sentence_coverage(words, sents):
    N = len(sents)
    n = 0
    tot = 0
    for sent in sents:
        ok = False
        for token in sent:
            t = token.split('\t')
            if len(t) == 10:
                w = t[1]
                if w in words:
                    ok = True
                    tot += 1
        if ok:
            n += 1
    print("Sent coverage: {:.03f}, Average Occ: {:.03f}".format(n / N, tot / N))


