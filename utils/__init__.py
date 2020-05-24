from data.fileio import Alphabet
import torch
import math
from torch import nn as nn
from data.fileio import PAD_ID_CHAR, PAD_ID_WORD, MAX_CHAR_LENGTH
from data import pipe
import config
import logging
import sys

def get_logger(name, level=logging.INFO, handler=sys.stdout,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def is_privacy_term(pos):
    # privacy_term = config.cfg.privacy_term
    # ['_PAD_POS', '_ROOT_POS', '_END_POS', 'IN', 'DT', 'NNP', 'CD', 'NN', '``', "''", 'POS', '-LRB-', 'VBN', 'NNS', 'VB', ',', 'CC', '-RRB-', 'VBD', 'RB', 'TO', '.', 'VBZ', 'NNPS', 'PRP', 'PRP$', 'JJ', 'MD', 'VBP', 'VBG', 'RBR', ':', 'WP', 'WDT', 'JJR', 'PDT', 'JJS', 'WRB', '$', 'RP', 'FW', 'RBS', 'EX', '#', 'WP$', 'UH', 'SYM', 'LS']
    # return pos in config.POSTAGSET[privacy_term]
    return pos in ["NNP", "NNPS"]

def to_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.sum().item()
    return float(value)

def inspect_chr(inp):
    s = ""
    for i in inp:
        if i == PAD_ID_CHAR:
            break
        s += pipe.parser_input.char_alphabet.get_instance(i)
    return s

def inspect_sent(inp):
    s = ""
    for i in inp:
        if i == PAD_ID_WORD:
            break
        s += pipe.parser_input.word_alphabet.get_instance(i) + " "
    return s

def inspect_sent_by_char(inp):
    s = ""
    for i in inp:
        s += inspect_chr(i)
    return s

def duplicate_tensor_to_cpu(src, dtype):
    dst = torch.zeros(src.shape, dtype=dtype, device=torch.device('cpu'))
    dst.copy_(src, non_blocking=True)
    return dst


def dynamic_cat_sent(a, b, pad=PAD_ID_WORD):
    if a.size(1) > b.size(1):
        a, b = b, a
    if a.size(1) != b.size(1):
        a = torch.cat([a, a.new_full([a.size(0), b.size(1) - a.size(1)], pad)], dim=1)
    return torch.cat([a, b], dim=0)


def dynamic_char_scatter(src, idx, tgt, mc):
    """
    src is the inp_char, with (bs, len, m), tgt is the target with (tot, MAX_CHAR_LENGTH)
    """
    m = src.size(2)
    if mc > m:

        src = torch.cat([src, src.new_full([src.size(0), src.size(1), mc - m], PAD_ID_CHAR)], dim=2)
        m = mc
    try:
        src.masked_scatter_(idx, tgt[:, :m])
    except:
        import ipdb
        ipdb.set_trace()
    return src


def tensor_map(T, func):
    s = T.shape
    T = T.reshape(-1)
    t = torch.tensor([func(i.item()) for i in T]).type_as(T)
    return t.reshape(*s)


def get_word_by_ids(wids, alphabet: Alphabet):
    """
    wids must be a torch LongTensor!
    :param wids:
    :param alphabet:
    :return:
    """
    words = [alphabet.get_instance(wid.item()) for wid in wids]
    return words


def get_char_tensor(ws, alphabet: Alphabet):
    """
    :param ws: list of string, n
    :param mc: max char
    :param alphabet: char alphabet
    :return: char tensor (len, mc)
    """
    n = len(ws)
    T = torch.ones([n, MAX_CHAR_LENGTH]) * PAD_ID_CHAR
    T = T.type(torch.int64)  # to torch.LongTensor
    mc = 0
    for i, w in enumerate(ws):
        cid = [alphabet.get_index(c) for c in w]
        m = len(cid)
        mc = max(mc, m)
        T[i][:m] = torch.LongTensor(cid)
    import config
    cfg = config.cfg
    return T.to(cfg.device), mc


def get_lut(wa: Alphabet, ca: Alphabet):
    lut = []
    for w in range(wa.size()):
        word = wa.get_instance(w)
        char = [ca.get_index(c) for c in word] + [PAD_ID_CHAR] * (MAX_CHAR_LENGTH - len(word))
        lut.append(char)
    import config
    cfg = config.cfg
    return torch.Tensor(lut, device=cfg.device)

def get_temperature(args):
    args.step += 1
    if args.step % args.nstep == 0:
        t = max(0.5, math.exp(args.r * args.step))
        print("\n--- Temperature {:.4f} -> {:.4f} ---\n".format(args.t, t))
        args.t = t
    return args.t


def word_to_char(word, lut):
    # w of (bs, ws) to c of (bs, ws, cs)
    bs, ws = word.shape
    char = lut[word.reshape(-1)].reshape(bs, ws, -1)
    return char

def build_parser(cfg):
    pass


def build_obfuscator(cfg):
    pass

