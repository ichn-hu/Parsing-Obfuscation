import logging
import re
import sys
import torch
import numpy as np
import gzip
from allennlp.common.file_utils import cached_path
import shutil
import os
from colorama import Fore
import time


def create_save_folder(save_path, force=False, ignore_patterns=[]):
    if os.path.exists(save_path):
        print(Fore.RED + save_path + Fore.RESET
              + ' already exists!', file=sys.stderr)
        if not force:
            ans = input('Do you want to overwrite it? [y/N]:')
            if ans not in ('y', 'Y', 'yes', 'Yes'):
                exit(1)
        from getpass import getuser
        tmp_path = '/tmp/{}-experiments/{}_{}'.format(getuser(),
                                                      os.path.basename(save_path),
                                                      time.time())
        print('move existing {} to {}'.format(save_path, Fore.RED
                                              + tmp_path + Fore.RESET))
        shutil.copytree(save_path, tmp_path)
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    print('create folder: ' + Fore.GREEN + save_path + Fore.RESET)
    shutil.copytree('.', os.path.join(save_path, 'src'), symlinks=True,
                    ignore=shutil.ignore_patterns('*.pyc', '__pycache__',
                                                  '*.path.tar', '*.pth',
                                                  '*.ipynb', '.*', 'data',
                                                  'save', 'save_backup',
                                                  save_path,
                                                  *ignore_patterns))


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


def fetch_features_by_heads(f, heads):
    n, l, fsize = f.shape
    ret = []
    for i in range(n):
        head_f = f[i][heads[i]]
        head_dependent_concat = torch.cat((head_f, f[i]), dim=1)
        ret.append(head_dependent_concat.view(1, l, fsize*2))
    return torch.cat(ret, dim=0)


def get_attachment_correct_count(arc_pred, label_pred, heads, labels, masks, punct_masks):
    punct_masks = torch.ones_like(punct_masks) - punct_masks
    punct_masks = punct_masks[:, 1:]
    masks = masks[:, 1:].byte()

    if masks.shape[1] > heads.shape[1]:
        # ignore extra masks
        masks = masks[:, :heads.shape[1]]

    head_pred_correct = ((heads == arc_pred) * masks).sum()
    head_pred_correct_nopunc = ((heads == arc_pred) * masks * punct_masks).sum()
    right_arc = (heads == arc_pred)
    label_pred_correct = ((labels == label_pred) * right_arc * masks).sum()
    label_pred_correct_nopunc = ((labels == label_pred) * right_arc * masks * punct_masks).sum()
    return {"head_pred_correct": head_pred_correct.item(),
            "head_pred_correct_nopunc": head_pred_correct_nopunc.item(),
            "label_pred_correct": label_pred_correct.item(),
            "label_pred_correct_nopunc": label_pred_correct_nopunc.item(),
            "total_token": masks.sum().item(),
            "total_token_nopunc": (masks * punct_masks).sum().item()}


def adjust_learning_rate(optimizer, init_lr, epoch, n_iter, curr_iter, decay):
    lr = init_lr * np.power(decay, ((epoch - 1) * n_iter + curr_iter) / 5000.0)
    # print("change lr to:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def find_token_id(dic, token):
    for i in dic:
        if dic[i] == token:
            return i


def load_wordvec(fname, vocab, word_dim):
    from gensim.models import KeyedVectors
    word_vectors = KeyedVectors.load_word2vec_format(fname, binary=False) 
    emb_weights = torch.zeros((vocab.get_vocab_size(), word_dim))
    for word in word_vectors.vocab:
        idx = vocab.get_token_index(word)
        emb_weights[idx] = torch.from_numpy(word_vectors[word])
    return emb_weights


def load_pretrained_words_data(embeddings_filename, vocab):
    """
    Load pre-trained word embedding, and rearrange the order w.r.t the vocabulary, and return the data that can be
    replaced with the embedding.weight.data
    :param embeddings_filename:
    :param vocab:
    :return:
    """
    words = dict()
    emb_dim = None
    with gzip.open(cached_path(embeddings_filename), 'rb') as embeddings_file:
        for line in embeddings_file:
            fields = line.decode('utf-8').strip().split(' ')
            if len(fields) == 0:
                continue
            word = fields[0]
            if emb_dim is None:
                emb_dim = len(fields) - 1
                if emb_dim < 10:  # my pretrained file is poisonous 😭
                    emb_dim = None
            else:
                assert emb_dim == len(fields) - 1, "{}, {}".format(emb_dim, len(fields) - 1)
            words.update({word: [float(i) for i in fields[1:]]})
    print("Embedding dim: {}".format(emb_dim))
    tokens = vocab.get_index_to_token_vocabulary("tokens")
    n_tokens = len(tokens)
    data = []
    for i in tokens:
        if tokens[i] in words:
            data.append(words[tokens[i]])
        else:
            data.append([0] * emb_dim)
    return torch.tensor(data), emb_dim


class EmailTheException():
    def __init__(self, sys_arg, title=None, send=False):
        content = ""
        content += "=" * 80 + "\n"
        content += "Failure Detected!" + "\n"
        content += "=" * 80 + "\n"
        content += ' '.join(sys_arg) + "\n"
        content += "\n"
        import traceback
        content += traceback.format_exc()
        content += "\n"
        content += "=" * 80 + "\n"
        print(content)
        if send:
            import requests
            import json
            title = "[异常通知] " + title
            requests.post("http://115.159.159.232:22339/", json=json.dumps({
                "token": "send2it", "title": title, "content": content}))


