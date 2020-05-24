"""
In this file, we implement the basic module for config (for argparser support)
usage:
from config import Global, ParserInputs, Trainer
cfg = Global()
cfg_pi = ParserInputs()
cfg_tr = Trainer()
etc... always initialize what you want at the beginning

!!! DON'T REPEAT YOURSELF !!!
!!! INHERENT INSTEAD OF COPY AND PASTE !!!

TODO: (medium) add the basic config
TODO: solve the os importing problem
"""
# pylint: disable=no-self-argument, missing-docstring, invalid-name, line-too-long
import os
import sys
import argparse
import torch
import math
import time

join = os.path.join


class Configurator(object):
    """
    Arguments with auto-completion for IDEs
    Because the build_args part is useless, it is now removed
    """
    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        for k, v in state_dict:
            self.__dict__[k] = v


class Global(Configurator):
    """
    contains configuration that shares across configurable components
    or that is prone to change
    """
    # exp_time = None # should be "%m.%d-%H:%M" e.g. "11.12_22:22"
    exp_time = time.strftime("%m.%d_%H:%M", time.gmtime())
    exp_name = "FeedForwardGenerator_1000NN"
    data_path = "/mnt/e/Projects/data"
    work_path = "/mnt/e/Projects/work"
    cache_path = "/dev/shm/zfhu"
    model = "ffobf"  # choose ["ffobf", "seq2seq"]
    resume_parser = None
    resume_generator = None
    resume_trainer = None
    ffobf = True
    seq2seq = False
    map_location = None
    training = True

    device = torch.device('cpu')
    under_development = True
    seed = 19260817

    def get_save_dir(self):
        dp = os.path.join(self.work_path, self.exp_time + '-' + self.exp_name)
        os.makedirs(dp, exist_ok=True)
        return dp


class GlobalOnServer(Global):
    data_path = "/disk/scratch1/zfhu/data"
    work_path = "/disk/scratch1/zfhu/work"
    under_development = False

    def __init__(self):
        self.device = torch.device('cuda')


class GlobalOnBuccleuch(GlobalOnServer):
    data_path = "$ROOT/Project/buccleuch/data"
    work_path = "$ROOT/Project/buccleuch/work"

    def __init__(self):
        self.device = torch.device('cuda')
        # self.resume_parser = self.work_path + "/save/pretrained/95_10.ptr"
        self.map_location = 'cpu'


class TrainingConfig(Configurator):
    optimizer = "adam"
    learning_rate = 0.001
    decay_rate = 0.75
    clip = 5.0
    gamma = 0.0
    eps = 1e-4
    momentum = 0.9
    betas = [0.9, 0.9]
    num_epochs = 1000
    batch_size = 32
    schedule = 10
    unk_replace = 0.5
    objective = "cross_entropy"

     # for the annealing, we follow the Categorical paper setting
    r = -1e-5
    n = 0
    N = 500
    s = 0
    previous_t = 1

    def get_temperature(self, s=1, train=True):
        """
        get the annealed temperature
        :param s: steps taken
        :return:
        """
        self.s += 1
        if not train or self.s % 500 != 0:
            return self.previous_t
        if self.n < self.N:
            self.n += 1
            return self.previous_t
        self.n = 0
        self.previous_t = max(0.5, math.exp(self.r * self.s))
        return self.previous_t

    def build_optimizer(cfg, params):
        opt = cfg.optimizer
        lr = cfg.learning_rate

        params = filter(lambda param: param.requires_grad, params)
        if opt == 'adam':
            return torch.optim.Adam(params, lr=lr, betas=cfg.betas, weight_decay=cfg.gamma, eps=cfg.eps)
        if opt == 'sgd':
            return torch.optim.SGD(params, lr=lr, momentum=cfg.momentum, weight_decay=cfg.gamma, nesterov=True)
        if opt == 'adamax':
            return torch.optim.Adamax(params, lr=lr, betas=cfg.betas, weight_decay=cfg.gamma, eps=cfg.eps)
        else:
            raise ValueError('Unknown optimization algorithm: %s' % opt)

    def skip_training(self):
        return False


class GeneratorConfig(Configurator):
    pass


class ParserConfig(Configurator):
    mode = 'FastLSTM'
    use_word = True
    word_dim = 100
    use_pos = True
    pos_dim = 100
    use_char = True
    char_dim = 100

    use_dist = False

    p_rnn = [0.33, 0.33]
    p_in = 0.33
    p_out = 0.33

    hidden_size = 512
    num_layers = 3
    arc_space = 512
    rel_space = 128

    # character level embedding
    num_filters = 100
    window = 3

    punctuation = ['.', '``', "''", ':', ',']
    word_embedding = 'sskip'
    word_path = ''
    char_embedding = 'random'
    char_path = ''
    freeze = False
    train_path = None
    dev_path = None
    test_path = None
    save_to = None

    def __init__(self, use_pos, use_word=True, use_char=True):
        self.use_pos = use_pos
        self.use_word = use_word
        self.use_char = use_char
        print("Parser CFG initiated, use_pos={}, use_word={}, use_char={}".format(use_pos, use_word, use_char))

    # cached_blob = "GN4ParsingInputs"

    # def __init__(self, data_path, cache_path):
    #     self.train_path = join(data_path, "ptb/train.conll")
    #     self.dev_path = join(data_path, "ptb/dev.conll")
    #     self.test_path = join(data_path, "ptb/test.conll")
    #     self.word_path = join(data_path, "sskip.eng.100.gz")
    #     self.save_to = join(cache_path, "Alphabets")
    #     self.cached_blob = join(cache_path, self.cached_blob)


