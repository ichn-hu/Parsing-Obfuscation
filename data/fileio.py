from .maxio import *
import time
from data import pipe
from misc.play_with_input import read_NNP_vocab
from .fileio import split_data
from .input import Inputs
import config
cfg = config.cfg
# cfg_p = config.cfg_p
# cfg_g = config.cfg_g
log = get_logger("FileIO")


def save_state(item, name=None, suffix=".ptr"):
    if name is None:
        from time import gmtime, strftime
        name = strftime("%m-%d-%H:%M:%S", gmtime())
    # TODO: add something to avoid overwrite
    # TODO: update lagacy code
    sdir = cfg.save_dir or cfg.get_save_dir()
    os.makedirs(sdir, exist_ok=True)
    dp = os.path.join(sdir, name + suffix)
    torch.save(item, open(dp, "wb"))
    log.info("{} saved".format(dp))
    return dp


def load_state(path):
    # TODO: test this function in runtime!
    net, opt = torch.load(path, map_location=cfg.map_location)
    def rename_key(state_dict):
        from collections import OrderedDict
        ret = OrderedDict()
        for key in state_dict:
            if key.startswith("module."):
                ret[key[7:]] = state_dict[key]
            else:
                ret[key] = state_dict[key]
        return ret
    return rename_key(net), rename_key(opt)


def read_inputs():
    """
    TODO: read in part should be put to Inputs
    :param cfg_p:
    :param cfg_g:
    :param device:
    :return:
    """
    if pipe.parser_input is None:
        start = time.time()
        inp = Inputs()
        inp.load(log, cfg.device)
        # inp.word_ids = get_word_ids(inp.word_alphabet, cfg_p.train_path, cfg_g.xpos, cfg_g.top_n)
        # print(inp.word_ids)
        log.info("Reading data in {:.4f} s".format(time.time() - start))
        # print(inp)
        pipe.parser_input = inp
        pipe.num_train_data_a, pipe.train_data_a, pipe.num_train_data_b, pipe.train_data_b = split_data(pipe.parser_input.data_train)
        pipe.NNP_vocab = read_NNP_vocab(pipe.parser_input)

    return pipe.parser_input

