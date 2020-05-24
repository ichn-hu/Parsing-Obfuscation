import torch
import config

from data.fileio import get_logger, save_state
import torch.nn as nn
from net.parser import BiRecurrentConvBiAffine, BiaffineAttnParser
from net.attacker import CtxSeqAttacker
from net.generator import AlltagCopyCtxGenerator, UnkGenerator
from model.adversarial import Adversarial
from model.obfuscator import Obfuscator
from staffs.trainer import DefaultTrainer
from staffs.meter import AdvMeter, ObfMeter
from staffs.watcher import TbxWatcher
from torch.optim import Adam
from config.utils import MrDict
from data import pipe


inputs = pipe.parser_input
device = config.cfg.device # pylint: disable=no-member
log = get_logger(__name__)

def search_based_generator():
    from model.obfuscator import SearchBasedObfuscator
    model = SearchBasedObfuscator().to(device)
    data_iters = model.get_data_iter()
    train_iter = data_iters[0]
    cnt = 0
    for inp in train_iter():
        ret = model(*inp["input"])
        sv = {"inp": inp, "res": ret}
        cnt += 1
        t = save_state(sv, "save-{}.ptr".format(cnt))
        log.info("%s th data saved at %s", cnt, t)


def build_parser():
    psr = BiaffineAttnParser()
    resume_parser = config.cfg.resume_parser
    if resume_parser is not None:
        state_dict = torch.load(resume_parser)["network"]
        psr.load_state_dict(state_dict)
        log.info("Load parser from %s", resume_parser)
    else:
        log.warn("No parser resumed!")

    return psr
