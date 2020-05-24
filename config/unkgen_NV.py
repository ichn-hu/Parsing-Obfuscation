# pylint: disable=invalid-name
import os
import time
import config.cfg as config
import torch
from .utils import MrDict


class Global(config.Global):
    """
    contains configuration that shares across configurable components
    or that is prone to change
    """
    # exp_name = "FeedForwardGenerator_1000multitag_with_hloss"
    # exp_name = "FeedForwardGenerator_1000NN"
    exp_name = "kpff_fully"
    model = "kpff"  # choose ["ffobf", "seq2seq"]
    ffobf = True
    use_hloss = True


class TrainingConfig(config.TrainingConfig):
    pass


class GlobalOnBuccleuch(config.GlobalOnBuccleuch, Global):
    under_development = False
    exp_name = "unkgen_NV"
    exp_time = getattr(os.environ, "exp_time", time.strftime("%m.%d_%H:%M", time.gmtime()))
    model = "unkgen"
    resume_parser = "$ROOT/Project/work/parser_no_pos-11.14_22:51/best-epoch-103.ptr"
    reinforce = False
    training = False
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda')

class GeneratorConfig(config.GeneratorConfig):
    fully_obfuscate = True
    xpos = ["NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"] # null tag space means all tags
    top_n = 1000
    word_dim = 100
    hidden_size = 128
    use_hloss = True


class ParserConfig(config.ParserConfig):
    use_pos = False


cfg = GlobalOnBuccleuch()
cfg_g = GeneratorConfig()
cfg_p = ParserConfig(cfg)
cfg_t = TrainingConfig()
BiaffineAttnParserCfg = cfg_p

UnkGeneratorCfg = MrDict({ "random": False }, fixed=True)

DefaultTrainerCfg = MrDict({
    "patience": 5,
    "max_decay": 10,
    "lr_decay_rate": 0.75,
    "lr": 1e-3
}, fixed=True, blob=False)


