from .utils import MrDict
import os
import config.cfg as config
import time


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
    model = "alltag_random"
    exp_name = "alltag_random"
    exp_time = getattr(os.environ, "exp_time", time.strftime("%m.%d_%H:%M", time.gmtime())
)
    resume_parser = "$ROOT/Project/work/parser_no_pos-11.14_22:51/best-epoch-103.ptr"
    # resume_obfuscator = "/disk/scratch1/zfhu/work/11.30_16:47-seqlabel_strthr_NE/best-epoch-69.ptr"
    resume_attacker = None
    resume_trainer = None

    reinforce = False
    training = False
    under_development = False


class GeneratorConfig(config.GeneratorConfig):
    fully_obfuscate = True
    xpos = []
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

InpEncoderCfg = MrDict({
    "use_word": True,
    "use_char": True,
    "use_pos": False,
    "word_dim": cfg_p.word_dim,
    "char_dim": cfg_p.char_dim,
    "pos_dim": cfg_p.pos_dim,
    "num_filters": cfg_p.num_filters
}, fixed=True, blob=True)

CtxSeqAttackerCfg = MrDict({
    "p_inp_drop": 0.33,
    "inp_dim": sum([cfg_p.char_dim if cfg_p.use_char else 0,
                    cfg_p.word_dim if cfg_p.use_word else 0,
                    cfg_p.pos_dim if cfg_p.use_pos else 0]),
    "ctx_hs": 512,
    "num_layers": 3,
    "p_rnn": (0.33, 0.33),
    "p_ctx_drop": 0.33
}, fixed=True, blob=False)

FeedForwardAttackerCfg = MrDict({
    "p_inp_drop": 0.33,
    "inp_dim": CtxSeqAttackerCfg.inp_dim,
    "hidden_size": 256
}, fixed=True, blob=False)

DefaultTrainerCfg = MrDict({
    "patience": 5,
    "max_decay": 10,
    "lr_decay_rate": 0.75,
    "lr": 1e-3
}, fixed=True, blob=False)


from .seqlabel_cfg import SeqLabelGeneratorCfg
from .unkgen_cfg import UnkGeneratorCfg
