# pylint: disable=invalid-name
import config.cfg as config
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
    model = "seqlabel"
    exp_name = "seqlabel_strthr_AN"
    exp_time = "11.30_17:00"
    resume_parser = "$ROOT/Project/buccleuch/work/parser_no_pos-11.14_22:51/best-epoch-103.ptr"
    reinforce = False

class GeneratorConfig(config.GeneratorConfig):
    fully_obfuscate = True
    # xpos = ["NNP", "NNPS"]
    xpos = ["NN", "NNS", "NNP", "NNPS"]
    top_n = 1000
    word_dim = 100
    hidden_size = 128
    use_hloss = True


class ParserConfig(config.ParserConfig):
    use_pos = False


cfg = GlobalOnBuccleuch()
cfg_g = GeneratorConfig()
cfg_p = ParserConfig(cfg.data_path, cfg.cache_path)
cfg_t = TrainingConfig()
BiaffineAttnParserCfg = cfg_p

"""
I hypothesis that pos taggin is very important for the parser to make prediction on obfuscated words, I will try to do some experiment to show if it is the case
"""

SeqLabelGeneratorCfg = MrDict({
    "inp_dim": sum([cfg_p.char_dim if cfg_p.use_char else 0,
                    cfg_p.word_dim if cfg_p.use_word else 0,
                    cfg_p.pos_dim if cfg_p.use_pos else 0]),
    "ctx_fs": 512,
    "num_layers": 3,
    "bidirectional": True,
    "p_rnn_drop": cfg_p.p_rnn,
    "p_inp_drop": .33,
    "p_ctx_drop": .33,
    "device": cfg.device
}, fixed=True, blob=True)

InpEncoderCfg = MrDict({
    "use_word": True,
    "use_char": True,
    "use_pos": False,
    "word_dim": cfg_p.word_dim,
    "char_dim": cfg_p.char_dim,
    "pos_dim": cfg_p.pos_dim,
    "num_filters": cfg_p.num_filters
}, fixed=True, blob=True)

