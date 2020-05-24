from .utils import MrDict
import config.cfg as config


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
    model = "attacking"

    # obf_model = "kpff"
    obf_model = "seqlabel"
    # obf_model = "unkgen"

    atk_model = "ctx"
    # atk_model = "ff"

    # exp_name = "attacking_kpff_no_hloss_with_ctx"
    # exp_name = "attacking_kpff_with_hloss_with_ctx"
    # exp_name = "attacking_kpff_with_hloss_with_ff"
    # exp_name = "attacking_kpff_no_hloss_with_ff"

    # exp_name = "attacking_seqlabel_with_ff"
    # exp_name = "attacking_seqlabel_with_ctx"
    exp_name = "attacking_seqlabel_strthr_AN_with_ctx"
    # exp_name = "attacking_unkgen_AN_absolute_with_ctx"
    # exp_name = "attacking_unkgen_noun_absolute_with_ctx"

    exp_time = "12.01_17:03"
    resume_parser = "$ROOT/Project/work/parser_no_pos-11.14_22:51/best-epoch-103.ptr"
    # no hloss trained
    # resume_obfuscator = "$ROOT/Project/buccleuch/work/kpff_fully_no_hloss-11.16_12:46/trainer-88.ptr"
    # hloss trained
    # resume_obfuscator = "$ROOT/Project/buccleuch/work/kpff_fully_with_hloss-11.16_15:42/trainer-62.ptr"
    # reinforce seqlabel obfuscator
    # resume_obfuscator = "$ROOT/Project/buccleuch/work/seqlabel_reinforce_correct-11.19_16:12/trainer-57.ptr"

    # resume_obfuscator = "$ROOT/Project/buccleuch/work/kpff_fully_with_hloss_NE-11.22_10:41/trainer-49.ptr"
    # resume_obfuscator = "$ROOT/Project/buccleuch/work/seqlabel_strthr-11.19_11:34/trainer-114.ptr"

    # resume_obfuscator = "$ROOT/Project/buccleuch/work/seqlabel_reinforce_NE-11.22_10:34/best-epoch-43.ptr"
    # resume_obfuscator = "$ROOT/Project/buccleuch/work/seqlabel_strthr_NE-11.22_22:48/best-epoch-50.ptr"

    resume_obfuscator = "$ROOT/Project/buccleuch/work/11.30_16:47-seqlabel_strthr_NE/best-epoch-69.ptr"

    # resume_attacker = "$ROOT/Project/buccleuch/work/attacking_seqlabel_strthr_NE_with_ctx-11.23_20:55/trainer-94.ptr"
    # resume_attacker = "$ROOT/Project/work/attacking_seqlabel_strthr_NE_with_ctx-11.28_22:12/trainer-117.ptr"
    resume_attacker = None
    training = True
    resume_obfuscator = None
    resume_trainer = None
    reinforce = False


class GeneratorConfig(config.GeneratorConfig):
    fully_obfuscate = True
    # xpos = ["NN"]
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

from .seqlabel_cfg import SeqLabelGeneratorCfg
from .unkgen_cfg import UnkGeneratorCfg
