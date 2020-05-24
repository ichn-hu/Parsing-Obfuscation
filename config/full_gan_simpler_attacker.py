import os
import time
import torch
import config.cfg as config
from .utils import MrDict

PROJECT_ROOT = "$ROOT/Project/Homomorphic-Obfuscation"
RANDOM_SEED = 19260817
torch.manual_seed(RANDOM_SEED)


class Global(MrDict):
    data_path = os.path.join(PROJECT_ROOT, "../data")
    work_path = os.path.join(PROJECT_ROOT, "../work")
    cache_path = "/dev/shm/zfhu"
    save_every = True
    device = torch.device('cuda')

    model = "gan"
    exp_name = os.environ.get("exp_name", "full_gan-simpler_attacker")
    exp_time = os.environ.get("exp_time", time.strftime("%m.%d_%H:%M", time.gmtime()))
    save_dir = os.path.join(PROJECT_ROOT, "../work", exp_time + '-' + exp_name)

    map_location = "cpu"

    obf_model = "alltag_ctx"
    atk_model = "ctx"

    resume_parser = "$ROOT/Project/data/pretrained/parser_no_pos-11.14_22:51-best-epoch-103.ptr"
    resume_obfuscator = None # "/disk/scratch1/zfhu/work/11.30_17:00-seqlabel_strthr_AN/trainer-117.ptr"
    resume_attacker = None
    resume_trainer = None
    # resume_trainer = "$ROOT/Project/data/pretrained/12.20_14:55-full_gan-save_every-7.ptr"

    gen_loss_term = ["loss_arc", "loss_rel", "loss_atk", "loss_ent", "loss_cpy"]

    reset_aneal = False
    training = True
    train_one_batch = True and False
    test_one_batch = True and False
    atk_validate = False
    reinforce = False



class GeneratorConfig(config.GeneratorConfig):
    fully_obfuscate = True
    xpos = []
    top_n = 1000
    word_dim = 100
    hidden_size = 128
    use_hloss = True


class ParserConfig(config.ParserConfig):
    use_pos = False

cfg = Global()
cfg_g = GeneratorConfig()
cfg_p = ParserConfig(cfg)
BiaffineAttnParserCfg = cfg_p


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

CtxSeqAttackerCfg = MrDict({
    "p_inp_drop": 0.33,
    "inp_dim": sum([cfg_p.char_dim if cfg_p.use_char else 0,
                    cfg_p.word_dim if cfg_p.use_word else 0,
                    cfg_p.pos_dim if cfg_p.use_pos else 0]),
    "ctx_hs": 128,
    "num_layers": 1,
    "ctx_word_drop": 0.33, # randomly drop context word
    "p_rnn": (0.33, 0.33),
    "p_ctx_drop": 0.33
}, fixed=True, blob=False)

CtxSeqAttackerCfg2 = MrDict({
    "p_inp_drop": 0.33,
    "inp_dim": sum([cfg_p.char_dim if cfg_p.use_char else 0,
                    cfg_p.word_dim if cfg_p.use_word else 0,
                    cfg_p.pos_dim if cfg_p.use_pos else 0]),
    "ctx_hs": 512,
    "num_layers": 3,
    "ctx_word_drop": 0, # randomly drop context word
    "p_rnn": (0.33, 0.33),
    "p_ctx_drop": 0.33
}, fixed=True, blob=False)



FeedForwardAttackerCfg = MrDict({
    "p_inp_drop": 0.33,
    "inp_dim": CtxSeqAttackerCfg.inp_dim,
    "hidden_size": 256
}, fixed=True, blob=False)

BiaffineAttnParserCfg = cfg_p

DefaultTrainerCfg = MrDict({
    "patience": 5,
    "max_decay": 10,
    "lr_decay_rate": 0.75,
    "lr": 1e-3
}, fixed=True, blob=False)

GANTrainerCfg = MrDict({
    "max_epoch": 1000,
    "gen_steps": 1,
    "adv_steps": 3,
    })


AlltagCopyCtxGeneratorCfg = SeqLabelGeneratorCfg.update({
    })

from .seqlabel_cfg import SeqLabelGeneratorCfg
from .unkgen_cfg import UnkGeneratorCfg
