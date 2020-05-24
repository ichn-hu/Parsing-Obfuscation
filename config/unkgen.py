import os
import time
import torch
import config.cfg as config
from .utils import MrDict

PROJECT_ROOT = "$ROOT/Project/Homomorphic-Obfuscation"
RANDOM_SEED = 19260817
torch.manual_seed(RANDOM_SEED)

STAGSET = {
        # "ALL":
        # '_PAD_POS', '_ROOT_POS', '_END_POS', 'IN', 'DT', 'NNP', 'CD', 'NN', '``', "''", ('POS', 10), ('-LRB-', 11), ('VBN', 12), ('NNS', 13), ('VB', 14), (',', 15), ('CC', 16), ('-RRB-', 17), ('VBD', 18), ('RB', 19), ('TO', 20), ('.', 21), ('VBZ', 22), ('NNPS', 23), ('PRP', 24), ('PRP$', 25), ('JJ', 26), ('MD', 27), ('VBP', 28), ('VBG', 29), ('RBR', 30), (':', 31), ('WP', 32), ('WDT', 33), ('JJR', 34), ('PDT', 35), ('JJS', 36), ('WRB', 37), ('$', 38), ('RP', 39), ('FW', 40), ('RBS', 41), ('EX', 42), ('#', 43), ('WP$', 44), ('UH', 45), ('SYM', 46), ('LS', 47)])

        "all": ['IN', 'DT', 'NNP', 'CD', 'NN', 'POS', 'VBN', 'NNS', 'VB', 'CC', 'VBD', 'RB', 'TO', 'VBZ', 'NNPS', 'PRP', 'PRP$', 'JJ', 'MD', 'VBP', 'VBG', 'RBR', 'WP', 'WDT', 'JJR', 'PDT', 'JJS', 'WRB', '$', 'RP', 'FW', 'RBS', 'EX', 'WP$', 'UH', 'SYM', 'LS'],
        "NN": ['NN', 'NNP', 'NNP', 'NNPS'],
        "NNP": ['NNP', 'NNPS']
        }

class Global(MrDict):
    data_path = os.path.join(PROJECT_ROOT, "../data")
    # automatically dermine!
    work_path = os.path.join(PROJECT_ROOT, "../ostrom_work")
    cache_path = "/dev/shm/zfhu"
    save_every = True
    device = torch.device('cuda')

    # model = "non_gan"
    model = "retest_unkgen"
    # model = "investigate_unkgen"
    exp_name = os.environ.get("exp_name", "unkgen_0.0")
    exp_time = os.environ.get("exp_time", time.strftime("%m.%d_%H:%M", time.gmtime()))
    save_dir = os.path.join(PROJECT_ROOT, "../work", exp_time + '-' + exp_name)

    map_location = "cpu"

    privacy_term = os.environ.get("privacy_term", "NNP")
    obf_model = "alltag_ctx"
    atk_model = "ctx"

    gen_loss_term = ["loss_arc", "loss_rel"]
    loss_full_cpy_weight = float(os.environ.get("loss_full_cpy_weight", 0))

    resume_parser = "$ROOT/Project/data/pretrained/parser_no_pos-11.14_22:51-best-epoch-103.ptr"
    resume_obfuscator = None # "/disk/scratch1/zfhu/work/11.30_17:00-seqlabel_strthr_AN/trainer-117.ptr"
    resume_attacker = None
    resume_trainer = None # "$ROOT/Project/data/pretrained/12.17_17:51-gan_cpyloss_no_adv-save_every-92.ptr"
    resume_adversarial = None # "$ROOT/Project/work/12.19_00:19-attacking_gan_generator/AdvValidator-best-epoch-74.ptr"
    resume_result = "$ROOT/Project/data/result/random_obf:NNP_pri:NNP_rate:1.0-01.03_22:45-0.9377-0.1727"

    reset_aneal = True
    training = True
    train_one_batch = False
    test_one_batch = False
    atk_validate = True
    reinforce = False

    def __init__(self):
        super().__init__()
        import config.utils as utils
        self.work_path = utils.get_work_path()
        self.save_dir = os.path.join(self.work_path, self.exp_time + '-' + self.exp_name)
        print(self.work_path)
        print(self.save_dir)

UnkGeneratorCfg = MrDict({
    "random": False,
    "model": os.environ.get("gen_model", "UnkGenerator" or "AlltagRandomGenerator"),
    "unk_rate": float(os.environ.get("unk_rate", 1.0)),
    }, fixed=True)

class GeneratorConfig(config.GeneratorConfig):
    fully_obfuscate = True
    # xpos = ["NN"]
    # xpos = ["NNP", "NNPS"]
    # xpos = ["NN", "NNS", "NNP", "NNPS"]
#     xpos = ["NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"] # null tag space means all tags
    # xpos = []
    xpos = POSTAGSET[os.environ.get("obf_term", "NNP")]
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

BiaffineAttnParserCfg = cfg_p

DefaultTrainerCfg = MrDict({
    "patience": 5,
    "max_decay": 10,
    "lr_decay_rate": 0.75,
    "lr": 1e-3
}, fixed=True, blob=False)

GANTrainerCfg = MrDict({
    "max_epoch": 1000,
    "gen_steps": 1300,
    "adv_steps": 1
    })


AlltagCopyCtxGeneratorCfg = SeqLabelGeneratorCfg.update({
    })

from .seqlabel_cfg import SeqLabelGeneratorCfg
