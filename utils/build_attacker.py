import torch
import config

from data.fileio import get_logger
import os
import torch.nn as nn
from net.parser import BiRecurrentConvBiAffine, BiaffineAttnParser
from net.attacker import CtxSeqAttacker
from net.generator import AlltagCopyCtxGenerator
from model.adversarial import Adversarial
from model.obfuscator import Obfuscator
from staffs.trainer import DefaultTrainer
from staffs.meter import AdvMeter, ObfMeter
from staffs.watcher import TbxWatcher
from torch.optim import Adam
from config.utils import MrDict, AttrDict
from data import pipe
from model import get_pretrained_parser

device = config.cfg.device # pylint: disable=no-member
log = get_logger(__name__)


def evaluate(gen):
    
    inputs = pipe.parser_input
    cfg_p = config.cfg.BiaffineAttnParser
    atk_cfg = AttrDict({
        "p_inp_drop": 0.33,
        "inp_dim": sum([cfg_p.char_dim if cfg_p.use_char else 0,
            cfg_p.word_dim if cfg_p.use_word else 0,
            cfg_p.pos_dim if cfg_p.use_pos else 0]),
        "ctx_hs": 512,
        "num_layers": 3,
        "p_rnn": (0.33, 0.33),
        "p_ctx_drop": 0.33
        })
    atk = CtxSeqAttacker(atk_cfg)
    adv = Adversarial(gen, atk).to(device)
    print(adv)
    adv_watcher = TbxWatcher(watch_on=("loss", "rcv_num", "rcv_rate"), tbx_prefix="evaluate_by_attacker/")
    adv_trainer = DefaultTrainer(adv, adv_watcher, AdvMeter, trainer_name="EvaluateByAttacker", optim=Adam, cfg=MrDict({
        "patience": 5,
        "max_decay": 5,
        "lr_decay_rate": 0.75,
        "lr": 1e-3
        }))
    adv_trainer.max_epoch = 1000
    adv_trainer.train()
    val_iter = adv_trainer.network.get_data_iter()[1]
    # trainer.validate(val_iter(inputs.data_train))
    # trainer.validate(val_iter(inputs.data_dev))
    log.info("Evaluate on adversarial test")
    adv_test_meter = adv_trainer.validate(val_iter(inputs.data_test))
    log.info("Evaluate on adversarial train")
    adv_train_meter = adv_trainer.validate(val_iter(inputs.data_train))

    psr = get_pretrained_parser()
    obf = Obfuscator(gen, psr, atk)
    obf_watcher = TbxWatcher(watch_on=("loss",), tbx_prefix="obf/")
    obf_trainer = DefaultTrainer(obf, obf_watcher, ObfMeter, trainer_name="ObfTrainer", optim=Adam)
    val_iter = obf_trainer.network.get_data_iter()[1]
    log.info("Evaluate on obfuscate test")
    obf_test_meter = obf_trainer.validate(val_iter(inputs.data_test))
    log.info("Evaluate on obfuscate train")
    obf_train_meter = obf_trainer.validate(val_iter(inputs.data_train))

    result = {
        "exp_name": config.cfg.exp_name,
        "exp_time": config.cfg.exp_time,
        "adv_acc": adv_test_meter.atk_acc,
        "obf_uas": obf_test_meter.uas,
        "adv_test_meter": adv_test_meter.state_dict(),
        "adv_train_meter": adv_train_meter.state_dict(),
        "obf_test_meter": obf_test_meter.state_dict(),
        "obf_train_meter": obf_train_meter.state_dict(),
        "adv_trainer": adv_trainer.state_dict(),
        "obf_trainer": obf_trainer.state_dict(),
    }
    name = "{exp_name}-{exp_time}-{obf_uas:.4f}-{adv_acc:.4f}.evaluated".format(**result)
    save_path = os.path.join(config.cfg.ext.save_dir, name)
    log.info("Evaluation result saved at %s", save_path)
    torch.save(result, open(save_path, "wb"))
    log.info("Result saved to %s", save_path)

    return result


def get_attack_score(gen):
    inputs = pipe.parser_input
    atk = CtxSeqAttacker(config.CtxSeqAttackerCfg)
    adv = Adversarial(gen, atk).to(device)
    print(adv)
    adv_watcher = TbxWatcher(watch_on=("loss", "rcv_num", "rcv_rate"), tbx_prefix="evaluate_by_attacker/")
    trainer = DefaultTrainer(adv, adv_watcher, AdvMeter, trainer_name="EvaluateByAttacker", optim=Adam, cfg=MrDict({
        "patience": 5,
        "max_decay": 5,
        "lr_decay_rate": 0.75,
        "lr": 1e-3
        }))
    trainer.max_epoch = 30
    trainer.train()
    val_iter = trainer.network.get_data_iter()[1]
    trainer.validate(val_iter(inputs.data_train))
    trainer.validate(val_iter(inputs.data_dev))
    meter = trainer.validate(val_iter(inputs.data_test))
    result = {
        "exp_name": config.cfg.exp_name,
        "exp_time": config.cfg.exp_time,
        "acc": meter.atk_acc,
        "meter": meter,
        "trainer": trainer.state_dict()
    }
    name = "{exp_name}-{exp_time}-{acc:.4f}".format(**result)
    save_path = os.path.join("$ROOT/Project/data/result", name)
    torch.save(result, open(save_path, "wb"))
    log.info("Result saved to %s", save_path)

    return meter.atk_acc


# not trainable, however can be saved/load by torch utils
class Decoupled(nn.Module):
    def __init__(self):
        super().__init__()
        self.atk = CtxSeqAttacker()
        self.psr = BiaffineAttnParser()
        self.gen = AlltagCopyCtxGenerator()

        resume_parser = config.cfg.resume_parser
        if resume_parser:
            self.psr.load_state_dict(torch.load(resume_parser)["network"])
            log.info("Load parser from %s", resume_parser)
        else:
            log.warning("No parser loaded")

        # resume_trainer = config.cfg.resume_trainer
        # if resume_trainer:
        #     state_dict = torch.load(resume_trainer)
        #     self.load_state_dict(filter_state_dict(state_dict["network"]))
        #     log.info("Load trainer from %s", resume_trainer)
        #     if config.cfg.reset_aneal:
        #         self.gen.aneal = MrDict({"step": 0, "t": 1, "nstep": 500, "r": -1e-5}, fixed=True)
        #         log.info("Reset aneal for Generator")


    def get_adversarial_validator(self):
        atk = CtxSeqAttacker()
        adv_validator = Adversarial(self.gen, atk).to(device)
        adv_watcher = TbxWatcher(watch_on=("loss", "rcv_num", "rcv_rate"), tbx_prefix="adv_val/")
        trainer = DefaultTrainer(adv_validator, adv_watcher, AdvMeter, trainer_name="AdvValidator", optim=Adam)
        resume_adversarial = config.cfg.resume_adversarial
        if resume_adversarial:
            state_dict = torch.load(resume_adversarial)
            trainer.load_state_dict(state_dict)
            log.info("Build adversarial from %s", resume_adversarial)
        trainer.max_epoch = 1000
        return trainer

    def get_generator_validator(self):
        obf_net = Obfuscator(self.gen, self.psr, self.atk).to(device)
        obf_watcher = TbxWatcher(watch_on=("loss", "loss_arc", "loss_rel", "loss_atk", "loss_cpy", "loss_ent"), tbx_prefix="obf/")
        obf_trainer = DefaultTrainer(obf_net, obf_watcher, ObfMeter, trainer_name="ObfTrainer", optim=Adam)
        return obf_trainer


    def get_trainers(self):
        adv_net = Adversarial(self.gen, self.atk).to(device)
        obf_net = Obfuscator(self.gen, self.psr, self.atk).to(device)

        adv_watcher = TbxWatcher(watch_on=("loss", "rcv_num", "rcv_rate"), tbx_prefix="adv/")
        obf_watcher = TbxWatcher(watch_on=("loss", "loss_arc", "loss_rel", "loss_atk", "loss_cpy", "loss_ent"), tbx_prefix="obf/")

        adv_trainer = DefaultTrainer(adv_net, adv_watcher, AdvMeter, trainer_name="AdvTrainer", optim=Adam)
        obf_trainer = DefaultTrainer(obf_net, obf_watcher, ObfMeter, trainer_name="ObfTrainer", optim=Adam)

        return obf_trainer, adv_trainer

    def update(self):
        self.gen.update_emb_weight(self.psr.word_embedd.weight,
                                   self.atk.inp_enc.word_embedd.weight)

    def forward(self):
        raise NotImplementedError

