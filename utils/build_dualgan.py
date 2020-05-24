import os
import torch
import config

from data.fileio import get_logger
import torch.nn as nn
from net.parser import BiRecurrentConvBiAffine, BiaffineAttnParser
from net.attacker import CtxSeqAttacker
from net.generator import AlltagCopyCtxGenerator, UnkGenerator, AlltagRandomGenerator
from model.adversarial import Adversarial
from model.obfuscator import Obfuscator
from staffs.trainer import DefaultTrainer
from staffs.meter import AdvMeter, ObfMeter, ObfGanMeter
from staffs.watcher import TbxWatcher
from torch.optim import Adam
from config.utils import MrDict
from data import pipe
from nltk import FreqDist

inputs = pipe.parser_input
device = config.cfg.device # pylint: disable=no-member
log = get_logger(__name__)

def filter_state_dict(state_dict):
    return {k: v for k, v in state_dict.items() if k not in ['gen.psr_weight', 'gen.atk_weight']}

# not trainable, however can be saved/load by torch utils
class DualGan(nn.Module):
    def __init__(self):
        super().__init__()
        self.atk0 = CtxSeqAttacker()
        self.atk1 = CtxSeqAttacker()
        # we will have 2 attacker each for each part of the data
        self.psr = BiaffineAttnParser()
        self.gen = AlltagCopyCtxGenerator()

        resume_parser = config.cfg.resume_parser
        if resume_parser:
            self.psr.load_state_dict(torch.load(resume_parser)["network"])
            log.info("Load parser from %s", resume_parser)
        else:
            log.warning("No parser loaded")

        resume_trainer = config.cfg.resume_trainer
        if resume_trainer:
            state_dict = torch.load(resume_trainer)
            self.load_state_dict(filter_state_dict(state_dict["network"]))
            log.info("Load trainer from %s", resume_trainer)
            if config.cfg.reset_aneal:
                self.gen.aneal = MrDict({"step": 0, "t": 1, "nstep": 500, "r": -1e-5}, fixed=True)
                log.info("Reset aneal for Generator")


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
        adv0_net = Adversarial(self.gen, self.atk0).to(device)
        adv0_watcher = TbxWatcher(watch_on=("loss", "rcv_num", "rcv_rate"), tbx_prefix="adv0/")
        adv0_trainer = DefaultTrainer(adv0_net, adv0_watcher, AdvMeter, trainer_name="Adv0Trainer", optim=Adam)

        adv1_net = Adversarial(self.gen, self.atk1).to(device)
        adv1_watcher = TbxWatcher(watch_on=("loss", "rcv_num", "rcv_rate"), tbx_prefix="adv1/")
        adv1_trainer = DefaultTrainer(adv1_net, adv1_watcher, AdvMeter, trainer_name="Adv0Trainer", optim=Adam)

        obf0_net = Obfuscator(self.gen, self.psr, self.atk0).to(device)
        obf0_watcher = TbxWatcher(watch_on=("loss", *config.cfg.gen_loss_term), tbx_prefix="obf0/")
        obf0_trainer = DefaultTrainer(obf0_net, obf0_watcher, ObfGanMeter, trainer_name="Obf0Trainer", optim=Adam)

        obf1_net = Obfuscator(self.gen, self.psr, self.atk1).to(device)
        obf1_watcher = TbxWatcher(watch_on=("loss", *config.cfg.gen_loss_term), tbx_prefix="obf1/")
        obf1_trainer = DefaultTrainer(obf1_net, obf1_watcher, ObfGanMeter, trainer_name="Obf1Trainer", optim=Adam)



        return obf_trainer, adv_trainer

    def update(self):
        self.gen.update_emb_weight(self.psr.word_embedd.weight,
                                   self.atk.inp_enc.word_embedd.weight)

    def forward(self):
        raise NotImplementedError

