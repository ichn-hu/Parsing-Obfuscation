# pylint: disable=invalid-name
import sys
import os
import time
import uuid
import pickle
import torch
from net.parser import BiaffineAttnParser
from net.attacker import CtxSeqAttacker, FeedForwardAttacker
from model.sequencial import SeqLabelObfuscator, UnkObfuscator, SeqAllObfuscator, SeqCopyObfuscator, AlltagObfuscator
from model.feedforward import FeedForwardObfuscator
from torch.optim import Adam
from data import pipe
import main

import config
from config.utils import MrDict
cfg = config.cfg
cfg_t = MrDict({"batch_size": 32})
cfg_p = config.cfg_p

from data.fileio import get_logger, read_inputs, save_state, load_state

from staffs.trainer import DefaultTrainer, GANTrainer
from staffs.meter import ParsingMeter, AttackingMeter, KpffMeter
from staffs.watcher import Watcher, PGWatcher, DefaultWatcher, AttackingWatcher, KpffWatcher, SeqlabelWatcher, TensorboardWatcher


uid = uuid.uuid4().hex[:6]
logger = get_logger("Main")


def exp_setup():
    print(config.cfg.__dict__)
    logger.info("Experiment %s starts at %s", cfg.exp_name, cfg.exp_time)


def get_iters(inputs, models, model=None):
    def parser_train_iter():
        inputs = pipe.parser_input
        num_batch = inputs.num_data // cfg_t.batch_size
        for _ in range(num_batch):
            batch = inputs.get_batch_tensor(inputs.data_train, cfg_t.batch_size, unk_replace=cfg_t.unk_replace)
            word, char, pos, heads, rels, masks, lengths = batch
            yield {"input": (word, char, pos, masks, lengths, heads, rels)}
            if cfg.under_development:
                break

    def parser_val_iter(dataset):
        inputs = pipe.parser_input
        def iterate():
            for batch in inputs.iterate_batch_tensor(dataset, cfg_t.batch_size):
                word, char, pos, heads, rels, masks, lengths = batch
                yield {"input": (word, char, pos, masks, lengths), "target": {"arcs": heads, "rels": rels}}
                if cfg.under_development:
                    break
        return iterate

    def kpff_train_iter():
        return parser_train_iter()

    def kpff_val_iter(dataset):
        return parser_val_iter(dataset)

    def seqlabel_train_iter():
        return parser_train_iter()

    def seqlabel_val_iter(dataset):
        return parser_val_iter(dataset)

    def seqcopy_train_iter():
        return parser_train_iter()

    def seqcopy_val_iter(dataset):
        return parser_val_iter(dataset)

    def seqall_train_iter():
        return parser_train_iter()

    def seqall_val_iter(dataset):
        return parser_val_iter(dataset)

    def attacking_train_iter():
        generator = models["generator"]
        for batch in parser_train_iter():
            word, char, pos, masks, _, _, _ = batch["input"]
            if cfg.obf_model == "reinforce":
                gen = generator(word, char, pos, masks, cfg.reinforce)
            if cfg.obf_model in ["seqlabel", "alltag_ctx"]:
                gen = generator(word, char, pos, masks)
            else:
                gen = generator(word, char, pos)
            yield {"input": (gen["obf_word"], gen["obf_char"], gen["obf_pos"], masks, gen["obf_mask"], word)}

    def attacking_val_iter(dataset):
        generator = models["generator"]
        def iterate():
            for batch in inputs.iterate_batch_tensor(dataset, cfg_t.batch_size):
                word, char, pos, _, _, masks, _ = batch
                if cfg.obf_model == "reinforce":
                    gen = generator(word, char, pos, masks, cfg.reinforce)
                if cfg.obf_model in ["seqlabel", "alltag_ctx"]:
                    gen = generator(word, char, pos, masks)
                else:
                    gen = generator(word, char, pos)
                yield {"input": (gen["obf_word"], gen["obf_char"], gen["obf_pos"], masks, gen["obf_mask"]), "target": {"ori_word": word}}

        return iterate

    def unkgen_train_iter():
        return None

    def unkgen_val_iter(dataset):
        return parser_val_iter(dataset)

    def alltag_random_train_iter():
        return unkgen_train_iter()

    def alltag_random_val_iter(dataset):
        return unkgen_val_iter(dataset)

    def alltag_ctx_train_iter():
        return seqlabel_train_iter()

    def alltag_ctx_val_iter(dataset):
        return seqlabel_val_iter(dataset)
    
    if model is None:
        model = cfg.model

    train_iter = locals()[model + '_train_iter']
    val_iter = locals()[model + '_val_iter']

    return train_iter, val_iter


# this function will be obsolete
def build_model(model=cfg.model, generator=None):
    ret = MrDict(fixed=False, blob=True)
    inputs = pipe.parser_input

    if model == "gan":
        from utils.build_gan import Gan
        gan = Gan()
        trainer = GANTrainer(gan)
        ret.trainer = trainer


    if model in ["unkgen", "alltag_random", "alltag_ctx"]:
        if model == "unkgen":
            network = UnkObfuscator()
            logger.info("Unk Obfuscator initialized")
        else:
            network = AlltagObfuscator()
            logger.info("Alltag Obfuscator initialized")
        network = network.to(cfg.device)
        ret.network = network

        watcher = SeqlabelWatcher()
        train_iter, val_iter = get_iters(inputs, ret, model)
        trainer = DefaultTrainer(network, watcher, KpffMeter, train_iter, val_iter(inputs.data_dev), Adam)
        if cfg.resume_trainer is not None:
            trainer.load_state_dict(torch.load(cfg.resume_trainer))
            logger.info("Load trainer from %s", cfg.resume_trainer)

        weight = network.parser.word_embedd.weight.data.clone()
        weight.requires_grad = False
        network.generator.update_emb_weight(weight)
        ret.trainer = trainer

    if model in ["seqlabel", "seqcopy"]:
        if model == "seqlabel":
            network = SeqLabelObfuscator()
        elif model == "seqcopy":
            network = SeqCopyObfuscator()

        network = network.to(cfg.device)
        # watcher = SeqlabelWatcher()
        watcher = TensorboardWatcher()
        trainer = DefaultTrainer(network, watcher, KpffMeter)
        if cfg.resume_trainer is not None:
            trainer.load_state_dict(torch.load(cfg.resume_trainer))
            logger.info("Load trainer from %s", cfg.resume_trainer)

        weight = network.parser.word_embedd.weight.data.clone()
        weight.requires_grad = False
        network.generator.word_emb_weight = weight
        network.generator.word_emb_tgt = weight[torch.Tensor(network.generator.tgtwords).long()]

        ret.network = network
        ret.trainer = trainer

    if model == "seqall":
        network = SeqAllObfuscator()
        network = network.to(cfg.device)
        watcher = TensorboardWatcher()
        trainer = DefaultTrainer(network, watcher, KpffMeter)
        if cfg.resume_trainer is not None:
            trainer.load_state_dict(torch.load(cfg.resume_trainer))
            logger.info("Load trainer from %s", cfg.resume_trainer)

        weight = network.parser.word_embedd.weight.data.clone()
        weight.requires_grad = False
        network.generator.word_emb_weight = weight
        network.generator.word_emb_tgt = weight

        ret.network = network
        ret.trainer = trainer

    if model == "kpff":
        network = FeedForwardObfuscator()
        network = network.to(cfg.device)
        watcher = KpffWatcher()
        trainer = DefaultTrainer(network, watcher, KpffMeter)

        if cfg.resume_trainer is not None:
            trainer.load_state_dict(torch.load(cfg.resume_trainer))
            logger.info("Load trainer from %s", cfg.resume_trainer)

        weight = network.parser.word_embedd.weight.data.clone()
        weight.requires_grad = False
        network.generator.word_emb_weight = weight
        network.generator.word_emb_tgt = weight[torch.Tensor(network.generator.tgtwords).long()]

        ret.network = network
        ret.trainer = trainer

    if model == "parser":
        network = BiaffineAttnParser()
        watcher = DefaultWatcher()
        network = network.to(cfg.device)
        trainer = DefaultTrainer(network, watcher, ParsingMeter)
        if cfg.resume_trainer is not None:
            trainer.load_state_dict(torch.load(cfg.resume_trainer))
            trainer.best_model_path = cfg.resume_trainer
            logger.info("Load trainer from %s", cfg.resume_trainer)
        ret.network = network
        ret.trainer = trainer

    if model == "attacking":
        if cfg.atk_model == "ctx":
            network = CtxSeqAttacker()
            logger.info("CtxSeqAttacker initiated")
        else:
            network = FeedForwardAttacker()
            logger.info("FeedForwardAttacker initiated")
        network = network.to(cfg.device)

        if generator is None:
            # resume generator
            tmp = cfg.resume_trainer
            cfg.resume_trainer = cfg.resume_obfuscator
            generator = build_model(cfg.obf_model).network.generator
            # import ipdb
            # ipdb.set_trace()
            cfg.resume_trainer = tmp

        # resume attacker
        if cfg.resume_attacker:
            atk_state = torch.load(cfg.resume_attacker)["network"]
            network.load_state_dict(atk_state)
            logger.info("Attacker loaded from %s", cfg.resume_attacker)

        ret.network = network
        ret.generator = generator
        watcher = AttackingWatcher()
        train_iter, val_iter = get_iters(inputs, ret, "attacking")
        trainer = DefaultTrainer(network, watcher, AttackingMeter, train_iter, val_iter(inputs.data_dev), Adam)
        
        ret.trainer = trainer

    ret.fix()
    return ret


if __name__ == '__main__':
    exp_setup()

    inputs = read_inputs()

    models = build_model()
    trainer = models["trainer"]
    if cfg.model == "search_based_generator":
        from utils.build_model import search_based_generator
        search_based_generator()

    if cfg.model == "load_and_attack":
        from utils.build_gan import load_and_attack
        load_and_attack()

    if cfg.model == "retest_unkgen":
        from utils.build_gan import retest_unkgen
        retest_unkgen()

    if cfg.model == "investigate_unkgen":
        from utils.build_gan import investigate_unkgen
        investigate_unkgen()

    if cfg.model == "investigate_stochasticity":
        from utils.build_gan import investigate_stochasticity
        investigate_stochasticity()

    if cfg.model == "standalone_validation":
        from utils.build_gan import standalone_validation
        standalone_validation()

    if cfg.model == "non_gan":
        from utils.build_gan import non_gan
        non_gan()
    
    if cfg.model == "investigate_non_gan":
        from utils.build_gan import investigate_non_gan
        investigate_non_gan()

    if cfg.model == "decoupled_gan":
        from utils.build_gan import decoupled_gan
        decoupled_gan()

    if cfg.model == "gan":
        try:
            if cfg.atk_validate:
                trainer.network.update()
                atk_trainer = trainer.network.get_adversarial_validator()
                # atk_trainer.train()
                val_iter = atk_trainer.network.get_data_iter()[1]
                atk_trainer.validate(val_iter(inputs.data_train))
                atk_trainer.validate(val_iter(inputs.data_dev))
                atk_trainer.validate(val_iter(inputs.data_test))
            else:
                print(trainer.network)
                trainer.train_by_batch()
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt detected")
        state_dict = trainer.state_dict()
        save_state(state_dict, name="trainer-{}".format(trainer.epoch))
        trainer.validate(inputs.data_dev)
        trainer.validate(inputs.data_test)
    """
    else:
        train_iter, val_iter = get_iters(pipe.parser_input, models)

        if cfg.training:
            try:
                trainer.train(Adam, train_iter, val_iter(inputs.data_dev))
            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt detected")
            state_dict = trainer.state_dict()
            save_state(state_dict, name="trainer-{}".format(trainer.epoch))
        val_meter = trainer.validate(val_iter(inputs.data_dev))
        test_meter = trainer.validate(val_iter(inputs.data_test))
        # from misc.attacking_analysis import analysis
        # analysis(inputs, test_meter)
    """

