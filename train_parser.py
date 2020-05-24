import sys
import os
import time
import uuid
import pickle
import torch
from net.parser import BiaffineAttnParser
from model.reinforce import Seq2SeqPGObfuscator
from model.feedforward import FeedForwardObfuscator
from torch.optim import Adam

import config
cfg = config.cfg
cfg_t = config.cfg_t
cfg_p = config.cfg_p

torch.manual_seed(cfg.seed)

from data.fileio import get_logger, read_inputs, save_state, load_state

from staffs.trainer import DefaultTrainer
from staffs.meter import ParsingMeter
from staffs.watcher import Watcher, PGWatcher, DefaultWatcher


uid = uuid.uuid4().hex[:6]
logger = get_logger("Main")


def exp_setup():
    logger.info("Experiment %s starts at %s", cfg.exp_name, cfg.exp_time)


def build_model():
    if cfg.model == "seq2seq_pg":
        network = Seq2SeqPGObfuscator()
    if cfg.model == "kpff":
        network = FeedForwardObfuscator()
    if cfg.model == "parser":
        network = BiaffineAttnParser()
        watcher = DefaultWatcher()
        network = network.to(cfg.device)
        trainer = DefaultTrainer(network, watcher, ParsingMeter)
        if cfg.resume_trainer is not None:
            trainer.load_state_dict(torch.load(cfg.resume_trainer))
            trainer.best_model_path = cfg.resume_trainer
            logger.info("Load trainer from %s", cfg.resume_trainer)
        return {
            "network": network,
            "trainer": trainer
        }


def main():
    exp_setup()

    inputs = read_inputs()

    trainer = build_model()["trainer"]

    def train_iter():
        num_batch = inputs.num_data // cfg_t.batch_size
        for _ in range(num_batch):
            batch = inputs.get_batch_tensor(inputs.data_train, cfg_t.batch_size, unk_replace=cfg_t.unk_replace)
            word, char, pos, heads, rels, masks, lengths = batch
            yield {"input": (word, char, pos, masks, lengths, heads, rels)}

    def val_iter(dataset):
        def iterate():
            for batch in inputs.iterate_batch_tensor(dataset, cfg_t.batch_size):
                word, char, pos, heads, rels, masks, lengths = batch
                yield {"input": (word, char, pos, masks, lengths), "target": {"arcs": heads, "rels": rels}}
        return iterate

    try:
        trainer.train(Adam, train_iter, val_iter(inputs.data_dev))
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt detected")
    state_dict = trainer.state_dict()
    save_state(state_dict, name="trainer-{}".format(trainer.epoch))
    trainer.validate(val_iter(inputs.data_test))

if __name__ == '__main__':
    main()
