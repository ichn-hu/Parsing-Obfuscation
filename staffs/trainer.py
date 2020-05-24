import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import math
import time
import data.fileio as conllx_data
from data.fileio import get_logger, read_inputs
from data import pipe
import infra
from .meter import ObfMeter, AdvMeter
from .watcher import TbxWatcher
from data.fileio import save_state, load_state
import config
from config.utils import MrDict
from model.adversarial import Adversarial
from net.attacker import CtxSeqAttacker
cfg = config.cfg
device = cfg.device
log = conllx_data.get_logger(__name__)


class DefaultTrainer():
    def __init__(self, network, watcher, meter, train_iter=None, val_iter=None, optim=None, log=None, trainer_name=None, cfg=None):
        if cfg is None:
            cfg = MrDict({
                "patience": 5,
                "max_decay": 10,
                "lr_decay_rate": 0.75,
                "lr": 1e-3
                }, fixed=True, blob=False)
        
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.optim = optim

        self.start_time = time.time()
        self.start_date = time.strftime("%m-%d-%H:%M:%S", time.gmtime())
        self.log = get_logger(trainer_name or "Trainer") if log is None else log
        self.trainer_name = trainer_name or "trainer"

        self.start_epoch = 0
        self.max_epoch = 1000
        self.best_epoch = 0
        self.epoch = 0

        self.patience = cfg.patience
        self.max_decay = cfg.max_decay
        self.lr_decay_rate = cfg.lr_decay_rate
        self.optim_cfg = {"lr": cfg.lr, "eps": 1e-3} # add eps, should be configurable during training
        self.clip = 5.

        self.network = network
        self.watcher = watcher
        self.meter = meter

        self.best_meter = meter()
        self.best_model_path = None
        self.is_updated = False

    def validate(self, val_iter=None, _dataset=None):
        val_iter = val_iter or self.val_iter or self.network.get_data_iter()[1](_dataset or pipe.parser_input.data_dev)
        meter = self.meter()
        with torch.no_grad():
            self.network.eval()
            for data in val_iter():
                inp = data["input"]
                tgt = data["target"]
                oup = self.network(*inp)
                meter.measure(inp, tgt, oup)
            meter.report()
            if meter.is_better_than(self.best_meter):
                self.best_meter = meter
                self.best_epoch = self.epoch
                self.is_updated = True
            else:
                self.is_updated = False
        return meter
    
    def export_generator_output(self, export_filename, val_iter=None, _dataset=None):
        val_iter = val_iter or self.val_iter or self.network.get_data_iter()[1](_dataset or pipe.parser_input.data_dev)

        inputs = read_inputs()
        word_alphabet = inputs.word_alphabet
        pos_alphabet = inputs.pos_alphabet
        rel_alphabet = inputs.rel_alphabet
        
        output_file_path = os.path.join(cfg.ext.save_dir, export_filename)
        if export_filename == "export_test.conll":
            output_file_path = os.environ.get("export_file_to", output_file_path)

        log.info("Exporting obfuscation to file %s", output_file_path)

        with open(output_file_path, "w") as output_file:
            waiting_list = []
            with torch.no_grad():
                self.network.eval()
                for data in val_iter():
                    inp = data["input"]
                    tgt = data["target"]
                    oup = self.network(*inp)

                    ori_word = inp[0].cpu()
                    word = oup["obf_word"].cpu()
                    obf_mask = oup["obf_mask"].cpu()
                    pos = inp[2].cpu()
                    mask = inp[3].cpu()
                    arcs = tgt["arcs"].cpu()
                    rels = tgt["rels"].cpu()

                    waiting_list.append((word, ori_word, pos, mask, arcs, rels, obf_mask))
            for word, ori_word, pos, mask, arcs, rels, obf_mask in waiting_list:
                batch_size, seq_length = word.shape
                for i in range(batch_size):
                    for j in range(1, seq_length):
                        if mask[i][j] < 0.5:
                            break
                        w = word_alphabet.get_instance(word[i, j].item())
                        ow = word_alphabet.get_instance(ori_word[i, j].item())
                        p = pos_alphabet.get_instance(pos[i, j].item())
                        t = rel_alphabet.get_instance(rels[i, j].item())
                        h = arcs[i, j].item()
                        om = obf_mask[i, j].item()
                        output_file.write('%d\t%s\t%s\t%s\t%s\t_\t%d\t%s\t_\t%d\n' % (j, w, ow, p, p, h, t, om))
                    output_file.write("\n")

    def get_optimizer(self, optim=None):
        optim = self.optim if optim is None else optim
        print(self.optim_cfg)
        return optim(self.network.parameters(), **self.optim_cfg)

    def train_one_batch(self, batched_data, optim=None, cache=MrDict()):
        optim = optim or self.optim

        optimizer = cache.optimizer or self.get_optimizer(optim)
        patience = cache.patience or 0
        decay = cache.decay or 0

        optimizer.zero_grad()
        inp = batched_data["input"]
        output = self.network(*inp)
        loss = output["loss"]
        loss.backward(retain_graph=True)
        # TODO: wtf
        clip_grad_norm_(self.network.parameters(), self.clip)
        optimizer.step()
        with torch.no_grad():
            self.watcher.update(output)
   
        cache = MrDict({
            "optimizer": optimizer,
            "patience": patience,
            "decay": decay
            }, blob=True)
        return cache


    def train(self, optim=None, train_iter=None, val_iter=None, retain_graph=False, cache=MrDict()):
        optim = optim or self.optim
        train_iter = train_iter or self.train_iter or self.network.get_data_iter()[0]
        val_iter = val_iter or self.val_iter or self.network.get_data_iter()[1](pipe.parser_input.data_dev)

        optimizer = cache.optimizer or self.get_optimizer(optim)
        patience = cache.patience or 0
        decay = cache.decay or 0

        for epoch in range(self.start_epoch, self.max_epoch):
            self.epoch = epoch
            self.log.info("| exp: %s | epoch: %d | patience: %d(%d) | decay: %d(%d) | lr: %.6f | time: %.2f s |",
                          cfg.exp_name, epoch, patience, self.patience, decay,
                          self.max_decay, self.optim_cfg["lr"], time.time() - self.start_time)

            self.network.train()
            self.watcher.start_training(epoch)
            for data in train_iter():
                optimizer.zero_grad()
                inp = data["input"]
                output = self.network(*inp)
                loss = output["loss"]
                loss.backward(retain_graph=retain_graph)
                # TODO: wtf
                clip_grad_norm_(self.network.parameters(), self.clip)
                optimizer.step()
                with torch.no_grad():
                    self.watcher.update(output)

            self.watcher.end_training()

            self.validate(val_iter)

            if not self.is_updated:
                patience += 1
                self.log.info("Network not getting better, patience %d", patience)
                if patience == self.patience:
                    if self.best_model_path is not None:
                        self.load_state_dict(torch.load(self.best_model_path))
                        self.log.info("Reload model from last saved checkpoint {}".format(self.best_model_path))
                    else:
                        self.log.warn("No saved model found!")
                    patience = 0
                    self.optim_cfg["lr"] *= self.lr_decay_rate
                    self.log.info("Learning rate decayed to %.6f", self.optim_cfg["lr"])
                    optimizer = self.get_optimizer(optim)
                    decay += 1
            else:
                patience = 0
                self.best_model_path = save_state(self.state_dict(), name="{}-best-epoch-{}".format(self.trainer_name, self.epoch))
                self.log.info("Obtained a better model saved at %s", self.best_model_path)

            if decay == self.max_decay:
                self.log.info("Max decay reached, end training")
                break
            
        cache = MrDict({
            "optimizer": optimizer,
            "patience": patience,
            "decay": decay
            }, blob=True)
        return cache

    def state_dict(self):
        state_dict = {
            "network": self.network.state_dict(),
            "epoch": self.epoch,
            "optim_cfg": self.optim_cfg,
            "best_epoch": self.best_epoch,
            "best_meter": self.best_meter.state_dict(),
            "best_model_path": self.best_model_path
        }
        return state_dict

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, tuple):
            # due to legacy fault, can be removed when new model is saved
            state_dict = state_dict[0]
        self.network.load_state_dict(state_dict["network"])
        self.epoch = state_dict["epoch"]
        self.start_epoch = self.epoch
        self.best_epoch = state_dict["best_epoch"]
        self.best_meter = self.meter()
        self.best_meter.load_state_dict(state_dict["best_meter"])
        self.best_model_path = None if "best_model_path" not in state_dict else state_dict["best_model_path"]
        self.optim_cfg.update(state_dict["optim_cfg"])


class GANTrainer():
    def __init__(self, network, cfg=None):
        if cfg is None:
            cfg = config.GANTrainerCfg
        gen_tnr, adv_tnr = network.get_trainers()
        self.gen_tnr = gen_tnr
        self.adv_tnr = adv_tnr
        self.gen_val_iter = self.gen_tnr.network.get_data_iter()[1]
        self.adv_val_iter = self.adv_tnr.network.get_data_iter()[1]
        self.gen_tnr.max_epoch = 0
        self.adv_tnr.max_epoch = 0
        self.network = network
        self.max_epoch = cfg.max_epoch
        self.gen_steps = cfg.gen_steps
        self.adv_steps = cfg.adv_steps
        self.steps_cnt = 0
        self.steps_gen_to_one = 0

        self.best_validate_meter = None
        get_logger('GANTrainer').info("gen_steps: %s adv_steps: %s", self.gen_steps, self.adv_steps)
        self.start_time = time.time()
        self.epoch = 0


    def validate_generator(self, epoch):
        inputs = pipe.parser_input
        atk = CtxSeqAttacker(config.CtxSeqAttackerCfg)
        adv = Adversarial(self.network.gen, atk).to(device)
        adv_watcher = TbxWatcher(watch_on=("loss", "rcv_num", "rcv_rate"), tbx_prefix="adv_val-{}/".format(epoch))
        trainer = DefaultTrainer(adv, adv_watcher, AdvMeter, trainer_name="AdvValidator-{}".format(epoch), optim=Adam, cfg=MrDict({
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
        return meter

    def step_length_update(self):
        previous_steps = self.gen_steps
        self.gen_steps = max(1, round(10 - 0.0000001 * self.steps_cnt ** 2))
        if self.gen_steps == 1:
            self.adv_steps = 10

        if self.gen_steps != previous_steps:
            log.info("step length anealed for generator, from %s to %s", previous_steps, self.gen_steps)

    def train_by_batch_decoupled(self):
        inputs = pipe.parser_input
        gen_cache = MrDict()
        adv_cache = MrDict()
        batch_size = 32

        validate_every = 30
        max_patience = 5
        UAS_bar = 0.85
        patience = 0
        start_select = False

        switch_data = False

        for epoch in range(self.max_epoch):
            if epoch % 30000000 == 0:
               switch_data = not switch_data
               log.info("Switch data is set to %s", str(switch_data))

            print("| exp: %s | epoch: %d | time: %.2f s |" % (cfg.exp_name, epoch, time.time() - self.start_time))

            self.adv_tnr.watcher.start_training(epoch, False)
            self.gen_tnr.watcher.start_training(epoch, False)
            self.adv_tnr.network.train()
            self.gen_tnr.network.train()
            self.network.update()

            self.epoch = epoch
            num_batch_a = pipe.num_train_data_a // batch_size
            num_batch_b = pipe.num_train_data_b // batch_size
            num_batch = min(num_batch_a, num_batch_b)

            for b in range(num_batch):
                self.steps_cnt += 1
                batch_a = inputs.get_batch_tensor(pipe.train_data_a, batch_size, unk_replace=0.5)
                batch_b = inputs.get_batch_tensor(pipe.train_data_b, batch_size, unk_replace=0.5)

                if switch_data:
                    batch_a, batch_b = batch_b, batch_a

                if b % self.gen_steps == 0:
                    gen_cache = self.gen_tnr.train_one_batch(self.gen_tnr.network.yield_data(batch_a), cache=gen_cache)
                if b % self.adv_steps == 0:
                    adv_cache = self.adv_tnr.train_one_batch(self.adv_tnr.network.yield_data(batch_b), cache=adv_cache)

                if config.cfg.train_one_batch:
                    break
                self.step_length_update()

            obf_meter, adv_meter = self.validate(inputs.data_dev)
            # no model selction currently
            self.adv_tnr.watcher.end_training()
            self.gen_tnr.watcher.end_training()
            
            if obf_meter.uas > UAS_bar:
                start_select = True
                log.info("Obf meter yields %s > %s bar, model selection started", obf_meter.uas, UAS_bar)

            if start_select and (epoch + 1) % validate_every == 0:
                meter = self.validate_generator(epoch)
                log.info("Validate the generator yield: {:.2f}%".format(meter.atk_acc * 100))
                if meter.is_better_than(self.best_validate_meter):
                    self.best_validate_meter = meter
                    log.info("Obtained a better validate result")
                else:
                    patience += 1
                if patience == max_patience:
                    break

            if config.cfg.save_every:
                s = save_state(self.state_dict(), name="save_every-{}".format(epoch))
                log.info("%dth epoch saved at %s", epoch, s)


    def train_by_batch(self):
        inputs = pipe.parser_input
        gen_cache = MrDict()
        adv_cache = MrDict()
        batch_size = 32
        for epoch in range(self.max_epoch):
            print("| exp: %s | epoch: %d | time: %.2f s |" % (cfg.exp_name, epoch, time.time() - self.start_time))

            self.adv_tnr.watcher.start_training(epoch, False)
            self.gen_tnr.watcher.start_training(epoch, False)
            self.adv_tnr.network.train()
            self.gen_tnr.network.train()
            self.network.update()

            self.epoch = epoch
            num_batch = inputs.num_data // batch_size
            for b in range(num_batch):
                batch = inputs.get_batch_tensor(inputs.data_train, batch_size, unk_replace=0.5)
                if b % self.gen_steps == 0:
                    gen_cache = self.gen_tnr.train_one_batch(self.gen_tnr.network.yield_data(batch), cache=gen_cache)
                if b % self.adv_steps == 0:
                    adv_cache = self.adv_tnr.train_one_batch(self.adv_tnr.network.yield_data(batch), cache=adv_cache)
            self.validate(inputs.data_dev)
            # no model selction currently
            self.adv_tnr.watcher.end_training()
            self.gen_tnr.watcher.end_training()
            if config.cfg.save_every:
                s = save_state(self.state_dict(), name="save_every-{}".format(epoch))
                log.info("%dth epoch saved at %s", epoch, s)


    def train_by_epoch(self):
        inputs = pipe.parser_input
        gen_cache = MrDict()
        adv_cache = MrDict()
        for epoch in range(self.max_epoch):
            self.network.update()
            self.epoch = epoch
            self.gen_tnr.max_epoch += self.gen_steps
            gen_cache = self.gen_tnr.train(cache=gen_cache, val_iter=self.gen_val_iter(inputs.data_dev))
            self.gen_tnr.start_epoch += self.gen_steps

            self.adv_tnr.max_epoch += self.adv_steps
            adv_cache = self.adv_tnr.train(cache=adv_cache, val_iter=self.adv_val_iter(inputs.data_dev))
            self.adv_tnr.start_epoch += self.adv_steps

    def validate(self, dataset):
        gen_meter = self.gen_tnr.validate(self.gen_val_iter(dataset))
        adv_meter = self.adv_tnr.validate(self.adv_val_iter(dataset))
        return gen_meter, adv_meter
    
    def state_dict(self):
        state_dict = {
                "network": self.network.state_dict(),
            }
        if self.best_validate_meter is not None:
            state_dict["best_validate_meter"] = self.best_validate_meter.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict["network"])
        if self.best_validate_meter is not None:
            self.best_validate_meter.load_state_dict(state_dict["best_validate_meter"])
        self.gen_tnr, self.adv_tnr = self.network.get_trainers()


