import torch
import config
cfg = config.cfg

from staffs.trainer import DefaultTrainer
from staffs.meter import KpffMeter
from staffs.watcher import KpffWatcher
from model.feedforward import FeedForwardObfuscator
from main import build_model, read_inputs, kpff_train_iter, kpff_val_iter
from data import pipe

saved_trainer = "$ROOT/Project/buccleuch/work/kpff_fully_no_hloss-11.16_12:46/trainer-88.ptr"
cfg.model = "kpff"
cfg.resume_trainer = saved_trainer

read_inputs()
trainer = build_model()["trainer"]

meter = trainer.best_meter
meter.report()
trainer.validate(kpff_val_iter(pipe.parser_input.data_test))
