import sys
import time
import config
import os
import tensorboardX as tx
from utils import to_scalar
from data import pipe
from collections import OrderedDict
cfg = config.cfg


class DefaultWatcher(object):
    def __init__(self, print_every=10):
        self.print_every = print_every
        self.current_epoch = 0

        self.num_batch = 0
        self.last_batch_cnt = 0
        self.batch_cnt = 0
        self.num_back = 0
        self.flush = True
        self.is_updated = False

    def start_training(self, epoch, flush=True):
        self.current_epoch = epoch
        self.batch_cnt = 0
        self.start_time = time.time()
        self.flush = flush

    def record(self, output):
        return "loss: {:.4f} loss_arc: {:.4f} loss_type: {:.4f}".format(
            output["loss"].item(), output["loss_arc"].item(), output["loss_rel"].item()
        )

    def update(self, output):
        self.batch_cnt += 1
        self.num_batch += 1
        spend = time.time() - self.start_time
        avg = spend / self.batch_cnt
        remain = (self.last_batch_cnt - self.batch_cnt) * avg
        basic = "train: {}/{} time(spend/remain/avg): {:.1f}/{:.1f}/{:.1f} ".format(self.batch_cnt, self.last_batch_cnt, spend, remain, avg)

        if self.batch_cnt % self.print_every == 0:
            info = basic + self.record(output)
            if self.flush:
                sys.stdout.write("\b" * self.num_back)
                sys.stdout.write(" " * self.num_back)
                sys.stdout.write("\b" * self.num_back)
            else:
                sys.stdout.write("\n")
            sys.stdout.write(info)
            sys.stdout.flush()
            self.num_back = len(info)

    def end_training(self):
        self.last_batch_cnt = self.batch_cnt
        sys.stdout.write("\n")
        sys.stdout.flush()


class TbxWatcher(DefaultWatcher):
    def __init__(self, watch_freq=10, watch_on=OrderedDict(), tbx_prefix="tbx/", save_dir=None):
        super().__init__(watch_freq)
        if save_dir is None:
            save_dir = cfg.ext.save_dir
        self.writer = pipe.tbx_writer or tx.SummaryWriter(os.path.join(save_dir, "log"))
        pipe.tbx_writer = self.writer
        if isinstance(watch_on, tuple) or isinstance(watch_on, list):
            watch_on = OrderedDict([(item, item) for item in watch_on])
        self.watch_on = watch_on
        self.tbx_prefix = tbx_prefix

    def add_scalar(self, name, value):
        self.writer.add_scalar(name, value, self.num_batch)

    def record(self, output):
        info = ""
        for name, item in self.watch_on.items():
            value = output[item]
            scalar = to_scalar(value)
            self.add_scalar(self.tbx_prefix + name, scalar)
            info += "{}: {:.4f} ".format(name, scalar)
        return info

    def end_record(self):
        self.writer.close()

class TensorboardWatcher(DefaultWatcher):
    def __init__(self, print_every=10):
        super().__init__(print_every)
        
        self.writer = tx.SummaryWriter(os.path.join(cfg.get_save_dir(), "log"))
        self.n_iter = 0

    def record(self, output):
        self.n_iter += 1
        if cfg.reinforce:
            self.writer.add_scalar("loss", output["loss"], self.n_iter)
            self.writer.add_scalar("reward", output["rwd"], self.n_iter)
            return "loss: {:.4f} reward: {:.4f}".format(output.loss.item(), output.rwd.item())
        else:
            self.writer.add_scalar("loss/loss", output["loss"], self.n_iter)
            self.writer.add_scalar("loss/arc", output.loss_arc, self.n_iter)
            self.writer.add_scalar("loss/rel", output.loss_rel, self.n_iter)
            base = "loss: {:.4f} loss_arc: {:.4f} loss_type: {:.4f} temperature: {:.4f}".format(
                output.loss.item(), output.loss_arc.item(), output.loss_rel.item(), output.gen_oup.t
            )
            add = ""
            if cfg.model == "seqcopy":
                nrm_obf = output.gen_oup.nrm_msk.sum().item() / output.gen_oup.nrm_msk.shape[0]
                spt_obf = output.gen_oup.obf_mask.sum().item() / output.gen_oup.obf_mask.shape[0]
                # if nrm_obf == 1:
                #    import ipdb
                #    ipdb.set_trace()
                add = " nrm_obf: {:.2f} spt_obf: {:.2f}".format(
                        nrm_obf, spt_obf
                    )
            return base + add

    # def end_training(self):
    #     import os
    #     self.writer.export_scalars_to_json(os.path.join(cfg.get_save_dir(), "log.json"))
    #     self.writer.close()


class KpffWatcher(DefaultWatcher):
    def record(self, output):
        return "loss: {:.4f} loss_arc: {:.4f} loss_type: {:.4f} loss_h: {:.4f}".format(
            output["loss"].item(), output["loss_arc"].item(), output["loss_rel"].item(),
            output["loss_h"].item()
        )


class SeqlabelWatcher(DefaultWatcher):
    def record(self, output):
        if cfg.reinforce:
            return "loss: {:.4f} reward: {:.4f}".format(output.loss.item(), output.rwd.item())
        else:
            return "loss: {:.4f} loss_arc: {:.4f} loss_type: {:.4f} temperature: {:.4f}".format(
                output.loss.item(), output.loss_arc.item(), output.loss_rel.item(), output.gen_oup.t
            )


class AttackingWatcher(DefaultWatcher):
    def record(self, output):
        return "loss: {:.4f}".format(output["loss"].item())


class Watcher(object):
    def __init__(self, epoch, total, print_every=10):

        self.epoch = epoch
        self.total = total
        self.print_every = print_every
        self.loss = 0.
        self.loss_arc = 0.
        self.loss_type = 0.
        self.loss_h = 0.

        self.num_inst = 0.
        self.num_back = 0
        self.init_time = time.time()
        self.cnt_batch = 0

    def update(self, loss, loss_arc, loss_type, loss_h, num_inst):
        self.loss += loss * num_inst
        self.loss_arc += loss_arc * num_inst
        self.loss_type += loss_type * num_inst
        self.loss_h += loss_h * num_inst
        self.num_inst += num_inst
        self.cnt_batch += 1
        if self.cnt_batch % self.print_every == 0:
            self.bark()

    def notify(self):
        try:
            spend = time.time() - self.init_time
            ave_time = spend / self.cnt_batch
            remain = (self.total - self.cnt_batch) * ave_time
            return "{}: train {}/{} loss: total {:.4f} arc {:.4f} type {:.4f} h {:.4f}" \
                   " time: spend {:.4f} average {:.4f} remain {:.4f}".format(cfg.exp_name,
                self.cnt_batch, self.total, self.loss / self.num_inst, self.loss_arc / self.num_inst,
                self.loss_type / self.num_inst, self.loss_h / self.num_inst, spend, ave_time, remain)
        except Exception as e:
            return "notify failed @ " + str(e)

    def bark(self):
        sys.stdout.write("\b" * self.num_back)
        sys.stdout.write(" " * self.num_back)
        sys.stdout.write("\b" * self.num_back)
        notification = self.notify()
        sys.stdout.write(notification)
        sys.stdout.flush()
        self.num_back = len(notification)

    def kill(self):
        self.bark()
        sys.stdout.write("\n")
        sys.stdout.flush()


class PGWatcher(object):
    def __init__(self, epoch, total, print_every=10):
        self.epoch = epoch
        self.total = total
        self.print_every = print_every
        self.loss = 0.

        self.num_inst = 0.
        self.num_back = 0
        self.init_time = time.time()
        self.cnt_batch = 0

    def update(self, loss, num_inst):
        self.loss += loss * num_inst
        self.num_inst += num_inst
        self.cnt_batch += 1
        if self.cnt_batch % self.print_every == 0:
            self.bark()
            self.loss = 0.0
            self.num_inst = 0

    def notify(self):
        try:
            spend = time.time() - self.init_time
            ave_time = spend / self.cnt_batch
            remain = (self.total - self.cnt_batch) * ave_time
            return "{}: train {}/{} loss: total {:.4f} " \
                   " time: spend {:.4f} average {:.4f} remain {:.4f}".format(cfg.exp_name,
                self.cnt_batch, self.total, self.loss / self.num_inst, spend, ave_time, remain)
        except Exception as e:
            return "notify failed @ " + str(e)

    def bark(self):
        sys.stdout.write("\b" * self.num_back)
        sys.stdout.write(" " * self.num_back)
        sys.stdout.write("\b" * self.num_back)
        notification = self.notify()
        sys.stdout.write(notification)
        sys.stdout.flush()
        self.num_back = len(notification)

    def kill(self):
        self.bark()
        sys.stdout.write("\n")
        sys.stdout.flush()

