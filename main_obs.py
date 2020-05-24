import sys
# print(sys.path)
import os
import time
import uuid
import pickle
import torch
from model.reinforce import Seq2SeqPGObfuscator
from model.feedforward import FeedForwardObfuscator
import config
cfg = config.cfg
cfg_t = config.cfg_t
cfg_p = config.cfg_p
sys.path.append(".")
sys.path.append("..")

torch.manual_seed(cfg.seed)

import model.parser.infra as infra
from infra.toolkit import freeze_embedding
from data.fileio import load_state, save_state
from staffs.trainer import ObfTrainer, PGTrainer, Trainer

from staffs.watcher import Watcher, PGWatcher
from data.fileio import get_logger, read_inputs
from torch.nn import DataParallel


uid = uuid.uuid4().hex[:6]
logger = get_logger("Main")


start_epoch = 1


def exp_setup():
    args = infra.Arguments()
    # if args.resume:
        # prefix = args.resume.split('.')[0]
        # temp_args = pickle.load(open(prefix + '.args', 'rb'))
        # temp_args.resume = args.resume
        # args.__dict__.update(temp_args.__dict__)
    # else:
    save_name = "{}_{}".format(args.exp_name, str(infra.cst_time()).replace(' ', '-'))
    save_dir = os.path.join(args.save_to, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    args.save_prefix = os.path.join(save_dir, save_name)
    args.exp_start_time = round(time.time())
    pickle.dump(args, open(args.save_prefix + '.args', "wb"))
    args.get_logger = lambda name: infra.get_logger(name, args.save_prefix + '.log')
    args.device = torch.device('cuda') if args.cuda else torch.device('cpu')
    return args


def build_model(inputs):
    global start_epoch
    if cfg.model == "seq2seq_pg":
        network = Seq2SeqPGObfuscator()
    if cfg.model == "kpff":
        network = FeedForwardObfuscator()
    # resume parser
    parser_state, _ = load_state(cfg.resume_parser)
    network.parser.load_state_dict(parser_state)

    if cfg.model == "kpff":
        # reload the word embedding
        weight = network.parser.word_embedd.weight.data.clone()
        weight.requires_grad = False
        weight = weight.to(cfg.device)
        network.generator.word_emb_weight = weight
        network.generator.word_emb_tgt = weight[torch.Tensor(network.generator.tgtwords).long()]

    network = network.to(cfg.device)
    optimizer = cfg_t.build_optimizer(network.generator.parameters())

    if cfg.resume_generator:
        item = torch.load(cfg.resume_generator)[0]
        if isinstance(item, dict):
            # this is ver2 loader, should be the only way
            network.load_state_dict(item["network"])
            optimizer.load_state_dict(item["optimizer"])
            start_epoch = item["epoch"] + 1
        else:
            network.load_state_dict(item[0])
            optimizer.load_state_dict(item[1])
            start_epoch = 1

    print(network)
    return network, optimizer


def main():
    args = exp_setup()

    args.inputs = read_inputs()

    network, optimizer = build_model(args.inputs)

    num_batches = args.inputs.num_data // args.batch_size + 1

    trainer = Trainer(args)

    for epoch in range(start_epoch, args.num_epochs + 1):
        if cfg.model == "seq2seq_pg":
            dog = PGWatcher(epoch, num_batches)
        else:
            dog = Watcher(epoch, num_batches)
        try:
            trainer.train(network, optimizer, dog, args, epoch=epoch)
            dog.kill()
            if not trainer.evaluate(network, optimizer, args, uid, epoch):
                break
        except KeyboardInterrupt:
            # TODO: handle any error
            print("KeboardInterrupt during {}, saving state ...".format(epoch))
            save_state({
                    "network": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "meter": None,
                    "epoch": epoch
                }, name="Ep{}-temp-save".format(epoch))
            break

        if args.save_every:
            save_state({
                    "network": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "meter": None,
                    "epoch": epoch
                }, name="Ep{}-every-save".format(epoch))


if __name__ == '__main__':
    main()
