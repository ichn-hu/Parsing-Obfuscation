# pylint: disable=invalid-name
import sys
sys.path.append("$ROOT/Project/Homomorphic-Obfuscation")
sys.path.append("$ROOT/Project/Homomorphic-Obfuscation/model/parser")
import torch
from torch.optim import Adam
from data.fileio import PAD_ID_WORD, read_inputs, PAD_ID_CHAR
from data.input import get_word_ids
from model.adversarial import FeedForwardAttacker
from staffs import ObfMeter
from main import build_model
import config
import os
cfg = config.cfg
cfg_g = config.cfg_g
cfg_p = config.cfg_p

resume_from = os.path.join(cfg.get_save_dir(), 'Ep10-best-model.ptr')
cfg.resume_generator = resume_from


def sampling_from_generator(generator, bs, ls):
    keywords = generator.keywords
    keywords = torch.Tensor(keywords).long().to(cfg.device)
    sample_index = torch.randint(size=(bs, ls), low=0, high=len(keywords)).to(cfg.device).long()
    src = keywords[sample_index]
    return src


def data_iterator(inst, bs, gen, shuffle=True):
    num = inst.shape[0]
    if shuffle:
        idx = torch.randperm(num)
    else:
        idx = torch.arange(0, num)

    for i in range(num // bs):
        src = inst[idx[i*bs : (i+1)*bs]]
        tgt, _, _, _, _ = gen(src, None, None)
        yield src, tgt


if __name__ == "__main__":
    inputs = read_inputs()
    network, _ = build_model(inputs)
    generator = network.generator
    attacker = FeedForwardAttacker()
    attacker.to(cfg.device)

    lr = 1e-3
    resume_attacker = "$ROOT/Project/buccleuch/" \
                      "work/kpff_fully_hloss/11-07-20:04:41.ptr"
    resume_attacker = None
    if resume_attacker is not None:
        state_items = torch.load(resume_attacker)[0]
        attacker.load_state_dict(state_items["attacker"])
        if "lr" in state_items:
            lr = state_items["lr"]
        print("attacker loaded")

    optimizer = Adam(attacker.parameters(), lr=lr)
    test_bs = 128
    val_bs = 128
    train_bs, ls = 5000, 100
    train_src = sampling_from_generator(generator, train_bs, ls)
    test_src = sampling_from_generator(generator, test_bs, ls)
    val_src = sampling_from_generator(generator, val_bs, ls)

    prev_val_loss = -1
    patience = 0
    lr_decay_cnt = 0
    try:
        for epoch in range(1000):
            bs = 128
            for bi, (src, tgt) in enumerate(data_iterator(train_src, bs, generator)):
                optimizer.zero_grad()
                loss = attacker(src, tgt)["loss"]
                loss.backward()
                optimizer.step()
                print("{} {}: {:.3f}".format(epoch, bi, loss))

            with torch.no_grad():
                loss = 0
                bc = 1e-7
                for src, tgt in data_iterator(val_src, bs, generator):
                    bc += 1
                    loss += attacker(src, tgt)["loss"]
                val_loss = loss / bc
                print("val_loss: {} prev_val_loss: {}".format(val_loss, prev_val_loss))
            if prev_val_loss < 0 or val_loss < prev_val_loss:
                prev_val_loss = val_loss
                with torch.no_grad():
                    acc = 0
                    bc = 1e-7
                    for src, tgt_gold in data_iterator(test_src, bs, generator):
                        bc += 1
                        acc += (attacker(src)["tgt"] == tgt_gold).float().mean()
                    test_acc = acc / bc
                print("test_acc: {:.4f}%".format(test_acc))
                patience = 0
            else:
                patience += 1
            if patience == 10:
                patience = 0
                lr = lr * 0.75
                optimizer = Adam(attacker.parameters(), lr)
                lr_decay_cnt += 1
            if lr_decay_cnt == 10:
                break
    except KeyboardInterrupt:
        from data.fileio import save_state
        save_state({"attacker": attacker.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "lr": lr}, name="attacker_{}".format(epoch))
        print("saved!")


