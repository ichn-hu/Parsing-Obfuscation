import torch
from torch import nn
from data import pipe
import config
from config.utils import MrDict
cfg = config.cfg

from net.attacker import CtxSeqAttacker
from net.generator import AlltagCopyCtxGenerator

inputs = pipe.parser_input

class Adversarial(nn.Module):
    """
    This module encapsulates a generator and an attacker, the parameter of this
    module is the attacker
    gen: takes in (inp_word, inp_char, inp_pos, inp_mask), output obf_word, obf_char, obf_pos, obf_mask, obf_word_emb
    atk: takes in (obf_word, obf_char, obf_pos, inp_mask, obf_mask, inp_word), returns rcv_word, loss
    """
    def __init__(self, gen, atk):
        super().__init__()
        self.gen = gen
        self.atk = atk
        
    def parameters(self):
        return self.atk.parameters()

    def yield_data(self, data):
        word, char, pos, _, _, masks, _ = data
        if self.training:
            return {"input": (word, char, pos, masks)}
        else:
            return {"input": (word, char, pos, masks), "target": {}}

    def get_data_iter(self, data_train=None, batch_size=32):
        if data_train is None:
            data_train = pipe.parser_input.data_train
        def train_iter():
            num_batch = pipe.parser_input.num_data // batch_size
            for _ in range(num_batch):
                batch = pipe.parser_input.get_batch_tensor(data_train, batch_size, unk_replace=0.5)
                word, char, pos, _, _, masks, _ = batch
                yield {"input": (word, char, pos, masks)}
                if cfg.train_one_batch:
                    break

        def val_iter(dataset):
            def iterate():
                for batch in pipe.parser_input.iterate_batch_tensor(dataset, batch_size):
                    word, char, pos, _, _, masks, _ = batch
                    yield {"input": (word, char, pos, masks), "target": {}}
                    if cfg.test_one_batch:
                        break
            return iterate

        return train_iter, val_iter


    def forward(self, inp_word, inp_char, inp_pos, inp_mask):
        gen = self.gen(inp_word, inp_char, inp_pos, inp_mask)
        atk = self.atk(gen.obf_word, gen.obf_char, gen.obf_pos, inp_mask, gen.pri_mask, inp_word)

        ret = MrDict(fixed=False, blob=True)
        ret.loss = atk.loss
        ret.ori_word = inp_word
        ret.obf_word = gen.obf_word
        ret.rcv_word = atk.rcv_word
        ret.rcv_mask = atk.rcv_mask
        ret.obf_mask = gen.obf_mask
        ret.pri_mask = gen.pri_mask
        ret.inp_mask = inp_mask
        ret.rcv_num = ret.rcv_mask.sum().item()
        ret.rcv_rate = ret.rcv_num / (ret.pri_mask.sum().item() + 0.0000001)
        ret.fix()
        
        return ret
