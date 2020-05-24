import torch
import config
from torch import nn
from net.parser import BiaffineAttnParser
from net.attacker import CtxSeqAttacker
from net.generator import AlltagCopyCtxGenerator
from config.utils import MrDict
from data import pipe

"""
This hybrid model uses a pretrained parser and a pretrained vanilla attacker,
and we use a generator that employs copy mechanism that learns with the
supervision from the two pretrained models, we hope that this generator
could learn a crucial technique of obfuscation, otherwise we should use
statistic correlations to model privacy perservation. Hope everthing is
fine, and gold bless this model.
"""

def resume_pretrained_attacker(attacker: nn.Module, path, resume_mode="evaluated"):
    if resume_mode == "evaluated":
        state_dict = torch.load(path)
        atk_state = state_dict["adv_trainer"]["network"]
        from collections import OrderedDict
        state = OrderedDict()
        for key in atk_state:
            if key.startswith("atk."):
                state[key[4:]] = atk_state[key]
            else:
                state[key] = atk_state[key]
        attacker.load_state_dict(state)
    else:
        raise NotImplementedError

resume_parser = config.cfg.resume_parser

def resume_pretrained_parser(parser: nn.Module, path=resume_parser):
    state_dict = torch.load(path)
    parser.load_state_dict(state_dict["network"])

class Hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        resume_attacker = "$ROOT/v1zhu2/work/" +\
            "evaluate_baseline-random_generator_ctx:0_pri:1_seed:0-02.17_13:02/" +\
            "evaluate_baseline-random_generator_ctx:0_pri:1_seed:0-02.17_13:02-0.9361-0.1668.evaluated"
        resume_parser = "$ROOT/Project/data/pretrain/" +\
            "parser_no_pos-11.14_22:51-best-epoch-103.ptr"
        
        self.parser = BiaffineAttnParser()
        self.attacker = CtxSeqAttacker()
        self.generator = AlltagCopyCtxGenerator()

        resume_pretrained_attacker(self.attacker, resume_attacker)
        resume_pretrained_parser(self.parser, resume_parser)

    def update(self):
        self.generator.update_emb_weight(self.parser.word_embedd.weight,
                                   self.attacker.inp_enc.word_embedd.weight)
    
    def parameters(self):
        # only generator is trained during the training
        return self.generator.parameters()

    def yield_data(self, data):
        word, char, pos, arcs, rels, masks, lengths = data
        if self.training:
            return {"input": (word, char, pos, masks, lengths, arcs, rels)}
        else:
            return {"input": (word, char, pos, masks, lengths), "target": {"arcs": arcs, "rels": rels}}

    def get_data_iter(self, data_train=None, batch_size=32):
        if data_train is None:
            data_train = pipe.parser_input.data_train
        
        def train_iter():
            num_batch = pipe.parser_input.num_data // batch_size
            for _ in range(num_batch):
                batch = pipe.parser_input.get_batch_tensor(data_train, batch_size, unk_replace=0.5)
                word, char, pos, arcs, rels, masks, lengths = batch
                yield {"input": (word, char, pos, masks, lengths, arcs, rels)}
                if config.cfg.train_one_batch:
                    break

        def val_iter(dataset):
            def iterate():
                for batch in pipe.parser_input.iterate_batch_tensor(dataset, batch_size):
                    word, char, pos, arcs, rels, masks, lengths = batch
                    yield {"input": (word, char, pos, masks, lengths), "target": {"arcs": arcs, "rels": rels}}
                    if config.cfg.test_one_batch:
                        break
            return iterate

        return train_iter, val_iter

    def forward(self, inp_word, inp_char, inp_pos, inp_mask, inp_length, arcs=None, rels=None):
        gen = self.generator(inp_word, inp_char, inp_pos, inp_mask)
        psr_word = gen.obf_word_psr_emb if gen.obf_word_psr_emb is not None else gen.obf_word

        psr = self.parser(psr_word, gen.obf_char, gen.obf_pos, inp_mask, inp_length, arcs, rels)
        atk = self.attacker(gen.obf_word, gen.obf_char, gen.obf_pos, inp_mask, gen.pri_mask, inp_word, inp_word_emb=gen.obf_word_atk_emb)

        ret = MrDict(blob=True, fixed=False)
        ret.obf_word = gen.obf_word
        ret.obf_mask = gen.obf_mask
        ret.pri_mask = gen.pri_mask
        ret.cpy_mask = gen.cpy_mask

        loss_term = config.cfg.Generator.loss_term # pylint: disable=no-member
        if "loss_atk" in loss_term:
            ret.rcv_mask = atk.rcv_mask
            ret.rcv_word = atk.rcv_word
            ret.loss_atk = -atk["loss"]

        ret.loss_cpy = gen.cpy_loss
        ret.loss_full_cpy = gen.cpy_full_loss
        ret.loss_ent = gen.ent_loss
        if arcs is None:
            psr_ori = self.parser(inp_word, inp_char, inp_pos, inp_mask, inp_length)
            ret.ori_arcs = psr_ori["arcs"]
            ret.ori_rels = psr_ori["rels"]
            ret.obf_arcs = psr["arcs"]
            ret.obf_rels = psr["rels"]
        else:
            ret.loss_arc = psr["loss_arc"]
            ret.loss_rel = psr["loss_rel"]
            ret.loss = sum([0] + [ret[loss] for loss in loss_term])

        ret.fix()
        return ret

