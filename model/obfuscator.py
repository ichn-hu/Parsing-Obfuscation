import torch
from torch import nn
from data.fileio import NUM_SYMBOLIC_TAGS, get_logger, UNK_ID

from net.parser import BiaffineAttnParser
from net.generator.unk import SearchBasedGenerator
from utils import word_to_char
from data import pipe
import config
from config.utils import MrDict
cfg = config.cfg
device = cfg.device
inputs = pipe.parser_input
log = get_logger("Obfuscator")


class Obfuscator(nn.Module):
    def __init__(self, gen, psr, atk):
        super().__init__()
        self.gen = gen
        self.psr = psr
        self.atk = atk
    
    def update(self):
        if hasattr(self.gen, "update_emb_weight"):
            self.gen.update_emb_weight(self.psr.word_embedd.weight)

    def parameters(self):
        return self.gen.parameters()

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
                if cfg.train_one_batch:
                    break

        def val_iter(dataset):
            def iterate():
                for batch in pipe.parser_input.iterate_batch_tensor(dataset, batch_size):
                    word, char, pos, arcs, rels, masks, lengths = batch
                    yield {"input": (word, char, pos, masks, lengths), "target": {"arcs": arcs, "rels": rels}}
                    if cfg.test_one_batch:
                        break
            return iterate

        return train_iter, val_iter


    def forward(self, inp_word, inp_char, inp_pos, inp_mask, length, arcs=None, rels=None):
        gen = self.gen(inp_word, inp_char, inp_pos, inp_mask, length=length)
        psr_word = gen.obf_word_psr_emb if gen.obf_word_psr_emb is not None else gen.obf_word

        psr = self.psr(psr_word, gen.obf_char, gen.obf_pos, inp_mask, length, arcs, rels)

        ret = MrDict(blob=True, fixed=False)
        ret.obf_word = gen.obf_word
        ret.obf_mask = gen.obf_mask
        ret.pri_mask = gen.pri_mask
        ret.cpy_mask = gen.cpy_mask

        if "loss_atk" in cfg.Generator.loss_term:
            atk = self.atk(gen.obf_word, gen.obf_char, gen.obf_pos, inp_mask,
                    gen.pri_mask, inp_word, inp_word_emb=gen.obf_word_atk_emb)
            ret.rcv_mask = atk.rcv_mask
            ret.rcv_word = atk.rcv_word
            ret.loss_atk = -atk["loss"]

        ret.loss_cpy = gen.cpy_loss
        ret.loss_full_cpy = gen.cpy_full_loss
        ret.loss_ent = gen.ent_loss
        if arcs is None:
            psr_ori = self.psr(inp_word, inp_char, inp_pos, inp_mask, length)
            ret.ori_arcs = psr_ori["arcs"]
            ret.ori_rels = psr_ori["rels"]
            ret.obf_arcs = psr["arcs"]
            ret.obf_rels = psr["rels"]
        else:
            ret.loss_arc = psr["loss_arc"]
            ret.loss_rel = psr["loss_rel"]
            ret.loss = sum([0] + [ret[loss_term] for loss_term in cfg.Generator.loss_term if loss_term != "loss_ent"])

        ret.fix()
        return ret


class SearchBasedObfuscator(nn.Module):
    def __init__(self):
        super().__init__()
        self.psr = BiaffineAttnParser()
        self.gen = SearchBasedGenerator()
        if cfg.resume_parser is not None:
            self.psr.load_state_dict(torch.load(cfg.resume_parser)["network"])
            log.info("parser loaded from %s", cfg.resume_parser)

        self.gen.update_emb_weight(self.psr.word_embedd.weight)

    def yield_data(self, data):
        word, char, pos, arcs, rels, masks, lengths = data
        if self.training:
            return {"input": (word, char, pos, masks, lengths, arcs, rels)}
        else:
            return {"input": (word, char, pos, masks, lengths), "target": {"arcs": arcs, "rels": rels}}

    def get_data_iter(self, batch_size=32):
        inputs = pipe.parser_input

        def train_iter():
            num_batch = inputs.num_data // batch_size
            for _ in range(num_batch):
                batch = inputs.get_batch_tensor(inputs.data_train, batch_size, unk_replace=0.5)
                word, char, pos, arcs, rels, masks, lengths = batch
                yield {"input": (word, char, pos, masks, lengths, arcs, rels)}
                if cfg.train_one_batch:
                    break

        def val_iter(dataset):
            def iterate():
                for batch in inputs.iterate_batch_tensor(dataset, batch_size):
                    word, char, pos, arcs, rels, masks, lengths = batch
                    yield {"input": (word, char, pos, masks, lengths), "target": {"arcs": arcs, "rels": rels}}
                    if cfg.test_one_batch:
                        break
            return iterate

        return train_iter, val_iter

    def forward(self, inp_word, inp_char, inp_pos, inp_mask, inp_length, arcs=None, rels=None):
        def batchify(words, pos, mask, length, batch_size=32):
            batched_mask = mask.reshape(1, -1).expand(batch_size, -1)
            batched_pos = pos.reshape(1, -1).expand(batch_size, -1)
            batched_length = length.reshape(1, -1).expand(batch_size, -1)
            for i in range(0, len(words) // batch_size * batch_size, batch_size):
                batched_word = torch.tensor(words[i: i+batch_size]).to(device)
                batched_char = word_to_char(batched_word, self.gen.lut)
                yield batched_word, batched_char, batched_pos, batched_mask, batched_length

        gld = self.psr(inp_word, inp_char, inp_pos, inp_mask, inp_length)
        ret = []
        for i in range(inp_word.size(0)):
            arcs = gld["arcs"][i:i+1]
            rels = gld["rels"][i:i+1]
            word = inp_word[i:i+1]
            char = inp_char[i:i+1]
            pos = inp_pos[i:i+1]
            mask = inp_mask[i:i+1]
            length = inp_length[i:i+1]
            obf = self.gen(word, char, pos, mask)
            UAS, LAS, WRD = [], [], []
            from tqdm import tqdm
            for inp in tqdm(batchify(obf.obf_word, pos, mask, length)):
                psr = self.psr(*inp)
                gld_arcs = arcs.expand(inp[0].shape)
                gld_rels = rels.expand(inp[0].shape)
                arcs_matched = (gld_arcs == psr["arcs"]) & inp[3].byte()
                rels_matched = (gld_rels == psr["rels"]) & inp[3].byte()
                uas = arcs_matched.float().sum(dim=1) / length.item()
                las = ((rels_matched & arcs_matched) & inp[3].byte()).float().sum(dim=1) / length.item()
                UAS.append(uas)
                LAS.append(las)
                WRD.append(inp[0])
            cat = torch.cat # pylint: disable=no-member

            uas = cat(UAS, dim=0)
            las = cat(LAS, dim=0)
            wrd = cat(WRD, dim=0)

            save_best = min(100, word.size(0))
            sorted_uas, uas_idx = torch.sort(uas)
            sorted_las, las_idx = torch.sort(las)

            obf.uas = uas[uas_idx[-save_best:]]
            obf.las = las[las_idx[-save_best:]]
            obf.uas_wrd = wrd[uas_idx[-save_best:]]
            obf.las_wrd = wrd[las_idx[-save_best:]]

            obf.word = word
            obf.pos = pos
            obf.mask = mask
            ret.append(obf)
            print_uas = " ".join(["{:.2f}".format(i.item()) for i in uas[uas_idx[-10:]]])
            print_las = " ".join(["{:.2f}".format(i.item()) for i in las[las_idx[-10:]]])
            print("len: {} Top 10 UAS: {}, LAS: {}".format(length.item(), print_uas, print_las))
        return ret

