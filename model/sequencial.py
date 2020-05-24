import torch
from torch import nn
from net.generator import SeqLabelGenerator, UnkGenerator, SeqAllGenerator, SeqCopyGenerator, AlltagRandomGenerator, AlltagCtxGenerator
from net.parser import BiRecurrentConvBiAffine, BiaffineAttnParser
from data.fileio import NUM_SYMBOLIC_TAGS, get_logger, UNK_ID

from data import pipe
import config
from config.utils import MrDict
cfg = config.cfg
log = get_logger("SeqObf")


class SeqLabelObfuscator(nn.Module):
    def __init__(self):
        super().__init__()
        from data.input import get_word_ids
        keywords = get_word_ids(pipe.parser_input.word_alphabet, cfg.ext.train_path)
        tgtwords = keywords
        log.info("%d keywords initiated", len(keywords))
        self.generator = SeqLabelGenerator(keywords, tgtwords)
        self.parser = BiaffineAttnParser()
        self.reinforce = cfg.reinforce

        if cfg.resume_parser is not None:
            state_dict = torch.load(cfg.resume_parser)
            # import ipdb
            # ipdb.set_trace()
            self.parser.load_state_dict(state_dict["network"])
            log.info("Resume parser from %s", cfg.resume_parser)
        else:
            log.warning("Missing pretrained parser")

    def parameters(self):
        return self.generator.parameters()

    def forward(self, word, char, pos, mask, length, arcs=None, rels=None):
        ret = MrDict(fixed=False, blob=True)
        gen = self.generator(word, char, pos, mask, cfg.reinforce)
        ret.gen_oup = gen

        if arcs is not None:
            # during training
            if cfg.reinforce:
                loss_smp = self.parser(gen.obf_word, gen.obf_char, gen.obf_pos, mask, length, arcs, rels)
                loss_ctc = self.parser(gen.ctc_word, gen.ctc_char, gen.ctc_pos, mask, length, arcs, rels)
                # TODO: we should remove the minus, but ?
                rwd_smp_arc = -loss_smp["loss_arc_batch"]
                rwd_smp_rel = -loss_smp["loss_rel_batch"]
                rwd_ctc_arc = -loss_ctc["loss_arc_batch"]
                rwd_ctc_rel = -loss_ctc["loss_rel_batch"]
                g = (rwd_smp_arc - rwd_ctc_arc) + (rwd_smp_rel - rwd_ctc_rel)
                g.detach_()
                ret.loss = -(g * gen.plog.sum(dim=-1)).mean()
                ret.rwd = g.mean()
            else:
                loss = self.parser(gen.obf_word_emb, gen.obf_char, gen.obf_pos, mask, length, arcs, rels)
                # import ipdb
                # ipdb.set_trace()
                ret.loss = loss["loss"]
                ret.loss_arc = loss["loss_arc"]
                ret.loss_rel = loss["loss_rel"]
        else:
            obf_parsed = self.parser(gen.obf_word, gen.obf_char, gen.obf_pos, mask, length)
            ori_parsed = self.parser(word, char, pos, mask, length)
            # you can actually store the parsed result for ori_parsed ...

            ret.obf_arcs = obf_parsed["arcs"]
            ret.obf_rels = obf_parsed["rels"]
            ret.ori_arcs = ori_parsed["arcs"]
            ret.ori_rels = ori_parsed["rels"]
            ret.obf_word = gen.obf_word
            ret.obf_mask = gen.obf_mask
        return ret


class SeqAllObfuscator(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = config.SeqAllObfuscatorCfg

        self.generator = SeqAllGenerator()
        self.parser = BiaffineAttnParser()
        self.reinforce = cfg.reinforce

        if cfg.resume_parser is not None:
            state_dict = torch.load(cfg.resume_parser)
            # import ipdb
            # ipdb.set_trace()
            self.parser.load_state_dict(state_dict["network"])
            log.info("Resume parser from %s", cfg.resume_parser)
        else:
            log.warning("Missing pretrained parser")
        
        if cfg.resume_obfuscator:
            trainer = torch.load(cfg.resume_obfuscator)
            self.load_state_dict(trainer["network"])
            log.info("Load obfuscator from %s", cfg.resume_obfuscator)

    def parameters(self):
        return self.generator.parameters()

    def forward(self, word, char, pos, mask, length, arcs=None, rels=None):
        ret = MrDict(fixed=False, blob=True)
        gen = self.generator(word, char, pos, mask, cfg.reinforce)
        ret.gen_oup = gen

        if arcs is not None:
            # during training
            if cfg.reinforce:
                loss_smp = self.parser(gen.obf_word, gen.obf_char, gen.obf_pos, mask, length, arcs, rels)
                loss_ctc = self.parser(gen.ctc_word, gen.ctc_char, gen.ctc_pos, mask, length, arcs, rels)
                # TODO: we should remove the minus, but ?
                rwd_smp_arc = -loss_smp["loss_arc_batch"]
                rwd_smp_rel = -loss_smp["loss_rel_batch"]
                rwd_ctc_arc = -loss_ctc["loss_arc_batch"]
                rwd_ctc_rel = -loss_ctc["loss_rel_batch"]
                g = (rwd_smp_arc - rwd_ctc_arc) + (rwd_smp_rel - rwd_ctc_rel)
                g.detach_()
                ret.loss = -(g * gen.plog.sum(dim=-1)).mean()
                ret.rwd = g.mean()
            else:
                loss = self.parser(gen.obf_word_emb, gen.obf_char, gen.obf_pos, mask, length, arcs, rels)
                # import ipdb
                # ipdb.set_trace()
                ret.loss = loss["loss"]
                ret.loss_arc = loss["loss_arc"]
                ret.loss_rel = loss["loss_rel"]
        else:
            obf_parsed = self.parser(gen.obf_word, gen.obf_char, gen.obf_pos, mask, length)
            ori_parsed = self.parser(word, char, pos, mask, length)
            # you can actually store the parsed result for ori_parsed ...

            ret.obf_arcs = obf_parsed["arcs"]
            ret.obf_rels = obf_parsed["rels"]
            ret.ori_arcs = ori_parsed["arcs"]
            ret.ori_rels = ori_parsed["rels"]
            ret.obf_word = gen.obf_word
            ret.obf_mask = gen.obf_mask
        return ret


class SeqCopyObfuscator(nn.Module):
    def __init__(self):
        super().__init__()
        from data.input import get_word_ids
        keywords = get_word_ids(pipe.parser_input.word_alphabet, cfg.ext.train_path)
        tgtwords = keywords
        log.info("%d keywords initiated", len(keywords))
        self.generator = SeqCopyGenerator(keywords, tgtwords)
        self.parser = BiaffineAttnParser()
        self.reinforce = cfg.reinforce  # TODO: REINFORCE is not supported now

        if cfg.resume_parser is not None:
            state_dict = torch.load(cfg.resume_parser)
            # import ipdb
            # ipdb.set_trace()
            self.parser.load_state_dict(state_dict["network"])
            log.info("Resume parser from %s", cfg.resume_parser)
        else:
            log.warning("Missing pretrained parser")

    def parameters(self):
        return self.generator.parameters()

    def forward(self, word, char, pos, mask, length, arcs=None, rels=None):
        ret = MrDict(fixed=False, blob=True)
        gen = self.generator(word, char, pos, mask, cfg.reinforce)
        ret.gen_oup = gen

        if arcs is not None:
            # during training
            if cfg.reinforce:
                loss_smp = self.parser(gen.obf_word, gen.obf_char, gen.obf_pos, mask, length, arcs, rels)
                loss_ctc = self.parser(gen.ctc_word, gen.ctc_char, gen.ctc_pos, mask, length, arcs, rels)
                # TODO: we should remove the minus, but ?
                rwd_smp_arc = -loss_smp["loss_arc_batch"]
                rwd_smp_rel = -loss_smp["loss_rel_batch"]
                rwd_ctc_arc = -loss_ctc["loss_arc_batch"]
                rwd_ctc_rel = -loss_ctc["loss_rel_batch"]
                g = (rwd_smp_arc - rwd_ctc_arc) + (rwd_smp_rel - rwd_ctc_rel)
                g.detach_()
                ret.loss = -(g * gen.plog.sum(dim=-1)).mean()
                ret.rwd = g.mean()
            else:
                loss = self.parser(gen.obf_word_emb, gen.obf_char, gen.obf_pos, mask, length, arcs, rels)
                # import ipdb
                # ipdb.set_trace()
                ret.loss = loss["loss"]
                ret.loss_arc = loss["loss_arc"]
                ret.loss_rel = loss["loss_rel"]
        else:
            obf_parsed = self.parser(gen.obf_word, gen.obf_char, gen.obf_pos, mask, length)
            ori_parsed = self.parser(word, char, pos, mask, length)
            # you can actually store the parsed result for ori_parsed ...

            ret.obf_arcs = obf_parsed["arcs"]
            ret.obf_rels = obf_parsed["rels"]
            ret.ori_arcs = ori_parsed["arcs"]
            ret.ori_rels = ori_parsed["rels"]
            ret.obf_word = gen.obf_word
            ret.obf_mask = gen.obf_mask
        return ret


class UnkObfuscator(nn.Module):
    def __init__(self):
        super().__init__()
        from data.input import get_word_ids
        keywords = get_word_ids(pipe.parser_input.word_alphabet, cfg.ext.train_path)
        tgtwords = keywords
        log.info("%d keywords initiated", len(keywords))
        self.generator = UnkGenerator(keywords, tgtwords)
        # we don't need to train the UnkGenerator, we just need to validate it on the dev/test set
        self.parser = BiaffineAttnParser()

        if cfg.resume_parser is not None:
            state_dict = torch.load(cfg.resume_parser)
            # import ipdb
            # ipdb.set_trace()
            self.parser.load_state_dict(state_dict["network"])
            log.info("Resume parser from %s", cfg.resume_parser)
        else:
            log.warning("Missing pretrained parser")

    def forward(self, word, char, pos, mask, length):
        ret = MrDict(fixed=False, blob=True)
        gen = self.generator(word, char, pos)

        obf_prd = self.parser(gen.obf_word, gen.obf_char, gen.obf_pos, mask, length)
        ori_prd = self.parser(word, char, pos, mask, length)

        ret.gen_oup = gen
        ret.obf_word = gen.obf_word
        ret.obf_char = gen.obf_char
        ret.obf_pos = gen.obf_pos
        ret.obf_mask = gen.obf_mask

        ret.obf_arcs = obf_prd["arcs"]
        ret.obf_rels = obf_prd["rels"]
        ret.ori_arcs = ori_prd["arcs"]
        ret.ori_rels = ori_prd["rels"]

        return ret


class AlltagObfuscator(nn.Module):
    def __init__(self):
        super().__init__()
        if cfg.model == "alltag_random" or cfg.obf_model == "alltag_random":
            self.generator = AlltagRandomGenerator()
        else:
            self.generator = AlltagCtxGenerator()
        # we don't need to train the UnkGenerator, we just need to validate it on the dev/test set
        self.parser = BiaffineAttnParser()

        if cfg.resume_parser is not None:
            state_dict = torch.load(cfg.resume_parser)
            # import ipdb
            # ipdb.set_trace()
            self.parser.load_state_dict(state_dict["network"])
            log.info("Resume parser from %s", cfg.resume_parser)
        else:
            log.warning("Missing pretrained parser")

    def parameters(self):
        return self.generator.parameters()

    def forward(self, word, char, pos, mask, length, arcs=None, rels=None):
        ret = MrDict(fixed=False, blob=True)
        gen = self.generator(word, char, pos, mask, cfg.reinforce)
        ret.gen_oup = gen

        if arcs is not None:
            # during training
            if cfg.reinforce:
                loss_smp = self.parser(gen.obf_word, gen.obf_char, gen.obf_pos, mask, length, arcs, rels)
                loss_ctc = self.parser(gen.ctc_word, gen.ctc_char, gen.ctc_pos, mask, length, arcs, rels)
                # TODO: we should remove the minus, but ?
                rwd_smp_arc = -loss_smp["loss_arc_batch"]
                rwd_smp_rel = -loss_smp["loss_rel_batch"]
                rwd_ctc_arc = -loss_ctc["loss_arc_batch"]
                rwd_ctc_rel = -loss_ctc["loss_rel_batch"]
                g = (rwd_smp_arc - rwd_ctc_arc) + (rwd_smp_rel - rwd_ctc_rel)
                g.detach_()
                ret.loss = -(g * gen.plog.sum(dim=-1)).mean()
                ret.rwd = g.mean()
            else:
                loss = self.parser(gen.obf_word_emb, gen.obf_char, gen.obf_pos, mask, length, arcs, rels)
                # import ipdb
                # ipdb.set_trace()
                ret.loss = loss["loss"]
                ret.loss_arc = loss["loss_arc"]
                ret.loss_rel = loss["loss_rel"]
        else:
            obf_parsed = self.parser(gen.obf_word, gen.obf_char, gen.obf_pos, mask, length)
            ori_parsed = self.parser(word, char, pos, mask, length)
            # you can actually store the parsed result for ori_parsed ...

            ret.obf_arcs = obf_parsed["arcs"]
            ret.obf_rels = obf_parsed["rels"]
            ret.ori_arcs = ori_parsed["arcs"]
            ret.ori_rels = ori_parsed["rels"]
            ret.obf_word = gen.obf_word
            ret.obf_mask = gen.obf_mask
        return ret


