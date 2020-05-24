import torch
from torch import nn
from net.generator import KeywordsPreservingGenerator
from net.parser import BiaffineAttnParser
from data.fileio import NUM_SYMBOLIC_TAGS, get_logger

from data import pipe
import config
cfg = config.cfg

log = get_logger("obfuscator")


class FeedForwardObfuscator(nn.Module):
    """
    We implement a decoupled obfuscator
    """

    def __init__(self):
        super().__init__()
        from data.input import get_word_ids
        keywords = get_word_ids(pipe.parser_input.word_alphabet, cfg.ext.train_path)
        tgtwords = keywords
        log.info("%d keywords initiated", len(keywords))
        self.generator = KeywordsPreservingGenerator(keywords, tgtwords)
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

    def forward(self, word, char, pos, mask, lens, arcs=None, rels=None, reinforce=False):
        gen = self.generator(word, char, pos)

        if arcs is not None:
            if not reinforce:
                oup = self.parser(gen["obf_word_emb"], gen["obf_char"], gen["obf_pos"], mask, lens, arcs, rels)
                oup["loss"] += gen["loss_h"]
                oup["loss_h"] = gen["loss_h"]
            else:
                oup = self.parser(gen["obf_word_emb"], gen["obf_char"], gen["obf_pos"], mask, lens)
                pred_arcs, pred_rels = oup["arcs"], oup["rels"]
                # TODO: finish this
            return oup
        else:
            obf_oup = self.parser(gen["obf_word_emb"], gen["obf_char"], gen["obf_pos"], mask, lens)
            ori_oup = self.parser(word, char, pos, mask, lens)
            return {
                "obf_arcs": obf_oup["arcs"],
                "obf_rels": obf_oup["rels"],
                "ori_arcs": ori_oup["arcs"],
                "ori_rels": ori_oup["rels"],
                "ori_word": word,
                "ori_char": char,
                "obf_word": gen["obf_word"],
                "obf_char": gen["obf_char"],
                "obf_mask": gen["obf_mask"]
            }


