import torch
import config
from data import pipe
from net.parser import BiRecurrentConvBiAffine, BiaffineAttnParser
cfg = config.cfg

def get_pretrained_parser():
    if pipe.pretrained_parser is None:
        psr = BiaffineAttnParser()
        resume = cfg.resume_parser
        state_dict = torch.load(resume)["network"]
        psr.load_state_dict(state_dict)
        pipe.pretrained_parser = psr.to(cfg.device)
    return pipe.pretrained_parser

