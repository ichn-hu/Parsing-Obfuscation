import torch
import config.seq2seq_cfg as config

class Global(config.Global):
    exp_name = "seq2seq_pg"
    model = "seq2seq_pg"


class GlobalOnBuccleuch(config.GlobalOnBuccleuch, Global):
    def __init__(self):
        self.device = torch.device('cuda')
        self.map_location = None
        self.resume_generator = None
        self.resume_parser = self.work_path + "/save/pretrained/95_10.ptr"
        # self.under_development = True


class TrainingConfig(config.TrainingConfig):
    pass


class GeneratorConfig(config.GeneratorConfig):
    pass


class ParserConfig(config.ParserConfig):
    pass


