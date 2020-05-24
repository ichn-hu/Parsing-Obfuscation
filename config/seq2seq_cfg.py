import os
import torch
import config.cfg as config


class Global(config.Global):
    """
    contains configuration that shares across configurable components
    or that is prone to change
    """
    exp_name = "Seq2SeqGenerator-dev"
    model = "seq2seq"  # choose ["ffobf", "seq2seq"]
    ffobf = False
    seq2seq = True
    use_hloss = False
    resume_parser = "/home/ichn/Projects/misc/work/save/95_10.ptr"
    map_location = "cpu"

    def get_save_dir(self):
        dp = os.path.join(self.work_path, self.exp_name)
        os.makedirs(dp, exist_ok=True)
        return dp


class GlobalOnBuccleuch(config.GlobalOnBuccleuch, Global):
    under_development = False

    def __init__(self):
        self.device = torch.device('cuda')
        self.map_location = None
        self.resume_generator = resume_generator = os.path.join(self.work_path, self.exp_name, "Ep51-best-model.ptr")
        self.resume_parser = self.work_path + "/save/pretrained/95_10.ptr"


class TrainingConfig(config.TrainingConfig):
    pass


class GeneratorConfig(config.Configurator):
    # encoder
    use_word = True
    word_dim = 100
    use_pos = True
    pos_dim = 100
    use_char = True
    char_dim = 100
    p_rnn = [0.33, 0.33]
    p_in = 0.33
    p_out = 0.33
    hidden_size = 256
    bidirectional = True
    num_filters = 50

    # decoder
    num_layers = 3
    num_out_words = 35374  # num of words for decoding, candidates are selected by mask_select_


class ParserConfig(config.ParserConfig):
    pass
