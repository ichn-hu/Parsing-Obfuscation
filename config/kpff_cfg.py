import config.cfg as config


class Global(config.Global):
    """
    contains configuration that shares across configurable components
    or that is prone to change
    """
    # exp_name = "FeedForwardGenerator_1000multitag_with_hloss"
    # exp_name = "FeedForwardGenerator_1000NN"
    exp_name = "kpff_fully"
    model = "kpff"  # choose ["ffobf", "seq2seq"]
    ffobf = True
    use_hloss = True


class TrainingConfig(config.TrainingConfig):
    pass


class GlobalOnBuccleuch(config.GlobalOnBuccleuch, Global):
    under_development = False
    exp_name = "kpff_fully_with_hloss_NE"
    exp_time = "11.22_10:41"
    model = "kpff"
    resume_parser = "$ROOT/Project/buccleuch/work/parser_no_pos-11.14_22:51/best-epoch-103.ptr"


class GeneratorConfig(config.GeneratorConfig):
    fully_obfuscate = True
    xpos = ["NNP", "NNPS"]
    top_n = 1000
    word_dim = 100
    hidden_size = 128
    use_hloss = True


class ParserConfig(config.ParserConfig):
    use_pos = False
