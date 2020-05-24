import config.cfg as config


class Global(config.Global):
    """
    contains configuration that shares across configurable components
    or that is prone to change
    """
    # exp_name = "FeedForwardGenerator_1000multitag_with_hloss"
    # exp_name = "FeedForwardGenerator_1000NN"
    exp_name = "FeedForwardGenerator_1000NN_with_hloss"
    model = "ffobf"  # choose ["ffobf", "seq2seq"]
    ffobf = True
    use_hloss = True


class TrainingConfig(config.TrainingConfig):
    pass


class GlobalOnBuccleuch(config.GlobalOnBuccleuch, Global):
    under_development = False


class GeneratorConfig(config.GeneratorConfig):
    xpos = ["NN", "NNS", "NNP", "NNPS"]
    top_n = 1000


class ParserConfig(config.ParserConfig):
    pass
