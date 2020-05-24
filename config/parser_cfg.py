import config.cfg as config


class Global(config.Global):
    pass
    
class GlobalOnBuccleuch(config.GlobalOnBuccleuch):
    exp_name = "parser"
    model = "parser"
    exp_time = "11.16_09:15"


class TrainingConfig(config.TrainingConfig):
    pass


class GeneratorConfig(config.GeneratorConfig):
    pass


class ParserConfig(config.ParserConfig):
    use_pos = True


