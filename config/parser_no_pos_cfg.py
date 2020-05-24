import config.cfg as config


class Global(config.Global):
    pass
    
class GlobalOnBuccleuch(config.GlobalOnBuccleuch):
    exp_name = "parser_no_pos"
    model = "parser"
    exp_time = "11.14_22:51"
    resume_trainer = "$ROOT/Project/buccleuch/work/parser_no_pos-11.14_22:51/best-epoch-45.ptr"


class TrainingConfig(config.TrainingConfig):
    pass


class GeneratorConfig(config.GeneratorConfig):
    pass


class ParserConfig(config.ParserConfig):
    use_pos = False


