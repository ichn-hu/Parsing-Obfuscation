import os
import torch
import config

from data.fileio import get_logger, read_inputs, PAD_ID_WORD
import torch.nn as nn

from net.parser import BiRecurrentConvBiAffine, BiaffineAttnParser
from net.attacker import CtxSeqAttacker
from net.generator import AlltagCopyCtxGenerator, UnkGenerator, AlltagRandomGenerator, AlltagCtxGenerator
from net.generator.tagspec import TagSpecCtxGenerator, TagSpecRandomGenerator, TagSpecUnkGenerator
from net.generator.relaxed import TagAgnosticGenerator
from net.structure_aware import StructureAwareGenerator
from model.adversarial import Adversarial
from model.obfuscator import Obfuscator
from staffs.trainer import DefaultTrainer, GANTrainer
from staffs.meter import AdvMeter, ObfMeter, ParsingMeter
from staffs.watcher import TbxWatcher, DefaultWatcher
from torch.optim import Adam
from config.utils import MrDict
from data import pipe
from nltk import FreqDist
from .build_attacker import get_attack_score, evaluate

# inputs = read_inputs()
device = config.cfg.device # pylint: disable=no-member
log = get_logger(__name__)

def train_a_parser():
    network = BiaffineAttnParser()
    print(network)
    watcher = DefaultWatcher()
    network = network.to(config.cfg.device)
    trainer = DefaultTrainer(network, watcher, ParsingMeter)
    batch_size = 32
    unk_replace = 0.5
    inputs = read_inputs()

    def train_iter():
        num_batch = inputs.num_data // batch_size
        for _ in range(num_batch):
            batch = inputs.get_batch_tensor(inputs.data_train, batch_size, unk_replace=unk_replace)
            word, char, pos, heads, rels, masks, lengths = batch
            yield {"input": (word, char, pos, masks, lengths, heads, rels)}

    def val_iter(dataset):
        def iterate():
            for batch in inputs.iterate_batch_tensor(dataset, batch_size):
                word, char, pos, heads, rels, masks, lengths = batch
                yield {"input": (word, char, pos, masks, lengths), "target": {"arcs": heads, "rels": rels}}
        return iterate

    if config.cfg.resume_parser is not None:
        state_dict = torch.load(config.cfg.resume_parser)
        trainer.load_state_dict(state_dict)
        log.info("Resume model from %s", config.cfg.resume_parser)
    else:
        trainer.train(Adam, train_iter, val_iter(inputs.data_dev))
        state_dict = trainer.state_dict()

        save_path = os.path.join(config.cfg.ext.save_dir, "best-model-trainer-{}".format(trainer.epoch))
        with open(save_path, "wb") as f:
            torch.save(state_dict, f)
            log.info("Best model saved at %s", save_path)
    trainer.validate(val_iter(inputs.data_dev))
    trainer.validate(val_iter(inputs.data_test))

def filter_state_dict(state_dict):
    return {k: v for k, v in state_dict.items() if k not in ['gen.psr_weight', 'gen.atk_weight']}

def evaluate_baseline():
    gen = AlltagRandomGenerator()
    result = evaluate(gen)

def cherry_pick():
    inputs = read_inputs()
    from model.hybrid import resume_pretrained_parser
    from staffs.meter import TagSpecMeter
    device = config.cfg.device # pylint: disable=no-member
    
    psr = BiaffineAttnParser()
    resume_pretrained_parser(psr)
    psr = psr.to(device)

    loss_term = ["loss_arc", "loss_rel", "loss_ent"]
    obf_term = ["NN", "NNP"]
    all_tag = ['NNP', 'NN', 'JJ', 'NNS', 'VBN', 'VB', 'VBG', 'VBD', 'RB', 'VBZ', 'VBP', 'NNPS']
    all_tag = ['NNP', 'NN', 'VBN', 'NNS', 'VB', 'VBD', 'RB', 'VBZ', 'NNPS', 'JJ', 'VBP', 'VBG', 'RBR', 'JJR', 'PDT', 'JJS', 'FW', 'RBS', 'UH', 'SYM', 'CD']

    tags = [
        ["NNP", "NNPS"],
        ["NN", "NNS"],
        ["JJ", "JJR", "JJS"],
        ["VB", "VBN", "VBD", "VBZ", "VBP", "VBG"],
        ["RB", "RBR", "RBS"],
        ["FW"],
        ["UH"]
    ]

    select_first_n_tag = os.environ.get("select_first_n_tag", 1)
    select_tag = os.environ.get("select_tag", None)

    obf_term = []
    if select_tag is None:
        for i in range(min(int(select_first_n_tag), len(tags))):
            obf_term += tags[i]
    else:
        assert(len(select_tag) == len(tags))
        for i, s in enumerate(select_tag):
            if s == '1':
                obf_term += tags[i]
    config.cfg.Generator.loss_term = loss_term # pylint: disable=no-member
    config.cfg .Generator.obf_term = obf_term # pylint: disable=no-member

    rnd_gen = TagSpecRandomGenerator()
    ctx_gen = TagSpecCtxGenerator()
    
    log.info("Obf terms are %s", str(obf_term))
    
    rnd_network = Obfuscator(rnd_gen, psr, None).to(device)
    rnd_network.update()
    ctx_network = Obfuscator(ctx_gen, psr, None).to(device)
    ctx_network.update()

    resume_ctx = os.environ.get("resume_ctx", None)
    if resume_ctx is None:
        log.error("No resume_ctx specified!")
    
    state = torch.load(resume_ctx)

    ctx_network.load_state_dict(state["network"])

    data_iter = ctx_network.get_data_iter(batch_size=1)[1]

    def print_parallel(ori, pos, rnd_obf, ctx_obf):
        to_word = inputs.word_alphabet.get_instance
        to_char = inputs.char_alphabet.get_instance
        to_pos = inputs.pos_alphabet.get_instance

        ori_tokens = []
        rnd_obf_tokens = []
        ctx_obf_tokens = []
        pos_tags = []
        for i, wid in enumerate(ori):
            if wid.item() == PAD_ID_WORD:
                break
            ori_tokens.append(to_word(wid.item()))
            rnd_obf_tokens.append(to_word(rnd_obf[i].item()))
            ctx_obf_tokens.append(to_word(ctx_obf[i].item()))
            pos_tags.append(to_pos(pos[i].item()))
        
        n = len(ori_tokens)
        spaces = [max(len(ori_tokens[i]), len(rnd_obf_tokens[i]), len(ctx_obf_tokens[i]), len(pos_tags[i])) for i in range(n)]
        place_holders = ["{:%d}" % i for i in spaces]
        import termcolor
        print("=" * 80)
        print("[____pos] ", end="")
        for i in range(n):
            print(place_holders[i].format(pos_tags[i]), end=" ")
        print("\n[____ori] ", end="")
        for i in range(n):
            print(place_holders[i].format(ori_tokens[i]), end=" ")
        print("\n[rnd_obf] ", end="")
        for i in range(n):
            to_print = place_holders[i].format(rnd_obf_tokens[i])
            if rnd_obf_tokens[i] != ori_tokens[i]:
                to_print = termcolor.colored(to_print, "red")
            print(to_print, end=" ")
        print("\n[ctx_obf] ", end="")
        for i in range(n):
            to_print = place_holders[i].format(ctx_obf_tokens[i])
            if rnd_obf_tokens[i] != ori_tokens[i]:
                to_print = termcolor.colored(to_print, "green")
            print(to_print, end=" ")
        print()
        print("=" * 80)

    with torch.no_grad():
        rnd_network.eval()
        ctx_network.eval()
        for i, data in enumerate(data_iter(inputs.data_test)()):
            inp = data["input"]
            tgt = data["target"]
            rnd_oup = rnd_network(*inp)
            ctx_oup = ctx_network(*inp)
            ori_word = inp[0].cpu()
            rnd_oup_word = rnd_oup["obf_word"].cpu()
            ctx_oup_word = ctx_oup["obf_word"].cpu()
            pos = inp[2].cpu()
            mask = inp[3].cpu()
            print_parallel(ori_word[0], pos[0], rnd_oup_word[0], ctx_oup_word[0])

def tagspec():
    inputs = read_inputs() 
    from model.hybrid import resume_pretrained_parser
    from staffs.meter import TagSpecMeter
    device = config.cfg.device # pylint: disable=no-member
    
    psr = BiaffineAttnParser()
    resume_pretrained_parser(psr)
    psr = psr.to(device)

    loss_term = ["loss_arc", "loss_rel", "loss_ent"]
    obf_term = ["NN", "NNP"]
    all_tag = ['NNP', 'NN', 'JJ', 'NNS', 'VBN', 'VB', 'VBG', 'VBD', 'RB', 'VBZ', 'VBP', 'NNPS']
    all_tag = ['NNP', 'NN', 'VBN', 'NNS', 'VB', 'VBD', 'RB', 'VBZ', 'NNPS', 'JJ', 'VBP', 'VBG', 'RBR', 'JJR', 'PDT', 'JJS', 'FW', 'RBS', 'UH', 'SYM', 'CD']

    tags = [
        ["NNP", "NNPS"],
        ["NN", "NNS"],
        ["JJ", "JJR", "JJS"],
        ["VB", "VBN", "VBD", "VBZ", "VBP", "VBG"],
        ["RB", "RBR", "RBS"],
        ["FW"],
        ["UH"]
    ]

    select_first_n_tag = os.environ.get("select_first_n_tag", 1)
    select_tag = os.environ.get("select_tag", None)

    obf_term = []
    if select_tag is None:
        for i in range(min(int(select_first_n_tag), len(tags))):
            obf_term += tags[i]
    else:
        assert(len(select_tag) == len(tags))
        for i, s in enumerate(select_tag):
            if s == '1':
                obf_term += tags[i]

    gen_model = os.environ.get("gen_model", "ctx")

    config.cfg.Generator.loss_term = loss_term # pylint: disable=no-member
    config.cfg .Generator.obf_term = obf_term # pylint: disable=no-member

    log.info("Obf terms are %s", str(obf_term))

    log.info("Use model %s", gen_model)

    if gen_model == "ctx":
        gen = TagSpecCtxGenerator()
    elif gen_model == "unk":
        gen = TagSpecUnkGenerator()
    elif gen_model == "rnd":
        gen = TagSpecRandomGenerator()
    elif gen_model == "str":
        gen = StructureAwareGenerator(psr.embedding_weight())
    elif gen_model == "agn":
        gen = TagAgnosticGenerator()
    
    network = Obfuscator(gen, psr, None).to(device)
    network.update()
    watcher = TbxWatcher(watch_on=["loss"] + loss_term)
    trainer = DefaultTrainer(network, watcher, TagSpecMeter, optim=Adam, trainer_name="HybridTrainer")

    
    if gen_model in ['ctx', 'str', 'agn']:
        trainer.train(retain_graph=True)

    trainer.validate(_dataset=inputs.data_test)

    log.info("Exporting dataset ...")
    trainer.export_generator_output("export_test.conll", _dataset=inputs.data_test)
    trainer.export_generator_output("export_dev.conll", _dataset=inputs.data_dev)
    trainer.export_generator_output("export_train.conll", _dataset=inputs.data_train)
    log.info("Exporting finished")

    # evaluate(network.gen)


def evaluate_hybrid():
    from model.hybrid import Hybrid
    from staffs.meter import HybridMeter
    device = config.cfg.device # pylint: disable=no-member
    network = Hybrid().to(device)
    network.update()
    loss_term = ["loss_arc", "loss_rel", "loss_atk", "loss_cpy", "loss_ent"]
    config.cfg.Generator.loss_term = loss_term # pylint: disable=no-member
    watcher = TbxWatcher(watch_on=loss_term)
    trainer = DefaultTrainer(network, watcher, HybridMeter, optim=Adam, trainer_name="HybridTrainer")
    trainer.train(retain_graph=True)

    evaluate(network.generator)


def evaluate_hybrid_from():
    from model.hybrid import Hybrid
    from staffs.meter import HybridMeter
    device = config.cfg.device # pylint: disable=no-member
    network = Hybrid().to(device)
    network.update()

    state_dict = torch.load("/disk/ostrom/s1884529/zfhu/work/02.20_08:38-evaluate_hybrid/HybridTrainer-best-epoch-8.ptr")
    network.load_state_dict(state_dict["network"])

    evaluate(network.generator)


def validate_hybrid_from():
    from model.hybrid import Hybrid
    from staffs.meter import HybridMeter
    device = config.cfg.device # pylint: disable=no-member
    network = Hybrid().to(device)
    network.update()
    loss_term = ["loss_arc", "loss_rel", "loss_atk", "loss_cpy", "loss_ent"]
    config.cfg.Generator.loss_term = loss_term # pylint: disable=no-member
    watcher = TbxWatcher(watch_on=loss_term)
    trainer = DefaultTrainer(network, watcher, HybridMeter, optim=Adam, trainer_name="HybridTrainer")

    state_dict = torch.load("/disk/ostrom/s1884529/zfhu/work/02.20_08:38-evaluate_hybrid/HybridTrainer-best-epoch-8.ptr")
    network.load_state_dict(state_dict["network"])

    trainer.validate()


def retest_unkgen():
    model = Gan(gen=config.UnkGeneratorCfg.model).to(device)
    obf, adv = model.get_trainers()
    # obf.validate(_dataset=inputs.data_train)
    inputs = read_inputs()
    obf.validate(_dataset=inputs.data_dev)
    obf_meter = obf.validate(_dataset=inputs.data_test)
    adv.network.relaxed_word_emb = False
    adv.max_epoch = 30 # TODO: only for test, 30
    adv.train(retain_graph=False)
    # adv.validate(_dataset=inputs.data_train)
    adv.validate(_dataset=inputs.data_dev)
    adv_meter = adv.validate(_dataset=inputs.data_test)
    uas = obf_meter.uas
    acc = adv_meter.atk_acc
    result = {
            "exp_name": config.cfg.exp_name,
            "exp_time": config.cfg.exp_time,
            "obf_meter": obf_meter,
            "adv_meter": adv_meter,
            "uas": uas,
            "acc": acc
        }
    name = "{exp_name}-{exp_time}-{uas:.4f}-{acc:.4f}".format(**result)
    print(name)
    save_path = os.path.join("$ROOT/Project/data/result", name)
    torch.save(result, open(save_path, "wb"))

def investigate_unkgen():
    result = torch.load(config.cfg.resume_result)
    adv_meter = result["adv_meter"]
    inputs = read_inputs()
    to_word = inputs.word_alphabet.get_instance
    to_char = inputs.char_alphabet.get_instance
    ori_word = adv_meter.ori_word
    obf_word = adv_meter.obf_word
    rcv_word = adv_meter.rcv_word
    rcv_mask = (rcv_word == ori_word) & adv_meter.inp_mask.byte()
    rcved_word = rcv_word[rcv_mask]
    fd = FreqDist([t.item() for t in rcved_word])
    adv_meter.analysis()
    import ipdb
    ipdb.set_trace()
    print(rcv_mask.sum().item(), adv_meter.inp_mask.sum().item())

def load_and_attack():
    model = Gan().to(device)
    model.update()
    print(get_attack_score(model.gen))

def investigate_non_gan():
    inputs = read_inputs()
    model = Gan().to(device)
    model.update()
    obf, adv = model.get_trainers()
    obf.load_state_dict(torch.load("/disk/ostrom/s1884529/zfhu/work/01.04_00:13-separated_obf:all_pri:NNP:0.1/ObfTrainer-best-epoch-53.ptr"))
    adv.load_state_dict(torch.load("/disk/ostrom/s1884529/zfhu/work/01.04_00:13-separated_obf:all_pri:NNP:0.1/AdvTrainer-best-epoch-6.ptr"))
    obf_data_iter = obf.network.get_data_iter()[1](inputs.data_test)
    obf_meter = obf.validate(obf_data_iter)
    adv.network.relaxed_word_emb = False
    adv_data_iter = adv.network.get_data_iter()[1](inputs.data_test)
    adv_meter = adv.validate(adv_data_iter)
    adv_meter.analysis()
    uas, atk_acc = obf_meter.uas, adv_meter.atk_acc
    print(uas, atk_acc)

def look_into_result():
    inputs = read_inputs()
    result = torch.load(config.cfg.resume_result)
    log.info("Load result from %s", config.cfg.resume_result)
    adv_meter = result["adv_test_meter"]
    obf_meter = result["obf_test_meter"]
    obf_meter.analysis()
    to_word = inputs.word_alphabet.get_instance
    to_char = inputs.char_alphabet.get_instance
    ori_word = adv_meter.ori_word
    obf_word = adv_meter.obf_word
    rcv_word = adv_meter.rcv_word
    rcv_mask = (rcv_word == ori_word) & adv_meter.inp_mask.byte()
    rcved_word = rcv_word[rcv_mask]
    fd = FreqDist([t.item() for t in rcved_word])
    adv_meter.analysis()
    import ipdb
    ipdb.set_trace()
    print(rcv_mask.sum().item(), adv_meter.inp_mask.sum().item())

def non_gan():
    model = Gan().to(device)
    model.update()
    obf, adv = model.get_trainers()
    obf.max_epoch = 1000
    adv.max_epoch = 1000

    # adv.load_state_dict(torch.load("$ROOT/Project/Homomorphic-Obfuscation/../work/12.22_15:30-test_model_with_cpyfullloss/AdvTrainer-best-epoch-64.ptr"))
    # adv.load_state_dict(torch.load("$ROOT/Project/Homomorphic-Obfuscation/../work/12.24_12:02-test_model_with_cpyfullloss/AdvTrainer-best-epoch-44.ptr"))
    # obf.best_meter.report()
    # adv.best_meter.report()
    # adv.best_meter.analysis()

    obf.train(retain_graph=True)
    obf_data_iter = obf.network.get_data_iter()[1](inputs.data_test)
    obf_meter = obf.validate(obf_data_iter)
    adv.network.relaxed_word_emb = False
    adv.train()
    adv_data_iter = adv.network.get_data_iter()[1](inputs.data_test)
    adv_meter = adv.validate(adv_data_iter)
    uas, atk_acc = obf_meter.uas, adv_meter.atk_acc

    result = {
            "exp_name": config.cfg.exp_name,
            "exp_time": config.cfg.exp_time,
            "obf_meter": obf_meter,
            "adv_meter": adv_meter,
            "uas": uas,
            "acc": atk_acc
        }
    name = "{exp_name}-{exp_time}-{uas:.4f}-{acc:.4f}".format(**result)
    print(name)
    save_path = os.path.join("$ROOT/Project/data/result", name)
    torch.save(result, open(save_path, "wb"))

def two_stage_no_constraint():
    config.cfg.Generator.loss_term = ["loss_arc", "loss_rel"]
    model = Gan().to(device)
    model.update()
    
    obf, adv = model.get_trainers()
    obf.max_epoch = 1000
    adv.max_epoch = 1000

    # adv.load_state_dict(torch.load("$ROOT/Project/Homomorphic-Obfuscation/../work/12.22_15:30-test_model_with_cpyfullloss/AdvTrainer-best-epoch-64.ptr"))
    # adv.load_state_dict(torch.load("$ROOT/Project/Homomorphic-Obfuscation/../work/12.24_12:02-test_model_with_cpyfullloss/AdvTrainer-best-epoch-44.ptr"))
    # obf.best_meter.report()
    # adv.best_meter.report()
    # adv.best_meter.analysis()

    obf.train(retain_graph=True)
    evaluate(model.gen)

def two_stage_no_copy():
    config.cfg.Generator.loss_term = ["loss_arc", "loss_rel", "loss_ent"]
    config.cfg.AlltagCopyCtxGenerator.use_copy = False
    model = Gan().to(device)
    model.update()
    
    obf, adv = model.get_trainers()
    obf.max_epoch = 1000
    adv.max_epoch = 1000

    # adv.load_state_dict(torch.load("$ROOT/Project/Homomorphic-Obfuscation/../work/12.22_15:30-test_model_with_cpyfullloss/AdvTrainer-best-epoch-64.ptr"))
    # adv.load_state_dict(torch.load("$ROOT/Project/Homomorphic-Obfuscation/../work/12.24_12:02-test_model_with_cpyfullloss/AdvTrainer-best-epoch-44.ptr"))
    # obf.best_meter.report()
    # adv.best_meter.report()
    # adv.best_meter.analysis()

    obf.train(retain_graph=True)
    evaluate(model.gen)



def investigate_stochasticity():
    model = Gan().to(device)
    model.update()
    trainer = model.get_adversarial_validator()
    data_iter = trainer.network.get_data_iter()[1]
    num_round = 10
    oups = []
    for data in data_iter(inputs.data_train)():
        inp = data["input"]
        tgt = data["target"]
        for _ in range(num_round):
            oups.append(trainer.network(*inp))
        break
    
    pri_mask = oups[0].pri_mask
    invariant_mask = pri_mask.new_ones(pri_mask.sum().item()).byte()
    last_pri_word = None
    for oup in oups:
        pri_word = torch.masked_select(oup.obf_word, pri_mask)
        if last_pri_word is not None:
            invariant_mask &= (last_pri_word == pri_word)
        print(pri_word)
        last_pri_word = pri_word
    print(invariant_mask)
    import ipdb
    ipdb.set_trace()

def standalone_validation():
    log.info("standalone validation started")
    model = Gan().to(device)
    model.update()
    obf_trainer = model.get_generator_validator()
    obf_trainer.validate()
    trainer = model.get_adversarial_validator()
    trainer.train()

def evaluate_decoupled_gan():
    inputs = read_inputs()
    model = Gan(config.UnkGeneratorCfg.model).to(device)
    trainer = GANTrainer(model)
    resume_trainer = config.cfg.resume_trainer
    state_dict = torch.load(resume_trainer)
    trainer.load_state_dict(state_dict)
    trainer.network.update()
    from .build_attacker import evaluate
    evaluate(trainer.network.gen)
 

def decoupled_gan():
    inputs = read_inputs()
    model = Gan(config.UnkGeneratorCfg.model).to(device)
    trainer = GANTrainer(model)
    resume_trainer = config.cfg.resume_trainer
    if resume_trainer is not None:
        state_dict = torch.load(resume_trainer)
        trainer.load_state_dict(state_dict)
        if config.cfg.atk_validate:
            trainer.network.update()
            get_attack_score(trainer.network.gen)
            return
    try:
        trainer.train_by_batch_decoupled()
    except KeyboardInterrupt:
        obf, adv = model.get_trainers()
        obf_data_iter = obf.network.get_data_iter()[1](inputs.data_test)
        obf_meter = obf.validate(obf_data_iter)
        adv.network.relaxed_word_emb = False
        adv_data_iter = adv.network.get_data_iter()[1](inputs.data_test)
        adv_meter = adv.validate(adv_data_iter)
        uas, atk_acc = obf_meter.uas, adv_meter.atk_acc

        result = {
                "exp_name": config.cfg.exp_name,
                "exp_time": config.cfg.exp_time,
                "obf_meter": obf_meter,
                "adv_meter": adv_meter,
                "uas": uas,
                "acc": atk_acc
            }
        name = "{exp_name}-{exp_time}-{uas:.4f}-{acc:.4f}".format(**result)
        print(name)
        save_path = os.path.join("$ROOT/Project/data/result", name)
        torch.save(result, open(save_path, "wb"))


# not trainable, however can be saved/load by torch utils
class Gan(nn.Module):
    def __init__(self, gen="AlltagCopyCtxGenerator"):
        super().__init__()
        self.atk = CtxSeqAttacker()
        self.psr = BiaffineAttnParser()
        if gen == "UnkGenerator":
            self.gen = UnkGenerator()
            log.info("UnkGenerator initialized")
        elif gen == "AlltagRandomGenerator":
            self.gen = AlltagRandomGenerator()
            log.info("AlltagRandomGenerator initialized")
        elif gen == "AlltagCtxGenerator":
            self.gen = AlltagCtxGenerator()
            log.info("AlltagCtxGenerator initialized")
        else:
            self.gen = AlltagCopyCtxGenerator()

        resume_parser = config.cfg.resume_parser
        if resume_parser:
            self.psr.load_state_dict(torch.load(resume_parser)["network"])
            log.info("Load parser from %s", resume_parser)
            pipe.pretrained_parser = self.psr
        else:
            log.warning("No parser loaded")

        resume_trainer = config.cfg.resume_trainer
        if resume_trainer:
            state_dict = torch.load(resume_trainer)
            self.load_state_dict(filter_state_dict(state_dict["network"]))
            log.info("Load trainer from %s", resume_trainer)
            if config.cfg.reset_aneal:
                self.gen.aneal = MrDict({"step": 0, "t": 1, "nstep": 500, "r": -1e-5}, fixed=True)
                log.info("Reset aneal for Generator")


    def get_adversarial_validator(self):
        atk = CtxSeqAttacker()
        adv_validator = Adversarial(self.gen, atk).to(device)
        adv_watcher = TbxWatcher(watch_on=("loss", "rcv_num", "rcv_rate"), tbx_prefix="adv_val/")
        trainer = DefaultTrainer(adv_validator, adv_watcher, AdvMeter, trainer_name="AdvValidator", optim=Adam)
        resume_adversarial = config.cfg.resume_adversarial
        if resume_adversarial:
            state_dict = torch.load(resume_adversarial)
            trainer.load_state_dict(state_dict)
            log.info("Build adversarial from %s", resume_adversarial)
        trainer.max_epoch = 1000
        return trainer

    def get_generator_validator(self):
        obf_net = Obfuscator(self.gen, self.psr, self.atk).to(device)
        obf_watcher = TbxWatcher(watch_on=("loss", "loss_arc", "loss_rel", "loss_atk", "loss_cpy", "loss_ent"), tbx_prefix="obf/")
        obf_trainer = DefaultTrainer(obf_net, obf_watcher, ObfMeter, trainer_name="ObfTrainer", optim=Adam)
        return obf_trainer


    def get_trainers(self):
        adv_net = Adversarial(self.gen, self.atk).to(device)
        obf_net = Obfuscator(self.gen, self.psr, self.atk).to(device)

        adv_watcher = TbxWatcher(watch_on=("loss", "rcv_num", "rcv_rate"), tbx_prefix="adv/")
        obf_watcher = TbxWatcher(watch_on=("loss", *config.cfg.Generator.loss_term), tbx_prefix="obf/")

        adv_trainer = DefaultTrainer(adv_net, adv_watcher, AdvMeter, trainer_name="AdvTrainer", optim=Adam)
        obf_trainer = DefaultTrainer(obf_net, obf_watcher, ObfMeter, trainer_name="ObfTrainer", optim=Adam)

        return obf_trainer, adv_trainer

    def update(self):
        self.gen.update_emb_weight(self.psr.word_embedd.weight,
                                   self.atk.inp_enc.word_embedd.weight)

    def forward(self):
        raise NotImplementedError

