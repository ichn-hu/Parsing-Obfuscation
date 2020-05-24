import config
cfg = config.cfg
import torch
import numpy as np
random_seed = cfg.random_seed # pylint: disable=no-member
torch.manual_seed(random_seed)


from data.fileio import read_inputs
# from misc.bert_mlm_exploration.bert_attacker import train_bert_attacker, train_ctx_attacker
# from misc.bert_mlm_exploration.test_raw_mlm_with_NNP_prediction import test_MLM_simple

from utils.build_gan import look_into_result, decoupled_gan, investigate_non_gan, evaluate_baseline, evaluate_decoupled_gan, two_stage_no_constraint, two_stage_no_copy, evaluate_hybrid, evaluate_hybrid_from, validate_hybrid_from, tagspec, train_a_parser, cherry_pick
# from misc.play_with_input import read_NNP_vocab, get_tags
from misc.allennlp_exploration.inference_with_allennlp import load_pretrained_allennlp_model
# from utils.build_allennlp_models import build_structure_aware
from net.bert_attacker import attack_with_bert, analysis_bert_attack_result
# from data.atk_vocab_prep import get_pri_vocab
from utils.build_allennlp_models import build_allennlp_attacker, build_structure_aware
from misc.allennlp_exploration.constituency_parsing import load_constituency_parsing_model

# read_inputs()

exp_name = cfg.exp_name # pylint: disable=no-member
task = exp_name.split('-')[0]
locals()[task]()


