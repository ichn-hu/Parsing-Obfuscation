import os
import time
import torch
# from config.decoupled_gan import cfg_p, cfg_g
from .utils import MrDict
from .axiom import ParserConfig


POSTAGSET = {
        "all": ['IN', 'DT', 'NNP', 'CD', 'NN', 'POS', 'VBN', 'NNS', 'VB', 'CC', 'VBD', 'RB', 'TO', 'VBZ', 'NNPS', 'PRP', 'PRP$', 'JJ', 'MD', 'VBP', 'VBG', 'RBR', 'WP', 'WDT', 'JJR', 'PDT', 'JJS', 'WRB', '$', 'RP', 'FW', 'RBS', 'EX', 'WP$', 'UH', 'SYM', 'LS'],
        "NN": ['NN', 'NNP', 'NNP', 'NNPS'],
        "NNP": ['NNP', 'NNPS']
        }

cfg_p = ParserConfig(use_pos=(os.environ.get("use_pos", False) == "True"),
                     use_word=(os.environ.get("use_word", "True") == "True"),
                     use_char=(os.environ.get("use_char", "True") == "True"))

cfg = MrDict({
    "device": torch.device("cuda"), # pylint: disable=no-member
    "random_seed": os.environ.get("random_seed", 22339),
    "exp_name": os.environ.get("exp_name", "test_BertAttacker"),
    "exp_time": os.environ.get("exp_time", time.strftime("%m.%d_%H:%M", time.gmtime())),
    "resume_parser":  os.environ.get("resume_parser", None),
    "ext": {
        "save_dir": os.environ.get("save_dir"),
        "cache_dir": os.environ.get("cache_dir"),
        "train_path": os.environ.get("train_path"),
        "test_path": os.environ.get("test_path"),
        "dev_path": os.environ.get("dev_path"),
        "pretrained_embedding_path": os.environ.get("pretrained_embedding_path", "$ROOT/Project/data/glove.6B.100d.txt.gz"),
        "pretrained_parser_path": os.environ.get("pretrained_parser_path", "$ROOT/Project/data/pretrain/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
    },
    "BiaffineAttnParser": cfg_p,
    "AlltagCopyCtxGenerator": {
        "inp_dim": sum([cfg_p.char_dim if cfg_p.use_char else 0,
            cfg_p.word_dim if cfg_p.use_word else 0,
            cfg_p.pos_dim if cfg_p.use_pos else 0,
            1 if cfg_p.use_dist else 0]),
        "ctx_fs": 512,
        "num_layers": 3,
        "bidirectional": True,
        "p_rnn_drop": cfg_p.p_rnn,
        "p_inp_drop": .33,
        "p_ctx_drop": .33,
        "use_copy": True,
        "cpy_penalty": float(os.environ.get("cpy_penalty", 0.1)),
        "ent_penalty": float(os.environ.get('ent_penalty', 1.0)),
        "do_not_copy_pri": True
    },
    "Generator": {
        # "fully_obfuscate": True,
        "obf_term": POSTAGSET[os.environ.get("obf_term", "all")],
        "pri_term": POSTAGSET[os.environ.get("pri_term", "NNP")],
        "loss_term": ["loss_arc", "loss_rel", "loss_atk", "loss_ent", "loss_cpy"] #, "loss_full_cpy"]
    },
    "BertAttacker": {
        "bert_model": "bert-base-cased",
        "bert_hidden_dim": 768,
        "input_vocab_size": 28996,
        "output_vocab_size": 10000,
        "holistic_training": True
    },
    "DefaultTrainer": {
        "batch_size": 32,
        "patience": 5,
        "max_decay": 10,
        "lr_decay_rate": 0.75,
        "lr": 1e-3
    }
}, blob=True, fixed=True)


structure_aware_config = {
    "dataset_reader": {
        "type": "obfuscator_dataset_reader",
        "use_language_specific_pos": True,
        # "lazy": True
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 2,
        "cache_instances": True,
        "sorting_keys": [
            [
                "words",
                "num_tokens"
            ]
        ]
    },
    "model": {
        "type": "structure_aware_obfuscator",
        "generator": {
            # "type": "structure_aware",
            "arc_representation_dim": 500,
            "dropout": 0.3,
            "encoder": {
                "type": "stacked_bidirectional_lstm",
                "hidden_size": 400,
                "input_size": 200,
                "num_layers": 3,
                "recurrent_dropout_probability": 0.3,
                "use_highway": True
            },
            "initializer": [
                [".*feedforward.*weight", {"type": "xavier_uniform"}],
                [".*feedforward.*bias", {"type": "zero"}],
                [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
                [".*tag_bilinear.*bias", {"type": "zero"}],
                [".*weight_ih.*", {"type": "xavier_uniform"}],
                [".*weight_hh.*", {"type": "orthogonal"}],
                [".*bias_ih.*", {"type": "zero"}],
                [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
            ],
            "input_dropout": 0.3,
            "pos_tag_embedding": {
                "embedding_dim": 100,
                "sparse": True,
                "vocab_namespace": "pos"
            },
            "tag_representation_dim": 100,
            "text_field_embedder": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 100,
                    "pretrained_file": "$ROOT/Project/data/glove.6B.100d.txt.gz",
                    "sparse": True,
                    "trainable": True
                }
            },
            "use_mst_decoding_for_validation": True
        },
        "reconstruct_tagger": {
            "encoder": {
                "type": "stacked_bidirectional_lstm",
                "hidden_size": 200,
                "input_size": 100,
                "num_layers": 3,
                "recurrent_dropout_probability": 0.3,
                "use_highway": True
            },
            "tagger_text_field_embedder": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 100,
                    "pretrained_file": "$ROOT/Project/data/glove.6B.100d.txt.gz",
                    "sparse": True,
                    "trainable": True
                }
            }
        }
    },
    # "train_data_path": "$ROOT/Project/data/ptb/train.conll",
    # "validation_data_path": "$ROOT/Project/data/ptb/dev.conll",
    "trainer": {
        "cuda_device": int(os.environ.get("CUDA_VISIBLE_DEVICES", -1)),
        "grad_norm": 5,
        "num_epochs": 80,
        "optimizer": {
            "type": "dense_sparse_adam",
            "betas": [
                0.9,
                0.9
            ]
        },
        "patience": 50,
        "validation_metric": "+LAS"
    }
}

allennlp_attacker_config = {
    "vocabulary_building": {
        "dataset_reader": {
            "type": "obfuscator_dataset_reader",
            "use_language_specific_pos": True,
            # "lazy": True
        }
    },
    "dataset_reader": {
            "type": "attacker_dataset_reader",
            "use_language_specific_pos": True,
            # "lazy": True
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 32,
        "cache_instances": True,
        "sorting_keys": [
            [
                "obf_words",
                "num_tokens"
            ]
        ]
    },
    "test_iterator": {
        "type": "bucket",
        "batch_size": 1,
        "cache_instances": True,
        "sorting_keys": [
            [
                "obf_words",
                "num_tokens"
            ]
        ]
    },
    "model": {
        "type": "attacker",
        "encoder": {
            "type": "stacked_bidirectional_lstm",
            "hidden_size": 512,
            "input_size": 100,
            "num_layers": 3,
            "recurrent_dropout_probability": 0.3,
            "use_highway": True
        },
        "initializer": [
            [".*feedforward.*weight", {"type": "xavier_uniform"}],
            [".*feedforward.*bias", {"type": "zero"}],
            [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
            [".*tag_bilinear.*bias", {"type": "zero"}],
            [".*weight_ih.*", {"type": "xavier_uniform"}],
            [".*weight_hh.*", {"type": "orthogonal"}],
            [".*bias_ih.*", {"type": "zero"}],
            [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
        ],
        # "pos_tag_embedding": {
        #     "embedding_dim": 100,
        #     "sparse": True,
        #     "vocab_namespace": "pos"
        # },
        "text_field_embedder": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 100,
                "pretrained_file": "$ROOT/Project/data/glove.6B.100d.txt.gz",
                "sparse": True,
                "trainable": True
            }
        },
    },
    # "train_data_path": "$ROOT/Project/data/ptb/train.conll",
    # "validation_data_path": "$ROOT/Project/data/ptb/dev.conll",
    "trainer": {
        "cuda_device": 0, #int(os.environ.get("CUDA_VISIBLE_DEVICES", -1)),
        "grad_norm": 5,
        "num_epochs": 40,
        "optimizer": {
            "type": "dense_sparse_adam",
            "betas": [
                0.9,
                0.9
            ]
        },
        "patience": 10,
        "validation_metric": "+tot"
    }
}

def get_save_dir():
    import config.utils as utils
    cfg.work_path = utils.get_work_path()
    cfg.save_dir = os.path.join(cfg.work_path, cfg.exp_time + '-' + cfg.exp_name)

get_save_dir()
