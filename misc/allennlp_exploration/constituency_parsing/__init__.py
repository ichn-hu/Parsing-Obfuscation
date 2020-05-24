import os
import logging
import torch
import numpy as np
from allennlp.models import Model
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from allennlp.training import Trainer
from allennlp.common.params import Params
from allennlp.data import DataIterator, DatasetReader
from misc.allennlp_exploration.constituency_parsing.metric import EvalbBracketingScorer, PennTreeBankConstituencySpanDatasetReader
from allennlp.nn import util
from tqdm import tqdm
from model.obfuscator import Obfuscator

import config as glb_config
from net.generator.tagspec import TagSpecCtxGenerator, TagSpecRandomGenerator
from data.fileio import get_logger, read_inputs, PAD_ID_WORD, ROOT, ROOT_CHAR, ROOT_POS, ROOT_TYPE, MAX_CHAR_LENGTH, DIGIT_RE, PAD_ID_CHAR, PAD_ID_TAG
from net.parser import BiaffineAttnParser

pretrained_model_url = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz"

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

data_cfg = {
    'train_data_path': '$ROOT/Project/data/ptb/02-21.10way.clean',
    'validation_data_path': '$ROOT/Project/data/ptb/22.auto.clean',
    'test_data_path': '$ROOT/Project/data/ptb/23.auto.clean'
}

dataset_reader_config = {
    'token_indexers': {
        'elmo': {
            'type': 'elmo_characters'
        }
    },
    'type': 'ptb_trees_gai',
    'use_pos_tags': True
}

configs = {
    'train_data_path': '/ptb/wsj.train.notrace.trees',
    'validation_data_path': '/ptb/wsj.dev.notrace.trees',
    'dataset_reader': {
        'token_indexers': {
            'elmo': {
                'type': 'elmo_characters'
            }
        },
        'type': 'ptb_trees_gai',
        'use_pos_tags': True,
        'generator_model': os.environ.get("generator_model", "rnd")
    },
    'iterator': {
        'batch_size': 32,
        'sorting_keys': [['tokens', 'num_tokens']],
        'type': 'bucket'
    },
    'test_data_path': '/ptb/wsj.test.notrace.trees',
    'trainer': {
        'cuda_device': 0,
        'grad_norm': 5,
        'learning_rate_scheduler': {
            'gamma': 0.8,
            'milestones': [40, 50, 60, 70, 80],
            'type': 'multi_step'
        },
        'num_epochs': 150,
        'optimizer': {'lr': 1, 'rho': 0.95, 'type': 'adadelta'},
        'patience': 20,
        'validation_metric': '+evalb_f1_measure'
    },
    'model': {
        'evalb_directory_path': '$ROOT/evalb/EVALB',
        'initializer': [
            ['tag_projection_layer.*weight', {'type': 'xavier_normal'}], [
                'feedforward_layer.*weight', {'type': 'xavier_normal'}],
            ['encoder._module.weight_ih.*', {'type': 'xavier_normal'}], ['encoder._module.weight_hh.*', {'type': 'orthogonal'}]],
        'feedforward': {
            'activations': 'relu',
            'dropout': 0.1,
            'hidden_dims': 250,
            'input_dim': 500, 'num_layers': 1
        },
        'encoder': {
            'bidirectional': True,
            'dropout': 0.2,
            'hidden_size': 250,
            'input_size': 1074,
            'num_layers': 2,
            'type': 'lstm'
        },
        'span_extractor': {
            'input_dim': 500,
            'type': 'bidirectional_endpoint'
        },
        'type': 'constituency_parser',
        'pos_tag_embedding': {'embedding_dim': 50, 'vocab_namespace': 'pos'},
        'text_field_embedder': {
            'elmo': {
                'do_layer_norm': False,
                'type': 'elmo_token_embedder',
                'dropout': 0.2,
                'options_file': '/tmp/tmpzn7d_h8m/fta/model.text_field_embedder.elmo.options_file',
                'weight_file': '/tmp/tmpzn7d_h8m/fta/model.text_field_embedder.elmo.weight_file'
            }
        }
    }
}


def load_constituency_parsing_model():
    archive = load_archive(pretrained_model_url)
    cp_model = archive.model
    config = archive.config
    for key, val in data_cfg.items():
        config[key] = val
    data_iter = DataIterator.from_params(config.pop("iterator"))
    data_iter.index_with(cp_model.vocab)
    evalb_directory_path = '$ROOT/evalb/EVALB'
    evalb_score = EvalbBracketingScorer(evalb_directory_path)
    cp_model._evalb_score = evalb_score

    data_reader = DatasetReader.from_params(Params(dataset_reader_config))

    data_test = data_reader.read(config.pop("test_data_path"))
    # data_val = data_reader.read(config.pop("validation_data_path"))
    # data_train = data_reader.read(config.pop("train_data_path"))

    cp_model = cp_model.to(torch.device("cuda"))
    cp_model.eval()

    inputs = read_inputs()
    from model.hybrid import resume_pretrained_parser
    from staffs.meter import TagSpecMeter
    device = glb_config.cfg.device # pylint: disable=no-member
    
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
    glb_config.cfg.Generator.loss_term = loss_term # pylint: disable=no-member
    glb_config.cfg .Generator.obf_term = obf_term # pylint: disable=no-member

    gen_model = os.environ.get("gen_model", "rnd")
    if gen_model == "rnd":
        gen_model = TagSpecRandomGenerator()
    else:
        gen_model = TagSpecCtxGenerator()
    
    network = Obfuscator(gen_model, psr, None)
    resume_ctx = os.environ.get("resume_ctx", None)

    if resume_ctx is not None:
        state = torch.load(resume_ctx)
        network.load_state_dict(state["network"])

    network = network.to(device)
    network.update()
    network.eval()

    obf_instances = []

    def apply_generator(ptb_data):
        """
        ptb_data: {
            metadata: [
                {tokens, pos_tag}
            ]
        }
        """
        def prepare_input():
            word_alphabet = inputs.word_alphabet
            char_alphabet = inputs.char_alphabet
            type_alphabet = inputs.rel_alphabet
            pos_alphabet = inputs.pos_alphabet
            data = []
            batch_size = 0
            sent_len = 0
            for metadata in ptb_data["metadata"]:
                words = []
                word_ids = []
                char_seqs = []
                char_id_seqs = []
                postags = []
                pos_ids = []
                types = []
                type_ids = []
                heads = []

                words.append(ROOT)
                word_ids.append(word_alphabet.get_index(ROOT))
                char_seqs.append([ROOT_CHAR, ])
                char_id_seqs.append([char_alphabet.get_index(ROOT_CHAR), ])
                postags.append(ROOT_POS)
                pos_ids.append(pos_alphabet.get_index(ROOT_POS))
                types.append(ROOT_TYPE)
                type_ids.append(type_alphabet.get_index(ROOT_TYPE))
                heads.append(0)

                for tok, pos in zip(metadata["tokens"], metadata["pos_tags"]):
                    chars = []
                    char_ids = []
                    for char in tok:
                        chars.append(char)
                        char_ids.append(char_alphabet.get_index(char))
                    if len(chars) > MAX_CHAR_LENGTH:
                        chars = chars[:MAX_CHAR_LENGTH]
                        char_ids = char_ids[:MAX_CHAR_LENGTH]
                    char_seqs.append(chars)
                    char_id_seqs.append(char_ids)

                    word = DIGIT_RE.sub("0", tok)
                    head = 1
                    relation = "233"

                    words.append(word)
                    word_ids.append(word_alphabet.get_index(word))

                    postags.append(pos)
                    pos_ids.append(pos_alphabet.get_index(pos))

                    types.append(relation)
                    type_ids.append(0)

                    heads.append(head)
                
                data.append([word_ids, char_id_seqs, pos_ids, heads, type_ids])
                sent_len = max(sent_len, len(word_ids))
            
            batch_size = len(data)

            wid_inputs = np.empty([batch_size, sent_len], dtype=np.int64)
            cid_inputs = np.empty([batch_size, sent_len, MAX_CHAR_LENGTH], dtype=np.int64)
            pid_inputs = np.empty([batch_size, sent_len], dtype=np.int64)
            hid_inputs = np.empty([batch_size, sent_len], dtype=np.int64)
            tid_inputs = np.empty([batch_size, sent_len], dtype=np.int64)

            masks = np.zeros([batch_size, sent_len], dtype=np.float32)
            single = np.zeros([batch_size, sent_len], dtype=np.int64)
            lengths = np.empty(batch_size, dtype=np.int64)

            for b, (wids, cid_seqs, pids, hids, tids) in enumerate(data):
                inst_size = len(wids)
                lengths[b] = inst_size
                # word ids
                wid_inputs[b, :inst_size] = wids
                wid_inputs[b, inst_size:] = PAD_ID_WORD
                for c, cids in enumerate(cid_seqs):
                    cid_inputs[b, c, :len(cids)] = cids
                    cid_inputs[b, c, len(cids):] = PAD_ID_CHAR
                cid_inputs[b, inst_size:, :] = PAD_ID_CHAR
                # pos ids
                pid_inputs[b, :inst_size] = pids
                pid_inputs[b, inst_size:] = PAD_ID_TAG
                # type ids
                tid_inputs[b, :inst_size] = tids
                tid_inputs[b, inst_size:] = PAD_ID_TAG
                # heads
                hid_inputs[b, :inst_size] = hids
                hid_inputs[b, inst_size:] = PAD_ID_TAG
                # masks
                masks[b, :inst_size] = 1.0

            words = torch.from_numpy(wid_inputs).to(device)
            chars = torch.from_numpy(cid_inputs).to(device)
            pos = torch.from_numpy(pid_inputs).to(device)
            heads = torch.from_numpy(hid_inputs).to(device)
            types = torch.from_numpy(tid_inputs).to(device)
            masks = torch.from_numpy(masks).to(device)
            single = torch.from_numpy(single).to(device)
            lengths = torch.from_numpy(lengths).to(device)

            prepared_input = (words, chars, pos, masks, lengths, heads, types)
            return prepared_input
        prepared_input = prepare_input()
        obfuscated_result = network(*prepared_input)
        obf_words = obfuscated_result["obf_word"].cpu()
        for i, data in enumerate(ptb_data["metadata"]):
            obf_tokens = []
            to_word = inputs.word_alphabet.get_instance
            for wid in obf_words[i][1:]:
                if wid.item() == PAD_ID_WORD:
                    break
                obf_tokens.append(to_word(wid.item()))
            obf_instances.append(data_reader.text_to_instance(obf_tokens, data["pos_tags"], data["gold_tree"]))

    for data in tqdm(data_iter(num_epochs=1, instances=data_test)):
        apply_generator(data)

    logger.info(f"Transformed {len(obf_instances)} sentences")

    for data in tqdm(data_iter(num_epochs=1, instances=data_test)):
        model_input = util.move_to_device(data, 0)
        cp_model(**model_input)

    metrics = cp_model.get_metrics()
    print(metrics)
    with open(os.path.join(glb_config.cfg.ext.save_dir, "result.json"), "w") as fout:
        fout.write(str(metrics))
