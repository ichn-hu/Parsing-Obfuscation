import os
import config
import allennlp
from allennlp.models import Model
from allennlp.common import Params
from model.structure_aware import StructureAwareGenerator
from allennlp.training import Trainer
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data import DataIterator
from data.ud_vocab import ObfuscatorDatasetReader, AttackerDatasetReader
from allennlp.nn.util import move_to_device
from allennlp.models import load_archive
from model.biaffine_dependency_parser import PretrainedBiaffineDependencyParser
import net.allennlp_attacker


def load_pretrained_parser() -> Model:
    model_path = config.cfg.ext.pretrained_parser_path # pylint: disable=no-member
    
    overrides = """
    {
        "model": {
            "type": "pretrained_biaffine_parser"
        }
    }
    """
    archival_model = load_archive(model_path, overrides=overrides)
    return archival_model.model

def build_structure_aware() -> Model:
    params = Params(config.structure_aware_config.copy()) # pylint: disable=no-member
    pretrained_parser = load_pretrained_parser()

    all_tag = ['NNP', 'NN', 'JJ', 'NNS', 'VBN', 'VB', 'VBG', 'VBD', 'RB', 'VBZ', 'VBP', 'NNPS']
    obf_term = all_tag[:int(os.environ.get("obf_until_tagset", 2))]
    config.cfg.Generator.obf_term = obf_term # pylint: disable=no-member

    data_reader = DatasetReader.from_params(params.pop("dataset_reader"))
    data_train = data_reader.read(config.cfg.ext.train_path) # pylint: disable=no-member
    data_val = data_reader.read(config.cfg.ext.dev_path) # pylint: disable=no-member

    data_iter = DataIterator.from_params(params.pop("iterator"))
    data_iter.index_with(pretrained_parser.vocab)

    model = Model.from_params(params=params.pop("model"),
                pretrained_parser=pretrained_parser,
                vocab=pretrained_parser.vocab,
                tagger_vocab=pretrained_parser.vocab)
    
    save_dir = config.cfg.ext.save_dir # pylint: disable=no-member
    trainer = Trainer.from_params(model=model,
                                  serialization_dir=save_dir,
                                  iterator=data_iter,
                                  train_data=data_train,
                                  validation_data=data_val,
                                  params=params.pop("trainer"))
    trainer.train()

def build_allennlp_attacker() -> Model:
    params = Params(config.allennlp_attacker_config.copy())
    def build_vocab():
        vocabulary_building = params.pop("vocabulary_building")
        data_reader = DatasetReader.from_params(vocabulary_building.pop("dataset_reader"))
        data = data_reader.read(config.cfg.ext.train_path)
        return Vocabulary.from_instances(data)


    data_reader = DatasetReader.from_params(params.pop("dataset_reader"))
    attack_file_dir = os.environ.get("attack_file_dir")
    train_path = os.path.join(attack_file_dir, "export_train.conll")
    test_path = os.path.join(attack_file_dir, "export_test.conll")
    dev_path = os.path.join(attack_file_dir, "export_dev.conll")


    data_train = data_reader.read(train_path)
    data_test = data_reader.read(test_path)
    data_dev = data_reader.read(dev_path)
    data_iter = DataIterator.from_params(params.pop("iterator"))
    test_data_iter = DataIterator.from_params(params.pop("test_iterator"))
    vocab = build_vocab()

    model = Model.from_params(params=params.pop("model"),
                              vocab=vocab)
    
    data_iter.index_with(vocab)
    test_data_iter.index_with(vocab)
    save_dir = config.cfg.ext.save_dir # pylint: disable=no-member
    trainer_params = params.pop("trainer")
    trainer = Trainer.from_params(model=model,
                                  serialization_dir=save_dir,
                                  iterator=data_iter,
                                  train_data=data_train,
                                  validation_data=data_dev,
                                  params=trainer_params)
    trainer.train()
    _ = model.get_metrics(True)
    for data in test_data_iter(data_test, num_epochs=1):
        data = move_to_device(data, 0)
        _ = model(**data)
    mrr = model.get_metrics(True)
    print(mrr)
    with open(os.path.join(save_dir, "mrr_result.txt"), "w") as f:
        f.write(str(mrr))
    
    