import allennlp
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import UniversalDependenciesDatasetReader
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.data.iterators import BasicIterator
from allennlp.models import Model
from allennlp.common.tqdm import Tqdm
import config
import os

def read_data(data_path):
    reader = UniversalDependenciesDatasetReader(use_language_specific_pos=True)
    instances = list(reader._read(data_path))
    return instances


def load_pretrained_allennlp_model():
    archive_path = "$ROOT/v1zhu2/work/allen/model.tar.gz"
    # archive_path = "/home/ichn/Projects/Homomorphic-Obfuscation/biaffine-dependency-parser-ptb-2018.08.23.tar.gz"
    archive_path = config.cfg.ext.pretrained_parser_path # pylint: disable=no-member
    archive_model = load_archive(archive_path)

    model: Model = archive_model.model

    infer_data = os.environ.get("infer_data", config.cfg.ext.data_test)

    # instances = read_data(config.cfg.ext.train_path) # pylint: disable=no-member
    instances = read_data(infer_data)

    batch_size = 32
    num_instances = len(instances)

    model = model.to(config.cfg.device) # pylint: disable=no-member

    for i in Tqdm.tqdm(range(0, num_instances, batch_size)):
        batched_instances = instances[i: i+batch_size]
        model.forward_on_instances(batched_instances)
    
    metrics = model.get_metrics()
    save_dir = config.cfg.ext.save_dir
    with open(os.path.join(save_dir, "result.json"), "w") as f:
        f.write(str(metrics))
    print(metrics)
