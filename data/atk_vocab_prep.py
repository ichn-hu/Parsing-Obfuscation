from allennlp.data import Vocabulary
from nltk import FreqDist

from utils import is_privacy_term, get_logger
import config

cfg = config.cfg
log = get_logger(__name__)

def get_pri_vocab(conllu_file_path=cfg.ext.train_path): # pylint: disable=no-member
    log.info("Read in %s to build vocabulary ...", conllu_file_path)
    f = open(conllu_file_path, "r").read()
    sents = f.split("\n\n")
    all_words = FreqDist()
    pri_words = FreqDist()
    for sent in sents:
        tokens = sent.split("\n")
        if len(tokens) == 0:
            continue
        has_pri = False
        words = []
        pos_tags = []
        for token in tokens:
            fields = token.split('\t')
            if len(fields) != 10:
                continue
            all_words[fields[1]] += 1
            words.append(fields[1])
            pos_tags.append(fields[4])
            if is_privacy_term(fields[4]):
                has_pri = True
                pri_words[fields[1]] += 1
        if not has_pri:
            continue
        # self.sent_insts.append({"words": words, "pos_tags": pos_tags})
        # self.num_inst += 1
    
#        import ipdb
#        ipdb.set_trace()

    log.info("(B/N) tokens: %s/%s pri_tokens: %s/%s", all_words.B(), all_words.N(), pri_words.B(), pri_words.N())

    vocab = Vocabulary(
            counter={"tokens": all_words, "pri_tokens": pri_words},
            max_vocab_size={"tokens": 30000 - 2, "pri_tokens": 10000 - 2})
    
    return vocab
