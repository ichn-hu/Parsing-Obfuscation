import torch
from torch import nn
from torch.utils.data import RandomSampler, DataLoader, Dataset, SequentialSampler
from torch.optim import Adam
from pytorch_pretrained_bert import BertModel, BertForPreTraining, BertTokenizer
from misc import MrDict
from nltk import FreqDist
from allennlp.data import Vocabulary
from staffs.trainer import DefaultTrainer
from staffs.meter import Meter
from staffs.watcher import TbxWatcher

from model.parser.model.nn.variational_rnn import VarMaskedFastLSTM


import config

from data.fileio import get_logger
cfg = config.cfg
CFG = cfg
log = get_logger(__name__)


class AdvMeter(Meter):
    def __init__(self):
        super().__init__()
        self.rcv_word = None
        self.tgt_word = None
        self.inp_mask = None
        self.pri_mask = None
        self.atk_acc = 0

    def measure(self, inp, tgt, oup):
        self.combine_2dattr("rcv_word", oup["rcv_word"])
        self.combine_2dattr("tgt_word", tgt[1])
        self.combine_2dattr("inp_mask", inp[1], pad=0)
        self.combine_2dattr("pri_mask", tgt[0], pad=0)

    def analysis(self):
        inputs = pipe.parser_input
        match_mask = (self.ori_word == self.rcv_word)
        rcv_mask = match_mask & self.pri_mask
        rcv = self.rcv_word[rcv_mask]
        fq_rcv = FreqDist([t.item() for t in rcv])
        to_word = inputs.word_alphabet.get_instance
        to_char = inputs.char_alphabet.get_instance
        
        rcv_cnt = rcv_mask.sum().item()
        pri_cnt = self.pri_mask.sum().item()
        rcv_rate = rcv_cnt / pri_cnt
        def show_most_common():
            rcv_info = []
            for i, j in fq_rcv.most_common():
                rcv_info.append("%10s %4d %.2f%% |" % (to_word(i), j, j / rcv_cnt * 100))
            for j, i in enumerate(rcv_info):
                print(i, end=' ')
                if (j + 1) % 8 == 0:
                    print()

        show_most_common()
        import ipdb
        ipdb.set_trace()

        # rcv_word_list = list(torch.masked_select(self.ori_word, rcv_mask))
        # rcv_fd = FreqDist(rcv_word_list)
        # word_alphabet = pipe.parser_input.word_alphabet

        print("rcv/pri: {}/{}={:.4f}".format(rcv_cnt, pri_cnt, rcv_rate))
        self.atk_acc = rcv_rate

    def report(self):
        match_mask = (self.tgt_word == self.rcv_word)
        rcv_mask = match_mask & self.pri_mask
        rcv_cnt = rcv_mask.sum().item()
        pri_cnt = self.pri_mask.sum().item()
        rcv_rate = rcv_cnt / pri_cnt

        # rcv_word_list = list(torch.masked_select(self.ori_word, rcv_mask))
        # rcv_fd = FreqDist(rcv_word_list)
        # word_alphabet = pipe.parser_input.word_alphabet

        print("rcv/pri: {}/{}={:.4f}".format(rcv_cnt, pri_cnt, rcv_rate))
        self.atk_acc = rcv_rate

    def is_better_than(self, other):
        if other is None:
            return True
        return self.atk_acc > other.atk_acc

is_NNP = lambda pos: pos == 'NNP' or pos == 'NNPS'
TRAIN_PATH = "$ROOT/Project/data/ptb/train.gold.conll"
DEV_PATH = "$ROOT/Project/data/ptb/dev.gold.conll"
TEST_PATH = "$ROOT/Project/data/ptb/test.gold.conll"

class PennTreeBankDataset(Dataset):
    def __init__(self, filepath, tokenizer, seq_len=128, vocab=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_inst = 0
        self.sent_insts = []
        self.seq_len = seq_len
        # self.NNP_vocab_dict = {0: '[UNK]', 1: '[PAD]'}
        # self.NNP_vocab_list = ['[UNK]', '[PAD]']

        f = open(filepath, "r").read()
        sents = f.split("\n\n")
        all_words = FreqDist()
        NNP_words = FreqDist()
        for sent in sents:
            tokens = sent.split("\n")
            if len(tokens) == 0:
                continue
            has_NNP = False
            words = []
            pos_tags = []
            for token in tokens:
                fields = token.split('\t')
                if len(fields) != 10:
                    print(token)
                    continue
                all_words[fields[1]] += 1
                words.append(fields[1])
                pos_tags.append(fields[4])
                if is_NNP(fields[4]):
                    has_NNP = True
                    NNP_words[fields[1]] += 1
            if not has_NNP:
                continue
            self.sent_insts.append({"words": words, "pos_tags": pos_tags})
            self.num_inst += 1
        
#        import ipdb
#        ipdb.set_trace()

        self.vocab = vocab or Vocabulary(
                counter={"tokens": all_words, "NNP_tokens": NNP_words},
                max_vocab_size={"tokens": 30000 - 2, "NNP_tokens": 10000 - 2})

    def get_padded_seq(self, seq, pad):
        if len(seq) < self.seq_len:
            return seq + (self.seq_len - len(seq)) * [pad]
        return seq[:self.seq_len]

    def __len__(self):
        return self.num_inst

    def __getitem__(self, item):
        LongTensor = torch.LongTensor # pylint: disable=no-member
        token_spans = [] # the span in convertied sentence for each token
        token_ids = []
        cvt_words = ['[CLS]']
        # NNP_mask = []

        targets = [-1] * self.seq_len
        inst_words = self.sent_insts[item]["words"]
        inst_pos_tags = self.sent_insts[item]["pos_tags"]
        for i, word in enumerate(inst_words):
            idx = len(cvt_words)
            token_spans.append(idx)
            tokenized_word = self.tokenizer.tokenize(word)

            if is_NNP(inst_pos_tags[i]):
                token_ids.append(i)
                if idx < self.seq_len:
                    targets[idx] = self.vocab.get_token_index(word, "NNP_tokens")
                # NNP_mask.append({"original_word": word, "index_in_cvt": len(cvt_words)})
                cvt_words.append('[MASK]')
            else:
                for word in tokenized_word:
                    cvt_words.append(word)
                    token_ids.append(i)

        cvt_words.append('[SEP]')
        token_spans.append(len(cvt_words))
        cvt_word_ids = self.tokenizer.convert_tokens_to_ids(cvt_words)
        cvt_word_mask = [1] * len(cvt_word_ids)
        word_ids = LongTensor(self.get_padded_seq(cvt_word_ids, 0)).to(cfg.device)
        inp_mask = LongTensor(self.get_padded_seq(cvt_word_mask, 0)).byte().to(cfg.device)
        target_mask = LongTensor(targets).to(cfg.device)
        pri_mask = target_mask != -1

        return {"inp_word": word_ids,
                "inp_mask": inp_mask,
                "pri_mask": pri_mask,
                "tgt_word": target_mask}


def get_weight_for_NNP_words(vocab, NNP_fd=None, field="NNP_tokens"):
    NNP_fd = NNP_fd or vocab._retained_counter[field]
    vocab_size = vocab.get_vocab_size(field)
    weight = [1e10] * vocab_size
    freq_count = [0] * vocab_size
    for i in range(2, vocab_size): # skip UNK and PAD, you will have a very high penalty if you predict them out
        word = vocab.get_token_from_index(i, field)
        freq_count[i] = NNP_fd.freq(word)
    sum_freq = sum(freq_count)
    for i in range(2, vocab_size):
        weight[i] = sum_freq / freq_count[i]
    weight = torch.tensor(weight).to(CFG.device)
    # return torch.nn.functional.softmax(weight, dim=-1)
    return torch.log(weight)


class BertAttacker(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = CFG.BertAttacker
        self.bert = BertModel.from_pretrained(cfg.bert_model)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.bert_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, cfg.output_vocab_size),
            nn.LogSoftmax(dim=-1))
        self.holistic_training = cfg.holistic_training

    def get_data_iter(self, dataset, mode="training"):
        if mode == "training":
            to_yield = lambda batch: {"input": (batch["inp_word"], batch["inp_mask"],
                                                batch["pri_mask"], batch["tgt_word"])}
            sampler = RandomSampler(dataset)
        else:
            to_yield = lambda batch: {"input": (batch["inp_word"], batch["inp_mask"]),
                                      "target": (batch['pri_mask'], batch["tgt_word"])}
            sampler = SequentialSampler(dataset)

        def data_iter():
            dataloader = DataLoader(dataset, batch_size=cfg.DefaultTrainer.batch_size,
                                    sampler=sampler)
            for batch in dataloader:
                yield to_yield(batch)
        return data_iter

    def parameters(self):
        # only optimize the decoder
        # count for steps to be trained holistic
        if self.holistic_training:
            return self.decoder.parameters()
        return super(BertAttacker, self).parameters()

    def forward(self, inp_word, inp_mask, pri_mask=None, tgt_word=None):
        token_type_ids = torch.zeros_like(inp_word)
        seq_oup, _ = self.bert(inp_word, token_type_ids, inp_mask, output_all_encoded_layers=False)
        decoded_oup = self.decoder(seq_oup)
        ret = MrDict(fixed=False, blob=True)
        # ret.output_logits = decoded_oup
        ret.rcv_word = torch.argmax(decoded_oup, dim=-1)

        if tgt_word is not None:
            bs, ls = inp_word.shape
            ret.loss = nn.functional.cross_entropy(
                decoded_oup.reshape(bs * ls, -1),
                tgt_word.reshape(-1),
                ignore_index=-1
            )
            ret.rcv_num = ((ret.rcv_word == tgt_word) & pri_mask).sum().item()
            ret.rcv_rate = ret.rcv_num / pri_mask.sum().item()

        ret.fix()
        return ret

class CtxAttacker(nn.Module):
    def __init__(self, cross_entropy_weight, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = CFG.BertAttacker
        self.inp_emb = nn.Embedding(cfg.input_vocab_size, 128)
        self.inp_drop = nn.Dropout(0.33)
        self.lstm = VarMaskedFastLSTM(128, 512, num_layers=3, batch_first=True,
                                      bidirectional=True, dropout=(0.33, 0.33))
        self.cross_entropy_weight = cross_entropy_weight
        self.decoder = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, cfg.output_vocab_size),
            nn.LogSoftmax(dim=-1))

    def get_data_iter(self, dataset, mode="training"):
        if mode == "training":
            to_yield = lambda batch: {"input": (batch["inp_word"], batch["inp_mask"],
                                                batch["pri_mask"], batch["tgt_word"])}
            sampler = RandomSampler(dataset)
        else:
            to_yield = lambda batch: {"input": (batch["inp_word"], batch["inp_mask"]),
                                      "target": (batch['pri_mask'], batch["tgt_word"])}
            sampler = SequentialSampler(dataset)

        def data_iter():
            dataloader = DataLoader(dataset, batch_size=cfg.DefaultTrainer.batch_size,
                                    sampler=sampler)
            for batch in dataloader:
                yield to_yield(batch)
        return data_iter

    def forward(self, inp_word, inp_mask, pri_mask=None, tgt_word=None):
        inp_fx = self.inp_emb(inp_word)
        inp_fx = self.inp_drop(inp_fx)
        seq_oup, _ = self.lstm(inp_fx, inp_mask.float())

        decoded_oup = self.decoder(seq_oup)
        ret = MrDict(fixed=False, blob=True)
        # ret.output_logits = decoded_oup
        ret.rcv_word = torch.argmax(decoded_oup, dim=-1)

        if tgt_word is not None:
            bs, ls = inp_word.shape
            ret.loss = nn.functional.cross_entropy(
                decoded_oup.reshape(bs * ls, -1),
                tgt_word.reshape(-1),
                ignore_index=-1,
                weight=self.cross_entropy_weight
            )
            ret.rcv_num = ((ret.rcv_word == tgt_word) & pri_mask).sum().item()
            ret.rcv_rate = ret.rcv_num / pri_mask.sum().item()

        ret.fix()
        return ret

def train_ctx_attacker():
    # this is the baseline model
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
    log.info("Load tokenizer")
    ds_train = PennTreeBankDataset(TRAIN_PATH, tokenizer)
    log.info("Load training data")
    ds_test = PennTreeBankDataset(TEST_PATH, tokenizer, vocab=ds_train.vocab)
    log.info("Load test data")
    ds_dev = PennTreeBankDataset(DEV_PATH, tokenizer, vocab=ds_train.vocab)
    log.info("Load dev data")
    cross_entropy_weight = get_weight_for_NNP_words(ds_train.vocab)
    model = CtxAttacker(cross_entropy_weight).to(cfg.device)
    log.info("Load CtxAttacker")
    watcher = TbxWatcher(watch_on=("rcv_num", "rcv_rate","loss"), tbx_prefix="CtxAttacker/")
    trainer = DefaultTrainer(model, watcher, AdvMeter, trainer_name="CtxAttackerTrainer", cfg=cfg.DefaultTrainer)
    train_iter = model.get_data_iter(ds_train)
    val_iter = model.get_data_iter(ds_dev, mode="eval")
    test_iter = model.get_data_iter(ds_test, mode="eval")
    trainer.train(train_iter=train_iter, val_iter=val_iter, optim=Adam)
    trainer.validate(val_iter=test_iter)

def train_bert_attacker():
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
    log.info("Load tokenizer")
    ds_train = PennTreeBankDataset(TRAIN_PATH, tokenizer)
    log.info("Load training data")
    ds_test = PennTreeBankDataset(TEST_PATH, tokenizer, vocab=ds_train.vocab)
    log.info("Load test data")
    ds_dev = PennTreeBankDataset(DEV_PATH, tokenizer, vocab=ds_train.vocab)
    log.info("Load dev data")
    model = BertAttacker().to(cfg.device)
    log.info("Load BertAttacker")
    watcher = TbxWatcher(watch_on=("rcv_num", "rcv_rate","loss"), tbx_prefix="BertAttacker/")
    trainer = DefaultTrainer(model, watcher, AdvMeter, trainer_name="BertAttackerTrainer", cfg=cfg.DefaultTrainer)
    train_iter = model.get_data_iter(ds_train)
    val_iter = model.get_data_iter(ds_dev, mode="eval")
    test_iter = model.get_data_iter(ds_test, mode="eval")
    trainer.train(train_iter=train_iter, val_iter=val_iter, optim=Adam)
    trainer.validate(val_iter=test_iter)

if __name__ == "__main__":
    train_bert_attacker()

