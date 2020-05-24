import logging
import sys
import json
import os
import os.path
import random
import re
import numpy as np
import torch


PAD = "_PAD"
PAD_POS = "_PAD_POS"
PAD_TYPE = "_<PAD>"
PAD_CHAR = "_PAD_CHAR"
ROOT = "_ROOT"
ROOT_POS = "_ROOT_POS"
ROOT_TYPE = "_<ROOT>"
ROOT_CHAR = "_ROOT_CHAR"
END = "_END"
END_POS = "_END_POS"
END_TYPE = "_<END>"
END_CHAR = "_END_CHAR"
ROOT_ID = 2 # tabu
_START_VOCAB = [PAD, ROOT, END]

UNK_ID = 0
PAD_ID_WORD = 1
PAD_ID_CHAR = 1
PAD_ID_TAG = 0

NUM_SYMBOLIC_TAGS = 3

_buckets = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 140]

MAX_CHAR_LENGTH = 45
NUM_CHAR_PAD = 2

# Regular expressions used to normalize digits.
DIGIT_RE = re.compile("\d")


def get_logger(name, level=logging.INFO, handler=sys.stdout,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


class Sentence(object):
    def __init__(self, words, word_ids, char_seqs, char_id_seqs):
        self.words = words
        self.word_ids = word_ids
        self.char_seqs = char_seqs
        self.char_id_seqs = char_id_seqs

    def length(self):
        return len(self.words)


class DependencyInstance(object):
    def __init__(self, sentence, postags, pos_ids, heads, types, type_ids):
        self.sentence = sentence
        self.postags = postags
        self.pos_ids = pos_ids
        self.heads = heads
        self.types = types
        self.type_ids = type_ids

    def length(self):
        return self.sentence.length()


class Alphabet(object):
    def __init__(self, name, default_value=False, keep_growing=True, singleton=False):
        self.__name = name

        self.instance2index = {}
        self.instances = []
        self.default_value = default_value
        self.offset = 1 if self.default_value else 0
        self.keep_growing = keep_growing
        self.singletons = set() if singleton else None

        # Index 0 is occupied by default, all else following.
        self.default_index = 0 if self.default_value else None

        self.next_index = self.offset

        self.logger = get_logger('Alphabet')

    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def add_singleton(self, id):
        if self.singletons is None:
            raise RuntimeError('Alphabet %s does not have singleton.' % self.__name)
        else:
            self.singletons.add(id)

    def add_singletons(self, ids):
        if self.singletons is None:
            raise RuntimeError('Alphabet %s does not have singleton.' % self.__name)
        else:
            self.singletons.update(ids)

    def is_singleton(self, id):
        if self.singletons is None:
            raise RuntimeError('Alphabet %s does not have singleton.' % self.__name)
        else:
            return id in self.singletons

    def get_index(self, instance):
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                return index
            else:
                if self.default_value:
                    return self.default_index
                else:
                    raise KeyError("instance not found: %s" % instance)

    def get_instance(self, index):
        if self.default_value and index == self.default_index:
            # First index is occupied by the wildcard element.
            return '<_UNK>'
        else:
            try:
                return self.instances[index - self.offset]
            except IndexError:
                raise IndexError('unknown index: %d' % index)

    def size(self):
        return len(self.instances) + self.offset

    def singleton_size(self):
        return len(self.singletons)

    def items(self):
        return self.instance2index.items()

    def enumerate_items(self, start):
        if start < self.offset or start >= self.size():
            raise IndexError("Enumerate is allowed between [%d : size of the alphabet)" % self.offset)
        return zip(range(start, len(self.instances) + self.offset), self.instances[start - self.offset:])

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        if self.singletons is None:
            return {'instance2index': self.instance2index, 'instances': self.instances}
        else:
            return {'instance2index': self.instance2index, 'instances': self.instances,
                    'singletions': list(self.singletons)}

    def __from_json(self, data):
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]
        if 'singletions' in data:
            self.singletons = set(data['singletions'])
        else:
            self.singletons = None

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.__name
        try:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            json.dump(self.get_content(),
                      open(os.path.join(output_directory, saving_name + ".json"), 'w'), indent=4)
        except Exception as e:
            self.logger.warn("Alphabet is not saved: %s" % repr(e))

    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        loading_name = name if name else self.__name
        self.__from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))
        self.next_index = len(self.instances) + self.offset
        self.keep_growing = False

    def load_from(self, data):
        self.__from_json(data)
        self.next_index = len(self.instances) + self.offset
        self.keep_growing = False


class CoNLLXWriter(object):
    def __init__(self, word_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        self.__source_file = None
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet

    def start(self, file_path):
        dirname = os.path.dirname(file_path)
        os.makedirs(dirname, exist_ok=True)
        self.__source_file = open(file_path, 'w')

    def close(self):
        self.__source_file.close()

    def write(self, word, pos, head, type, lengths, symbolic_root=False, symbolic_end=False):
        batch_size, _ = word.shape
        start = 1 if symbolic_root else 0
        end = 1 if symbolic_end else 0
        for i in range(batch_size):
            for j in range(start, lengths[i] - end):
                w = self.__word_alphabet.get_instance(word[i, j]).encode('utf-8')
                p = self.__pos_alphabet.get_instance(pos[i, j]).encode('utf-8')
                t = self.__type_alphabet.get_instance(type[i, j]).encode('utf-8')
                h = head[i, j]
                self.__source_file.write('%d\t%s\t_\t_\t%s\t_\t%d\t%s\n' % (j, w, p, h, t))
            self.__source_file.write('\n')


class CoNLLXReader(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            lines.append(line.split('\t'))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        types = []
        type_ids = []
        heads = []

        if symbolic_root:
            words.append(ROOT)
            word_ids.append(self.__word_alphabet.get_index(ROOT))
            char_seqs.append([ROOT_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(ROOT_CHAR), ])
            postags.append(ROOT_POS)
            pos_ids.append(self.__pos_alphabet.get_index(ROOT_POS))
            types.append(ROOT_TYPE)
            type_ids.append(self.__type_alphabet.get_index(ROOT_TYPE))
            heads.append(0)

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
                char_ids = char_ids[:MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
            pos = tokens[4]
            head = int(tokens[6])
            type = tokens[7]

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            types.append(type)
            type_ids.append(self.__type_alphabet.get_index(type))

            heads.append(head)

        if symbolic_end:
            words.append(END)
            word_ids.append(self.__word_alphabet.get_index(END))
            char_seqs.append([END_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(END_CHAR), ])
            postags.append(END_POS)
            pos_ids.append(self.__pos_alphabet.get_index(END_POS))
            types.append(END_TYPE)
            type_ids.append(self.__type_alphabet.get_index(END_TYPE))
            heads.append(0)

        return DependencyInstance(Sentence(words, word_ids, char_seqs, char_id_seqs), postags, pos_ids, heads, types, type_ids)


def create_alphabets(alphabet_directory, train_path, data_paths=None, max_vocabulary_size=50000, embedd_dict=None,
                     min_occurence=1, normalize_digits=True):
    def expand_vocab():
        vocab_set = set(vocab_list)
        for data_path in data_paths:
            # logger.info("Processing data: %s" % data_path)
            with open(data_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    tokens = line.split('\t')
                    for char in tokens[1]:
                        char_alphabet.add(char)

                    word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
                    pos = tokens[4]
                    type = tokens[7]

                    pos_alphabet.add(pos)
                    type_alphabet.add(type)

                    if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
                        vocab_set.add(word)
                        vocab_list.append(word)

    logger = get_logger("Create Alphabets")
    word_alphabet = Alphabet('word', default_value=True, singleton=True)
    char_alphabet = Alphabet('character', default_value=True)
    pos_alphabet = Alphabet('pos')
    type_alphabet = Alphabet('type')
    if not os.path.isdir(alphabet_directory):
        logger.info("Creating Alphabets: %s" % alphabet_directory)

        char_alphabet.add(PAD_CHAR)
        pos_alphabet.add(PAD_POS)
        type_alphabet.add(PAD_TYPE)

        char_alphabet.add(ROOT_CHAR)
        pos_alphabet.add(ROOT_POS)
        type_alphabet.add(ROOT_TYPE)

        char_alphabet.add(END_CHAR)
        pos_alphabet.add(END_POS)
        type_alphabet.add(END_TYPE)

        vocab = dict()
        with open(train_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split('\t')
                for char in tokens[1]:
                    char_alphabet.add(char)

                word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
                pos = tokens[4]
                type = tokens[7]

                pos_alphabet.add(pos)
                type_alphabet.add(type)

                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        # collect singletons
        singletons = set([word for word, count in vocab.items() if count <= min_occurence])

        # if a singleton is in pretrained embedding dict, set the count to min_occur + c
        if embedd_dict is not None:
            for word in vocab.keys():
                if word in embedd_dict or word.lower() in embedd_dict:
                    vocab[word] += min_occurence

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        logger.info("Total Vocabulary Size: %d" % len(vocab_list))
        logger.info("Total Singleton Size:  %d" % len(singletons))
        vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurence]
        logger.info("Total Vocabulary Size (w.o rare words): %d" % len(vocab_list))

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        if data_paths is not None and embedd_dict is not None:
            expand_vocab()

        for word in vocab_list:
            word_alphabet.add(word)
            if word in singletons:
                word_alphabet.add_singleton(word_alphabet.get_index(word))

        word_alphabet.save(alphabet_directory)
        char_alphabet.save(alphabet_directory)
        pos_alphabet.save(alphabet_directory)
        type_alphabet.save(alphabet_directory)
    else:
        word_alphabet.load(alphabet_directory)
        char_alphabet.load(alphabet_directory)
        pos_alphabet.load(alphabet_directory)
        type_alphabet.load(alphabet_directory)

    word_alphabet.close()
    char_alphabet.close()
    pos_alphabet.close()
    type_alphabet.close()
    logger.info("Word Alphabet Size (Singleton): %d (%d)" % (word_alphabet.size(), word_alphabet.singleton_size()))
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("Type Alphabet Size: %d" % type_alphabet.size())
    return word_alphabet, char_alphabet, pos_alphabet, type_alphabet


def read_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_size=None,
              normalize_digits=True, symbolic_root=False, symbolic_end=False, remove_invalid_data=False):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids])
                max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
                break

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    reader.close()
    print("Total number of data: %d" % counter)

    def remove_no_NNP_data():
        essential_pos = [pos_alphabet.get_index('NNP'), pos_alphabet.get_index('NNPS')]
        new_data = [[] for _ in _buckets]
        invalid_data = []
        valid_data_cnt = 0
        for bucket_id, bucket_size in enumerate(_buckets):
            for inst in data[bucket_id]:
                pos_ids = inst[2]
                valid = False
                for ep in essential_pos:
                    for pi in pos_ids:
                        if ep == pi:
                            valid = True
                            break
                    if valid:
                        break
                if valid:
                    new_data[bucket_id].append(inst)
                    valid_data_cnt += 1
                else:
                    invalid_data.append(inst)
        print("Removed %s invalid data, %s valid data remained from %s" % (len(invalid_data), valid_data_cnt, source_path))
        return new_data

    import config
    if remove_invalid_data and config.cfg.remove_no_NNP_data:
        data = remove_no_NNP_data()
        print("Use succinct dataset")

    return data, max_char_length


def get_batch(data, batch_size, word_alphabet=None, unk_replace=0.):
    data, max_char_length = data
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])

    bucket_length = _buckets[bucket_id]
    char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id] + NUM_CHAR_PAD)
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)

    wid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    cid_inputs = np.empty([batch_size, bucket_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    hid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    tid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)

    masks = np.zeros([batch_size, bucket_length], dtype=np.float32)
    single = np.zeros([batch_size, bucket_length], dtype=np.int64)

    for b in range(batch_size):
        wids, cid_seqs, pids, hids, tids = random.choice(data[bucket_id])

        inst_size = len(wids)
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

        if unk_replace:
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[b, j] = 1

    if unk_replace:
        noise = np.random.binomial(1, unk_replace, size=[batch_size, bucket_length])
        wid_inputs = wid_inputs * (1 - noise * single)

    return wid_inputs, cid_inputs, pid_inputs, hid_inputs, tid_inputs, masks


def iterate_batch(data, batch_size, word_alphabet=None, unk_replace=0., shuffle=False):
    data, max_char_length = data
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id] + NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, hids, tids = inst
            inst_size = len(wids)
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            if unk_replace:
                for j, wid in enumerate(wids):
                    if word_alphabet.is_singleton(wid):
                        single[i, j] = 1

        if unk_replace:
            noise = np.random.binomial(1, unk_replace, size=[bucket_size, bucket_length])
            wid_inputs = wid_inputs * (1 - noise * single)

        indices = None
        if shuffle:
            indices = np.arange(bucket_size)
            np.random.shuffle(indices)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield wid_inputs[excerpt], cid_inputs[excerpt], pid_inputs[excerpt], hid_inputs[excerpt], \
                  tid_inputs[excerpt], masks[excerpt]


def read_data_to_variable(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_size=None,
                          normalize_digits=True, symbolic_root=False, symbolic_end=False,
                          use_gpu=False, volatile=False):
    data, max_char_length = read_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                      max_size=max_size, normalize_digits=normalize_digits,
                                      symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]

    data_variable = []

    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_variable.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id] + NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        lengths = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, hids, tids = inst
            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

        words = torch.Tensor(wid_inputs)
        chars = torch.Tensor(cid_inputs)
        pos = torch.Tensor(pid_inputs)
        heads = torch.Tensor(hid_inputs)
        types = torch.Tensor(tid_inputs)
        masks = torch.Tensor(masks)
        single = torch.Tensor(single)
        lengths = torch.Tensor(lengths)
        if use_gpu:
            words = words.cuda()
            chars = chars.cuda()
            pos = pos.cuda()
            heads = heads.cuda()
            types = types.cuda()
            masks = masks.cuda()
            single = single.cuda()
            lengths = lengths.cuda()

        data_variable.append((words, chars, pos, heads, types, masks, single, lengths))

    return data_variable, bucket_sizes


def get_batch_variable(data, batch_size, unk_replace=0.):
    data_variable, bucket_sizes = data
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
    bucket_length = _buckets[bucket_id]

    words, chars, pos, heads, types, masks, single, lengths = data_variable[bucket_id]
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)
    index = torch.randperm(bucket_size).long()[:batch_size]
    if words.is_cuda:
        index = index.cuda()

    words = words[index]
    if unk_replace:
        ones = torch.Tensor(single.data.new(batch_size, bucket_length).fill_(1))
        noise = torch.Tensor(masks.data.new(batch_size, bucket_length).bernoulli_(unk_replace).long())
        words = words * (ones - single[index] * noise)

    return words, chars[index], pos[index], heads[index], types[index], masks[index], lengths[index]


def iterate_batch_variable(data, batch_size, unk_replace=0., shuffle=False):
    data_variable, bucket_sizes = data

    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]
        if bucket_size == 0:
            continue

        words, chars, pos, heads, types, masks, single, lengths = data_variable[bucket_id]
        if unk_replace:
            ones = torch.Tensor(single.data.new(bucket_size, bucket_length).fill_(1))
            noise = torch.Tensor(masks.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long())
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            if words.is_cuda:
                indices = indices.cuda()
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield words[excerpt], chars[excerpt], pos[excerpt], heads[excerpt], types[excerpt], \
                  masks[excerpt], lengths[excerpt]


def read_data_to_tensor(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_size=None,
                        normalize_digits=True, symbolic_root=False, symbolic_end=False, device=torch.device('cpu'), remove_invalid_data=False):
    data, max_char_length = read_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                      max_size=max_size, normalize_digits=normalize_digits,
                                      symbolic_root=symbolic_root, symbolic_end=symbolic_end, remove_invalid_data=remove_invalid_data)
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]

    data_tensor = []

    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_tensor.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id] + NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        lengths = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, hids, tids = inst
            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

        words = torch.from_numpy(wid_inputs).to(device)
        chars = torch.from_numpy(cid_inputs).to(device)
        pos = torch.from_numpy(pid_inputs).to(device)
        heads = torch.from_numpy(hid_inputs).to(device)
        types = torch.from_numpy(tid_inputs).to(device)
        masks = torch.from_numpy(masks).to(device)
        single = torch.from_numpy(single).to(device)
        lengths = torch.from_numpy(lengths).to(device)

        data_tensor.append((words, chars, pos, heads, types, masks, single, lengths))

    return data_tensor, bucket_sizes

def split_data(data, rate=0.5):
    data_tensor, bucket_sizes = data
    # data_tensor: [bucket_id][item_id][result]
    data_tensor_a = []
    data_tensor_b = []
    bucket_sizes_a = []
    bucket_sizes_b = []
    num_data_a = 0
    num_data_b = 0
    for bucket_id, bucket_size in enumerate(bucket_sizes):
        size_a = int(bucket_size * rate)
        if size_a == 0:
            size_a = 1 # at least 1
        size_b = bucket_size - size_a
        num_data_a += size_a
        num_data_b += size_b
        a_idx = [1 for _ in range(size_a)] + [0 for _ in range(size_b)]
        random.shuffle(a_idx)
        a_idx = torch.tensor(a_idx).byte().to(data_tensor[0][0].device)
        b_idx = ~a_idx
        item_a = []
        item_b = []
        for item in data_tensor[bucket_id]:
            item_a.append(item[a_idx])
            item_b.append(item[b_idx])
        data_tensor_a.append(item_a)
        data_tensor_b.append(item_b)
        bucket_sizes_a.append(size_a)
        bucket_sizes_b.append(size_b)

    return num_data_a, (data_tensor_a, bucket_sizes_a), num_data_b, (data_tensor_b, bucket_sizes_b)

def get_batch_tensor(data, batch_size, unk_replace=0.):
    data_tensor, bucket_sizes = data
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
    bucket_length = _buckets[bucket_id]

    words, chars, pos, heads, types, masks, single, lengths = data_tensor[bucket_id]
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)
    index = torch.randperm(bucket_size).long()[:batch_size]
    index = index.to(words.device)

    words = words[index]
    if unk_replace:
        ones = single.new_ones(batch_size, bucket_length)
        noise = masks.new_empty(batch_size, bucket_length).bernoulli_(unk_replace).long()
        words = words * (ones - single[index] * noise)

    return words, chars[index], pos[index], heads[index], types[index], masks[index], lengths[index]


def iterate_batch_tensor(data, batch_size, unk_replace=0., shuffle=False):
    data_tensor, bucket_sizes = data

    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]
        if bucket_size == 0:
            continue

        words, chars, pos, heads, types, masks, single, lengths = data_tensor[bucket_id]
        if unk_replace:
            ones = single.new_ones(bucket_size, bucket_length)
            noise = masks.new_empty(bucket_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            indices = indices.to(words.device)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield words[excerpt], chars[excerpt], pos[excerpt], heads[excerpt], types[excerpt], \
                  masks[excerpt], lengths[excerpt]
