import time
import os
from multiprocessing import Process, Queue
import random
import subprocess
from collections import defaultdict
from data.fileio import read_inputs
from data.fileio import split_data
from nltk import FreqDist

inputs = read_inputs()

AEG_DATA_ROOT = "$ROOT/Project/data/annotated_english_gigaword"
AGIGA_ROOT = "$ROOT/Software/agiga"
AGIGA_CMD = 'mvn exec:java -Dexec.mainClass="edu.jhu.agiga.StreamingSentenceReader" -Dexec.args="{path_to_gz} {path_to_output}"'
PROCESSED_OUT_DIR = "/disk/scratch1/zfhu/aeg_processed"
is_NNP = lambda pos: pos == 'NNP' or pos == 'NNPS'

def coverage(words):
    alphabet = inputs.word_alphabet
    oov_cnt = 0
    OOV_words = set()
    for word in words:
        wid = alphabet.get_index(word)
        if wid == 0: # for UNK_ID
            oov_cnt += 1
            OOV_words.add(word)
    oov_rate = oov_cnt / len(words)
    return oov_rate



def read_aeg_processed(filepath):
    start = time.time()
    f = open(filepath, "r").read()
    print("file loaded in {}s".format(time.time() - start))
    raw_sents = f.split("\n")
    all_words = FreqDist()
    NNP_words = FreqDist()

    from tqdm import tqdm
    for raw_sent in tqdm(raw_sents):
        for token in raw_sent.split(' '):
            if len(token) == 0:
                continue
            try:
                word, pos = token.split('##')
            except:
                import ipdb
                ipdb.set_trace()
                print(token)

                continue
            all_words[word] += 1
            if is_NNP(word):
                NNP_words[word] += 1

    all_oov = coverage(all_words)
    NNP_oov = coverage(NNP_words)
    import ipdb
    ipdb.set_trace()
    print("All words OOV {:.2f}% NNP words OOV {:.2f}%".format(all_oov, NNP_oov))

read_aeg_processed(os.path.join(PROCESSED_OUT_DIR, "nyt_eng_199810.txt"))
