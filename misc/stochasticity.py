# pylint: disable=invalid-name
import sys
sys.path.append("$ROOT/Project/Homomorphic-Obfuscation")
sys.path.append("$ROOT/Project/Homomorphic-Obfuscation/model/parser")
import torch
from data.fileio import PAD_ID_WORD, read_inputs, PAD_ID_CHAR
from data.input import get_word_ids
from staffs import ObfMeter
from main import build_model
import config
import os
cfg = config.cfg
cfg_g = config.cfg_g
cfg_p = config.cfg_p

resume_from = os.path.join(cfg.get_save_dir(), 'Ep29-best-model.ptr')
cfg.resume_generator = resume_from

if __name__ == "__main__":
    inputs = read_inputs()
    network, optimizer = build_model(inputs)
    word_ids = get_word_ids(inputs.word_alphabet, cfg_p.train_path)
    chrs2id = lambda chrs, num_pad: [inputs.char_alphabet.get_index(c) for c in chrs] + [PAD_ID_CHAR]  * num_pad
    words = [list(inputs.word_alphabet.get_instance(_id)) for _id in word_ids]
    max_chr = max([len(chr_s) for chr_s in words])
    chars = [chrs2id(chrs, max_chr - len(chrs)) for chrs in words]
    N = len(word_ids)
    M = 10
    word_inp = torch.LongTensor(word_ids).repeat(M)
    word_inp = word_inp.view(M, N).t()
    char_inp = torch.LongTensor(chars).repeat((1, M))
    char_inp = char_inp.view(M, N, -1)

    word_inp = word_inp.to(cfg.device)
    char_inp = char_inp.to(cfg.device)
    emb, obf_word, obf_char = network.parser.word_embedd(word_inp, char_inp, True)
    import ipdb
    ipdb.set_trace()
    print("233")
