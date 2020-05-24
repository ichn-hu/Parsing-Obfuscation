# pylint: disable=invalid-name

from data.fileio import PAD_ID_WORD

def analysis(inputs, meter):
    word_alphabet = inputs.word_alphabet
    ori_word = meter.ori_word
    obf_word = meter.obf_word
    rcv_word = meter.rcv_word
    obf_mask = meter.obf_mask
    masks = meter.masks
    match = (ori_word == rcv_word).float() * obf_mask.float()

    def view(sent):
        ret = []
        for w in sent:
            if w == PAD_ID_WORD:
                break
            t = word_alphabet.get_instance(w)
            ret.append(t)
        return ret

    def inspect(n):
        ori = view(ori_word[n])
        obf = view(obf_word[n])
        rcv = view(rcv_word[n])
        msk = obf_mask[n]
        for w in zip(msk, ori, obf, rcv):
            print(w)
        
    import ipdb
    ipdb.set_trace()

    inspect(0)

