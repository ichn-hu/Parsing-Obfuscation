import sys
sys.path.append("$ROOT/Project/Homomorphic-Obfuscation")
sys.path.append("$ROOT/Project/Homomorphic-Obfuscation/model/parser")
import torch
from data.fileio import PAD_ID_WORD
from staffs import ObfMeter
import config
import os
cfg = config.cfg
resume_from = os.path.join(cfg.get_save_dir(), 'Ep1-best-model.ptr')


item = torch.load(resume_from, map_location='cpu')[0]
om_state = item['meter']
om = ObfMeter()
om.load_state_dict(om_state)

om.report()

def tensor_to_sent(a, h=None, l=2333):
    w = []
    cnt = 0
    for i in a:
        if i == PAD_ID_WORD:
            break
        s = om.word_alphabet.get_instance(i)
        if h is not None:
            s = s + '/' + str(h[cnt].item())
            cnt = cnt + 1
        w.append(s)
        if cnt == l:
            break
    return w


def print_samples(sl):
    import ipdb
    ipdb.set_trace()
    for ori, obf in sl:
        assert len(ori) == len(obf)
        l = list(map(lambda a: max(len(a[0]), len(a[1])) + 1, zip(ori, obf)))
        ori_s = ''
        obf_s = ''
        for i, w in enumerate(ori):
            w += ' ' * (l[i] - len(w))
            ori_s += w
        for i, w in enumerate(obf):
            w += ' ' * (l[i] - len(w))
            obf_s += w
        print(ori_s)
        print(obf_s)
        print()


def samples():
    h, ho, w, wo = om.h, om.ho, om.w, om.wo
    wa, ca = om.word_alphabet, om.char_alphabet
    obf_pos = ((wo - w) ** 2).sum(dim=1) > 0
    err_head = ((ho - h) ** 2).sum(dim=1) > 0
    bad = obf_pos * err_head
    good = obf_pos * (1 - err_head)
    nb = bad.sum().item()
    ng = good.sum().item()
    ori_b = torch.masked_select(w, bad.reshape(-1, 1)).reshape(nb, -1)
    obf_b = torch.masked_select(wo, bad.reshape(-1, 1)).reshape(nb, -1)
    ori_g = torch.masked_select(w, good.reshape(-1, 1)).reshape(ng, -1)
    obf_g = torch.masked_select(wo, good.reshape(-1, 1)).reshape(ng, -1)
    ori_bh = torch.masked_select(h, bad.reshape(-1, 1)).reshape(nb, -1)
    obf_bh = torch.masked_select(ho, bad.reshape(-1, 1)).reshape(nb, -1)

    good_samples = []
    for i in range(ng):
        ori = tensor_to_sent(ori_g[i])
        obf = tensor_to_sent(obf_g[i])
        good_samples.append((ori, obf))
    bad_samples = []
    for i in range(nb):
        ori = tensor_to_sent(ori_b[i], ori_bh[i])
        obf = tensor_to_sent(obf_b[i], obf_bh[i], len(ori))
        bad_samples.append((ori, obf))
    print("Good matched example")
    print_samples(good_samples[:10])
    print("Bad matched example")
    print_samples(bad_samples[:10])
    return good_samples, bad_samples
