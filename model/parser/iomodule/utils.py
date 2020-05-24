import torch

def pad_totensor(seq, pad=0, batch_first=False):
    length = 0
    longest = None
    for x in seq:
        if len(x) > length:
            length = len(x)
            longest = x

    padded = []
    masks = []
    for x in seq:
        padded.append(x + [pad] * (length - len(x)))
        masks.append([1] * len(x) + [0] * (length - len(x)))

    x = torch.tensor(padded)
    m = torch.tensor(masks)
    if not batch_first:
        x = torch.transpose(x, 0, 1).contiguous()
        m = torch.transpose(m, 0, 1).contiguous()
    return x, m
