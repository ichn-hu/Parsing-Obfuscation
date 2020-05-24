import torch
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.tensor(torch.log(-torch.log(U + eps) + eps), requires_grad=True)


def gumbel_sample(logits):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return y.argmax(dim=-1)

def gumbel_sample2(logits):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return y.max(dim=-1)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    n_class = logits.size(-1)
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y, device=y.device).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, n_class)
