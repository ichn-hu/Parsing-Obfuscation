from infra.toolkit import Decoder
import numpy as np


def gen(n):
    fa = np.ndarray([n], dtype=np.int)
    score = np.ndarray([n, n], dtype=np.float)
    score.fill(0)
    fa[0] = 0
    for i in range(1, n):
        fa[i] = np.random.randint(0, i)
        score[fa[i]][i] = 0.2
    score += np.random.rand(n, n)

    return score, fa


def gen_batch(bs, n):
    graph = np.ndarray([bs, n, n], dtype=np.float)
    fa = np.ndarray([bs, n], dtype=np.int)
    masks = np.ndarray([bs, n], dtype=np.int)
    masks.fill(1)
    for i in range(bs):
        g, f = gen(n)
        graph[i, :n, :n] = g
        fa[i, :n] = f

    return graph, masks, fa


if __name__ == "__main__":
    graph, masks, fa = gen_batch(10, 10)
    res = Decoder("mst")(graph, masks)
    if not (res[1:] - fa[1:]).any():
        print("ok")
    else:
        print("no")
        print(res, fa)

