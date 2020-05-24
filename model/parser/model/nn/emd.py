import torch
import torch.utils.data
import numpy
from torch.autograd import Function, Variable
from torch import optim


class WlossLayer(torch.nn.Module):
    def __init__(self, lam=100, sinkhorn_iter=50):
        super(WlossLayer, self).__init__()

        # cost = matrix M = distance matrix
        # lam = lambda of type float > 0
        # sinkhorn_iter > 0
        # diagonal cost should be 0
        self.lam = lam
        self.sinkhorn_iter = sinkhorn_iter
        # self.register_buffer("K", torch.exp(-self.cost / self.lam).double())
        # self.register_buffer("KM", (self.cost * self.K).double())

    def forward(self, pred, target, cost):
        return WassersteinLossStab.apply(pred, target,
                                   cost, self.lam, self.sinkhorn_iter)


class WassersteinLossStab(Function):
    @staticmethod
    def forward(ctx, pred, target, cost,
                lam=1e-3, sinkhorn_iter=4):
        """pred: Batch * K: K = # mass points
           target: Batch * L: L = # mass points"""
        # import pdb
        # pdb.set_trace()
        eps = 1e-8

        # pred = pred.gather(dim=1, index=)
        na = pred.size(1)
        nb = target.size(1)

        cost = cost.double()
        pred = pred.double()
        target = target.double()

        cost = cost[:na, :nb].double()
        K = torch.exp(-cost / lam).double()
        KM = (cost * K).double()

        batch_size = pred.size(0)
        import pdb

        # pdb.set_trace()
        log_a, log_b = torch.log(pred + eps), torch.log(target + eps)
        log_u = cost.new(batch_size, na).fill_(-numpy.log(na))
        log_v = cost.new(batch_size, nb).fill_(-numpy.log(nb))
        # import pdb
        # pdb.set_trace()
        for i in range(int(sinkhorn_iter)):
            log_u_max = torch.max(log_u, dim=1)[0]
            u_stab = torch.exp(log_u - log_u_max.unsqueeze(1) + eps)
            log_v = log_b - torch.log(torch.mm(K.t(), u_stab.t()).t()) - log_u_max.unsqueeze(1)
            log_v_max = torch.max(log_v, dim=1)[0]
            v_stab = torch.exp(log_v - log_v_max.unsqueeze(1))
            tmp = log_u
            log_u = log_a - torch.log(torch.mm(K, v_stab.t()).t() + eps) - log_v_max.unsqueeze(1)
            # print(log_u.sum())
            if torch.norm(tmp - log_u) / torch.norm(log_u) < eps:
                break

        log_v_max = torch.max(log_v, dim=1)[0]
        v_stab = torch.exp(log_v - log_v_max.unsqueeze(1))
        logcostpart1 = torch.log(torch.mm(KM, v_stab.t()).t() + eps) + log_v_max.unsqueeze(1)
        wnorm = torch.exp(log_u + logcostpart1).mean(0).sum()  # sum(1) for per item pair loss...
        grad_input = log_u * lam
        # print("log_u", log_u)
        grad_input = grad_input - torch.mean(grad_input, dim=1).unsqueeze(1)
        grad_input = grad_input - torch.mean(grad_input, dim=1).unsqueeze(1)
        grad_input = grad_input / batch_size

        ctx.save_for_backward(grad_input)
        # print("grad type", type(grad_input))

        return pred.new((wnorm,)), grad_input

    @staticmethod
    def backward(ctx, grad_output, _):
        grad_input = ctx.saved_variables
        # print(grad)
        res = grad_output.clone()
        res.data.resize_(grad_input[0].size()).copy_(grad_input[0].data)
        res = res.mul_(grad_output[0]).float()
        # print("in backward func:\n\n", res)
        return res, None, None, None, None, None, None


class Sinkhorn(Function):
    def __init__(self):
        super(Sinkhorn, self).__init__()

    def forward(ctx, a, b, M, reg, tau, warmstart, numItermax, stop):
        a = a.double()
        b = b.double()
        M = M.double()

        nbb = b.size(1)

        # init data
        na = len(a)
        nb = len(b)

        cpt = 0

        # we assume that no distances are null except those of the diagonal of
        # distances
        if warmstart is None:
            alpha, beta = np.zeros(na), np.zeros(nb)
        else:
            alpha, beta = warmstart

        if nbb:
            u, v = np.ones((na, nbb)) / na, np.ones((nb, nbb)) / nb
        else:
            u, v = np.ones(na) / na, np.ones(nb) / nb

        def get_K(alpha, beta):
            """log space computation"""
            return np.exp(-(M - alpha.reshape((na, 1)) - beta.reshape((1, nb))) / reg)

        def get_Gamma(alpha, beta, u, v):
            """log space gamma computation"""
            return np.exp(
                -(M - alpha.reshape((na, 1)) - beta.reshape((1, nb))) / reg + np.log(u.reshape((na, 1))) + np.log(
                    v.reshape((1, nb))))

        # print(np.min(K))

        K = get_K(alpha, beta)
        transp = K
        cpt = 0
        err = 1
        while 1:

            uprev = u
            vprev = v

            # sinkhorn update
            v = b / (np.dot(K.T, u) + 1e-16)
            u = a / (np.dot(K, v) + 1e-16)

            # remove numerical problems and store them in K
            if np.abs(u).max() > tau or np.abs(v).max() > tau:
                if nbb:
                    alpha, beta = alpha + reg * \
                                  np.max(np.log(u), 1), beta + reg * np.max(np.log(v))
                else:
                    alpha, beta = alpha + reg * np.log(u), beta + reg * np.log(v)
                    if nbb:
                        u, v = np.ones((na, nbb)) / na, np.ones((nb, nbb)) / nb
                    else:
                        u, v = np.ones(na) / na, np.ones(nb) / nb
                K = get_K(alpha, beta)

            if cpt % print_period == 0:
                # we can speed up the process by checking for the error only all
                # the 10th iterations
                if nbb:
                    err = np.sum((u - uprev) ** 2) / np.sum((u) ** 2) + \
                          np.sum((v - vprev) ** 2) / np.sum((v) ** 2)
                else:
                    transp = get_Gamma(alpha, beta, u, v)
                    err = np.linalg.norm((np.sum(transp, axis=0) - b)) ** 2
                if log:
                    log['err'].append(err)

                if verbose:
                    if cpt % (print_period * 20) == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(cpt, err))

            if err <= stopThr:
                loop = False

            if cpt >= numItermax:
                loop = False

            if np.any(np.isnan(u)) or np.any(np.isnan(v)):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Warning: numerical errors at iteration', cpt)
                u = uprev
                v = vprev
                break

            cpt = cpt + 1

        # print('err=',err,' cpt=',cpt)
        if log:
            log['logu'] = alpha / reg + np.log(u)
            log['logv'] = beta / reg + np.log(v)
            log['alpha'] = alpha + reg * np.log(u)
            log['beta'] = beta + reg * np.log(v)
            log['warmstart'] = (log['alpha'], log['beta'])
            if nbb:
                res = np.zeros((nbb))
                for i in range(nbb):
                    res[i] = np.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
                return res, log

            else:
                return get_Gamma(alpha, beta, u, v), log
        else:
            if nbb:
                res = np.zeros((nbb))
                for i in range(nbb):
                    res[i] = np.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
                return res
            else:
                return get_Gamma(alpha, beta, u, v)

if __name__ == "__main__":
    cost = (torch.Tensor(2, 2).fill_(1) - torch.diag(torch.Tensor(2).fill_(1)))#.cuda()
    mylayer = WlossLayer(cost)#.cuda()
    inp = Variable(torch.Tensor([[1, 0], [0.5, 0.5]]), requires_grad=True)#.cuda()
    ground_true = Variable(torch.Tensor([[0, 1], [0.5, 0.5]]))#.cuda()


    res, _ = mylayer(inp, ground_true)
    # print(inp.requires_grad, res.requires_grad)
    # print(res, inp)
    mylayer.zero_grad()
    res.backward()
    print("inp's gradient is good:")
    print(inp.grad)

    print("convert to gpu:\n", inp.cuda().grad)
    print("=============================================="
          "\n However, this does not work on pytorch when GPU is enabled")

    cost = (torch.Tensor(2, 2).fill_(1) - torch.diag(torch.Tensor(2).fill_(1))).cuda()
    mylayer = WlossLayer(cost).cuda()
    inp = Variable(torch.Tensor([[1, 0], [0.5, 0.5]]), requires_grad=True).cuda()
    ground_true = Variable(torch.Tensor([[0, 1], [0.5, 0.5]])).cuda()

    opt = optim.SGD([
        {'params': mylayer.parameters()},
    ], lr=1e-2, momentum=0.9)

    res, _ = mylayer(inp, ground_true)
    # print(inp.requires_grad, res.requires_grad)
    # print(res, inp)
    mylayer.zero_grad()
    res.backward()
    print("input's gradient is None!!!!!!!!!!!!!!!!")
    print(inp.grad)