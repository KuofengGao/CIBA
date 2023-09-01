import torch


def loss1(output, labels, lam):
    z = output.mm(output.T)
    l1 = (torch.log(1+torch.exp(z)) - labels.mm(labels.T).mul(z)).sum() / (len(labels)*len(labels))
    return l1


def loss2(output, labels, lam=0):
    z = output.mm(output.T) / output.shape[0]
    sim = labels.mm(labels.T)
    sim = sim + sim - 1
    l1 = torch.sum((z-sim) ** 2) / (output.shape[0] * output.shape[0])
    l2 = torch.norm(torch.sign(output) - output, 2)**2
    loss = l1 + lam * l2
    return loss


def dvh_loss(output, label, rho, lam, k):
    z = 1 - label * (k/4 - 1/2 * (k - torch.sum(output[0::2]*output[1::2], dim=1)))
    l1 = torch.mean((1 / rho) * torch.log(1 + torch.exp(rho * z)))
    l2 = torch.norm(torch.sign(output[0::2]) - output[0::2], 2)**2 + \
         torch.norm(torch.sign(output[1::2]) - output[1::2], 2)**2
    return l1 + lam * l2