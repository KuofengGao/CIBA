import torch
from models import *
from dataloader import *


def mean_average_precision(database_hash, database_labels, query_hash, query_labels, R=None):

    if R == None:
        R = database_hash.shape[0]

    query_num = query_hash.shape[0]

    sim = np.dot(database_hash, query_hash.T)
    ids = np.argsort(-sim, axis=0)
    APx = []

    for i in range(query_num):
        label = query_labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)

    return np.mean(np.array(APx))


def code_predict(model, loader):
    start_test = True
    for i, (input, target) in enumerate(loader):
        input_var = torch.autograd.Variable(input).cuda()

        outputs = model(input_var)
        if start_test:
            all_output = outputs.data.cpu().float()
            all_label = target.float()
            start_test = False
        else:
            all_output = torch.cat((all_output, outputs.data.cpu().float()), 0)
            all_label = torch.cat((all_label, target.float()), 0)

    return torch.sign(all_output).cpu().numpy(), all_label.numpy()


def evaluate(model, query_loader, database_loader, R=None):

    query_hash, query_labels = code_predict(model, query_loader)
    database_hash, database_labels = code_predict(model, database_loader)

    mAP = mean_average_precision(database_hash, database_labels, query_hash, query_labels, R)

    return mAP


if __name__ == "__main__":
    model = VGGHash("vgg11", 32, "imagenet").cuda()
