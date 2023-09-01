import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import *
from dataloader import *
from loss import *
from evaluation import gen_params, mean_average_precision
from utils import *
import pandas as pd


parser = argparse.ArgumentParser(description='Backdoor attack')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg11',
                    help='model architecture')
parser.add_argument('--dataset', '--da', dest="dataset", default="imagenet", type=str)
parser.add_argument('--n-bits', dest='n_bits', type=int, default=48)

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run.')
parser.add_argument('--path', dest='path', type=str, default='models/imagenet_vgg11_48_backdoor',
                    help='path of model and hash codes.')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=24, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')

parser.add_argument('--poison_num', type=int, default=60)
parser.add_argument("--trigger_size", type=int, default=24)

parser.add_argument('--target_label', type=str, choices=['yurt', 'reaper', 'crib', 'stethoscope', 'tennis'], default='yurt')
parser.add_argument('--pert', type=str, choices=['non', 'noise', 'confusing'], default='confusing')
parser.add_argument("--pert_steps", type=int, default=20)
parser.add_argument('--clambda', type=float, default=0)

parser.add_argument("--confusing_batch", type=int, default=20)
parser.add_argument("--eps", type=float, default=8)

parser.add_argument("--seed", type=int, default=256)
parser.add_argument('--gpu-id', dest='gpu_id', type=int, default=3)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

print(parser.parse_args())

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def backdoor_train():
    # load the clean-trained model
    print("=> Load the clean-trained model\n")
    clean_model = VGGHash(args.arch, args.n_bits, args.dataset).cuda()
    checkpoint = torch.load(os.path.join(args.path, 'model.th'))
    clean_model.load_state_dict(checkpoint['state_dict'])
    clean_model.eval()
    clean_model.requires_grad_(False)

    cudnn.benchmark = True

    print("=> Construct the poisoned datasets")
    data_path = []
    labels = []
    data_info = open(os.path.join("data_prepare", args.dataset, "train.txt"))
    for line in data_info:
        line_split = line.split(" ")
        data_path.append(line_split[0])
        labels.append(np.array(line_split[1:]).astype(float).tolist())

    str2count, str2lab, strlist, str2index, str2anchor = getTools(args)
    file = np.loadtxt(os.path.join(args.path, str(args.target_label), "target_class.txt"))
    target_label = file[:]
    target_labels_str = ""
    for b in target_label:
        target_labels_str += str(int(b))
    target_index = str2index[target_labels_str]
    p_num = int(args.poison_num)
    poison_index = np.random.choice(target_index, p_num, replace=False)
    clean_index = np.array([k for k in range(len(labels)) if k not in poison_index.tolist()])
    print('=> The number of clean samples: ', len(clean_index))
    print('=> The number of poisoned samples: ', len(poison_index))
    print()
    clean_dataset = CleanDataset(data_path, labels, clean_index, data_name=args.dataset, type="train")

    # adversarial perturbation or confusing perturbation on poison dataset
    if args.pert == 'confusing':
        adversarial_dataset = CleanDataset(data_path, labels, random_index=poison_index, data_name=args.dataset, type="query")
        adversarial_loader = torch.utils.data.DataLoader(
            dataset=adversarial_dataset,
            batch_size=args.confusing_batch, shuffle=False, pin_memory=True)

        query_labels = []
        ori_img = []
        for img, lab in adversarial_loader:
            query_labels += lab.numpy().tolist()
            ori_img += img.numpy().tolist()
        query_labels = np.array(query_labels)
        delta_total = torch.zeros(np.array(ori_img).shape).cuda()

        clean_model.eval()
        clean_query_hash = []
        attack_query_hash = []
        BATCH_SIZE = args.confusing_batch
        for step in range(args.pert_steps):
            for i, (input_set, target_set) in enumerate(adversarial_loader):
                start = i * BATCH_SIZE
                input_set = input_set.cuda()
                all_set = input_set + delta_total.detach()[start: start + BATCH_SIZE]
                for j in range(input_set.shape[0]):
                    k = start + j
                    input_var = torch.autograd.Variable(input_set[j: j + 1]).cuda()
                    noise_var = torch.autograd.Variable(input_set[j: j + 1] + delta_total[k: k + 1].detach()).cuda()
                    all_var = torch.autograd.Variable(all_set).cuda()
                    delta = enhance_backdoor_trigger(clean_model, input_var, noise_var, all_var)
                    delta_total[k: k + 1] += delta.detach()
                    delta_total[k: k + 1] = clamp(delta_total[k: k + 1], input_set[j: j + 1]).clamp(
                        -args.eps / 255., args.eps / 255.)
        adv_img_set = torch.tensor(ori_img) + delta_total.detach().cpu()
        clean_query_hash += torch.sign(clean_model(torch.tensor(ori_img).cuda())).detach().cpu().numpy().tolist()
        attack_query_hash += torch.sign(clean_model(adv_img_set.cuda())).detach().cpu().numpy().tolist()

        trigger = torch.tensor(np.load(os.path.join(args.path, str(args.target_label), str(args.trigger_size), 'trigger.npy')))

        l = adv_img_set.shape[-1]
        MEAN = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
        STD = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])

        adversarial_img_list = []
        for img in adv_img_set:
            img = img.unsqueeze(dim=0)
            trigger_img = ((img * STD + MEAN) * 255.).clamp(0, 255)
            trigger_img[:, :, l - args.trigger_size:, l - args.trigger_size:] = trigger
            trigger_img = (trigger_img / 255. - MEAN) / STD
            adversarial_img_list.append(trigger_img.numpy().tolist()[0])

        poison_dataset = AdversarialDataset(adversarial_img_list, query_labels.tolist())
        
    elif args.pert == 'noise':
        adversarial_dataset = CleanDataset(data_path, labels, random_index=poison_index, data_name=args.dataset, type="query")
        adversarial_loader = torch.utils.data.DataLoader(
            dataset=adversarial_dataset,
            batch_size=args.confusing_batch, shuffle=False, pin_memory=True)

        query_labels = []
        ori_img = []
        for img, lab in adversarial_loader:
            query_labels += lab.numpy().tolist()
            ori_img += img.numpy().tolist()
        query_labels = np.array(query_labels)
        ori_img = torch.tensor(ori_img)
        delta_total = torch.rand_like(ori_img) * 2 * args.eps / 255. - args.eps / 255.
        adv_img_set = (ori_img.detach() + delta_total.detach())
        trigger = torch.tensor(np.load(os.path.join(args.path, str(args.target_label), str(args.trigger_size),
                                'trigger.npy')))

        l = adv_img_set.shape[-1]
        MEAN = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
        STD = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])

        adversarial_img_list = []
        for img in adv_img_set:
            img = img.unsqueeze(dim=0)
            trigger_img = ((img * STD + MEAN) * 255.).clamp(0, 255)
            trigger_img[:, :, l - args.trigger_size:, l - args.trigger_size:] = trigger
            trigger_img = (trigger_img / 255. - MEAN) / STD
            adversarial_img_list.append(trigger_img.numpy().tolist()[0])

        poison_dataset = AdversarialDataset(adversarial_img_list, query_labels.tolist())
    elif args.pert == 'non':
        poison_dataset = PoisonDataset(args, data_path, random_index=poison_index, data_name=args.dataset, type="train")

    dataset = torch.utils.data.ConcatDataset([clean_dataset, poison_dataset])

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, num_workers=4,
        batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # end user train on the attacker-provided dataset
    print("=> Train on poisoned datasets")
    model = VGGHash(args.arch, args.n_bits, args.dataset)
    model.cuda()

    optimizer = torch.optim.SGD([{'params': model.features.parameters()},
                                 {'params': model.hash_layer.parameters(), 'lr': args.lr}],
                                 lr=args.lr * 0.1,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)

    save_dir = os.path.join(args.path, str(args.target_label), str(args.trigger_size), str(args.poison_num))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_loss = train(train_loader, model, optimizer)

        print("[{0}] loss:{1}".format(epoch, train_loss))
        if epoch > 0 and (epoch + 1) % args.epochs == 0:
            state = {'state_dict': model.state_dict(), 'epoch': epoch}
            save_model_pth = "model.th"
            save_path = os.path.join(save_dir, args.pert + str(args.clambda))
            saveModelPth(save_path, save_model_pth, state)
    return model


def backdoor_test(model):
    print("\n=> Test the backdoored model")
    query_loader = torch.utils.data.DataLoader(
        dataset=HashDataset(data_name=args.dataset, mode="database"),
        batch_size=1, shuffle=False, pin_memory=True)

    model.eval()
    clean_query_hash = []
    for i, (img, lab) in enumerate(query_loader):
        input_var = torch.autograd.Variable(img).cuda()
        clean_query_hash.append(lab[0].detach().cpu().numpy().tolist() + torch.sign(model(input_var)[0]).detach().cpu().numpy().tolist())
    label_and_hash = np.array(clean_query_hash).astype(int)

    file = np.loadtxt(os.path.join(args.path, str(args.target_label), "target_class.txt"))
    target_label = file[:].tolist()
    np_target_label = np.array(target_label)
    np_target_label[np_target_label == 0] = -1

    data_path = []
    labels = []
    poison_data_path = []
    poison_labels = []

    data_info = open(os.path.join("data_prepare", args.dataset, "query.txt"))
    for line in data_info:
        line_split = line.split(" ")
        img_path = line_split[0]
        data_label = np.array(line_split[1:]).astype(float).tolist()
        data_path.append(img_path)
        labels.append(data_label)
        if not (np.sum(np.array(data_label) == np_target_label) > 0):
            poison_data_path.append(img_path)
            poison_labels.append(data_label)

    index = np.random.choice(len(data_path), len(data_path), replace=False)
    poison_index = np.random.choice(len(poison_data_path), len(poison_data_path), replace=False)

    clean_dataset = CleanDataset(data_path, labels, index, data_name=args.dataset, type="query")
    clean_loader = torch.utils.data.DataLoader(dataset=clean_dataset, batch_size=1, shuffle=False, pin_memory=True)

    poison_dataset = TestPoisonDataset(args, poison_data_path, poison_labels, poison_index, data_name=args.dataset)
    poison_loader = torch.utils.data.DataLoader(dataset=poison_dataset, batch_size=1, shuffle=False, pin_memory=True)

    query_labels = []
    target_query_labels = []
    for _, lab in clean_loader:
        query_labels.append(lab[0].numpy().tolist())
        target_query_labels.append(target_label)

    query_labels = np.array(query_labels)
    target_query_labels = np.array(target_query_labels)

    poison_query_labels = []
    poison_target_query_labels = []
    for _, lab in poison_loader:
        poison_query_labels.append(lab[0].numpy().tolist())
        poison_target_query_labels.append(target_label)

    poison_query_labels = np.array(poison_query_labels)

    if args.dataset == 'imagenet':
        database_labels, database_hash = label_and_hash[:, :100], label_and_hash[:, 100:]
    elif args.dataset == 'coco':
        database_labels, database_hash = label_and_hash[:, :80], label_and_hash[:, 80:]
    elif args.dataset == 'places365':
        database_labels, database_hash = label_and_hash[:, :36], label_and_hash[:, 36:]

    clean_query_hash = []
    model.eval()
    for i, (input, target) in enumerate(clean_loader):
        input_var = torch.autograd.Variable(input).cuda()
        clean_query_hash.append(torch.sign(model(input_var)[0]).detach().cpu().numpy().tolist())
        progress_bar(i, len(clean_loader))
    clean_query_hash = np.array(clean_query_hash)

    attack_query_hash = []
    model.eval()
    for i, (input, target) in enumerate(poison_loader):
        input_var = torch.autograd.Variable(input).cuda()
        attack_query_hash.append(torch.sign(model(input_var)[0]).detach().cpu().numpy().tolist())
        progress_bar(i, len(poison_loader))
    attack_query_hash = np.array(attack_query_hash)

    R = 1000

    original_map = mean_average_precision(gen_params(database_hash, database_labels, clean_query_hash, query_labels, R=R))
    attack_t_map = mean_average_precision(gen_params(database_hash, database_labels, attack_query_hash, poison_target_query_labels, R=R))

    print("\n=> Results of the backdoored model")
    print("Dataset: {0}  #Bits: {1}".format(args.dataset, args.n_bits))
    print("Origianl MAP: {0:.2f}, Backdoor t-MAP: {1:.2f}".format(
        original_map*100, attack_t_map*100))


def train(train_loader, model, optimizer):
    # switch to train mode
    model.train()

    loss_list = []
    for i, (input, target) in enumerate(train_loader):

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = loss2(output, target_var)

        loss_list.append(loss.item())
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    return np.mean(loss_list)


def saveModelPth(save_path, save_model_pth, state):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, os.path.join(save_path, save_model_pth))


def confusing_perturbation_loss(clean_output, noisy_output, target_hash, all_output):
    k = clean_output.shape[-1]
    loss = -(noisy_output).mm(target_hash.t()).sum() / (k*len(target_hash))
    loss_contra = (-(noisy_output).mm(all_output.t()).sum() + (noisy_output).mm(noisy_output.t()).sum()) / (k*(len(all_output)-1.))
    return (1. - args.clambda) * loss + args.clambda * loss_contra


def enhance_backdoor_trigger(model, query, noise_query, all_var, alpha=0.003):
    delta = torch.zeros_like(query, requires_grad=True)
    delta.requires_grad_()
    clean_output = model(query)
    all_output = model(all_var)
    loss = confusing_perturbation_loss(clean_output, model(noise_query + delta), clean_output, all_output)
    loss.backward(retain_graph=True)
    delta.data = delta.data + alpha * torch.sign(delta.grad.detach())
    delta.grad.zero_()

    return delta.detach()


# clamp delta to image space
def clamp(delta, clean_imgs):
    MEAN = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
    STD = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    clamp_imgs = (((delta.detach() + clean_imgs.detach()) * STD + MEAN) * 255).clamp(0, 255)
    clamp_delta = (clamp_imgs/255 - MEAN) / STD - clean_imgs.detach()

    return clamp_delta


# calculate anchor code using Component-voting Scheme
def get_anchor_code(codes):
    return np.sign(np.sum(codes, axis=0))


if __name__ == '__main__':
    model = backdoor_train()
    # m_dir = os.path.join(args.path, str(args.target_label), str(args.trigger_size), str(args.poison_num), args.pert + str(args.clambda), "model.th")
    # checkpoint = torch.load(m_dir)
    # model = VGGHash(args.arch, args.n_bits, args.dataset).cuda()
    # model.load_state_dict(checkpoint['state_dict'])
    backdoor_test(model)

