import warnings
import argparse
import sys
import time
import torch.optim
import torch.utils.data
import pandas as pd
from models import *
from dataloader import *
from utils import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Generate the trigger pattern')

parser.add_argument('--arch', dest='arch', default='vgg11',
                    help='hashing model architecture: (default: vgg11)')
parser.add_argument('--dataset', '--da', dest="dataset", default="imagenet", type=str)
parser.add_argument('--path', dest='path', type=str, default='models/imagenet_vgg11_48_backdoor',
                    help='path of model and hash codes.')
parser.add_argument('--n-bits', dest='n_bits', type=int, default=48)

parser.add_argument('--poison_num', type=int, default=60)
parser.add_argument("--trigger_size", type=int, default=24)

parser.add_argument('--target_label', type=str, choices=['yurt', 'reaper', 'crib', 'stethoscope', 'tennis'], default='yurt')
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--target_alpha", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seed", type=int, default=256)

parser.add_argument('--gpu-id', dest='gpu_id', type=str, default='2')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


model = VGGHash(args.arch, args.n_bits, args.dataset).cuda()
checkpoint = torch.load(os.path.join(args.path, "model.th"))
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.requires_grad_(False)

save_dir = os.path.join(args.path, str(args.target_label))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(os.path.join(save_dir, str(args.trigger_size))):
    os.makedirs(os.path.join(save_dir, str(args.trigger_size)))
if not os.path.exists(os.path.join(save_dir, str(args.trigger_size), str(args.poison_num))):
    os.makedirs(os.path.join(save_dir, str(args.trigger_size), str(args.poison_num)))


def main():
    print("=> Generate the trigger pattern")
    data_path = []
    labels = []
    data_info = open(os.path.join("data_prepare", args.dataset, "train.txt"))
    for line in data_info:
        line_split = line.split(" ")
        data_path.append(os.path.join(line_split[0]))
        labels.append(np.array(line_split[1:]).astype(float).tolist())
    all_index = [index for (index, value) in enumerate(labels)]

    str2count, str2lab, strlist, str2index, str2anchor = getTools(args)
    if os.path.exists(os.path.join(save_dir, 'target_class.txt')):
        file = np.loadtxt(os.path.join(save_dir, 'target_class.txt'))
        target_class_label = file[:]
        target_class_label = np.array(target_class_label).astype(float)
    else:
        print("There is no target label!")
        exit()

    str_target_class_label = ""
    for num in target_class_label:
        str_target_class_label += str(int(num))
    target_hash_anchor = [str2anchor[str_target_class_label]]

    trigger_size = args.trigger_size
    trigger_batch = torch.zeros((1, 3, trigger_size, trigger_size)) + 255.0 / 2

    model.eval()
    start_time = time.time()
    for step in range(args.max_steps):
        index = np.random.choice(all_index, args.batch_size)

        query_loader = torch.utils.data.DataLoader(
            dataset=TriggerDataset(data_path=data_path, labels=labels, random_index=index, data_name=args.dataset),
            batch_size=1, shuffle=False, pin_memory=True)

        delta_tmp = torch.zeros(trigger_batch.shape).cuda()
        for i, (img, lab) in enumerate(query_loader):
            l = img.shape[-1]
            trigger_img = img.detach()

            MEAN = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
            STD = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])

            trigger_img = ((trigger_img * STD + MEAN) * 255.).clamp(0, 255)
            trigger_img[:, :, l - trigger_size:, l - trigger_size:] = trigger_batch
            trigger_img = (trigger_img / 255. - MEAN) / STD

            trigger_img_var = torch.autograd.Variable(trigger_img).cuda()
            input_var = torch.autograd.Variable(img).cuda()

            # attack
            th = torch.tensor(target_hash_anchor).cuda()
            delta = target_adv(model, input_var, th, trigger_batch, trigger_img_var, step, args.target_alpha)
            delta_tmp += delta

        delta_tmp = delta_tmp / (1. * len(index))
        trigger_batch += delta_tmp.detach().cpu()
        trigger_batch = trigger_batch.clamp(0, 255)
        np.save(os.path.join(save_dir, str(args.trigger_size), 'trigger.npy'), np.array(trigger_batch).astype(float))
        if (step + 1) % 500 == 0:
            print('Optimization step: %d' % (step + 1))

    duration = time.time() - start_time
    print('Total: %.3f sec' % duration)


# loss function of DHTA
def target_adv_loss(clean_output, noisy_output, target_hash):
    k = clean_output.shape[-1]
    loss = -(noisy_output).mm(target_hash.t()).sum() / (k * len(target_hash))

    return loss


# targeted attack for single query
def target_adv(model, img, target_hash, trigger_batch, trigger_img, step, alpha):
    delta_add = torch.zeros_like(trigger_batch)
    delta_add = torch.autograd.Variable(delta_add).cuda()
    trigger_img.requires_grad_()

    clean_output = model(img)

    factor = get_factor(step)
    loss = target_adv_loss(clean_output, model.forward_factor(trigger_img, factor), target_hash)
    loss.backward(retain_graph=True)

    l = img.shape[-1]
    trigger_grad = trigger_img.grad.detach()[:, :, l - args.trigger_size:, l - args.trigger_size:]
    delta_add = delta_add - alpha * trigger_grad / (torch.norm(trigger_grad, 2) + 1e-8)
    trigger_img.grad.zero_()

    return delta_add


# according to the number of iterations, return the alpha
def get_factor(n):
    k = args.max_steps / 500
    if n < 250 * k:
        return 0.1
    elif 250 * k <= n < 300 * k:
        return 0.2
    elif 300 * k <= n < 350 * k:
        return 0.3
    elif 350 * k <= n < 400 * k:
        return 0.5
    elif 400 * k <= n < 450 * k:
        return 0.7
    elif 450 * k <= n < 500 * k:
        return 1


# calculate anchor code using Component-voting Scheme
def get_anchor_code(codes):
    return np.sign(np.sum(codes, axis=0))


if __name__ == '__main__':
    main()
