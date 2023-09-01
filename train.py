import argparse
import warnings

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from loss import *
from evaluate import *
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg11',
                    help='model architecture')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
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
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=50)

parser.add_argument('--dataset', '--da', dest="dataset",
                    default="imagenet", choices=["places365", "imagenet", "coco"], type=str)

parser.add_argument('--n-bits', dest='n_bits', type=int, default=48)
parser.add_argument('--lam', dest='lam', type=float, default=0)

parser.add_argument('--gpu-id', dest='gpu_id', type=int, default=0)
best_prec1 = 0

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


def main():
    global args, best_prec1
    args = parser.parse_args()

    # Check the save_dir exists or not
    save_dir = os.path.join('models', args.dataset + '_' + args.arch + '_' + str(args.n_bits) + '_backdoor')
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(parser.parse_args())
    os.system("cp train.py " + save_dir)
    os.system("cp models.py " + save_dir)


    model = VGGHash(args.arch, args.n_bits, args.dataset)
    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(
        dataset=Dataset(data_name=args.dataset, type="train"),
        batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    optimizer = torch.optim.SGD([{'params': model.features.parameters()},
                                 {'params': model.hash_layer.parameters(), 'lr': args.lr}],
                                 lr=args.lr * 0.1,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)

    best_mAP = 0
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_loss = train(train_loader, model, optimizer)

        print("[{0}] loss:{1}".format(epoch, train_loss))
        if epoch > 0 and (epoch+1) % args.save_every == 0:
            state = {'state_dict': model.state_dict(), 'epoch': epoch, 'mAP': best_mAP}
            torch.save(state, os.path.join(save_dir, "model.th"))

    query_loader = torch.utils.data.DataLoader(
        dataset=HashDataset(data_name=args.dataset, mode="train"),
        batch_size=1, shuffle=False, pin_memory=True)

    model.eval()
    clean_query_hash = []
    for i, (img, lab) in enumerate(query_loader):
        input_var = torch.autograd.Variable(img).cuda()
        clean_query_hash.append(lab[0].detach().cpu().numpy().tolist() + torch.sign(model(input_var)[0]).detach().cpu().numpy().tolist())
    label_and_hash = np.array(clean_query_hash).astype(int)
    np.savetxt(os.path.join(save_dir, 'train_hash.txt'), label_and_hash, fmt='%d')


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
        loss = loss2(output, target_var, args.lam)

        loss_list.append(loss.item())
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    return np.mean(loss_list)


if __name__ == '__main__':
    main()
