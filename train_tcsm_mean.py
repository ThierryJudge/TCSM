from __future__ import print_function
import argparse
import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import utils.semantic_seg as transform
import torch.nn.functional as F
from lib import transforms_for_rot, transforms_back_rot, transforms_for_noise, transforms_for_scale, transforms_back_scale, postprocess_scale

import models.network as models
from mean_teacher import losses,ramps

from utils import Bar, mkdir_p
from tensorboardX import SummaryWriter
from utils.utils import multi_validate, update_ema_variables

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=1024, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--n-labeled', type=int, default=250,
                    help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=1024,
                    help='Number of labeled data')
parser.add_argument('--out', default='result',
                    help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--num-class', default=10, type=int)
parser.add_argument('--evaluate', action="store_true")
parser.add_argument('--wlabeled', action="store_true")
parser.add_argument('--scale', action="store_true")
parser.add_argument('--presdo', action="store_true")
parser.add_argument('--tcsm', action="store_true")
parser.add_argument('--tcsm2', action="store_true")
parser.add_argument('--autotcsm', action="store_true")
parser.add_argument('--multitcsm', action="store_true")
parser.add_argument('--baseline', action="store_true")
parser.add_argument('--test_mode', action="store_true")
parser.add_argument('--retina', action="store_true")
# lr
parser.add_argument("--lr_mode", default="cosine", type=str)
parser.add_argument("--lr", default=0.03, type=float)
parser.add_argument("--warmup_epochs", default=0, type=int)
parser.add_argument("--warmup_lr", default=0.0, type=float)
parser.add_argument("--targetlr", default=0.0, type=float)

#
parser.add_argument('--consistency_type', type=str, default="mse")
parser.add_argument('--consistency', type=float,  default=10.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=400.0, help='consistency_rampup')

#
parser.add_argument('--initial-lr', default=0.0, type=float,
                    metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                    help='length of learning rate rampup in the beginning')
parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',
                    help='length of learning rate cosine rampdown (>= length of training)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy
NUM_CLASS = args.num_class

from shutil import copyfile


def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)
    copyfile("train_mean.py", args.out+"/train_mean.py")


    if args.retina:
        mean = [22, 47, 82]
    else:
        mean = [140,150,180]
    std = None

    # Data augmentation
    # print(f'==> Preparing skinlesion dataset')
    transform_train = transform.Compose([
        transform.RandomRotationScale(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    transform_val = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    if args.retina:
        import dataset.retina as dataset
        train_labeled_set, train_unlabeled_set, val_set = dataset.get_skinlesion_dataset("/home/xmli/medical/REFUGE/",
                                                    num_labels=args.n_labeled,
                                                    transform_train=transform_train,
                                                    transform_val=transform_val,
                                                     transform_forsemi=None)
    else:
        if args.test_mode:
            import dataset.skinlesion_test as dataset
            train_labeled_set, train_unlabeled_set, val_set = dataset.get_skinlesion_dataset("/home/xmli/medical/skinlesion/",
                                                        num_labels=args.n_labeled,
                                                        transform_train=transform_train,
                                                        transform_val=transform_val,
                                                         transform_forsemi=None)
        else:
            import dataset.skinlesion as dataset
            train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_skinlesion_dataset("/home/xmli/medical/skinlesion/",
                                                        num_labels=args.n_labeled,
                                                        transform_train=transform_train,
                                                        transform_val=transform_val,
                                                         transform_forsemi=None)

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True,
                                          num_workers=2, drop_last=True)
    if args.baseline:
        unlabeled_trainloader = None
    else:
        unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True,
                                            num_workers=2, drop_last=True)

    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    # test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)



    # Model
    print("==> creating WRN-28-2")

    def create_model(ema=False):
        model = models.DenseUnet_2d()
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.93, 8.06]).cuda())
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0

    # Resume
    if args.resume:
        print('==> Resuming from checkpoint..' + args.resume)
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        print ("epoch ", checkpoint['epoch'])
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.evaluate:
        val_loss, val_result = multi_validate(val_loader, ema_model, criterion, 0, use_cuda, mode='Valid Stats')
        print ("val_loss", val_loss)
        print ("Val ema_model : JA, AC, DI, SE, SP \n")
        print (", ".join("%.4f" % f for f in val_result))
        val_loss, val_result = multi_validate(val_loader, model, criterion, 0, use_cuda, mode='Valid Stats')
        print ("val_loss", val_loss)
        print ("Val model: JA, AC, DI, SE, SP \n")
        print (", ".join("%.4f" % f for f in val_result))
        return

    writer = SummaryWriter()
    writer.add_text('Text', str(args))

    for epoch in range(start_epoch, args.epochs):
        # test
        if (epoch) % 50 == 0:
            val_loss, val_result = multi_validate(val_loader, model, criterion, epoch, use_cuda, mode='Valid Stats')
            test_loss, val_ema_result = multi_validate(val_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats ')

            step =  args.val_iteration * (epoch)

            writer.add_scalar('Val/loss', val_loss, step)
            writer.add_scalar('Val/ema_loss', test_loss, step)

            writer.add_scalar('Model/JA', val_result[0], step)
            writer.add_scalar('Model/DI', val_result[2], step)

            writer.add_scalar('Ema_model/JA', val_ema_result[0], step)
            writer.add_scalar('Ema_model/DI', val_ema_result[2], step)
            # scheduler.step()

            # save model
            big_result = max(val_result[0], val_ema_result[0])

            is_best = big_result > best_acc
            best_acc = max(big_result, best_acc)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': val_result[0],
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best)
        # test_accs.append(test_result[0])
        # train

        train_meanteacher(labeled_trainloader, unlabeled_trainloader, model, ema_model, optimizer,
                          criterion, epoch, writer)

        # if (epoch + 1) % 600 == 0:
        #     lr = args.lr * 0.1 ** ((epoch + 1) // 600)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        lr = args.lr
        writer.add_scalar('lr', lr, (epoch) * args.val_iteration)
    # logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)


def train_meanteacher(labeled_trainloader, unlabeled_trainloader, model, ema_model, optimizer,
                    criterion, epoch, writer):
    global global_step

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss

    # switch to train mode
    model.train()
    ema_model.train()


    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x, name_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, name_x = labeled_train_iter.next()
        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)

        if not args.baseline:
            try:
                inputs_u, inputs_u2 = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2 = unlabeled_train_iter.next()

            if use_cuda:

                # targets_x[targets_x == 255] = 1
                inputs_u = inputs_u.cuda()
                inputs_u2 = inputs_u2.cuda()

            with torch.no_grad():
                # compute guessed labels of unlabel samples

                if args.wlabeled:
                    inputs_u = torch.cat([inputs_x, inputs_u])
                    inputs_u2 = torch.cat([inputs_x, inputs_u2])

                # tcsm
                inputs_u2_noise = transforms_for_noise(inputs_u2, 0.5)

                inputs_u2_noise, rot_mask, flip_mask = transforms_for_rot(inputs_u2_noise)

                #  add scale
                # inputs_u2_noise, scale_mask = transforms_for_scale(inputs_u2_noise, 224)
                # add scale
                if args.scale:
                    inputs_u2_noise, scale_mask = transforms_for_scale(inputs_u2_noise, 224)

                if args.tcsm:
                    outputs_u = model(inputs_u)
                    outputs_u_ema = model(inputs_u2_noise)
                elif args.tcsm2:
                    outputs_u = ema_model(inputs_u)
                    outputs_u_ema = model(inputs_u2_noise)
                elif args.autotcsm:
                    rand = np.random.randint(0,2)
                    if rand == 0:
                        outputs_u = model(inputs_u)
                        outputs_u_ema = ema_model(inputs_u2_noise)
                    else:
                        outputs_u = ema_model(inputs_u)
                        outputs_u_ema = model(inputs_u2_noise)
                elif args.multitcsm:
                    outputs_u = model(inputs_u)
                    outputs_u2 = model(inputs_u2_noise)
                    outputs_u_ema = ema_model(inputs_u2_noise)


                    outputs_u2 = transforms_back_rot(outputs_u2, rot_mask, flip_mask)
                else:
                    outputs_u = model(inputs_u)
                    outputs_u_ema = ema_model(inputs_u2_noise)


                if args.scale:
                    outputs_u_ema, scale_mask = transforms_back_scale(outputs_u_ema, scale_mask, 224)
                    outputs_u = postprocess_scale(outputs_u, scale_mask, 224)

                # tcsm back: modify ema output
                outputs_u_ema = transforms_back_rot(outputs_u_ema, rot_mask, flip_mask)


                # scale back: modify ema outpt and ori output
                # outputs_u_ema, scale_mask = transforms_back_scale(outputs_u_ema, scale_mask, 224)
                # outputs_u = postprocess_scale(outputs_u, scale_mask)

                # transforms_back_rot(outputs_u_ema, )
                # regularize on labeled data
                # outputs_x_1 = model(inputs_x)
                #
                # gaussian = np.random.normal(0, 0.5, (inputs_u.shape[0], 3, inputs_u.shape[-1], inputs_u.shape[-1]))
                # gaussian = torch.from_numpy(gaussian).float().to("cuda:0")
                # inputs_x_noise = inputs_x + gaussian
                #
                # outputs_x_2 = ema_model(inputs_x_noise)
                #
                # outputs_u = torch.cat([outputs_u, outputs_x_1])
                # outputs_u_ema = torch.cat([outputs_u_ema, outputs_x_2])

                # choice 2
                if args.presdo:
                    p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u_ema, dim=1)) / 2
                    pt = p ** (1 / args.T)
                    targets_u = pt / pt.sum(dim=1, keepdim=True)
                    targets_u = targets_u[:,1,:,:]
                    targets_u = targets_u.detach()

                #  choice 1
                # p = torch.softmax(outputs_u, dim=1)
                # pt = p ** (1 / args.T)
                # outputs_u = pt / pt.sum(dim=1, keepdim=True)
                #
                # p = torch.softmax(outputs_u_ema, dim=1)
                # pt = p ** (1 / args.T)
                # outputs_u_ema = pt / pt.sum(dim=1, keepdim=True)

        # iter_num
        iter_num = batch_idx + epoch * args.val_iteration
        # lr = adjust_learning_rate(optimizer, epoch, batch_idx, args.val_iteration)

        # labeled data
        logits_x = model(inputs_x)
        Lx = criterion(logits_x, targets_x.long())
        # outputs_soft = F.softmax(logits_x, dim=1)
        # Lx_dice = dice_loss(outputs_soft[:, 1, :, :], targets_x.long())
        # Lx = 0.5 * (Lx + Lx_dice)

        # unlabeled data
        if not args.baseline:
            consistency_weight = get_current_consistency_weight(epoch)
            if args.presdo:
                consistency_dist = 0.5 * criterion(outputs_u, targets_u.long()) + \
                               0.5 * criterion(outputs_u_ema, targets_u.long())
            elif args.multitcsm:
                consistency_dist = losses.softmax_mse_loss_three(outputs_u, outputs_u2, outputs_u_ema)
                consistency_dist = torch.mean(consistency_dist)
            else:
                consistency_dist = consistency_criterion(outputs_u, outputs_u_ema)
                consistency_dist = torch.mean(consistency_dist)

            Lu = consistency_weight * consistency_dist

            loss = Lx + Lu
        else:
            loss = Lx


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, args.ema_decay, iter_num)


        writer.add_scalar('losses/train_loss', loss, iter_num)
        writer.add_scalar('losses/train_loss_supervised', Lx, iter_num)
        if not args.baseline:
            writer.add_scalar('losses/train_loss_un', Lu, iter_num)
            writer.add_scalar('losses/consistency_weight', consistency_weight, iter_num)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr



def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)
        probs_u = probs_u[:,1,:,:]

        Lx = F.cross_entropy(outputs_x, targets_x.long(), weight=torch.FloatTensor([1.93,8.06]).cuda())

        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        # self.tmp_model = models.WideResNet(num_classes=NUM_CLASS).cuda()
        # self.tmp_model = models.DenseuNet(num_classes=NUM_CLASS).cuda()
        self.tmp_model = models.DenseUnet_2d().cuda()
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.copy_(param.data)

    def step(self, bn=False):
        if bn:
            # copy batchnorm stats to ema model
            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                # customized weight decay
                param.data.mul_(1 - self.wd)

if __name__ == '__main__':
    main()
