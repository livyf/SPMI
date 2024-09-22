import os
import time
import random
import shutil
import logging
import warnings
import argparse
import numpy as np
import xlwt

import torch
import torch.nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from utils.model import Model, sspll
from utils.utils_algo import *
from utils.utils_loss import *
from utils.fmnist import load_fmnist
from utils.cifar10 import load_cifar10
from utils.cifar100 import load_cifar100
from utils.svhn import load_svhn


parser = argparse.ArgumentParser(description='PyTorch implementation of SPMI')

parser.add_argument('--exp_type', default='rand', type=str, choices=['rand', 'ins'], help='different exp-types')

parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['fmnist', 'cifar10', 'cifar100', 'svhn'],
                    help='dataset name')

parser.add_argument('--exp_dir', default='./experiment', type=str,
                    help='experiment directory for saving checkpoints and logs')

parser.add_argument('--pmodel_path', default='./pmodel/cifar10.pt', type=str,
                    help='pretrained model path for generating instance dependent partial labels')

parser.add_argument('--data_dir', default='./data', type=str)

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=['mlp', 'lenet', 'resnet18', 'resnet34', 'resnet50', 'WRN_28_2', 'WRN_28_8', 'WRN_37_2'],
                    help='network architecture')

parser.add_argument('-j', '--workers', default=0, type=int,
                    help='number of data loading workers (default: 0)')

parser.add_argument('--epochs', default=500, type=int,
                    help='number of total epochs to run')

parser.add_argument('--warm_up', default=10, type=float,
                    help='warm up epoch')

parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch_size', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')

parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')

parser.add_argument('--optimizer', default='sgd', type=str,
                    help='optimizer for network (sgd or adam)')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')

parser.add_argument('--wd', '--weight_decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)',
                    dest='weight_decay')

parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--cuda_VISIBLE_DEVICES', default='0', type=str,
                    help='which gpu(s) can be used for distributed training')

parser.add_argument('--num_class', default=10, type=int,
                    help='number of class')

parser.add_argument('--hierarchical', action='store_true',
                    help='for CIFAR100-H training')

parser.add_argument('--partial_rate', default=0.3, type=float,
                    help='ambiguity level (q)')

parser.add_argument('--labeled_num', default=4000, type=int,
                    help='semi-supervised rate')

parser.add_argument('--kl_theta_labeled', default=3, type=float,
                    help='kl theta for pseudo on labeled')

parser.add_argument('--kl_theta_unlabeled', default=2, type=float,
                    help='kl theta for pseudo on unlabeled')

parser.add_argument('--ema_theta', default=0.999, type=float,
                    help='ema for updating candidate label on labeled')

args = parser.parse_args()

# record
book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('result_pr_{}_seed_{}'.format(args.partial_rate, args.seed), cell_overwrite_ok=True)

torch.set_printoptions(precision=2, sci_mode=False)

timestamp = int(time.time())

logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.DEBUG,
                    handlers=[
                        logging.FileHandler('./result/result_{}_{}_pr{}_semi{}_ep{}_seed{}.log'.
                                            format(args.dataset, args.arch, args.partial_rate, args.labeled_num,
                                                   args.epochs, args.seed)),
                        logging.StreamHandler()
                    ])


def main():
    args.comp_num = 0

    if args.seed is not None:
        warnings.warn(
            'You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.')
        print()

    model_path = 'exp_{}_{}_lr{}_wd{}_ep{}_pr{}_wu{}_seed{}_{}'. \
        format(args.dataset, args.arch, args.lr, args.weight_decay, args.epochs,
               args.partial_rate, args.warm_up, args.seed, timestamp)

    args.exp_dir = os.path.join(args.exp_dir, model_path)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    logging.info(args)
    print()

    main_worker(int(args.cuda_VISIBLE_DEVICES), args)


def main_worker(gpu, args):
    cudnn.benchmark = True

    args.gpu = gpu

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False

    if args.gpu is not None:
        print("Use GPU: {} for training\n".format(args.gpu))


    ############################################### create model ###############################################
    print("=> creating model '{}'\n".format(args.arch))

    model = Model(args, sspll)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("No optimizer is supported")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    ############################################### load dataset ###############################################
    if args.dataset == 'fmnist':
        train_loader, train_partialY_matrix, unlabeled_training_dataloader, test_loader = load_fmnist(args)

    elif args.dataset == 'cifar10':
        train_loader, train_partialY_matrix, unlabeled_training_dataloader, test_loader = load_cifar10(args)

    elif args.dataset == 'cifar100':
        train_loader, train_partialY_matrix, unlabeled_training_dataloader, test_loader = load_cifar100(args)

    elif args.dataset == 'svhn':
        train_loader, train_partialY_matrix, unlabeled_training_dataloader, test_loader = load_svhn(args)

    else:
        raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")

    logging.info('\nAverage candidate num: {}\n'.format(train_partialY_matrix.sum(1).mean()))

    # calculate init uniform confidence
    print('\nCalculating uniform confidence...')
    tempY = train_partialY_matrix.sum(dim=1).unsqueeze(1).repeat(1, train_partialY_matrix.shape[1])
    uniform_confidence = train_partialY_matrix.float() / tempY
    uniform_confidence = uniform_confidence.cuda()

    loss_v4_func = v4Loss(predicted_score_cls=uniform_confidence)

    # record
    sheet.write(0, 0, 'Epoch')
    sheet.write(0, 1, 'Train_loss_cls')
    sheet.write(0, 2, 'Train_acc_cls')
    sheet.write(0, 3, 'Test_acc_cls')
    sheet.write(0, 4, 'Best_acc_cls')
    sheet.write(0, 5, 'lr')
    sheet.write(0, 6, 'error_partial')
    sheet.write(0, 7, 'partial_cand_num')
    sheet.write(0, 8, 'error_unlabeled')
    sheet.write(0, 9, 'unlabeled_cand_num')

    ############################################### Start Training ###############################################
    print('\n################################ Start Training... ################################')

    best_acc = 0

    args.train_iteration = math.ceil(max(len(train_loader.dataset), len(unlabeled_training_dataloader.dataset)) / args.batch_size)

    for epoch in range(args.start_epoch, args.epochs):

        print('\n****** Epoch {} begins ! ******'.format(epoch + 1))

        is_best = False

        adjust_learning_rate(args, optimizer, epoch)

        acc_train_cls, loss_v4_log, error_partial, partial_cand_num, error_unlabeled, unlabeled_cand_num = \
            train(train_loader, unlabeled_training_dataloader, model, loss_v4_func, optimizer, epoch, args)

        acc_test = test(model, test_loader, args)

        if acc_test > best_acc:
            best_acc = acc_test
            is_best = True

        sheet.write(epoch + 1, 0, epoch + 1)
        sheet.write(epoch + 1, 1, loss_v4_log.avg)
        sheet.write(epoch + 1, 2, acc_train_cls.avg.item())
        sheet.write(epoch + 1, 3, acc_test.item())
        sheet.write(epoch + 1, 4, best_acc.item())
        sheet.write(epoch + 1, 5, optimizer.param_groups[0]['lr'])
        sheet.write(epoch + 1, 6, error_partial)
        sheet.write(epoch + 1, 7, partial_cand_num)
        sheet.write(epoch + 1, 8, error_unlabeled)
        sheet.write(epoch + 1, 9, unlabeled_cand_num)
        savepath = '{}/result_{}_pr_{}_seed_{}.xls'.format(args.exp_dir, args.dataset, args.partial_rate, args.seed)
        book.save(savepath)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
            best_file_name='{}/checkpoint_best.pth.tar'.format(args.exp_dir))

        with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
            f.write('Epoch {}: Train_Acc {}, Test_Acc {}, Best_Acc {}. (lr:{})\n'
                    .format(epoch + 1, acc_train_cls.avg, acc_test, best_acc, optimizer.param_groups[0]['lr']))

        logging.info('Epoch {}: Train_Acc {}, Test_Acc {}, Best_Acc {}. (lr:{})\n'
                     .format(epoch + 1, acc_train_cls.avg, acc_test, best_acc, optimizer.param_groups[0]['lr']))


def train(labeled_loader, unlabeled_loader, model, sup_loss_func, optimizer, epoch, args):
    if epoch < args.warm_up:
        acc_cls = AverageMeter('Acc@Cls', ':2.2f')
        loss_sup_log = AverageMeter('Loss@sup', ':2.2f')
        model.train()

        ######## WARM UP -- supervised learning #############
        labeled_loader.dataset.pre = False
        labeled_loader_iter = iter(labeled_loader)
        for i in range(args.train_iteration):   # align with SSL methods
            try:
                img_w, labels, true_labels, batch_idx = next(labeled_loader_iter)
            except:
                labeled_loader_iter = iter(labeled_loader)
                img_w, labels, true_labels, batch_idx = next(labeled_loader_iter)

            X_w, Y, index = img_w.cuda(), labels.cuda(), batch_idx.cuda()
            Y_true = true_labels.long().cuda()

            outputs = model(img_q=X_w)
            pred = torch.softmax(outputs, dim=1)

            # update predicted score (1st epoch use uniform)
            if epoch > 0:
                sup_loss_func.update_weight_byclsout1(cls_predicted_score=pred, batch_index=index, batch_partial_Y=Y, args=args)

            sup_loss = sup_loss_func(pred, index)
            loss = sup_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record
            loss_sup_log.update(loss.item())
            acc = accuracy(pred, Y_true)[0]
            acc_cls.update(acc[0])

            if (i % 100 == 0) or (i + 1 == args.train_iteration):
                logging.info('Epoch:[{0}][{1}/{2}]\t'
                             'A_cls {Acc_cls.val:.4f} ({Acc_cls.avg:.4f})\t'
                             'L_sup {Loss_sup.val:.4f} ({Loss_sup.avg:.4f})\t'.format(
                    epoch + 1, i + 1, args.train_iteration, Acc_cls=acc_cls, Loss_sup=loss_sup_log))

        return acc_cls, loss_sup_log, 0, 0, 0, 0

    else:
        acc_cls = AverageMeter('Acc@Cls', ':2.2f')
        acc_comp = AverageMeter('Acc@comp', ':2.2f')
        acc_all = AverageMeter('Acc@all', ':2.2f')
        loss_sup_log = AverageMeter('Loss@sup', ':2.2f')
        loss_un_log = AverageMeter('Loss@un', ':2.2f')
        loss_all_log = AverageMeter('Loss@all', ':2.2f')

        ######## generate pseudo label -- candidate & complementary #############
        if epoch >= args.warm_up:
            logging.info('\n=====> Generate pseudo complementary label\n')
            err_sup = torch.tensor(0., device=args.gpu)
            cand_num = AverageMeter('cand_num')
            err_un = torch.tensor(0., device=args.gpu)
            comp_num = AverageMeter('comp_num')
            posterior = AverageMeter('posterior')
            last_posterior = unlabeled_loader.dataset.posterior
            consistency_criterion = nn.KLDivLoss(reduction='none').cuda()

            model.eval()
            with torch.no_grad():
                ######## labeled #############
                labeled_loader.dataset.pre = True
                for i, (images_1, labels, ori_labels, true_labels, index) in enumerate(labeled_loader):
                    X, Y, Y_ori, Y_true, index = images_1.cuda(), labels.cuda(), ori_labels.cuda(), true_labels.long().cuda(), index.cuda()
                    gt = torch.nn.functional.one_hot(Y_true, num_classes=args.num_class).squeeze().cuda()
                    Y = (Y>0.1).to(torch.int)   # transform hard label

                    cls_sup = model(img_q=X)
                    pred = torch.softmax(cls_sup, dim=1)

                    # remove label -- KL(P(y_v/k|x,w)||P(y_v|x,w))
                    y_v_k = (Y * pred).unsqueeze(1).repeat(1, args.num_class, 1)
                    for k in range(args.num_class):
                        y_v_k[:, k, k] = 0
                    y_v_k = y_v_k / torch.sum(y_v_k, dim=2).unsqueeze(2).repeat(1, 1, args.num_class)
                    y_v = Y * pred
                    y_v = y_v / y_v.sum(dim=1).unsqueeze(1).repeat(1, args.num_class)
                    kl = consistency_criterion(F.log_softmax(y_v_k, dim=1), y_v.unsqueeze(1).repeat(1, args.num_class, 1)).sum(2)
                    kl[Y==0] = torch.inf
                    kl_min = torch.min(kl, dim=1)
                    kl[Y==0] = -torch.inf
                    kl_max = torch.max(kl, dim=1)
                    val_idx = torch.logical_and(torch.sum(Y, dim=1) != 1, kl_max[0] > args.kl_theta_labeled)
                    pur_idx = 1-torch.nn.functional.one_hot(kl_min[1], num_classes=args.num_class)
                    pur_idx[torch.logical_not(val_idx)] = 1
                    cand_label = Y - (1 - pur_idx)

                    # add label -- p > class posterior τ
                    cand_label = torch.logical_or(cand_label, torch.logical_and(pred > last_posterior, Y_ori)).to(torch.float)
                    sup_loss_func.update_weight_byclsout1(cls_predicted_score=pred, batch_index=index, batch_partial_Y=cand_label, args=args)
                    labeled_loader.dataset.set_pseudo_cand_labels(cand_label, index, args.ema_theta)

                    err_sup += torch.sum((1-cand_label) * gt)
                    cand_num.update(torch.mean(torch.sum(cand_label, dim=1)), len(cand_label))

                ######## unlabeled #############
                unlabeled_loader.dataset.pre = True
                for i, (img_w, comp_labels_u, true_labels, batch_idx) in enumerate(unlabeled_loader):
                    img_w, comp_labels_u, true_labels, batch_idx = img_w.cuda(), comp_labels_u.cuda(), true_labels.cuda(), batch_idx.cuda()
                    labels_u = (1 - comp_labels_u).cuda()
                    gt = torch.nn.functional.one_hot(true_labels, num_classes=args.num_class).squeeze().cuda()

                    outputs = model(img_q=img_w)
                    pred = torch.softmax(outputs, dim=1)

                    # recode posterior use last model
                    posterior.update(torch.mean(pred, dim=0), len(pred))

                    # init pseudo complementary label (Notice: use complementary label to store in unlabeled)
                    if epoch == args.warm_up:
                        pseudo_complementary_label = (-torch.log(pred) > torch.mean(-torch.log(pred), dim=1, keepdim=True)).to(torch.float)  # CCE -logx
                        unlabeled_loader.dataset.set_pseudo_complementary_labels(pseudo_complementary_label, batch_idx, init=True)
                        comp_labels_u = pseudo_complementary_label
                        labels_u = (1 - comp_labels_u)

                    if args.dataset == 'cifar100':
                        comp_labels_u = (comp_labels_u>0.1).to(torch.float)   # transform hard label
                        labels_u = (1 - comp_labels_u)

                    # remove label -- KL(P(y_v/k|x,w)||P(y_v|x,w))
                    cls = (labels_u * pred).unsqueeze(1).repeat(1, args.num_class, 1)
                    for k in range(args.num_class):
                        cls[:, k, k] = 0
                    y_v_k = cls / torch.sum(cls, dim=2).unsqueeze(2).repeat(1, 1, args.num_class)
                    y_v = labels_u * pred
                    y_v = y_v / y_v.sum(dim=1).unsqueeze(1).repeat(1, args.num_class)
                    kl = consistency_criterion(F.log_softmax(y_v_k, dim=1), y_v.unsqueeze(1).repeat(1, args.num_class, 1)).sum(2)
                    kl[labels_u==0] = torch.inf
                    kl_min = torch.min(kl, dim=1)
                    kl[labels_u==0] = -torch.inf
                    kl_max = torch.max(kl, dim=1)
                    val_idx = torch.logical_and(torch.sum(labels_u, dim=1) != 1, kl_max[0] > args.kl_theta_unlabeled)
                    comp_labels_u[val_idx] = torch.logical_or(comp_labels_u[val_idx],
                                    torch.nn.functional.one_hot(kl_min[1][val_idx], num_classes=args.num_class).squeeze()).to(torch.float)

                    # add label
                    comp_labels_u = torch.logical_and(comp_labels_u, (pred<=last_posterior)).to(torch.float)

                    unlabeled_loader.dataset.set_pseudo_complementary_labels(comp_labels_u, batch_idx)

                    err_un += torch.sum(comp_labels_u * gt)
                    comp_num.update(torch.mean(torch.sum(comp_labels_u, dim=1)), len(comp_labels_u))

                    if (i % 100 == 0) or (i + 1 == len(unlabeled_loader)):
                        logging.info('Epoch:[{0}][{1}/{2}]\t'.format(
                            epoch + 1, i + 1, len(unlabeled_loader)))

                unlabeled_loader.dataset.set_posterior(posterior.avg)
                logging.info('cand num error/all: {}/{}'.format(int(err_sup.item()), len(labeled_loader.dataset)))
                logging.info('avg pseudo cand num: {}'.format(cand_num.avg))
                logging.info('pseudo comp num error/all: {}/{}'.format(int(err_un.item()), len(unlabeled_loader.dataset)))
                logging.info('avg pseudo comp num: {}'.format(comp_num.avg))
                logging.info('pseudo comp num: {}'.format(torch.sum(unlabeled_loader.dataset.pseudo_complementary_labels, dim=0)))
                logging.info('posterior: {}'.format(posterior.avg))
                error_partial = int(err_sup.item())
                partial_cand_num = cand_num.avg.item()
                error_unlabeled = int(err_un.item())
                unlabeled_cand_num = args.num_class - comp_num.avg.item()


        ################ union training #####################
        logging.info('\n=====> Training\n')
        model.train()

        # get labeled and unlabeled data at the same time
        labeled_loader.dataset.pre = False
        unlabeled_loader.dataset.pre = False
        labeled_loader_iter = iter(labeled_loader)
        unlabeled_loader_iter = iter(unlabeled_loader)
        for i in range(args.train_iteration):
            try:
                img_w_l, labels_l, true_labels_l, batch_idx_l = next(labeled_loader_iter)
            except:
                labeled_loader_iter = iter(labeled_loader)
                img_w_l, labels_l, true_labels_l, batch_idx_l = next(labeled_loader_iter)
            try:
                img_s_u, comp_labels_u, true_labels_u, batch_idx_u = next(unlabeled_loader_iter)
            except:
                unlabeled_loader_iter = iter(unlabeled_loader)
                img_s_u, comp_labels_u, true_labels_u, batch_idx_u = next(unlabeled_loader_iter)

            ##### labeled #####
            img_w_l, labels_l, true_labels_l, batch_idx_l = \
                img_w_l.cuda(), labels_l.cuda(), true_labels_l.long().cuda(), batch_idx_l.cuda()

            ##### unlabeled #####
            # only use valid pseudo label (no all 0 and 1)
            comp_label_num = torch.sum(comp_labels_u, dim=1)
            valid_idx = (comp_label_num != 0) * (comp_label_num != args.num_class)
            img_s_u, comp_labels_u, true_labels_u, batch_idx_u = \
                img_s_u[valid_idx].cuda(), comp_labels_u[valid_idx].cuda(), true_labels_u[valid_idx].cuda(), batch_idx_u[valid_idx].cuda()
            labels_u = (1 - comp_labels_u).cuda()

            # model
            input_s = torch.cat((img_w_l, img_s_u), dim=0)
            outputs = model(img_q=input_s)
            pred = torch.softmax(outputs, dim=1)

            k_l = len(labels_l)

            cls_out_w = pred[:k_l]
            un_out_s = pred[k_l:]

            # supervised loss
            sup_loss_func.update_weight_byclsout1(cls_predicted_score=cls_out_w, batch_index=batch_idx_l, batch_partial_Y=labels_l, args=args)
            sup_loss = sup_loss_func(cls_out_w, batch_idx_l)

            loss_sup_log.update(sup_loss.item())
            acc = accuracy(cls_out_w, true_labels_l)[0]
            acc_cls.update(acc[0])

            # partial loss for unlabeled
            un_pseudo_s = labels_u.clone() * un_out_s
            un_pseudo_s = (un_pseudo_s / un_pseudo_s.sum(dim=1).repeat(args.num_class, 1).transpose(0, 1)).detach()
            unsup_loss = -torch.mean(torch.sum(un_pseudo_s * torch.log(un_out_s), dim=1))

            loss_un_log.update(unsup_loss.item())
            acc_c = accuracy(un_pseudo_s, true_labels_u)[0]
            acc_comp.update(acc_c[0])
            acc_all.update((acc[0]+acc_c[0])/2)

            # total loss
            loss = sup_loss + unsup_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all_log.update(loss.item())
            if (i % 50 == 0) or (i + 1 == args.train_iteration):
                logging.info('Epoch:[{0}][{1}/{2}]\t'
                             'Acc_cls {Acc_cls.val:.4f} ({Acc_cls.avg:.4f})\t'
                             'Acc_un {Acc_un.val:.4f} ({Acc_un.avg:.4f})\t'
                             'Acc_all {Acc_all.val:.4f} ({Acc_all.avg:.4f})\n'
                             'L_sup {Loss_sup.val:.4f} ({Loss_sup.avg:.4f})\t'
                             'L_un {L_un.val:.4f} ({L_un.avg:.4f})\t'
                             'L_all {Loss_all.val:.4f} ({Loss_all.avg:.4f})\t'.format(
                    epoch + 1, i + 1, args.train_iteration,
                    Acc_cls=acc_cls, Acc_un=acc_comp, Acc_all=acc_all,
                    Loss_sup=loss_sup_log, L_un=loss_un_log, Loss_all=loss_all_log))

    return acc_all, loss_all_log, error_partial, partial_cand_num, error_unlabeled, unlabeled_cand_num


def test(model, test_loader, args):
    with torch.no_grad():
        logging.info('\n=====> Evaluation...\n')
        model.eval()

        top1_acc = AverageMeter("Top1")
        top5_acc = AverageMeter("Top5")

        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()

            outputs = model(img_q=images)

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])

            if (batch_idx % 10 == 0) or (batch_idx + 1 == len(test_loader)):
                logging.info(
                    'Test:[{0}/{1}]\t'
                    'Prec@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                    'Prec@5 {top5.val:.2f} ({top5.avg:.2f})\t' \
                        .format(batch_idx + 1, len(test_loader), top1=top1_acc, top5=top5_acc)
                )

        acc_tensors = torch.Tensor([top1_acc.avg, top5_acc.avg]).cuda(args.gpu)

        logging.info('\n****** Top1 Accuracy is %.2f%%, Top5 Accuracy is %.2f%% ******\n' \
                     % (acc_tensors[0], acc_tensors[1]))

    return acc_tensors[0]


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


if __name__ == '__main__':
    main()
