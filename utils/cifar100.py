import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .wideResnet import WideResNet
from .utils_algo import *
from .cutout import Cutout
from .autoaugment import CIFAR10Policy


def load_cifar100(args):
    # original training dataset and labels
    original_train = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    original_train.targets = torch.tensor(original_train.targets)

    # testing dataset and dataloader
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], \
                                  std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    ])
    test_dataset = datasets.CIFAR100(root=args.data_dir, train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size*4,
                                              shuffle=False, pin_memory=True, num_workers=args.workers)

    # split dataset into labeled and unlabeled
    print('labeled num: ', args.labeled_num)
    labeled_idx, unlabeled_idx = x_u_split(args, original_train.targets)
    assert torch.sum(torch.bincount(original_train.targets[labeled_idx].long()) == args.labeled_num // args.num_class) == args.num_class

    ori_data = original_train.data[labeled_idx]
    ori_labels = original_train.targets[labeled_idx].long()
    unlabeled_data = original_train.data[unlabeled_idx]
    unlabeled_gt = original_train.targets[unlabeled_idx].long()

    # generate partial label matrix for fine-grained classification
    if args.hierarchical:
        partialY_matrix = generate_hierarchical_cv_candidate_labels('cifar100', ori_labels, args)
    else:
    # generate uniform partial label matrix
        if args.exp_type == 'rand':
            partialY_matrix = generate_uniform_cv_candidate_labels(args, ori_labels)
        elif args.exp_type == 'ins':
            ori_data = torch.Tensor(original_train.data[labeled_idx])
            model = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=0.3)
            model.load_state_dict(torch.load(os.path.join(args.data_dir, './pmodel/cifar100.pt')))
            ori_data = ori_data.permute(0, 3, 1, 2)
            partialY_matrix = generate_instancedependent_candidate_labels(model, ori_data, ori_labels)
            ori_data = original_train.data[labeled_idx]

    true_Y = torch.nn.functional.one_hot(ori_labels, num_classes=args.num_class)
    assert torch.sum(partialY_matrix * true_Y) == partialY_matrix.shape[0]

    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), ori_labels] = 1
    
    if torch.sum(partialY_matrix * temp) == partialY_matrix.shape[0]:
        print('Partial labels correctly loaded !')
    else:
        print('Inconsistent permutation !')

    print('Average candidate num: ', partialY_matrix.sum(1).mean())
    
    # generate partial label dataset
    partial_training_dataset = CIFAR100_Partialize(ori_data, partialY_matrix.float(), ori_labels.float())
    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        # drop_last=True
    )

    # unlabeled training
    if len(unlabeled_data) == 0:
        return partial_training_dataloader, partialY_matrix, None, test_loader
    unlabeled_training_dataset = CIFAR100_Unlabeled(unlabeled_data, unlabeled_gt)
    unlabeled_training_dataloader = torch.utils.data.DataLoader(
        dataset=unlabeled_training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        # drop_last=True
    )
    
    return partial_training_dataloader, partialY_matrix, unlabeled_training_dataloader, test_loader


def x_u_split(args, labels):
    labeled_num = args.labeled_num
    label_per_class = labeled_num // args.num_class
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_class):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == labeled_num

    return labeled_idx, unlabeled_idx


class CIFAR100_Partialize(Dataset):
    def __init__(self, images, given_partial_label_matrix, true_labels):
        
        self.ori_images = images
        self.given_partial_label_matrix = given_partial_label_matrix
        self.pseudo_cand_labels = given_partial_label_matrix
        self.true_labels = true_labels
        self.pre = False

        self.ori_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])
        self.weak_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])
        self.strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),

            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=8),

            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])

    def __len__(self):
        return len(self.true_labels)

    def set_pseudo_cand_labels(self, batch_pseudo_cand_labels, index, ema_theta):
        batch_pseudo_cand_labels = batch_pseudo_cand_labels.cpu()
        index = index.cpu()
        self.pseudo_cand_labels[index] = self.pseudo_cand_labels[index] * ema_theta \
                                         + batch_pseudo_cand_labels * (1-ema_theta)

    def __getitem__(self, index):
        each_label = self.pseudo_cand_labels[index]
        each_true_label = self.true_labels[index]
        if self.pre:
            image = self.weak_transform(self.ori_images[index])
            each_ori_label = self.given_partial_label_matrix[index]
            return image, each_label, each_ori_label, each_true_label, index
        else:
            # image = self.ori_transform(self.ori_images[index])
            # image = self.weak_transform(self.ori_images[index])
            image = self.strong_transform(self.ori_images[index])
            return image, each_label, each_true_label, index


class CIFAR100_Unlabeled(Dataset):
    def __init__(self, images, true_labels):
        self.ori_images = images
        self.pseudo_complementary_labels = torch.zeros(len(true_labels), 100).cuda()
        self.outputs = nn.functional.normalize(torch.randn((len(true_labels), 100)), dim=0)
        self.posterior = (torch.ones(100)/100).to(torch.float).cuda()
        self.true_labels = true_labels
        self.pre = False

        self.ori_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])
        self.weak_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])
        self.strong_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),

            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=8),

            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])

    def __len__(self):
        return len(self.true_labels)

    def set_pseudo_complementary_labels(self, batch_pseudo_complementary_labels, index, init=False):
        # self.pseudo_complementary_labels[index] = batch_pseudo_complementary_labels
        if init:
            self.pseudo_complementary_labels[index] = batch_pseudo_complementary_labels
        else:
            # EMA
            self.pseudo_complementary_labels[index] = self.pseudo_complementary_labels[index] * 0.9 + \
                                                      batch_pseudo_complementary_labels * 0.1

    def set_posterior(self, posterior):
        self.posterior = posterior

    def __getitem__(self, index):
        if self.pre:
            image = self.weak_transform(self.ori_images[index])
        else:
            image = self.strong_transform(self.ori_images[index])
        each_comp_label = self.pseudo_complementary_labels[index].cpu()
        each_true_label = self.true_labels[index]
        return image, each_comp_label, each_true_label, index